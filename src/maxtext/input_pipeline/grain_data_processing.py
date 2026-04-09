# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input pipeline using Grain."""

import glob
from pathlib import Path
import functools
import ml_collections
from concurrent import futures
import json

import jax

from grain.experimental import BestFitPackIterDataset, pick_performance_config
import grain.python as grain

from maxtext.input_pipeline import input_pipeline_utils
from maxtext.input_pipeline import grain_tokenizer
from maxtext.input_pipeline import multihost_dataloading
from maxtext.input_pipeline import tokenizer
from maxtext.input_pipeline._mmap_datasource import MMapDatasetConfig, get_mmap_dataset, get_mmap_npy_dataset
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging


def find_data_files(data_file_pattern):
  """Find data files matching the pattern."""
  if data_file_pattern.startswith("gs://"):
    data_files = gcs_utils.gcs_glob_pattern(data_file_pattern)
  else:
    # Local files
    data_files = glob.glob(str(Path(data_file_pattern).expanduser().resolve()))
  if not data_files:
    raise FileNotFoundError(f"No files found matching pattern: {data_file_pattern}")
  max_logging.log(f"Found {len(data_files)} files for train/eval with grain")
  return data_files


def _apply_mapdataset_transforms(
    dataset,
    shuffle,
    shuffle_seed,
    num_epoch,
    dataloading_host_index,
    dataloading_host_count,
    grain_num_threads,
    grain_prefetch_buffer_size,
):
  """Apply standard shuffle, repeat, shard, and iter conversion transforms."""
  if shuffle:
    dataset = dataset.shuffle(seed=shuffle_seed)
  dataset = dataset.repeat(num_epoch)
  dataset = dataset[dataloading_host_index::dataloading_host_count]  # sharding
  dataset = dataset.to_iter_dataset(
      read_options=grain.ReadOptions(
          num_threads=grain_num_threads,
          prefetch_buffer_size=grain_prefetch_buffer_size,
      )
  )
  return dataset


def _build_dataset_config(config, num_samples=None, seed=1234, split_ratio=None, split_index=0):
  """Build format-specific dataset config from global config.

  Returns MMapDatasetConfig for mmap/mmap_npy formats, None otherwise.
  """
  if config.grain_file_type not in ("mmap", "mmap_npy"):
    return None
  return MMapDatasetConfig(
      max_target_length=config.max_target_length,
      eod_id=config.mmap_eod_id,
      mmap_split_sentences=config.mmap_split_sentences,
      blend_cache_dir=config.blend_cache_dir,
      blend_index_dir=config.blend_index_dir,
      num_samples=num_samples,
      seed=seed,
      split_ratio=split_ratio,
      split_index=split_index,
  )


def get_datasets(
    data_file_pattern,
    data_file_type,
    shuffle,
    shuffle_seed,
    shuffle_buffer_size,
    num_epoch,
    dataloading_host_index,
    dataloading_host_count,
    grain_worker_count,
    grain_num_threads,
    grain_prefetch_buffer_size,
    grain_data_source_max_workers,
    mixture_config_path=None,
    dataset_config=None,
    split="train",
):
  """Load dataset from array_record files for using with grain

  Args:
      dataset_config: Optional config object providing format-specific dataset
          parameters (eod_id, max_target_length, mmap_split_sentences,
          blend_cache_dir, blend_index_dir, num_samples, seed, split_ratio,
          split_index).  Required for mmap / mmap_npy.
      split: 'train' or 'eval', used by mmap_npy to select the blend split.
  """
  if data_file_type == "arrayrecord":
    # Helper function to find files, create data source, and wrap in MapDataset
    def create_dataset_from_pattern(pattern):
      files = find_data_files(pattern)
      source = grain.ArrayRecordDataSource(files)
      return grain.MapDataset.source(source)

    # Handle mixture config with named datasets, allows flexibility in recovering checkpoints
    if mixture_config_path:
      with open(mixture_config_path, "r", encoding="utf-8") as f:
        mixture_config = json.load(f)

      paths = [config["path"] for config in mixture_config.values()]
      weights = [float(config["weight"]) for config in mixture_config.values()]

      executor = futures.ThreadPoolExecutor(max_workers=grain_data_source_max_workers)
      dataset_list = list(executor.map(create_dataset_from_pattern, paths))
      executor.shutdown(wait=True)

      datasets_dict = dict(zip(mixture_config.keys(), dataset_list))

      for name, ds in datasets_dict.items():
        datasets_dict[name] = _apply_mapdataset_transforms(
            ds,
            shuffle,
            shuffle_seed,
            num_epoch,
            dataloading_host_index,
            dataloading_host_count,
            grain_num_threads,
            grain_prefetch_buffer_size,
        )

      # Normalize weights
      total_weight = sum(weights)
      weights_dict = {name: weight / total_weight for name, weight in zip(mixture_config.keys(), weights)}

      dataset = grain.IterDataset.mix(datasets_dict, weights_dict)
      return dataset
    elif ";" in data_file_pattern:
      data_file_patterns, weights = zip(*[pattern.split(",") for pattern in data_file_pattern.split(";")])
      assert len(data_file_patterns) == len(weights), "Number of data file patterns and weights must match"
      weights = [float(weight) for weight in weights]
      weights = [round(weight / sum(weights), 4) for weight in weights]

      # Parallelize file finding (globbing), data source creation, and dataset wrapping
      # File finding and source creation are I/O-bound operations that release the GIL
      executor = futures.ThreadPoolExecutor(max_workers=grain_data_source_max_workers)
      dataset_list = list(executor.map(create_dataset_from_pattern, data_file_patterns))
      executor.shutdown(wait=True)

      # Apply shuffle, repeat, sharding, and conversion to IterDataset to each dataset before mixing
      for d, _ in enumerate(dataset_list):
        dataset_list[d] = _apply_mapdataset_transforms(
            dataset_list[d],
            shuffle,
            shuffle_seed,
            num_epoch,
            dataloading_host_index,
            dataloading_host_count,
            grain_num_threads,
            grain_prefetch_buffer_size,
        )
      # Use IterDataset.mix instead of MapDataset.mix in order to have per-mixture component checkpoints
      # for supporting changing the mixture after checkpointing
      dataset = grain.IterDataset.mix(dataset_list, weights)
      return dataset
    else:
      # Single pattern case - no need for parallelization
      dataset = create_dataset_from_pattern(data_file_pattern)
      dataset = _apply_mapdataset_transforms(
          dataset,
          shuffle,
          shuffle_seed,
          num_epoch,
          dataloading_host_index,
          dataloading_host_count,
          grain_num_threads,
          grain_prefetch_buffer_size,
      )
      return dataset
  elif data_file_type == "tfrecord":
    data_files = find_data_files(data_file_pattern)
    dataset = grain.MapDataset.source(data_files)
    if shuffle:
      dataset = dataset.shuffle(seed=shuffle_seed)
    dataset = dataset.repeat(num_epoch)
    dataset = dataset[dataloading_host_index::dataloading_host_count]  # sharding
    dataset = dataset.map(input_pipeline_utils.make_tfrecord_iter_dataset)
    files_per_host = max(len(data_files) // dataloading_host_count, 1)
    cycle_length = min(files_per_host, grain_num_threads)
    dataset = grain.experimental.InterleaveIterDataset(dataset, cycle_length=cycle_length)
    if shuffle:
      dataset = grain.experimental.WindowShuffleIterDataset(dataset, window_size=shuffle_buffer_size, seed=shuffle_seed)
    return dataset
  elif data_file_type == "parquet":
    data_files = find_data_files(data_file_pattern)
    dataset = grain.MapDataset.source(data_files)
    if shuffle:
      dataset = dataset.shuffle(seed=shuffle_seed)
    dataset = dataset.repeat(num_epoch)
    dataset = dataset[dataloading_host_index::dataloading_host_count]  # sharding
    assert grain_worker_count <= len(dataset), (
        f"grain worker count is currently {grain_worker_count}, exceeding the max allowable value {len(dataset)} "
        f"(file shard count of a data loading host) for your dataset. "
        f"Please lower grain_worker_count or increase file shard count."
    )
    dataset = dataset.map(grain.experimental.ParquetIterDataset)
    cycle_length = min(len(dataset) // num_epoch, grain_num_threads)
    dataset = grain.experimental.InterleaveIterDataset(dataset, cycle_length=cycle_length)
    if shuffle:
      dataset = grain.experimental.WindowShuffleIterDataset(dataset, window_size=shuffle_buffer_size, seed=shuffle_seed)
    return dataset
  elif data_file_type == "mmap":
    return get_mmap_dataset(
        data_file_pattern,
        dataset_config.mmap_split_sentences,
        dataset_config.max_target_length,
        dataset_config.eod_id,
        shuffle,
        shuffle_seed,
        num_epoch,
        dataloading_host_index,
        dataloading_host_count,
        grain_num_threads,
        grain_prefetch_buffer_size,
        apply_transforms=_apply_mapdataset_transforms,
    )
  elif data_file_type == "mmap_npy":
    return get_mmap_npy_dataset(
        data_file_pattern,
        dataset_config.mmap_split_sentences,
        dataset_config.max_target_length,
        dataset_config.eod_id,
        num_epoch,
        dataloading_host_index,
        dataloading_host_count,
        grain_num_threads,
        grain_prefetch_buffer_size,
        dataset_config.blend_cache_dir or None,
        dataset_config.blend_index_dir or None,
        split,
        apply_transforms=_apply_mapdataset_transforms,
        num_samples=dataset_config.num_samples,
        seed=dataset_config.seed,
        split=dataset_config.split_ratio,
        split_index=dataset_config.split_index,
    )
  else:
    raise ValueError(
        f"grain pipeline supports (arrayrecord, tfrecord, parquet, mmap, mmap_npy) as grain_file_type, "
        f"but got {data_file_type}"
    )


def _make_multiprocessing_options(dataset, config, grain_worker_count, grain_per_worker_buffer_size):
  """Build Grain multiprocessing options for mp_prefetch."""
  return (
      pick_performance_config(
          ds=dataset,
          ram_budget_mb=config.grain_ram_budget_mb,
          max_workers=None,
          max_buffer_size=None,
      ).multiprocessing_options
      if grain_worker_count == -1
      else grain.MultiprocessingOptions(
          num_workers=grain_worker_count,
          per_worker_buffer_size=grain_per_worker_buffer_size,
      )
  )


def _standard_pretrain_pipeline(
    dataset,
    config,
    text_column,
    tokenize,
    grain_worker_count,
    grain_per_worker_buffer_size,
):
  """Standard pretrain pipeline for arrayrecord / tfrecord / parquet formats."""
  tokenizer_model = tokenizer.build_tokenizer(
      config.tokenizer_path,
      config.tokenizer_type,
      config.add_bos,
      config.add_eos,
      config.hf_access_token,
  )
  if tokenizer_model.pad_id is not None:
    pad_id = tokenizer_model.pad_id
  elif tokenizer_model.unk_id is not None:
    pad_id = tokenizer_model.unk_id
  else:
    pad_id = -1

  if tokenize:
    if config.use_truncation:
      dataset = dataset.map(grain_tokenizer.TokenizeAndTrim(text_column, config.max_target_length, tokenizer_model))
    else:
      dataset = dataset.apply(grain_tokenizer.TokenizeAndChunk(text_column, config.max_target_length, tokenizer_model))

  data_columns = ("inputs", "targets")
  rekey_dict = {col: text_column for col in data_columns}
  dataset = dataset.map(input_pipeline_utils.Rekey(rekey_dict))

  # Pack and Batch examples.
  batch_size = config.global_batch_size_to_load // jax.process_count()
  if config.expansion_factor_real_data > 1:
    # global_batch_size_to_load has been expanded in pyconfig.py when expansion_factor_real_data > 1.
    # But when using Grain, we want to keep the batch_size consistent with that in the checkpoint.
    # We revert the batch_size expansion here, but load multiple batches per step in multihost_dataloading.py.
    batch_size = int(batch_size // config.expansion_factor_real_data)

  if config.packing:
    length_struct = {col: config.max_target_length for col in data_columns}
    max_segments = config.max_segments_per_seq
    if max_segments is not None and max_segments <= 0:
      max_segments = None
    if config.grain_packing_type == "first_fit":
      dataset = grain.experimental.FirstFitPackIterDataset(
          dataset,
          length_struct=length_struct,
          num_packing_bins=batch_size,
          max_sequences_per_bin=max_segments,
      )
    elif config.grain_packing_type == "best_fit":
      dataset = BestFitPackIterDataset(dataset, length_struct=length_struct, num_packing_bins=batch_size)
    elif config.grain_packing_type == "concat_then_split":
      if config.add_bos and hasattr(tokenizer_model, "bos_id"):
        dataset = grain.experimental.ConcatThenSplitIterDataset(
            dataset,
            length_struct=length_struct,
            bos_handling=grain.experimental.BOSHandling.REPLACE_FIRST_TOKEN_WITH_BOS,
            bos_token_id=tokenizer_model.bos_id,
        )
      else:
        dataset = grain.experimental.ConcatThenSplitIterDataset(dataset, length_struct=length_struct)
    else:
      raise ValueError(f"Unknown packing type: {config.packing}")

    rekey_dict = {
        "targets_segmentation": "targets_segment_ids",
        "inputs_segmentation": "inputs_segment_ids",
        "targets_position": "targets_positions",
        "inputs_position": "inputs_positions",
    }
    dataset = dataset.map(input_pipeline_utils.Rekey(rekey_dict))
  else:
    dataset = dataset.map(input_pipeline_utils.PadOrTrimToMaxLength(config.max_target_length, pad_id))

  batch_fn = functools.partial(grain.experimental.batch_and_pad, batch_size=batch_size, pad_value=pad_id)
  dataset = dataset.batch(batch_size, batch_fn=batch_fn)

  dataset = dataset.map(
      input_pipeline_utils.ShiftData(
          ignored_ids=[pad_id],
          axis=1,
      )
  )
  mp_options = _make_multiprocessing_options(dataset, config, grain_worker_count, grain_per_worker_buffer_size)
  dataset = dataset.mp_prefetch(mp_options)
  return dataset


def _mmap_pretrain_pipeline(
    dataset,
    config,
    text_column,
    grain_worker_count,
    grain_per_worker_buffer_size,
):
  """Pretrain pipeline for Megatron-compatible mmap / mmap_npy pre-tokenized formats."""
  eod_id = config.mmap_eod_id
  is_npy = config.grain_file_type == "mmap_npy"

  # Split or rekey
  if is_npy:
    # MegatronNpyDataSource returns seq_length+1 tokens; split into
    # inputs[:-1] / targets[1:] with EOD-aware segmentation.
    dataset = dataset.map(
        input_pipeline_utils.MegatronSplitInputsTargets(
            eod_id=eod_id,
            reset_attention_mask=config.reset_attention_mask,
            eod_mask_loss=config.eod_mask_loss,
        )
    )
  else:
    data_columns = ("inputs", "targets")
    rekey_dict = {col: text_column for col in data_columns}
    dataset = dataset.map(input_pipeline_utils.Rekey(rekey_dict))
    # Samples are already exactly max_target_length with EOD tokens between
    # documents.  Generate doc-boundary-aware segmentation, skip packing.
    dataset = dataset.map(
        input_pipeline_utils.GenerateDocSegmentIds(
            eod_id=eod_id,
            reset_attention_mask=config.reset_attention_mask,
            eod_mask_loss=config.eod_mask_loss,
        )
    )

  batch_size = config.global_batch_size_to_load // jax.process_count()
  if config.expansion_factor_real_data > 1:
    # global_batch_size_to_load has been expanded in pyconfig.py when expansion_factor_real_data > 1.
    # But when using Grain, we want to keep the batch_size consistent with that in the checkpoint.
    # We revert the batch_size expansion here, but load multiple batches per step in multihost_dataloading.py.
    batch_size = int(batch_size // config.expansion_factor_real_data)

  mp_options = _make_multiprocessing_options(dataset, config, grain_worker_count, grain_per_worker_buffer_size)

  # mmap_npy: mp_prefetch BEFORE batch to preserve sample ordering in
  # blend-then-shard.  Grain's mp_prefetch does per-worker sharding; placing
  # it before batch ensures each worker processes a contiguous chunk and the
  # round-robin interleaver reconstructs global order.
  if is_npy:
    dataset = dataset.mp_prefetch(mp_options)

  batch_fn = functools.partial(grain.experimental.batch_and_pad, batch_size=batch_size, pad_value=eod_id)
  dataset = dataset.batch(batch_size, batch_fn=batch_fn)

  # mmap: shift needed (Rekey produced identical inputs/targets);
  # mmap_npy: skip (MegatronSplitInputsTargets already split).
  if not is_npy:
    if not config.eod_mask_loss:
      max_logging.log(
          "WARNING: mmap mode with eod_mask_loss=False uses mmap_eod_id as both "
          "padding and EOD sentinel. ShiftData will zero targets_segmentation "
          "at all EOD positions, effectively masking EOD from loss regardless "
          "of eod_mask_loss. Use mmap_npy mode for correct eod_mask_loss=False behavior."
      )
    dataset = dataset.map(input_pipeline_utils.ShiftData(ignored_ids=[eod_id], axis=1))
    dataset = dataset.mp_prefetch(mp_options)

  return dataset


def pretrain_preprocessing_pipeline(
    dataset,
    config,
    data_columns,
    tokenize,
    grain_worker_count,
    grain_per_worker_buffer_size,
):
  """Use grain pipeline to pre-process the dataset and return iterators for pretrain"""
  if config.grain_file_type in ("arrayrecord", "tfrecord"):
    dataset = dataset.map(input_pipeline_utils.ParseFeatures(data_columns, tokenize))
    dataset = dataset.map(input_pipeline_utils.NormalizeFeatures(data_columns, tokenize))
  else:
    dataset = dataset.map(input_pipeline_utils.KeepFeatures(feature_names=data_columns))

  assert len(data_columns) == 1
  text_column = data_columns[0]

  if config.grain_file_type in ("mmap", "mmap_npy"):
    if tokenize:
      max_logging.log(
          f"grain_file_type='{config.grain_file_type}' implies pre-tokenized data; overriding tokenize to False"
      )
    return _mmap_pretrain_pipeline(dataset, config, text_column, grain_worker_count, grain_per_worker_buffer_size)

  return _standard_pretrain_pipeline(
      dataset,
      config,
      text_column,
      tokenize,
      grain_worker_count,
      grain_per_worker_buffer_size,
  )


def dpo_preprocessing_pipeline(
    dataset,
    config,
    data_columns,
    tokenize,
    grain_worker_count,
    grain_per_worker_buffer_size,
):
  """Use grain to pre-process the dataset and return iterators for dpo fine-tuning"""
  if config.grain_file_type in ("arrayrecord", "tfrecord"):
    dataset = dataset.map(input_pipeline_utils.ParseFeatures(data_columns, tokenize))
    dataset = dataset.map(input_pipeline_utils.NormalizeFeatures(data_columns, tokenize))
  tokenizer_model = tokenizer.build_tokenizer(
      config.tokenizer_path,
      config.tokenizer_type,
      config.add_bos,
      config.add_eos,
      config.hf_access_token,
  )
  if tokenizer_model.pad_id is not None:
    pad_id = tokenizer_model.pad_id
  elif tokenizer_model.unk_id is not None:
    pad_id = tokenizer_model.unk_id
  else:
    pad_id = -1

  if tokenize:
    dataset = dataset.map(grain_tokenizer.TokenizeAndTrim(data_columns, config.max_target_length, tokenizer_model))

  dataset = dataset.map(input_pipeline_utils.PadOrTrimToMaxLength(config.max_target_length, pad_id))
  batch_size = config.global_batch_size_to_load // jax.process_count()
  batch_fn = functools.partial(grain.experimental.batch_and_pad, batch_size=batch_size, pad_value=pad_id)
  dataset = dataset.batch(batch_size, batch_fn=batch_fn)
  multiprocessing_options = (
      pick_performance_config(
          ds=dataset,
          ram_budget_mb=config.grain_ram_budget_mb,
          max_workers=None,
          max_buffer_size=None,
      ).multiprocessing_options
      if grain_worker_count == -1
      else grain.MultiprocessingOptions(
          num_workers=grain_worker_count,
          per_worker_buffer_size=grain_per_worker_buffer_size,
      )
  )
  dataset = dataset.mp_prefetch(multiprocessing_options)
  return dataset


def make_grain_train_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices,
):
  """Load, preprocess dataset and return iterators"""
  assert (
      config.global_batch_size_to_load % global_mesh.size == 0
  ), "Batch size should be divisible by number of global devices."
  # For mmap_npy: compute num_samples for auto-rebuild of npy indices.
  # When steps > 0, the index builder can derive the correct num_epochs
  # from num_samples, enabling automatic cache management.
  mmap_npy_num_samples = None
  if config.grain_file_type == "mmap_npy" and getattr(config, "steps", 0) > 0:
    mmap_npy_num_samples = config.steps * config.global_batch_size_to_load
  dataset_config = _build_dataset_config(
      config,
      num_samples=mmap_npy_num_samples,
      seed=config.data_shuffle_seed,
      split_ratio=config.mmap_npy_split or None,
      split_index=0,
  )
  if not config.colocated_python_data_input and not 0 < config.expansion_factor_real_data < 1:
    train_ds = get_datasets(
        config.grain_train_files,
        config.grain_file_type,
        shuffle=config.enable_data_shuffling,
        shuffle_seed=config.data_shuffle_seed,
        shuffle_buffer_size=config.grain_shuffle_buffer_size,
        num_epoch=config.num_epoch,
        dataloading_host_index=process_indices.index(jax.process_index()),
        dataloading_host_count=len(process_indices),
        grain_worker_count=config.grain_worker_count,
        grain_num_threads=config.grain_num_threads,
        grain_prefetch_buffer_size=config.grain_prefetch_buffer_size,
        grain_data_source_max_workers=config.grain_data_source_max_workers,
        mixture_config_path=config.grain_train_mixture_config_path,
        dataset_config=dataset_config,
    )
    if config.use_dpo:
      train_dataloader = dpo_preprocessing_pipeline(
          train_ds,
          config,
          data_columns=config.train_data_columns,
          tokenize=config.tokenize_train_data,
          grain_worker_count=config.grain_worker_count,
          grain_per_worker_buffer_size=config.grain_per_worker_buffer_size,
      )
    else:
      train_dataloader = pretrain_preprocessing_pipeline(
          train_ds,
          config,
          data_columns=config.train_data_columns,
          tokenize=config.tokenize_train_data,
          grain_worker_count=config.grain_worker_count,
          grain_per_worker_buffer_size=config.grain_per_worker_buffer_size,
      )
    return multihost_dataloading.MultiHostDataLoadIterator(
        train_dataloader,
        global_mesh,
        config.generate_padding_batch_train,
        expansion_loading_factor_for_grain=config.expansion_factor_real_data,
    )
  else:
    get_ds_fn = functools.partial(
        get_datasets,
        config.grain_train_files,
        config.grain_file_type,
        shuffle=config.enable_data_shuffling,
        shuffle_seed=config.data_shuffle_seed,
        shuffle_buffer_size=config.grain_shuffle_buffer_size,
        num_epoch=config.num_epoch,
        grain_worker_count=config.grain_worker_count,
        grain_num_threads=config.grain_num_threads,
        grain_prefetch_buffer_size=config.grain_prefetch_buffer_size,
        grain_data_source_max_workers=config.grain_data_source_max_workers,
        dataset_config=dataset_config,
    )
    if config.use_dpo:
      preprocessing_fn = functools.partial(
          pretrain_preprocessing_pipeline,
          config=config,
          data_columns=config.train_data_columns,
          tokenize=config.tokenize_train_data,
          grain_worker_count=config.grain_worker_count,
          grain_per_worker_buffer_size=config.grain_per_worker_buffer_size,
      )
    else:
      preprocessing_fn = functools.partial(
          pretrain_preprocessing_pipeline,
          config=config,
          data_columns=config.train_data_columns,
          tokenize=config.tokenize_train_data,
          grain_worker_count=config.grain_worker_count,
          grain_per_worker_buffer_size=config.grain_per_worker_buffer_size,
      )
    if config.colocated_python_data_input:
      global_shape = (config.global_batch_size_to_load, config.max_target_length)
      return multihost_dataloading.RemoteIterator(get_ds_fn, preprocessing_fn, global_mesh, global_shape)
    else:
      # config.expansion_factor_real_data is between 0 and 1
      num_dataloader_to_restore = int(1 / config.expansion_factor_real_data)
      train_dataloader_list = []
      dataloading_host_count = len(process_indices) * num_dataloader_to_restore
      for i in range(num_dataloader_to_restore):
        dataloading_host_index = len(process_indices) * i + process_indices.index(jax.process_index())
        train_ds = get_ds_fn(dataloading_host_index=dataloading_host_index, dataloading_host_count=dataloading_host_count)
        train_dataloader = preprocessing_fn(train_ds)
        train_dataloader_list.append(train_dataloader)
      return [
          multihost_dataloading.MultiHostDataLoadIterator(x, global_mesh, config.generate_padding_batch_train)
          for x in train_dataloader_list
      ]


def make_grain_eval_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices,
):
  """Load, preprocess dataset and return iterators"""
  assert (
      config.global_batch_size_to_load_eval % global_mesh.size == 0
  ), "Batch size should be divisible by number of global devices."
  # Eval blend must cover ALL eval rounds over the entire training run.
  # Megatron formula: ceil(steps / eval_interval) * eval_iters * global_batch.
  mmap_npy_eval_num_samples = None
  if config.grain_file_type == "mmap_npy" and hasattr(config, "eval_steps") and config.eval_steps > 0:
    eval_interval = getattr(config, "eval_interval", 0)
    if eval_interval > 0 and hasattr(config, "steps") and config.steps > 0:
      eval_rounds = -(-config.steps // eval_interval)  # ceil division
    else:
      eval_rounds = 1
    mmap_npy_eval_num_samples = eval_rounds * config.eval_steps * config.global_batch_size_to_load
  dataset_config = _build_dataset_config(
      config,
      num_samples=mmap_npy_eval_num_samples,
      seed=config.data_shuffle_seed,
      split_ratio=config.mmap_npy_split or None,
      split_index=1 if config.mmap_npy_split else 0,
  )

  if not config.colocated_python_data_input:
    eval_ds = get_datasets(
        config.grain_eval_files,
        config.grain_file_type,
        shuffle=False,
        shuffle_seed=config.data_shuffle_seed,
        shuffle_buffer_size=config.grain_shuffle_buffer_size,
        num_epoch=1,
        dataloading_host_index=process_indices.index(jax.process_index()),
        dataloading_host_count=len(process_indices),
        grain_worker_count=config.grain_worker_count_eval,
        grain_num_threads=config.grain_num_threads_eval,
        grain_prefetch_buffer_size=config.grain_prefetch_buffer_size_eval,
        grain_data_source_max_workers=config.grain_data_source_max_workers,
        dataset_config=dataset_config,
        split="eval",
    )
    if config.use_dpo:
      eval_dataloader = dpo_preprocessing_pipeline(
          eval_ds,
          config,
          data_columns=config.eval_data_columns,
          tokenize=config.tokenize_eval_data,
          grain_worker_count=config.grain_worker_count_eval,
          grain_per_worker_buffer_size=config.grain_per_worker_buffer_size_eval,
      )
    else:
      eval_dataloader = pretrain_preprocessing_pipeline(
          eval_ds,
          config,
          data_columns=config.eval_data_columns,
          tokenize=config.tokenize_eval_data,
          grain_worker_count=config.grain_worker_count_eval,
          grain_per_worker_buffer_size=config.grain_per_worker_buffer_size_eval,
      )
    return multihost_dataloading.MultiHostDataLoadIterator(
        eval_dataloader, global_mesh, config.generate_padding_batch_eval
    )
  else:
    get_ds_fn = functools.partial(
        get_datasets,
        config.grain_eval_files,
        config.grain_file_type,
        shuffle=False,  # No shuffle for eval
        shuffle_seed=config.data_shuffle_seed,
        shuffle_buffer_size=config.grain_shuffle_buffer_size,
        num_epoch=1,
        grain_worker_count=config.grain_worker_count_eval,
        grain_num_threads=config.grain_num_threads_eval,
        grain_prefetch_buffer_size=config.grain_prefetch_buffer_size_eval,
        grain_data_source_max_workers=config.grain_data_source_max_workers,
        dataset_config=dataset_config,
        split="eval",
    )
    if config.use_dpo:
      preprocessing_fn = functools.partial(
          dpo_preprocessing_pipeline,
          config=config,
          data_columns=config.eval_data_columns,
          tokenize=config.tokenize_eval_data,
          grain_worker_count=config.grain_worker_count_eval,
          grain_per_worker_buffer_size=config.grain_per_worker_buffer_size_eval,
      )
    else:
      preprocessing_fn = functools.partial(
          pretrain_preprocessing_pipeline,
          config=config,
          data_columns=config.eval_data_columns,
          tokenize=config.tokenize_eval_data,
          grain_worker_count=config.grain_worker_count_eval,
          grain_per_worker_buffer_size=config.grain_per_worker_buffer_size_eval,
      )
    global_shape = (config.global_batch_size_to_load, config.max_target_length)
    return multihost_dataloading.RemoteIterator(get_ds_fn, preprocessing_fn, global_mesh, global_shape)
