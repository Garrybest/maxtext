"""Iterator factories for antllm .scatter/.lazy data in MaxText.

Supports:
  - All three loader modes: sliding_window, map, pack
  - Multi-dataset weighted blending (LazyBlendedDataSource)
  - Two data configuration modes:
    A) Separate train/valid/test file specification
    B) Split-from-train with ratio (e.g. "0.98,0.01,0.01")
"""

from __future__ import annotations

import functools
import math
import os

import jax
import grain.python as grain
import numpy as np

from maxtext.input_pipeline import input_pipeline_utils
from maxtext.input_pipeline._lazy_datasource import MultiShardLazyDataSource
from maxtext.input_pipeline._lazy_blending import LazyBlendedDataSource
from maxtext.input_pipeline.multihost_dataloading import MultiHostDataLoadIterator


# ---------------------------------------------------------------------------
# Config parsing helpers
# ---------------------------------------------------------------------------


def _parse_datasets_and_weights(files_str: str):
  """Parse lazy file specification into (paths, weights).

  Formats:
    "w1 path1 w2 path2 ..."  -> ([path1, path2], [w1, w2])
    "path"                    -> ([path], None)
  """
  if not files_str or not files_str.strip():
    return [], None
  parts = files_str.strip().split()
  if len(parts) == 1:
    return [parts[0]], None
  # Try "weight path weight path ..." format
  try:
    float(parts[0])
    is_weighted = True
  except ValueError:
    is_weighted = False
  if is_weighted:
    if len(parts) % 2 != 0:
      raise ValueError(f"Weighted format expects pairs of 'weight path', got {len(parts)} parts")
    weights = [float(parts[i]) for i in range(0, len(parts), 2)]
    paths = [parts[i] for i in range(1, len(parts), 2)]
    return paths, weights
  return parts, None


def _parse_split(split_str: str) -> list[float] | None:
  """Parse split ratio string like '0.98,0.01,0.01' into [train, valid, test]."""
  if not split_str or not split_str.strip():
    return None
  parts = split_str.replace("/", ",").split(",")
  splits = [float(s) for s in parts]
  total = sum(splits)
  if total < 1.0:
    splits.append(1.0 - total)
  while len(splits) < 3:
    splits.append(0.0)
  splits = splits[:3]
  final_sum = sum(splits)
  return [s / final_sum for s in splits]


def _should_split(split: list[float] | None) -> bool:
  if split is None:
    return False
  return max(split) / sum(split) != 1.0


# ---------------------------------------------------------------------------
# Data source construction
# ---------------------------------------------------------------------------


def _resolve_mode(config, scatter_path: str) -> str:
  """Resolve loader mode for a dataset, supporting per-dataset pack via lazy_bfd_pack.

  Matches antllm's SampleLazyLoader behavior: when bfd_pack is 'ALL' or
  contains the dataset name, the dataset uses 'pack' mode; otherwise
  falls back to 'sliding_window'.  When lazy_bfd_pack is empty, the
  global lazy_loader_mode is used (backward compatible).
  """
  bfd_pack = config.lazy_bfd_pack.strip()
  if not bfd_pack:
    return config.lazy_loader_mode
  ds_name = os.path.basename(scatter_path).replace(".scatter", "")
  if bfd_pack == "ALL" or ds_name in {n.strip() for n in bfd_pack.split(",")}:
    return "pack"
  return "sliding_window"


def _build_source(config, scatter_path: str, stage: str = "train", process_index: int = 0):
  """Build a MultiShardLazyDataSource for a single scatter directory."""
  # antllm's SampleLazyLoaderConfig.__post_init__ does seq_length += 1
  seq_length = config.max_target_length + 1
  mode = _resolve_mode(config, scatter_path)

  return MultiShardLazyDataSource(
      scatter_dir=scatter_path,
      mode=mode,
      seq_length=seq_length,
      eos_token_id=config.lazy_eos_token_id,
      data_type=config.lazy_data_type,
      cls_token_id=config.lazy_cls_token_id if config.lazy_cls_token_id >= 0 else None,
      add_cls=config.lazy_add_cls,
      drop_last=config.lazy_drop_last if stage == "train" else False,
      index_mapping_path=config.lazy_index_mapping_path or None,
      bin_index_base_path=config.lazy_bin_index_path or None,
      loader_online_shuffle=config.lazy_loader_online_shuffle if stage == "train" else False,
      loader_seed=config.lazy_loader_seed,
      loader_scatter=config.lazy_loader_scatter,
      process_index=process_index,
      bfd_pack_sort_by_lens=config.lazy_bfd_pack_sort_by_lens,
      pack_divisible_by=config.lazy_pack_divisible_by,
  )


def _build_blend(config, paths: list[str], weights: list[float] | None, stage: str = "train", process_index: int = 0):
  """Build a blended (or single) data source from dataset paths and weights."""
  data_root = config.lazy_data_root
  sources = []
  for path in paths:
    if data_root:
      scatter_path = f"{data_root}/{path}.scatter"
    else:
      scatter_path = path
    sources.append(_build_source(config, scatter_path, stage, process_index))

  if len(sources) == 1 and weights is None:
    return sources[0]

  if weights is None:
    weights = [1.0] * len(sources)

  weight_mode = config.lazy_dataset_weight_mode
  if stage in ("valid", "test"):
    weight_mode = "epoch"

  if weight_mode == "epoch":
    raw_weights = [w * len(src) for w, src in zip(weights, sources)]
    total = sum(raw_weights)
    size = int(total)
    # antllm configure_data.py:265 normalizes by int(sum), not float sum.
    # Using float sum would cause greedy blending indices to diverge after
    # a few iterations due to float64 precision differences.
    norm_weights = np.array([w / size for w in raw_weights], dtype=np.float64)
  elif weight_mode == "ratio":
    norm_weights = np.array(weights, dtype=np.float64)
    norm_weights = norm_weights / np.sum(norm_weights)
    scatter = max(abs(config.lazy_loader_scatter), 1)
    size = int(math.ceil(config.lazy_data_size_B_tokens * 1e9 / config.max_target_length / scatter))
  else:
    raise ValueError(f"Unknown lazy_dataset_weight_mode: {weight_mode}")

  if size <= 0:
    raise ValueError(f"Computed blend size is {size}, must be positive")

  if len(sources) == 1:
    return sources[0]

  shuffle_seed = config.lazy_blend_shuffle_seed if stage == "train" else -1
  shuffle_only_dataset = config.lazy_blend_shuffle_only_dataset if stage == "train" else False
  cache_dir = config.lazy_blend_cache_dir or None
  return LazyBlendedDataSource(
      sources,
      norm_weights,
      size,
      shuffle_seed=shuffle_seed,
      shuffle_only_dataset=shuffle_only_dataset,
      cache_dir=cache_dir,
      process_index=process_index,
      loader_scatter=config.lazy_loader_scatter,
  )


def _build_pipeline(config, source, global_mesh, process_indices, files_str=""):
  """Wrap a data source in the standard Grain preprocessing pipeline.

  Args:
    files_str: The active dataset file specification (e.g. config.lazy_train_files
      or config.lazy_valid_files). Used to resolve no_attnmask_data dataset names
      to blend indices matching the source's dataset ordering.
  """
  eod_id = config.lazy_eos_token_id
  dataset = grain.MapDataset.source(source)

  # Host sharding — fold by scatter (antllm configure_data.py:311-313)
  scatter = abs(config.lazy_loader_scatter)
  if scatter > 1:
    host_count = len(process_indices) // scatter
    host_index = process_indices.index(jax.process_index()) // scatter
  else:
    host_count = len(process_indices)
    host_index = process_indices.index(jax.process_index())
  dataset = dataset[host_index::host_count]

  # Resolve no_attnmask_data dataset names to blend indices.
  # Only meaningful when reset_attention_mask=True and blending multiple datasets.
  no_attnmask_ids = set()
  no_attnmask_str = getattr(config, "lazy_no_attnmask_data", "")
  if no_attnmask_str and config.reset_attention_mask and files_str:
    no_attnmask_names = {n.strip() for n in no_attnmask_str.split(",") if n.strip()}
    paths, _ = _parse_datasets_and_weights(files_str)
    for idx, path in enumerate(paths):
      ds_name = os.path.basename(path).replace(".scatter", "")
      if ds_name in no_attnmask_names:
        no_attnmask_ids.add(idx)

  # The lazy sources output seq_length+1 tokens (antllm convention).
  # Use MegatronSplitInputsTargets to split into inputs[:-1] / targets[1:].
  dataset = dataset.map(
      input_pipeline_utils.MegatronSplitInputsTargets(
          eod_id=eod_id,
          reset_attention_mask=config.reset_attention_mask,
          eod_mask_loss=config.eod_mask_loss,
          no_attnmask_dataset_ids=no_attnmask_ids if no_attnmask_ids else None,
      )
  )

  batch_size = config.global_batch_size_to_load // jax.process_count()

  mp_options = grain.MultiprocessingOptions(
      num_workers=config.grain_worker_count,
      per_worker_buffer_size=config.grain_per_worker_buffer_size,
  )
  dataset = dataset.mp_prefetch(mp_options)

  batch_fn = functools.partial(grain.experimental.batch_and_pad, batch_size=batch_size, pad_value=eod_id)
  dataset = dataset.batch(batch_size, batch_fn=batch_fn)

  return dataset


# ---------------------------------------------------------------------------
# Split support
# ---------------------------------------------------------------------------


class SplitDataSource:
  """Replicate antllm's SplitDataset for split-from-train mode."""

  def __init__(self, source, start_index: int, end_index: int):
    self._source = source
    self._start = start_index
    self._size = end_index - start_index

  def __len__(self):
    return self._size

  def __getitem__(self, idx):
    return self._source[idx % self._size + self._start]

  def set_num_epochs(self, num_epochs: int):
    """Delegate to the inner source."""
    _set_source_num_epochs(self._source, num_epochs)


def _split_source(source, split_ratios: list[float]):
  """Split a source into train/valid/test by ratio (antllm SplitDataset logic)."""
  total = len(source)
  start_idx = 0
  residual_idx = 0.0
  results = [None, None, None]
  for i, ratio in enumerate(split_ratios):
    if ratio != 0:
      proportion = total * ratio
      residual_idx += proportion % 1
      split_size = int(int(proportion) + residual_idx)
      results[i] = SplitDataSource(source, start_idx, start_idx + max(split_size, 1))
      start_idx += split_size
      residual_idx %= 1
  return results


# ---------------------------------------------------------------------------
# Epoch computation
# ---------------------------------------------------------------------------


def _compute_num_epochs(config, total_samples: int, stage: str = "train") -> int:
  """Compute the number of epochs the data source should expose.

  Determines how many times the full dataset needs to be repeated so that
  Grain can iterate enough indices to cover all training steps.

  Resolution order (for train stage):
    1. config.num_epoch > 1  → use it directly (user override).
    2. config.steps > 0      → auto-compute from steps and per-group
       consumption rate:
         scatter = max(abs(loader_scatter), 1)
         consumed_per_group = steps * (global_batch_size / scatter)
         num_epochs = ceil(consumed_per_group / total_samples) + 1
       The +1 provides a safety margin: when shards are unevenly sized
       across scatter groups, the smallest group may exhaust its samples
       earlier than the average estimate predicts.
    3. Otherwise             → 1 (single epoch, Grain stops at the end).

  For eval stages, always returns 1.

  Why per-group instead of global:
    With loader_scatter, the global_batch_size is distributed evenly across
    abs(scatter) groups. Each group independently consumes its own subset
    of shards. ``total_samples`` here is the count for *this* group (after
    scatter shard filtering), so the consumption rate must also be per-group:
      samples_per_step_per_group = global_batch_size / abs(scatter)

  Example:
    total_samples=50_000 (this group), steps=10_000, global_batch_size=32,
    loader_scatter=-4
    → per_group_rate = 32/4 = 8
    → consumed = 10_000 * 8 = 80_000
    → num_epochs = ceil(80_000/50_000) + 1 = 3
  """
  if stage != "train":
    return 1

  # User-specified num_epoch takes precedence
  if config.num_epoch > 1:
    return config.num_epoch

  # Auto-compute from training steps
  if getattr(config, "steps", 0) > 0 and total_samples > 0:
    scatter = max(abs(config.lazy_loader_scatter), 1)
    # Each scatter group consumes global_batch / scatter samples per step
    consumed_per_group = config.steps * (config.global_batch_size_to_load // scatter)
    # +1 safety margin for uneven shard sizes across scatter groups
    return max(1, math.ceil(consumed_per_group / total_samples) + 1)

  return 1


def _set_source_num_epochs(source, num_epochs: int):
  """Set num_epochs on the outermost data source.

  After the source is fully constructed (blending + split), we know the
  real total_samples and can compute the correct num_epochs.  This function
  calls set_num_epochs() on the **outermost** source so that Grain sees
  the inflated __len__ and iterates enough indices for multi-epoch training.

  Key insight: epoch inflation must happen at the outermost level that
  Grain directly iterates.  Setting it on sub-sources inside a blend is
  useless because Grain only sees the blend's __len__.

  For blended sources, the blend epoch is propagated to sub-sources via
  sample_idx offset in LazyBlendedDataSource.__getitem__, which triggers
  per-epoch shuffle rotation in sub-sources with loader_online_shuffle.
  """
  if num_epochs <= 1:
    return

  if hasattr(source, "set_num_epochs"):
    source.set_num_epochs(num_epochs)


# ---------------------------------------------------------------------------
# Public iterator factories
# ---------------------------------------------------------------------------


def make_lazy_train_iterator(config, global_mesh, process_indices):
  """Create a training data iterator from .scatter/.lazy data."""
  paths, weights = _parse_datasets_and_weights(config.lazy_train_files)
  if not paths:
    raise ValueError("lazy_train_files must be specified for dataset_type='lazy'")

  process_index = jax.process_index()

  source = _build_blend(config, paths, weights, "train", process_index)

  split = _parse_split(config.lazy_split)
  if _should_split(split):
    # If separate valid/test files are specified, zero out those split ratios
    if config.lazy_valid_files:
      split[1] = 0.0
    if config.lazy_test_files:
      split[2] = 0.0
    final_sum = sum(split)
    split = [s / final_sum for s in split]
    parts = _split_source(source, split)
    if parts[0] is None:
      raise ValueError("Split resulted in empty train partition")
    source = parts[0]

  # Compute and apply num_epochs after the source is fully constructed (incl.
  # blending + split), so we have the accurate total_samples for the formula.
  # _total_samples is the single-epoch sample count; _num_epochs inflates
  # __len__ so Grain iterates enough indices for the full training run.
  num_epochs = _compute_num_epochs(config, len(source), "train")
  _set_source_num_epochs(source, num_epochs)

  dataset = _build_pipeline(config, source, global_mesh, process_indices, files_str=config.lazy_train_files)
  return MultiHostDataLoadIterator(dataset, global_mesh, config.generate_padding_batch_train)


def make_lazy_eval_iterator(config, global_mesh, process_indices):
  """Create an eval data iterator from .scatter/.lazy data."""
  # Mode A: separate eval files
  process_index = jax.process_index()
  if config.lazy_valid_files:
    paths, weights = _parse_datasets_and_weights(config.lazy_valid_files)
    source = _build_blend(config, paths, weights, "valid", process_index)
    dataset = _build_pipeline(config, source, global_mesh, process_indices, files_str=config.lazy_valid_files)
    return MultiHostDataLoadIterator(dataset, global_mesh, config.generate_padding_batch_eval)

  # Mode B: split-from-train
  split = _parse_split(config.lazy_split)
  if _should_split(split) and split[1] > 0:
    paths, weights = _parse_datasets_and_weights(config.lazy_train_files)
    if not paths:
      return None
    source = _build_blend(config, paths, weights, "train", process_index)
    parts = _split_source(source, split)
    if parts[1] is None:
      return None
    dataset = _build_pipeline(config, parts[1], global_mesh, process_indices, files_str=config.lazy_train_files)
    return MultiHostDataLoadIterator(dataset, global_mesh, config.generate_padding_batch_eval)

  return None
