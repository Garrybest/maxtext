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

"""Operations used by Grain"""

import dataclasses
import warnings
from threading import current_thread
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
  import datasets
  import tensorflow as tf

import grain.python as grain
import numpy as np
from grain._src.python.dataset.sources.tfrecord_dataset import _TFRecordReader, _TFRecordDatasetIterator  # pylint: disable=protected-access
from grain.experimental import TFRecordIterDataset
from maxtext.input_pipeline.protos import example_pb2
from maxtext.input_pipeline import tokenizer
from maxtext.multimodal import processor as mm_processor
from maxtext.multimodal import utils as mm_utils
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging

Features = dict[str, Any]
INPUT_TOKENS_KEY = "input_ids"

########## Functions used by TFDS pipeline


def normalize_features(x, column_name):
  return {"inputs": x[column_name], "targets": x[column_name]}


def get_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token=None):
  # Load tokenizer
  tokenizer_model = tokenizer.build_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token)
  return tokenizer_model


def truncate_to_max_allowable_length(x, max_length):
  return {k: v[:max_length] for k, v in x.items()}


def shift_data_by_truncation(x):
  x["inputs"] = x["inputs"][:-1]
  x["targets"] = x["targets"][1:]
  return x


def add_segmentation_and_position(x, data_columns, padding_token=0):
  import tensorflow as tf  # pylint: disable=import-outside-toplevel

  for data_column in data_columns:
    x[f"{data_column}_segmentation"] = tf.cast(x[data_column] != padding_token, tf.int32)
    x[f"{data_column}_position"] = tf.broadcast_to(
        tf.range(x[data_column].shape[-1], dtype=np.int32)[None, :], x[data_column].shape
    )
  return x


def TokenizeOp(tokenizer_model, features: Features, data_keys: Iterable[str] = ("inputs", "targets")) -> Features:
  """Op for tokenization"""
  import tensorflow as tf  # pylint: disable=import-outside-toplevel

  def _process_string(string_tensor):
    # Extract string value and decode it if necessary
    string_value = string_tensor.numpy().decode("utf-8")
    # encode and extract the tokenized integers
    modified_string = tokenizer_model.encode(string_value)
    return [modified_string]

  for k in data_keys:
    features[k] = tf.py_function(_process_string, [features[k]], Tout=[tf.int32])[0]
  return features


########## Functions used by HF pipeline


def reformat_prompt(example, column, image_placeholder, model_name):
  """reformat prompt for multimodal SFT"""
  if isinstance(example["images"], list):
    num_images = len(example["images"])
  else:
    num_images = 1
  example[column] = mm_processor.reformat_prompt(example[column], image_placeholder, model_name, num_images)
  return example


def reformat_response(example, column, model_name):
  """reformat response for multimodal SFT"""
  example[column] = mm_processor.reformat_response(example[column][0], model_name)
  return example


def merge_image_columns(example, image_columns, max_num_images_per_example):
  """Merge multiple image columns into a single list of images."""
  images = []
  for col in image_columns:
    if isinstance(example[col], list):
      images.extend(example[col])
    else:
      images.append(example[col])

  example["images"] = images[:max_num_images_per_example] if max_num_images_per_example > 0 else images
  return example


def pre_process_image_sft(example, image_column, model_name):
  """pre-process image for multimodal SFT"""

  def _process_image_fn(image):
    if isinstance(image, list):
      image = [np.array(mm_utils.convert_to_RGB(img)) for img in image]
    else:
      image = np.array(mm_utils.convert_to_RGB(image))

    image = mm_processor.preprocess_image_for_training(image, model_name)
    return image

  example[image_column] = _process_image_fn(example[image_column])
  return example


def prepare_text_for_image_fusion(example, column_name, config):
  """prepare text for image fusion for multimodal SFT"""
  example[column_name] = mm_processor.prepare_text_for_image_fusion(
      tokens=example[column_name], config=config, processor_output=example["images"]
  )
  return example


def combine_columns(example, columns, data_column):
  """Combine columns such as 'prompt' and 'completion' for sft training"""
  assert len(columns) > 1
  combined = []
  for i in range(len(example[columns[0]])):
    for c in columns:
      combined.append(example[c][i])
  example[data_column] = combined
  return example


def is_conversational(features, data_columns):
  """Check if data is in a conversational format.
  Examples:

  features = {'prompt': [{'content': Value(dtype='string', id=None), 'role': Value(dtype='string', id=None)}],
              'completion': [{'content': Value(dtype='string', id=None), 'role': Value(dtype='string', id=None)}]}
  data_columns = ["prompt", "completion"]
  is_conversational(features, data_columns) return True.

  features = {'prompt': [Value(dtype='string', id=None)], 'completion': [Value(dtype='string', id=None)]}
  data_columns = ["prompt", "completion"]
  is_conversational(features, data_columns) returns False.
  """
  import datasets  # pylint: disable=import-outside-toplevel

  for column in data_columns:
    messages = features[column]
    if isinstance(messages, datasets.Sequence):
      if isinstance(messages.feature, dict) and "role" in messages.feature and "content" in messages.feature:
        return True

  return False


def _get_completion_in_chat_template(tokenizer_model, round_msgs):
  """
  Calculates the completion part of a conversation turn when formatted with a chat template.

  This function handles both older and current Hugging Face tokenizers. Modern tokenizers
  may return a `BatchEncoding` object instead of a simple list of token IDs.

  Args:
    tokenizer_model: The tokenizer instance.
    round_msgs: A list of messages for the current conversational turn, including the assistant's response.

  Returns:
    A string representing the completion formatted by the chat template.
  """
  prompt_completion_tokens = tokenizer_model.apply_chat_template(round_msgs, add_generation_prompt=False, tokenize=True)
  # include generation_prompt as part of the prompt tokens
  prompt_tokens = tokenizer_model.apply_chat_template(round_msgs[:-1], add_generation_prompt=True, tokenize=True)

  # attention masks in BatchEncoding are effectively ignored
  if hasattr(prompt_completion_tokens, INPUT_TOKENS_KEY):
    prompt_completion_ids = getattr(prompt_completion_tokens, INPUT_TOKENS_KEY)
    prompt_ids = getattr(prompt_tokens, INPUT_TOKENS_KEY)
  elif isinstance(prompt_completion_tokens, dict) and INPUT_TOKENS_KEY in prompt_completion_tokens:
    prompt_completion_ids = prompt_completion_tokens[INPUT_TOKENS_KEY]
    prompt_ids = prompt_tokens[INPUT_TOKENS_KEY]
  elif isinstance(prompt_completion_tokens, list):
    prompt_completion_ids = prompt_completion_tokens
    prompt_ids = prompt_tokens
  else:
    raise ValueError(f"Can't handle the chat template output of type {type(prompt_completion_tokens)}")

  completion_tokens = prompt_completion_ids[len(prompt_ids) :]
  completion_in_chat_template = tokenizer_model.decode(completion_tokens, skip_special_tokens=False)
  return completion_in_chat_template


def apply_chat_template(example, tokenizer_model, data_column_name):
  """Formats conversational data by applying the tokenizer's chat template
  and identifying prompt/completion segments for SFT masking.

  Args:
    example: A dictionary containing conversational data. It is expected to have a key
      specified by `data_column_name` that holds a list of messages.
    tokenizer_model: The tokenizer instance associated with the language model,
      which contains the specific chat template.
    data_column_name: The name of the column in the `example` dictionary
      that contains the list of messages.

  Returns:
    The modified `example` dictionary.
      - The `data_column_name` column will be updated to a list of
        messages, each formatted according to the tokenizer's chat template.
      - A new column "is_prompt" is added, where `True` indicates the
        tokens contain the system message, user message, and generation
        prompt (if applicable). `False` indicates the expected LLM
        completion, excluding the assistant's start tokens.
  """
  messages = []
  is_prompt = []
  round_msgs = []
  try:
    for idx, message in enumerate(example[data_column_name]):
      if message["role"] == "system":
        if idx != 0:
          raise ValueError(f"System message found at index {idx}. System messages must be at index 0.")
        round_msgs.append(message)
      elif message["role"] == "user":
        round_msgs.append(message)
        prompt_in_chat_template = tokenizer_model.apply_chat_template(
            round_msgs, add_generation_prompt=True, tokenize=False
        )
        messages.append(prompt_in_chat_template)
        is_prompt.append(True)
      elif message["role"] == "assistant":
        round_msgs.append(message)
        messages.append(_get_completion_in_chat_template(tokenizer_model, round_msgs))
        is_prompt.append(False)
        # Round ended, clearing the buffer.
        round_msgs.clear()
  except ValueError as e:
    max_logging.log(f"Unable to apply chat template: {e}")
    raise e
  example["is_prompt"] = is_prompt
  example[data_column_name] = messages
  return example


def tokenization(example, hf_tokenizer, truncation, max_length, column_names):
  """Tokenize a HuggingFace dataset"""
  for column_name in column_names:
    if isinstance(example[column_name], list):
      example[column_name] = [
          hf_tokenizer(x, truncation=truncation, max_length=max_length)["input_ids"] for x in example[column_name]
      ]
    elif isinstance(example[column_name], str):
      example[column_name] = hf_tokenizer(example[column_name], truncation=truncation, max_length=max_length)["input_ids"]
  return example


@dataclasses.dataclass
class SFTPromptMasking(grain.MapTransform):
  """Construct inputs and targets for SFT training. Concat prompt and completion to generate inputs.
  For targets, if train on completion only, the prompt will be masked by unk_id. Otherwise the same as inputs.
  """

  def __init__(self, text_column_name, completion_only, max_target_length, unk_id=0):
    self.text_column_name = text_column_name
    self.completion_only = completion_only
    self.max_target_length = max_target_length
    self.unk_id = unk_id

  def map(self, element):
    """
    Maps a single dataset element to an SFT training instance.
    It concatenates the prompt and completion to form the `inputs` sequence.
    For the `targets` sequence:
    - If `self.completion_only` is `True`, the prompt portion of the
      concatenated sequence is masked using `self.unk_id`.
    - If `self.completion_only` is `False`, the target sequence is
      identical to the input sequence.
    """
    inputs, targets = [], []
    for i, text in enumerate(element[self.text_column_name]):
      inputs += text
      targets += [self.unk_id] * len(text) if self.completion_only and element["is_prompt"][i] else text
    return {
        "inputs": np.asarray(inputs[: self.max_target_length], dtype=np.int32),
        "targets": np.asarray(targets[: self.max_target_length], dtype=np.int32),
    }


@dataclasses.dataclass
class SFTPromptMaskingVision(grain.MapTransform):
  """SFT prompt masking for multimodal"""

  def __init__(self, query_column, response_column, max_target_length, unk_id):
    self.query_column = query_column
    self.response_column = response_column
    self.max_target_length = max_target_length
    self.unk_id = unk_id

  def map(self, element):
    inputs = np.concatenate((element[self.query_column], element[self.response_column]))
    targets = np.concatenate((np.asarray([self.unk_id] * len(element[self.query_column])), element[self.response_column]))
    return {
        "inputs": np.asarray(inputs[: self.max_target_length], dtype=np.int32),
        "targets": np.asarray(targets[: self.max_target_length], dtype=np.int32),
        "images": element["images"],
    }


@dataclasses.dataclass
class HFNormalizeFeatures(grain.MapTransform):
  """Normalize feature keys for HuggingFace input"""

  def __init__(self, column_name):
    self.column_name = column_name

  def map(self, element):
    return {
        "inputs": np.asarray(element[self.column_name], dtype=np.int32),
        "targets": np.asarray(element[self.column_name], dtype=np.int32),
    }


class HFDataSource(grain.RandomAccessDataSource):
  """A class that makes HuggingFace IterableDataset a grain datasource without random access support"""

  def __init__(
      self,
      dataset: "datasets.IterableDataset",
      dataloading_host_index: int,
      dataloading_host_count: int,
      num_threads: int,
      max_target_length: int,
      data_column_names: list[str],
  ):
    from datasets.distributed import split_dataset_by_node  # pylint: disable=import-outside-toplevel

    self._split_dataset_by_node = split_dataset_by_node
    self.dataset = dataset
    self.num_threads = num_threads
    self.dataloading_host_count = dataloading_host_count
    self.dataloading_host_index = dataloading_host_index
    self.max_target_lenth = max_target_length
    self.data_column_names = data_column_names
    if hasattr(dataset, "n_shards"):
      self.n_shards = dataset.n_shards
    else:
      self.n_shards = 1
    self._check_shard_count()
    self.dataset_shards = [dataloading_host_index * self.num_threads + i for i in range(self.num_threads)]
    self.datasets = [self._split_dataset_by_node(dataset, world_size=self.n_shards, rank=x) for x in self.dataset_shards]
    self.data_iters = []

  def _check_shard_count(self):
    if self.n_shards < (self.dataloading_host_count * self.num_threads):
      warnings.warn(
          f"WARNING: Inefficient dataloading. Your train or eval dataset contains {self.n_shards} shards, "
          "smaller than number of host loading data. This is known to lead to inefficient dataloading. See"
          "github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md#multihost-dataloading-best-practice"
      )
      self.n_shards = self.dataloading_host_count * self.num_threads

  def _update_shard(self, idx):
    """update shard"""
    new_shard = self.dataset_shards[idx] + self.dataloading_host_count * self.num_threads
    if new_shard < self.n_shards:
      max_logging.log(
          f"Updating host {self.dataloading_host_index} dataset {idx}, was on shard {self.dataset_shards[idx]}"
      )
      max_logging.log(f"New shard is {new_shard}")
      self.dataset_shards[idx] = new_shard
      self.datasets[idx] = self._split_dataset_by_node(
          self.dataset, world_size=self.n_shards, rank=self.dataset_shards[idx]
      )
      self.data_iters[idx] = iter(self.datasets[idx])
    else:
      raise StopIteration(f"Run out of shards on host {self.dataloading_host_index}, shard {new_shard} is not available")

  def __len__(self):
    """Return length of the HF dataset. Since HuggingFace IterableDataset does not have length,
    a fake length bigger than the dataset is returned"""
    return 10_000_000_000

  def __getitem__(self, index):
    """Since HuggingFace IterableDataset does not support random access by index.
    The next item in the iterator is returned."""
    if not self.data_iters:
      self.data_iters = [iter(x) for x in self.datasets]
    idx = int(current_thread().name.split("_")[1])

    while True:
      try:
        data = next(self.data_iters[idx])
        return data
      except StopIteration:
        self._update_shard(idx)


########## Functions used by Grain pipeline


class _GCSTFRecordReader(_TFRecordReader):
  """Extends Grain's _TFRecordReader to open TFRecord files from GCS via streaming BlobReader."""

  def __init__(self, path: str):
    # Skip parent __init__ (which calls open(path, "rb")) and open via GCS BlobReader instead.
    bucket_name, blob_name = gcs_utils.parse_gcs_bucket_and_prefix(path)
    self._reader = gcs_utils.storage.Client().bucket(bucket_name).blob(blob_name).open("rb")


class _GCSTFRecordDatasetIterator(_TFRecordDatasetIterator):
  """Extends Grain's _TFRecordDatasetIterator to use _GCSTFRecordReader for GCS paths."""

  def __init__(self, path: str):
    # Skip parent __init__ (which creates _TFRecordReader); use GCS-aware reader instead.
    grain.DatasetIterator.__init__(self)
    self._reader = _GCSTFRecordReader(path)


class GCSTFRecordIterDataset(TFRecordIterDataset):
  """Extends Grain's TFRecordIterDataset to support GCS paths."""

  def __iter__(self) -> grain.DatasetIterator:  # pylint: disable=non-iterator-returned
    return _GCSTFRecordDatasetIterator(self._path)


def make_tfrecord_iter_dataset(path: str):
  """Returns the appropriate TFRecordIterDataset for local or GCS paths."""
  if path.startswith("gs://"):
    return GCSTFRecordIterDataset(path)
  return TFRecordIterDataset(path)


@dataclasses.dataclass
class ParseFeatures(grain.MapTransform):
  """Parse serialized example"""

  def __init__(self, data_columns, tokenize):
    self.data_columns = data_columns
    self.tokenize = tokenize

  def map(self, element):
    """Parse a serialized tf.train.Example proto and extract features."""
    example = example_pb2.Example()
    example.ParseFromString(element)
    features = example.features.feature

    parsed = {}
    for col in self.data_columns:
      if col in features:
        f = features[col]
        if self.tokenize:
          parsed[col] = np.array(f.bytes_list.value, dtype=object)
        else:
          parsed[col] = np.array(f.int64_list.value, dtype=np.int32)
    return parsed


@dataclasses.dataclass
class NormalizeFeatures(grain.MapTransform):
  """Normalize text feature keys."""

  def __init__(self, column_names, tokenize):
    self.column_names = column_names
    self.tokenize = tokenize

  def map(self, element):
    if self.tokenize:
      return {col: element[col][0].decode() for col in self.column_names}
    else:
      return {col: element[col] for col in self.column_names}


@dataclasses.dataclass
class KeepFeatures(grain.MapTransform):
  """Keep only specified features in the dataset element.

  This transform filters the input dictionary, retaining only the keys
  that are present in `feature_names`.
  """

  def __init__(self, feature_names: list[str]):
    """Initializes the KeepFeatures transform.

    Args:
      feature_names: A list of strings, where each string is the name of a
        feature to be kept in the dataset element.
    """
    self.feature_names = feature_names

  def map(self, element: dict[str, Any]) -> dict[str, Any]:
    """Applies the feature filtering to the input element."""
    return {k: v for k, v in element.items() if k in self.feature_names}


@dataclasses.dataclass
class Rekey(grain.MapTransform):
  """Rename keys according to a mapping dict"""

  def __init__(self, mapping_dict, keep_old_keys=False):
    self.mapping_dict = mapping_dict
    self.keep_old_keys = keep_old_keys

  def map(self, element):
    old_keys = set()
    for new_key, old_key in self.mapping_dict.items():
      element[new_key] = element[old_key]
      old_keys.add(old_key)
    if not self.keep_old_keys:
      for key in old_keys:
        del element[key]
    return element


@dataclasses.dataclass
class ReformatPacking(grain.MapTransform):
  """Reformat packing outputs."""

  def __init__(self, column_names):
    self.column_names = column_names

  def map(self, element):
    ret = {}
    for col in self.column_names:
      ret[f"{col}"] = element[0][col]
      ret[f"{col}_segmentation"] = element[1][col]
      ret[f"{col}_position"] = element[2][col]
    return ret


@dataclasses.dataclass
class PadOrTrimToMaxLength(grain.MapTransform):
  """Pads or trims each input to the specified length.
  And optionally add true length for the input."""

  def __init__(
      self,
      max_length: int,
      pad_id: int = 0,
      model_name: str | None = None,
      add_true_length: bool = False,
      max_num_images_per_example: int = -1,
  ):
    self.max_length = max_length
    self.pad_id = pad_id
    self.model_name = model_name
    self.add_true_length = add_true_length
    self.max_num_images_per_example = max_num_images_per_example

  def _pad_text(self, x: np.ndarray, max_length: int, pad_id: int) -> np.ndarray:
    pad_amount = max(max_length - x.shape[0], 0)
    pad_amount = [(0, pad_amount)] + [(0, 0)] * (len(x.shape) - 1)
    return np.pad(x, pad_amount, constant_values=pad_id)[: self.max_length]

  def _pad_image_and_mask(self, preprocessed_image: mm_utils.PreprocessorOutput) -> mm_utils.PreprocessorOutput:
    """Pads the input tensors (image and mask) of a PreprocessorOutput to a maximum number of items.

    This function unifies padding logic for image tensors (standard or tiled) and
    mask tensors. It determines the tensor type based on its dimensions and applies
    the appropriate padding along the first axis.

    The maximum number of items is calculated based on model constraints or a
    user-defined limit, ensuring that sequence length limits are respected while
    reserving space for at least one text token. If the input tensor has fewer
    items than this maximum, it is padded with zeros.

    Args:
        preprocessed_image (multimodal_utils.PreprocessorOutput): The input numpy arrays to pad.
            - For masks, the expected shape is (num_masks, num_tiles).
            - For standard images, the shape is (num_images, H, W, C).
            - For tiled images, the shape is (num_images, num_tiles, H, W, C).

    Returns:
        np.ndarray: The tensor, padded with zeros up to the maximum number of
        items along the first axis.

    Raises:
        ValueError: If the input tensor's dimension is not 2, 4, or 5.
        ValueError: If the number of items in the input tensor exceeds the
        allowed maximum.

    Notes:
      - The computation of maximum images ensures that space is reserved in the sequence
        for at least one text token.
      - The dummy images used for padding are based on the image shape for initialization
        of this model (ignoring batch size).
    """
    if not isinstance(preprocessed_image, mm_utils.PreprocessorOutput):
      raise TypeError(f"Input must be multimodal_utils.PreprocessorOutput, but got {type(preprocessed_image)}")

    if preprocessed_image.pixel_values is None:
      raise ValueError("Input preprocessed_image must have pixel_values to pad images.")

    # Determine the maximum number of images/masks allowed.
    image_offsets = mm_processor.get_image_offsets(self.model_name, preprocessed_image)
    single_image_offset = image_offsets // preprocessed_image.pixel_values.shape[0]

    # Reserve space for at least one text token.
    max_num_items = (self.max_length - 1) // single_image_offset
    if self.max_num_images_per_example > 0:
      max_num_items = min(self.max_num_images_per_example, max_num_items)

    image_tensor = preprocessed_image.pixel_values
    mask_tensor = preprocessed_image.pixel_mask

    def _pad(tensor: np.ndarray) -> np.ndarray:
      # Validate tensor dimensions.
      if tensor.ndim in (4, 5):  # Standard or Tiled Image
        tensor_type = "images"
      elif tensor.ndim == 2:  # Mask
        tensor_type = "masks"
      else:
        raise ValueError(
            "Input tensor must be 2D (mask), 4D (image), or 5D (tiled image), " f"but got {tensor.ndim} dimensions."
        )

      # Assert that the input tensor does not exceed the maximum size.
      if tensor.shape[0] > max_num_items:
        raise ValueError(f"Number of {tensor_type} ({tensor.shape[0]}) exceeds the maximum allowed ({max_num_items}).")

      # Apply padding if the tensor is smaller than the maximum size.
      if tensor.shape[0] < max_num_items:
        pad_size = max_num_items - tensor.shape[0]
        pad_shape_suffix = tensor.shape[1:]
        pad_shape = (pad_size,) + pad_shape_suffix
        pad_tensor = np.zeros(pad_shape, dtype=tensor.dtype)

        if tensor.size > 0:
          tensor = np.concatenate([tensor, pad_tensor], axis=0)
        else:
          # If the input tensor is empty, the result is just the padding.
          tensor = pad_tensor

      return tensor

    preprocessed_image.pixel_values = _pad(image_tensor)

    if mask_tensor is not None:
      preprocessed_image.pixel_mask = _pad(mask_tensor)

    return preprocessed_image

  def map(
      self, element: dict[str, np.ndarray | mm_utils.PreprocessorOutput]
  ) -> dict[str, np.ndarray | mm_utils.PreprocessorOutput]:
    """map to each element"""
    data_columns = list(element.keys())
    for data_column in data_columns:
      if data_column != "images":
        if isinstance(element[data_column], mm_utils.PreprocessorOutput):
          raise TypeError("Only 'images' column can be of type PreprocessorOutput.")

        element[f"{data_column}_segmentation"] = element[data_column] != self.pad_id
        element[f"{data_column}_segmentation"] = element[f"{data_column}_segmentation"].astype(np.int32)
        element[f"{data_column}_position"] = np.arange(element[data_column].shape[0], dtype=np.int32)
        if self.add_true_length:
          element[f"{data_column}_true_length"] = np.array([element[data_column].shape[0]], dtype=np.int32)

    for key, _ in element.items():
      if key == "images":
        if self.model_name is None:
          raise ValueError("model_name must be provided when padding images")

        element["images"] = self._pad_image_and_mask(element["images"])

      elif "true_length" not in key:
        element[key] = self._pad_text(element[key], self.max_length, self.pad_id)
    return element


@dataclasses.dataclass
class ExtractImagesAndMasks(grain.MapTransform):
  """Extracts images and masks from a PreprocessorOutput object.

  This transform is used in multi-modal data pipelines to extract the image
  tensors and their corresponding masks from a PreprocessorOutput object.
  The extracted images and masks are then added to the data element under
  the keys 'images' and 'image_masks', respectively.

  If the 'images' key is not present in the input element, the transform
  returns the element unchanged.
  """

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Applies the extraction transformation to the 'images' field if present."""
    preprocessed_image = element.get("images")
    if preprocessed_image is None:
      return element

    if not isinstance(preprocessed_image, mm_utils.PreprocessorOutput):
      raise TypeError(f"'images' must be of type PreprocessorOutput, but got {type(preprocessed_image)}")

    output = element.copy()
    output["images"] = preprocessed_image.pixel_values
    if preprocessed_image.pixel_mask is not None:
      output["image_masks"] = preprocessed_image.pixel_mask

    return output


@dataclasses.dataclass
class FoldImagesIntoBatch(grain.MapTransform):
  """Folds the 'image' dimension into the batch dimension.

  This transform is used in multi-modal data pipelines where each data example
  might have multiple associated images. For model processing, it's often
  efficient to treat each image as a separate item in a larger batch.

  This operation reshapes the 'images' tensor from a shape like
  (B, N, T, H, W, C) to (B * N, T, H, W, C), where B is the batch size, N is
  the number of images per example, and T is the number of image tiles.

  The transformation is triggered only if the input 'images' tensor has more
  dimensions than the expected batched image tensor.
  """

  model_name: str | None = None

  def __post_init__(self):
    """Initializes the target shape after the dataclass is created."""
    self.target_shape = mm_processor.get_dummy_image_shape_for_init(self.model_name)

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Applies the folding transformation to the 'images' field if present."""
    images = element.get("images")
    if images is None:
      return element

    # If ndim is greater than the expected ndim for a batched image tensor,
    # it implies an extra dimension (e.g., number of images per example)
    # that needs to be folded into the batch dimension.
    if images.ndim > len(self.target_shape):
      # Compute the new shape by merging the batch and image count dimensions.
      trailing_dims = self.target_shape[1:]

      # Reshape merges the leading dimensions (B, N) into one (-1) and
      # appends the correct trailing dimensions.
      element["images"] = images.reshape(-1, *trailing_dims)

    return element


def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  slices = [
      slice(None),
  ] * len(x.shape)
  slices[axis] = slice(0, -1)
  padded = np.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(0))
  return padded[tuple(slices)]


def shift_left(x, pad_id, axis=1):
  """Shift to the left and pad."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (0, 1)
  slices = [
      slice(None),
  ] * len(x.shape)
  slices[axis] = slice(1, None)
  padded = np.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(pad_id))
  return padded[tuple(slices)]


def shift_and_refine(x, ignored_ids, axis=1):
  """Shift inputs, set segmentation to 0 when target element is in ignored_ids if provided"""
  x["targets"] = shift_left(x["targets"], ignored_ids[0], axis=axis)
  x["targets_segmentation"] = shift_left(x["targets_segmentation"], 0, axis=axis)
  for ignore_id in ignored_ids:
    x["targets_segmentation"] = np.where(x["targets"] != ignore_id, x["targets_segmentation"], 0)

  return x


@dataclasses.dataclass
class ShiftData(grain.MapTransform):
  """Shift inputs and refine annotations."""

  def __init__(self, ignored_ids, axis=1):
    self.ignored_ids = ignored_ids
    self.axis = axis

  def map(self, element):
    return shift_and_refine(element, ignored_ids=self.ignored_ids, axis=self.axis)


@dataclasses.dataclass
class ComputeQwen3OmniPositions(grain.MapTransform):
  """Computes 3D position IDs for Qwen3-Omni multimodal sequences.

  This transform replaces the standard 1D sequential positions with 3D
  positions (temporal, height, width) for multimodal models like Qwen3-Omni.

  For text-only sequences, all 3 dimensions receive the same sequential values.
  For multimodal sequences with vision/audio, vision tokens get true 3D positions
  and text tokens continue sequentially from max(vision_pos) + 1.

  The actual position computation is delegated to multimodal_utils.get_rope_index(),
  which can be tested and modified independently.
  """

  def __init__(
      self,
      data_column: str = "inputs",
      spatial_merge_size: int = 2,
      position_id_per_seconds: int = 25,
      use_audio_in_video: bool = False,
  ):
    """Initialize the Qwen3-Omni position computation transform.

    Args:
      data_column: Name of the data column to compute positions for (default: "inputs").
      spatial_merge_size: Number of patches merged spatially (e.g., 2 for 2x2→1).
      position_id_per_seconds: Temporal granularity (tokens per second, typically 25).
      use_audio_in_video: If True, audio tokens are interleaved with video tokens.
    """
    self.data_column = data_column
    self.spatial_merge_size = spatial_merge_size
    self.position_id_per_seconds = position_id_per_seconds
    self.use_audio_in_video = use_audio_in_video

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute 3D position IDs for the batch element.

    Args:
      element: Dictionary containing:
        - {data_column}: Token IDs with shape (batch, seq_len)
        - {data_column}_segmentation: Attention mask (1=real, 0=padding)
        - image_grid_thw: Optional (num_images, 3) array
        - video_grid_thw: Optional (num_videos, 3) array
        - audio_lengths: Optional (num_audios,) array
        - second_per_grids: Optional (num_videos,) array

    Returns:
      element with {data_column}_position updated to shape (3, batch, seq_len)
      for 3D positions (always 3D, even for text-only sequences).
    """

    # Extract inputs and metadata
    input_ids = element[self.data_column]
    attention_mask = element.get(f"{self.data_column}_segmentation")

    # Extract multimodal metadata (if present)
    image_grid_thw = element.get("image_grid_thw")
    video_grid_thw = element.get("video_grid_thw")
    audio_lengths = element.get("audio_lengths")
    second_per_grids = element.get("second_per_grids")

    # Call the standalone get_rope_index function from multimodal_utils
    from maxtext.multimodal import processor_qwen3_omni  # pylint: disable=import-outside-toplevel

    # TODO(jfacevedo/hengtaoguo): Now get_rope_index is Qwen3-Omni specific. We should generalize it for other models
    position_ids, mrope_position_deltas = processor_qwen3_omni.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        use_audio_in_video=self.use_audio_in_video,
        audio_lengths=audio_lengths,
        second_per_grids=second_per_grids,
        spatial_merge_size=self.spatial_merge_size,
        position_id_per_seconds=self.position_id_per_seconds,
    )

    # Update element with 3D positions
    # Shape: (3, batch, seq_len) for multimodal, or (batch, seq_len) for text-only
    element[f"{self.data_column}_position"] = position_ids
    element[f"{self.data_column}_mrope_deltas"] = mrope_position_deltas

    return element


@dataclasses.dataclass
class GenerateDocSegmentIds(grain.MapTransform):
  """Generate segmentation and position arrays from EOD tokens within samples.

  Detects EOD tokens (``eod_id``) within each sample and generates proper
  ``_segmentation`` and ``_position`` arrays.

  This is used with ``MMapSampleIndexDataSource`` which inserts EOD tokens
  between concatenated documents.

  Args:
    eod_id: Token ID that marks the end of a document.
    reset_attention_mask: Controls how document boundaries affect attention.

      * ``True`` (default) -- attention resets at every document boundary.
        EOD belongs to the preceding document (same segment ID), and a new
        segment starts after EOD.  Positions continue through EOD and reset
        after EOD.

        ::

            tokens:        [tok tok tok EOD tok tok EOD tok tok tok tok tok]
            segmentation:  [ 1   1   1   1   2   2   2   3   3   3   3   3]
            positions:     [ 0   1   2   3   0   1   2   0   1   2   3   4]

      * ``False`` -- cross-document attention is allowed.  All non-EOD
        tokens share the same segment ID (``1``), and positions are a
        continuous ``arange`` over the whole sequence.

        ::

            tokens:        [tok tok tok EOD tok tok EOD tok tok tok tok tok]
            segmentation:  [ 1   1   1   0   1   1   0   1   1   1   1   1]  (eod_mask_loss=True)
            segmentation:  [ 1   1   1   1   1   1   1   1   1   1   1   1]  (eod_mask_loss=False)
            positions:     [ 0   1   2   3   4   5   6   7   8   9  10  11]

    eod_mask_loss: When True, EOD tokens get segmentation=0 (excluded from loss).
      When False (default), EOD tokens get segmentation=1 (included in loss).
      Only applies when reset_attention_mask=False.
  """

  def __init__(self, eod_id: int, reset_attention_mask: bool = True, eod_mask_loss: bool = False):
    self.eod_id = eod_id
    self.reset_attention_mask = reset_attention_mask
    self.eod_mask_loss = eod_mask_loss

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply EOD-based segmentation and loss masking to each column."""
    for col, tokens in list(element.items()):
      seq_len = tokens.shape[0]
      is_eod = tokens == self.eod_id

      if self.reset_attention_mask:
        # EOD belongs to the preceding document: keep current seg_id and
        # continue position counter.  New segment starts AFTER EOD.
        segmentation = np.zeros(seq_len, dtype=np.int32)
        position = np.zeros(seq_len, dtype=np.int32)
        seg_id = 1
        pos_in_doc = 0
        for i in range(seq_len):
          if is_eod[i]:
            # EOD keeps the preceding document's seg_id and position
            segmentation[i] = seg_id
            position[i] = pos_in_doc
            pos_in_doc += 1
            # New segment starts after EOD
            seg_id += 1
            pos_in_doc = 0
          else:
            segmentation[i] = seg_id
            position[i] = pos_in_doc
            pos_in_doc += 1
      else:
        if self.eod_mask_loss:
          segmentation = np.where(is_eod, np.int32(0), np.int32(1))
        else:
          segmentation = np.ones(seq_len, dtype=np.int32)
        position = np.arange(seq_len, dtype=np.int32)

      element[f"{col}_segmentation"] = segmentation
      element[f"{col}_position"] = position
    return element


@dataclasses.dataclass
class MegatronSplitInputsTargets(grain.MapTransform):
  """Split seq_length+1 tokens into inputs/targets following Megatron-LM convention.

  Input:  ``{"text": tokens}`` where ``len(tokens) == seq_length + 1``
  Output: ``{"inputs": tokens[:-1], "targets": tokens[1:],
            "inputs_segmentation": ..., "targets_segmentation": ...,
            "inputs_position": ..., "targets_position": ...}``

  This gives valid prediction targets at every position without padding,
  unlike the ShiftData approach which wastes the last position.

  The three output array groups are constructed independently to match
  Megatron-LM's separate attention_mask / loss_mask / position_ids:

  - ``inputs_segmentation`` (attention): controlled by ``reset_attention_mask``.
    When True, EOD belongs to its preceding document (same segment ID) and
    a new segment starts after EOD; when False, all tokens share segment 1.
  - ``targets_segmentation`` (loss): controlled by ``eod_mask_loss`` only.
    When True, positions where ``inputs == eod_id`` get segmentation=0
    (excluded from loss); when False, all positions get segmentation=1.
  - ``inputs_position`` (RoPE): controlled by ``reset_attention_mask``.
    When True, EOD continues the preceding document's position counter and
    resets to 0 after EOD; when False, positions are sequential.

  Args:
    eod_id: Token ID that marks the end of a document.
    reset_attention_mask: Controls how document boundaries affect attention
      and position encoding. When True, attention is blocked across
      document boundaries and position IDs reset after each EOD token.
    eod_mask_loss: When True, positions where inputs contain EOD get
      targets_segmentation=0 (excluded from loss). When False (default),
      all positions participate in loss.
  """

  def __init__(
      self,
      eod_id: int,
      reset_attention_mask: bool = True,
      eod_mask_loss: bool = False,
      no_attnmask_dataset_ids: set[int] | None = None,
  ):
    self.eod_id = eod_id
    self.reset_attention_mask = reset_attention_mask
    self.eod_mask_loss = eod_mask_loss
    self.no_attnmask_dataset_ids = no_attnmask_dataset_ids or set()
    self._has_no_attnmask = bool(self.no_attnmask_dataset_ids)

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Split tokens into input/target pairs with EOD-based segmentation."""
    tokens = element["text"]
    inputs = tokens[:-1]
    targets = tokens[1:]

    seq_len = inputs.shape[0]
    is_eod = inputs == self.eod_id

    # Per-sample override: datasets in no_attnmask_dataset_ids bypass
    # attention mask reset (antllm no_attnmask_data behavior).
    effective_reset = self.reset_attention_mask
    if self._has_no_attnmask:
      dataset_id = element.get("dataset_id", None)
      if dataset_id is not None and int(dataset_id) in self.no_attnmask_dataset_ids:
        effective_reset = False

    # --- inputs_segmentation (attention mask) ---
    if effective_reset:
      input_segmentation = np.zeros(seq_len, dtype=np.int32)
      position = np.zeros(seq_len, dtype=np.int32)
      seg_id = 1
      pos_in_doc = 0
      for i in range(seq_len):
        if is_eod[i]:
          # EOD belongs to the preceding document: keep current seg_id,
          # continue position counter.  New segment starts AFTER EOD.
          input_segmentation[i] = seg_id
          position[i] = pos_in_doc
          pos_in_doc = 0
          seg_id += 1
        else:
          input_segmentation[i] = seg_id
          position[i] = pos_in_doc
          pos_in_doc += 1
    else:
      input_segmentation = np.ones(seq_len, dtype=np.int32)
      position = np.arange(seq_len, dtype=np.int32)

    # --- targets_segmentation (loss mask) ---
    # Independent of reset_attention_mask, matching Megatron's separate
    # loss_mask which only depends on eod_mask_loss.
    if self.eod_mask_loss:
      target_segmentation = np.where(is_eod, np.int32(0), np.int32(1))
    else:
      target_segmentation = np.ones(seq_len, dtype=np.int32)

    return {
        "inputs": inputs,
        "targets": targets,
        "inputs_segmentation": input_segmentation,
        "targets_segmentation": target_segmentation,
        "inputs_position": position,
        "targets_position": position,
    }
