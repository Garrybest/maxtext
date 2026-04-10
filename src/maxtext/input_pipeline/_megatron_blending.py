"""Megatron-compatible dataset blending for Grain MapDatasets."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Sequence, TypeVar

import grain.python as grain
import numpy as np

from maxtext.input_pipeline import _mmap_index_utils


logger = logging.getLogger(__name__)

_DATASET_INDEX_SUFFIX = "dataset_index.npy"
_DATASET_SAMPLE_INDEX_SUFFIX = "dataset_sample_index.npy"

# Exceptions that indicate corrupted/missing/invalid index files,
# as opposed to programming errors which should propagate immediately.
_RECOVERABLE_ERRORS = (FileNotFoundError, OSError, ValueError, EOFError)


def build_blending_indices(
    dataset_index: np.ndarray,
    dataset_sample_index: np.ndarray,
    weights: np.ndarray,
    num_datasets: int,
    size: int,
) -> None:
  """Build blend indices with Megatron-LM greedy error minimization.

  This follows `megatron/core/datasets/helpers.cpp::build_blending_indices`:
  - sample_idx_double = max(sample_idx, 1.0)
  - error = weights[j] * sample_idx_double - current_samples[j]
  - ties keep the smaller dataset index (strict `>` comparison in C++).
  """
  if size < 0:
    raise ValueError(f"size must be non-negative, got {size}")
  if dataset_index.shape != (size,):
    raise ValueError(f"dataset_index must have shape ({size},), got {dataset_index.shape}")
  if dataset_sample_index.shape != (size,):
    raise ValueError(f"dataset_sample_index must have shape ({size},), got {dataset_sample_index.shape}")
  if num_datasets <= 0:
    raise ValueError(f"num_datasets must be positive, got {num_datasets}")

  weights = np.asarray(weights, dtype=np.float64)
  if weights.shape != (num_datasets,):
    raise ValueError(f"weights must have shape ({num_datasets},), got {weights.shape}")
  if np.any(weights <= 0):
    raise ValueError(f"weights must all be positive, got {weights.tolist()}")
  if not np.isclose(np.sum(weights), 1.0, rtol=1e-6, atol=1e-8):
    raise ValueError(f"weights must sum to 1.0, got sum={float(np.sum(weights))} for {weights.tolist()}")

  current_samples = np.zeros(num_datasets, dtype=np.int64)
  for sample_idx in range(size):
    sample_idx_double = max(float(sample_idx), 1.0)
    errors = weights * sample_idx_double - current_samples
    max_error_index = int(np.argmax(errors))
    dataset_index[sample_idx] = max_error_index
    dataset_sample_index[sample_idx] = current_samples[max_error_index]
    current_samples[max_error_index] += 1


_T = TypeVar("_T")


def _normalize_and_filter_weights(
    datasets: Sequence[_T],
    weights: Sequence[float],
    dataset_lengths: Sequence[int] | None = None,
) -> tuple[list[_T], np.ndarray, list[int] | None]:
  """Normalize weights to sum to 1 and filter out zero-weight datasets."""
  if len(datasets) != len(weights):
    raise ValueError(f"datasets/weights length mismatch: {len(datasets)} vs {len(weights)}")
  if dataset_lengths is not None and len(dataset_lengths) != len(datasets):
    raise ValueError(f"datasets/dataset_lengths length mismatch: {len(datasets)} vs {len(dataset_lengths)}")
  if not datasets:
    raise ValueError("At least one dataset is required for blending")

  raw_weights = np.asarray(weights, dtype=np.float64)
  if np.any(raw_weights < 0):
    raise ValueError(f"Negative weight detected in {list(weights)}")
  if np.sum(raw_weights) <= 0:
    raise ValueError(f"weights must sum to a positive value, got {list(weights)}")

  # Filter out zero-weight datasets, then normalize once to match
  # Megatron BlendedDataset.__init__ before build_blending_indices.
  keep = raw_weights > 0
  filtered_datasets = [dataset for dataset, keep_i in zip(datasets, keep) if keep_i]
  filtered_weights = raw_weights[keep]
  filtered_weights = filtered_weights / np.sum(filtered_weights)
  filtered_lengths = (
      [int(length) for length, keep_i in zip(dataset_lengths, keep) if keep_i] if dataset_lengths is not None else None
  )

  if not filtered_datasets:
    raise ValueError("All datasets were filtered out by zero weights")

  if len(filtered_weights) > 1:
    ratio = float(np.min(filtered_weights) / np.max(filtered_weights))
    if ratio < 0.01:
      logger.warning("Extreme blend weight ratio detected: %s", filtered_weights.tolist())

  return filtered_datasets, filtered_weights, filtered_lengths


def _infer_size_from_lengths(weights: np.ndarray, dataset_lengths: Sequence[int]) -> int:
  """Infer total blend size from dataset lengths and weights."""
  if len(dataset_lengths) != len(weights):
    raise ValueError(f"dataset_lengths/weights length mismatch: {len(dataset_lengths)} vs {len(weights)}")
  if any(length <= 0 for length in dataset_lengths):
    raise ValueError(f"All dataset lengths must be positive, got {dataset_lengths}")

  capacities = [float(dataset_lengths[i]) / float(weights[i]) for i in range(len(dataset_lengths))]
  size = int(np.floor(min(capacities)))
  if size <= 0:
    raise ValueError(
        f"Inferred blend size {size} is not positive for lengths={dataset_lengths}, " f"weights={weights.tolist()}"
    )
  return size


def _megatron_blend_size(requested_size: int, weights: np.ndarray) -> int:
  """Compute the blend size the same way Megatron's builder does.

  Megatron computes per-dataset target counts as ceil(requested_size * w_i)
  and uses their sum as the total blend size.  Because of ceiling, the result
  can be up to (num_datasets - 1) larger than *requested_size*.

  See ``_get_size_per_split_per_dataset`` in
  ``megatron/core/datasets/blended_megatron_dataset_builder.py``.
  """
  return sum(math.ceil(requested_size * float(w)) for w in weights)


def _build_cache_key(
    dataset_lengths: Sequence[int] | None,
    weights: np.ndarray,
    size: int,
) -> str:
  """Build a deterministic MD5-based cache key from blend parameters."""
  payload = {
      "dataset_lengths": list(dataset_lengths) if dataset_lengths is not None else None,
      "weights": [float(weight) for weight in weights.tolist()],
      "size": int(size),
  }
  payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
  return hashlib.md5(payload_json.encode("utf-8"), usedforsecurity=False).hexdigest()


def _build_cache_paths(cache_dir: str, cache_key: str, split: str) -> tuple[Path, Path]:
  base = f"{cache_key}-BlendedDataset-{split}"
  cache_path = Path(cache_dir)
  return (
      cache_path / f"{base}-{_DATASET_INDEX_SUFFIX}",
      cache_path / f"{base}-{_DATASET_SAMPLE_INDEX_SUFFIX}",
  )


def _validate_indices(
    dataset_index: np.ndarray,
    dataset_sample_index: np.ndarray,
    num_datasets: int,
    dataset_lengths: Sequence[int] | None = None,
    expected_size: int | None = None,
) -> None:
  """Validate blend index arrays for correctness and bounds."""
  if dataset_index.ndim != 1:
    raise ValueError(f"dataset_index must be 1-D, got shape {dataset_index.shape}")
  if dataset_sample_index.ndim != 1:
    raise ValueError(f"dataset_sample_index must be 1-D, got shape {dataset_sample_index.shape}")
  if dataset_index.shape[0] != dataset_sample_index.shape[0]:
    raise ValueError(
        "dataset_index and dataset_sample_index must have equal length, got "
        f"{dataset_index.shape[0]} and {dataset_sample_index.shape[0]}"
    )
  if expected_size is not None and dataset_index.shape[0] != expected_size:
    raise ValueError(
        f"Blend indices have length {dataset_index.shape[0]}, expected {expected_size}. "
        f"The indices may have been built for a different dataset size."
    )
  if not np.issubdtype(dataset_index.dtype, np.integer):
    raise ValueError(f"dataset_index must be integer dtype, got {dataset_index.dtype}")
  if not np.issubdtype(dataset_sample_index.dtype, np.integer):
    raise ValueError(f"dataset_sample_index must be integer dtype, got {dataset_sample_index.dtype}")
  if num_datasets <= 0:
    raise ValueError(f"num_datasets must be positive, got {num_datasets}")

  dataset_index_i64 = np.asarray(dataset_index, dtype=np.int64)
  dataset_sample_index_i64 = np.asarray(dataset_sample_index, dtype=np.int64)

  if np.any(dataset_index_i64 < 0) or np.any(dataset_index_i64 >= num_datasets):
    min_value = int(np.min(dataset_index_i64))
    max_value = int(np.max(dataset_index_i64))
    raise ValueError("dataset_index out of range: " f"min={min_value}, max={max_value}, num_datasets={num_datasets}")
  if np.any(dataset_sample_index_i64 < 0):
    min_value = int(np.min(dataset_sample_index_i64))
    raise ValueError(f"dataset_sample_index contains negative values (min={min_value})")

  for dataset_id in range(num_datasets):
    used_sample_ids = dataset_sample_index_i64[dataset_index_i64 == dataset_id]
    if used_sample_ids.size == 0:
      continue
    expected = np.arange(used_sample_ids.size, dtype=np.int64)
    if not np.array_equal(used_sample_ids, expected):
      raise ValueError(
          "dataset_sample_index must be contiguous per dataset. "
          f"dataset={dataset_id}, expected [0..{used_sample_ids.size - 1}], "
          f"got first={used_sample_ids[:5].tolist()}, last={used_sample_ids[-5:].tolist()}"
      )

  if dataset_lengths is None:
    return

  if len(dataset_lengths) != num_datasets:
    raise ValueError(f"dataset_lengths length mismatch: expected {num_datasets}, got {len(dataset_lengths)}")
  required_samples = np.bincount(dataset_index_i64, minlength=num_datasets)
  for dataset_id, required in enumerate(required_samples.tolist()):
    available = int(dataset_lengths[dataset_id])
    if required > available:
      raise ValueError(
          f"Dataset {dataset_id} has only {available} samples but blend indices "
          f"require {required} samples. Increase num_epochs or adjust weights."
      )


def _find_index_pair_in_dir(index_dir: str, split: str) -> tuple[Path, Path]:
  """Locate a dataset_index / dataset_sample_index .npy pair in a directory."""
  directory = Path(index_dir)
  if not directory.is_dir():
    raise FileNotFoundError(f"Blend index directory does not exist: {index_dir}")

  direct_dataset_index = directory / _DATASET_INDEX_SUFFIX
  direct_sample_index = directory / _DATASET_SAMPLE_INDEX_SUFFIX
  if direct_dataset_index.is_file() and direct_sample_index.is_file():
    return direct_dataset_index, direct_sample_index

  all_pairs: list[tuple[Path, Path]] = []
  for dataset_index_path in sorted(directory.glob(f"*{_DATASET_INDEX_SUFFIX}")):
    sample_index_name = dataset_index_path.name.replace(_DATASET_INDEX_SUFFIX, _DATASET_SAMPLE_INDEX_SUFFIX)
    sample_index_path = dataset_index_path.with_name(sample_index_name)
    if sample_index_path.is_file():
      all_pairs.append((dataset_index_path, sample_index_path))

  if not all_pairs:
    raise FileNotFoundError(f"Could not find blend index file pairs in directory: {index_dir}")

  split_tag = f"-BlendedDataset-{split}-"
  split_pairs = [pair for pair in all_pairs if split_tag in pair[0].name]
  if len(split_pairs) == 1:
    return split_pairs[0]
  if len(split_pairs) > 1:
    raise ValueError(
        f"Multiple blend index pairs match split '{split}' in {index_dir}: " f"{[pair[0].name for pair in split_pairs]}"
    )

  if len(all_pairs) == 1:
    return all_pairs[0]
  raise ValueError(
      f"Multiple blend index pairs found in {index_dir}; provide split-specific files. "
      f"Candidates: {[pair[0].name for pair in all_pairs]}"
  )


def _load_or_build_blend_indices(
    num_datasets: int,
    weights: np.ndarray,
    size: int,
    dataset_lengths: Sequence[int] | None = None,
    cache_dir: str | None = None,
    blend_index_dir: str | None = None,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
  """Load pre-built blend indices or compute them from scratch."""
  if blend_index_dir:
    try:
      dataset_index_path, dataset_sample_index_path = _find_index_pair_in_dir(blend_index_dir, split)
      dataset_index = np.load(dataset_index_path, allow_pickle=False, mmap_mode="r")
      dataset_sample_index = np.load(dataset_sample_index_path, allow_pickle=False, mmap_mode="r")
      _validate_indices(
          dataset_index,
          dataset_sample_index,
          num_datasets,
          dataset_lengths,
          expected_size=size,
      )
      logger.info(
          "Loaded pre-generated blend indices from %s and %s",
          dataset_index_path,
          dataset_sample_index_path,
      )
      return dataset_index, dataset_sample_index
    except _RECOVERABLE_ERRORS as error:
      logger.warning(
          "Failed to load pre-generated blend indices from %s, falling back to cache/computation. Error: %s",
          blend_index_dir,
          error,
      )

  cache_paths = None
  if cache_dir:
    cache_key = _build_cache_key(dataset_lengths, weights, size)
    cache_paths = _build_cache_paths(cache_dir, cache_key, split)
    if cache_paths[0].is_file() and cache_paths[1].is_file():
      try:
        dataset_index = np.load(cache_paths[0], allow_pickle=False, mmap_mode="r")
        dataset_sample_index = np.load(cache_paths[1], allow_pickle=False, mmap_mode="r")
        _validate_indices(
            dataset_index,
            dataset_sample_index,
            num_datasets,
            dataset_lengths,
            expected_size=size,
        )
        logger.info(
            "Loaded cached blend indices from %s and %s",
            cache_paths[0],
            cache_paths[1],
        )
        return dataset_index, dataset_sample_index
      except _RECOVERABLE_ERRORS as error:
        logger.warning(
            "Blend index cache validation failed for %s / %s; recomputing. Error: %s",
            cache_paths[0],
            cache_paths[1],
            error,
        )

  dataset_index = np.zeros(size, dtype=np.int16)
  dataset_sample_index = np.zeros(size, dtype=np.int64)
  build_blending_indices(
      dataset_index=dataset_index,
      dataset_sample_index=dataset_sample_index,
      weights=weights,
      num_datasets=num_datasets,
      size=size,
  )

  if cache_paths and _mmap_index_utils.is_primary_process():
    os.makedirs(cache_dir, exist_ok=True)
    try:
      _mmap_index_utils.save_npy_atomic(cache_paths[0], dataset_index)
      _mmap_index_utils.save_npy_atomic(cache_paths[1], dataset_sample_index)
      logger.info(
          "Saved blend indices to cache: %s and %s",
          cache_paths[0],
          cache_paths[1],
      )
    except OSError as error:
      logger.warning("Failed to save blend cache to %s: %s", cache_dir, error)

  return dataset_index, dataset_sample_index


class MegatronBlendedDataSource:
  """Megatron-compatible blend at the MapDataset level.

  Operates on MapDatasets so that host sharding can be applied
  AFTER blending, ensuring global batch alignment with Megatron-LM.

  Usage:
    blended = grain.MapDataset.source(
        MegatronBlendedDataSource(map_datasets, weights, ...)
    )
    blended = blended[host_index::num_hosts]   # shard after blend
    dataset = blended.to_iter_dataset(...)
  """

  def __init__(
      self,
      map_datasets: Sequence[grain.MapDataset],
      weights: Sequence[float],
      size: int | None = None,
      dataset_lengths: Sequence[int] | None = None,
      cache_dir: str | None = None,
      blend_index_dir: str | None = None,
      split: str = "train",
  ):
    filtered = _normalize_and_filter_weights(map_datasets, weights, dataset_lengths)
    self._datasets, self._weights, self._dataset_lengths = filtered
    self._num_datasets = len(self._datasets)

    if self._dataset_lengths is None:
      self._dataset_lengths = [len(ds) for ds in self._datasets]

    if size is None:
      self._size = _infer_size_from_lengths(self._weights, self._dataset_lengths)
    else:
      self._size = _megatron_blend_size(int(size), self._weights)
      if self._size <= 0:
        raise ValueError(f"size must be positive, got {self._size}")

    self._dataset_index, self._dataset_sample_index = _load_or_build_blend_indices(
        num_datasets=self._num_datasets,
        weights=self._weights,
        size=self._size,
        dataset_lengths=self._dataset_lengths,
        cache_dir=cache_dir,
        blend_index_dir=blend_index_dir,
        split=split,
    )

  def __len__(self) -> int:
    return self._size

  def __getitem__(self, idx):
    ds_id = int(self._dataset_index[idx])
    sample_id = int(self._dataset_sample_index[idx])
    return self._datasets[ds_id][sample_id]

  def __str__(self) -> str:
    return f"MegatronBlendedDataSource([{self._num_datasets} datasets], " f"size={self._size})"
