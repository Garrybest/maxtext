"""Blending layer for lazy data sources, replicating antllm BlendableDataset.

Reuses MaxText's ``build_blending_indices()`` which implements the same
greedy error minimization algorithm as antllm's C++ helper.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from typing import Sequence

import grain.python as grain
from grain.python import SharedMemoryArray
import numpy as np

from maxtext.input_pipeline._megatron_blending import build_blending_indices


def _new_shm(shape, dtype) -> SharedMemoryArray:
  """Allocate a zeroed SharedMemoryArray in POSIX shared memory."""
  shm_arr = SharedMemoryArray(shape, np.dtype(dtype))
  shm_arr.fill(0)
  shm_arr.unlink_on_del()
  return shm_arr


def _take_shm(arr: np.ndarray, indices: np.ndarray) -> SharedMemoryArray:
  """Fancy-index ``arr[indices]`` directly into a new SharedMemoryArray (zero-copy)."""
  shm_arr = SharedMemoryArray(indices.shape, arr.dtype)
  np.take(arr, indices, out=shm_arr)
  shm_arr.unlink_on_del()
  return shm_arr


logger = logging.getLogger(__name__)


def _blend_cache_key(
    weights: np.ndarray,
    size: int,
    num_datasets: int,
    shuffle_seed: int,
    shuffle_only_dataset: bool,
    scatter_id: int = 0,
    source_lengths: list[int] | None = None,
) -> str:
  """Build a deterministic MD5 cache key from all parameters that affect blending indices.

  The key captures every input to ``build_blending_indices`` plus the
  post-shuffle configuration.  Any change in weights, size, shuffle
  behaviour, or dataset count produces a different key.

  ``scatter_id`` (= ``process_index % abs(loader_scatter)``) and
  ``source_lengths`` ensure per-scatter-group uniqueness: each scatter
  group loads different shards, producing different source lengths and
  blend sizes.  All processes in the same scatter group share a single
  cache entry.

  Weights are rounded to 15 significant digits (full float64 precision)
  to avoid spurious mismatches from repr artifacts while still catching
  any real weight change.
  """
  payload = {
      "weights": [round(float(w), 15) for w in weights],
      "size": size,
      "num_datasets": num_datasets,
      "shuffle_seed": shuffle_seed,
      "shuffle_only_dataset": shuffle_only_dataset,
      "scatter_id": scatter_id,
  }
  if source_lengths is not None:
    payload["source_lengths"] = source_lengths
  payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
  return hashlib.md5(payload_json.encode("utf-8"), usedforsecurity=False).hexdigest()


def _try_load_blend_cache(cache_dir: str, cache_key: str, size: int):
  """Try to load cached blending indices via mmap.

  Returns (dataset_index, dataset_sample_index) or None.
  Uses ``mmap_mode='r'`` so the arrays are memory-mapped from disk,
  avoiding a full copy into RAM for large blend indices.
  """
  di_path = os.path.join(cache_dir, f"{cache_key}.dataset_index.npy")
  dsi_path = os.path.join(cache_dir, f"{cache_key}.dataset_sample_index.npy")
  if not (os.path.isfile(di_path) and os.path.isfile(dsi_path)):
    return None
  try:
    dataset_index = np.load(di_path, mmap_mode="r")
    dataset_sample_index = np.load(dsi_path, mmap_mode="r")
    if dataset_index.shape != (size,) or dataset_sample_index.shape != (size,):
      logger.warning("Blend cache shape mismatch, recomputing (expected %d)", size)
      return None
    logger.info("Loaded blend indices from cache (mmap): %s", cache_dir)
    return dataset_index, dataset_sample_index
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("Failed to load blend cache: %s", e)
    return None


def _save_blend_cache(cache_dir: str, cache_key: str, dataset_index: np.ndarray, dataset_sample_index: np.ndarray):
  """Atomically save blending indices to cache directory.

  Each array is first written to a temporary file in the same directory,
  then renamed to its final path.  ``os.rename`` is atomic on POSIX
  when source and destination are on the same filesystem, so concurrent
  readers never see a partially-written file.
  """
  try:
    os.makedirs(cache_dir, exist_ok=True)
    di_path = os.path.join(cache_dir, f"{cache_key}.dataset_index.npy")
    dsi_path = os.path.join(cache_dir, f"{cache_key}.dataset_sample_index.npy")

    # Write to temp files then atomically rename
    for arr, final_path in [(dataset_index, di_path), (dataset_sample_index, dsi_path)]:
      fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".tmp.npy")
      try:
        os.close(fd)
        np.save(tmp_path, arr)
        os.rename(tmp_path, final_path)
      except BaseException:
        # Clean up temp file on any failure
        try:
          os.unlink(tmp_path)
        except OSError:
          pass
        raise

    logger.info("Saved blend indices to cache: %s", cache_dir)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("Failed to save blend cache: %s", e)


class LazyBlendedDataSource(grain.RandomAccessDataSource):
  """Weighted blend of multiple lazy data sources.

  Constructs ``dataset_index`` and ``dataset_sample_index`` arrays using
  Megatron-LM's greedy error minimization, then routes each global index
  to the appropriate source + local sample.

  Optional ``shuffle_seed`` (antllm ``dataset_index_shuffle_seed``)
  shuffles the blending indices after construction so that consecutive
  global indices no longer follow the deterministic greedy order.
  """

  def __init__(
      self,
      sources: Sequence[grain.RandomAccessDataSource],
      weights: np.ndarray,
      size: int,
      shuffle_seed: int = -1,
      shuffle_only_dataset: bool = False,
      cache_dir: str | None = None,
      process_index: int = 0,
      loader_scatter: int = 0,
  ):
    if len(sources) != len(weights):
      raise ValueError(f"sources/weights length mismatch: {len(sources)} vs {len(weights)}")
    if size <= 0:
      raise ValueError(f"size must be positive, got {size}")
    for i, src in enumerate(sources):
      if len(src) == 0:
        raise ValueError(f"Source {i} is empty (length 0); all sources must have positive length")

    self._sources = list(sources)
    self._size = size
    num_datasets = len(sources)

    # Scatter group identity: all processes with the same scatter_id load
    # identical shards and produce identical blend indices → share one cache.
    abs_scatter = max(abs(loader_scatter), 1)
    scatter_id = process_index % abs_scatter
    # Only the first process in each scatter group writes the cache;
    # others will read it (possibly after a brief wait for the file).
    is_cache_writer = process_index < abs_scatter

    # antllm BlendableDataset does NOT re-normalize weights — the caller
    # (configure_data.py) already normalizes by int(sum).  Re-dividing by
    # np.sum(weights) here introduces ~1e-13 float64 drift that makes the
    # greedy blending indices diverge after a few iterations.
    weights = np.asarray(weights, dtype=np.float64)

    # Cache key covers all inputs that determine the final index arrays.
    # Uses scatter_id (not process_index) so the entire scatter group
    # shares a single cache entry.
    source_lengths = [len(src) for src in sources]
    cache_key = _blend_cache_key(
        weights,
        size,
        num_datasets,
        shuffle_seed,
        shuffle_only_dataset,
        scatter_id=scatter_id,
        source_lengths=source_lengths,
    )
    self._blend_cache_dir = cache_dir
    self._blend_cache_key = cache_key

    cached = _try_load_blend_cache(cache_dir, cache_key, size) if cache_dir else None

    if cached is not None:
      self._dataset_index, self._dataset_sample_index = cached
    else:
      # Build blending indices directly into shared memory (zero-copy).
      logger.info("Building blending indices: size=%d, num_datasets=%d ...", size, num_datasets)
      self._dataset_index = _new_shm((size,), np.int16)
      self._dataset_sample_index = _new_shm((size,), np.int64)
      build_blending_indices(
          dataset_index=self._dataset_index,
          dataset_sample_index=self._dataset_sample_index,
          weights=weights,
          num_datasets=num_datasets,
          size=size,
      )

      # Optional shuffle (antllm BlendableDataset lines 204-218)
      if shuffle_seed > 0:
        rng = np.random.RandomState(shuffle_seed)
        inds = np.arange(size, dtype=np.int64)
        rng.shuffle(inds)
        self._dataset_index = _take_shm(self._dataset_index, inds)
        if shuffle_only_dataset:
          # antllm: only shuffle dataset_index, regenerate dataset_sample_index
          # as contiguous [0, 1, 2, ...] per dataset (lines 213-216)
          del inds
          for i in range(num_datasets):
            mask = self._dataset_index == i
            self._dataset_sample_index[mask] = np.arange(mask.sum())
        else:
          self._dataset_sample_index = _take_shm(self._dataset_sample_index, inds)
          del inds

      # Only one process per scatter group writes; atomic save prevents
      # concurrent readers from seeing partial files.
      if cache_dir and is_cache_writer:
        _save_blend_cache(cache_dir, cache_key, self._dataset_index, self._dataset_sample_index)

    logger.info(
        "LazyBlendedDataSource: %d datasets, size=%d, weights=%s, shuffle_seed=%d, shuffle_only_dataset=%s",
        num_datasets,
        size,
        weights.tolist(),
        shuffle_seed,
        shuffle_only_dataset,
    )

  def __len__(self) -> int:
    return self._size

  def __getitem__(self, idx: int):
    idx_in_epoch = idx % self._size
    ds_idx = int(self._dataset_index[idx_in_epoch])
    sample_idx = int(self._dataset_sample_index[idx_in_epoch])
    result = self._sources[ds_idx][sample_idx]
    result["dataset_id"] = np.int32(ds_idx)
    return result

  @property
  def sources(self) -> list:
    """The constituent data sources."""
    return self._sources

  @property
  def dataset_index(self) -> np.ndarray:
    return self._dataset_index

  @property
  def dataset_sample_index(self) -> np.ndarray:
    return self._dataset_sample_index

  def __getstate__(self):
    state = {
        "sources": self._sources,
        "size": self._size,
        "blend_cache_dir": self._blend_cache_dir,
        "blend_cache_key": self._blend_cache_key,
    }
    if isinstance(self._dataset_index, SharedMemoryArray):
      state["dataset_index"] = self._dataset_index
      state["dataset_sample_index"] = self._dataset_sample_index
    return state

  def __setstate__(self, state):
    self._sources = state["sources"]
    self._size = state["size"]
    self._blend_cache_dir = state.get("blend_cache_dir")
    self._blend_cache_key = state.get("blend_cache_key")
    if "dataset_index" in state:
      self._dataset_index = state["dataset_index"]
      self._dataset_sample_index = state["dataset_sample_index"]
    else:
      cached = _try_load_blend_cache(self._blend_cache_dir, self._blend_cache_key, self._size)
      if cached is None:
        raise RuntimeError(f"Blend cache missing in worker: dir={self._blend_cache_dir}, key={self._blend_cache_key}")
      self._dataset_index, self._dataset_sample_index = cached
