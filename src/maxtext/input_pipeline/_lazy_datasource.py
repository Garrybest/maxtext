"""Grain-compatible data sources for antllm .scatter/.lazy format.

Replicates the exact consumption order of antllm's SampleLazyLoader and
AllSampleLazyLoader so that MaxText on TPU produces identical training
sequences for a given global index.

On-disk format (per scatter shard):
  {name}.scatter/{shard_id}.lazy/
    text           — raw int32 token binary (concatenated documents)
    text.lens.npy  — per-document token counts (int64/int32)
"""

from __future__ import annotations

import logging
import os
import sys
from bisect import bisect_right
from concurrent.futures import as_completed, ThreadPoolExecutor  # pylint: disable=no-name-in-module
from pathlib import Path

import grain.python as grain
import numpy as np

logger = logging.getLogger(__name__)


class LazyIndexedDataset:
  """Low-level reader for a single .lazy shard's binary tokens + lens.

  Uses ``os.pread()`` for thread-safe reads without mmap (avoids SIGBUS on
  shared filesystems, matches MaxText _mmap_datasource.py convention).
  """

  def __init__(self, lazy_dir: str, data_type: str = "text", dtype=np.int32):
    self._lazy_dir = lazy_dir
    self._data_type = data_type
    self._dtype = np.dtype(dtype)
    self._element_size = self._dtype.itemsize
    self._bin_fd = None

    lens_path = os.path.join(lazy_dir, f"{data_type}.lens.npy")
    self._sizes = np.load(lens_path, mmap_mode="r")
    self._ends = np.cumsum(self._sizes, dtype=np.int64)
    self._num_docs = len(self._sizes)
    self._total_tokens = int(self._ends[-1]) if self._num_docs > 0 else 0

    bin_path = os.path.join(lazy_dir, data_type)
    self._bin_path = bin_path
    self._bin_fd = os.open(bin_path, os.O_RDONLY)

  @property
  def sizes(self) -> np.ndarray:
    return self._sizes

  @property
  def ends(self) -> np.ndarray:
    return self._ends

  @property
  def total_tokens(self) -> int:
    return self._total_tokens

  @property
  def num_docs(self) -> int:
    return self._num_docs

  def get(self, idx: int) -> np.ndarray:
    """Read all tokens for document ``idx``."""
    if idx < 0 or idx >= self._num_docs:
      raise IndexError(f"Document index {idx} out of range [0, {self._num_docs})")
    start = int(self._ends[idx - 1]) if idx > 0 else 0
    length = int(self._sizes[idx])
    return self._read_tokens(start, length)

  def read_range(self, start: int, end: int) -> np.ndarray:
    """Read raw tokens in the half-open range ``[start, end)``."""
    length = end - start
    if length <= 0:
      return np.array([], dtype=self._dtype)
    return self._read_tokens(start, length)

  def _read_tokens(self, offset: int, count: int) -> np.ndarray:
    byte_offset = offset * self._element_size
    nbytes = count * self._element_size
    data = os.pread(self._bin_fd, nbytes, byte_offset)
    if len(data) != nbytes:
      raise IOError(
          f"Short read from {self._bin_path}: expected {nbytes} bytes " f"at offset {byte_offset}, got {len(data)}"
      )
    return np.frombuffer(data, dtype=self._dtype).copy()

  def close(self):
    if self._bin_fd is not None:
      os.close(self._bin_fd)
      self._bin_fd = None

  def __del__(self):
    self.close()

  def __getstate__(self):
    return {
        "lazy_dir": self._lazy_dir,
        "data_type": self._data_type,
        "dtype": self._dtype,
    }

  def __setstate__(self, state):
    self.__init__(state["lazy_dir"], state["data_type"], state["dtype"])


class LazySlidingWindowDataSource(grain.RandomAccessDataSource):
  """Grain data source replicating antllm ``fetch_data_sliding_window``.

  Key details from antllm:
  - ``SampleLazyLoaderConfig.__post_init__`` does ``seq_length += 1``, so
    the caller should pass the *incremented* seq_length.
  - ``mapping_ends`` equals ``ends`` (cumsum of original doc lens) in
    sliding_window mode.
  - Sample count = ``total_tokens // seq_length`` (drop_last=True).
  - EOS is injected dynamically at document boundaries; CLS optionally
    prepended per document segment.
  - Trailing tokens padded with EOS, then truncated to ``seq_length``.
  """

  def __init__(
      self,
      dataset: LazyIndexedDataset,
      seq_length: int,
      eos_token_id: int,
      cls_token_id: int | None = None,
      add_cls: bool = False,
      drop_last: bool = True,
  ):
    self._dataset = dataset
    self._seq_length = seq_length
    self._eos_token_id = eos_token_id
    self._cls_token_id = cls_token_id
    self._add_cls = add_cls
    self._drop_last = drop_last
    self._ends = dataset.ends
    total = dataset.total_tokens
    if drop_last:
      self._num_samples = total // seq_length
    else:
      self._num_samples = -(-total // seq_length)  # ceil division

  def __len__(self) -> int:
    return self._num_samples

  def __getitem__(self, index: int):
    seq_length = self._seq_length
    st = index * seq_length
    ed = st + seq_length
    ends = self._ends
    total = self._dataset.total_tokens

    # Read raw tokens (may be shorter than seq_length for last sample)
    raw_data = self._dataset.read_range(st, min(ed, total))

    # Find document boundaries that overlap [st, ed)
    # antllm uses self.ends (== mapping_ends in sliding_window mode)
    st_id = bisect_right(ends, st)
    ed_id = min(bisect_right(ends, ed), len(ends) - 1)

    # Compute per-document token counts in this window
    data_lens = []
    for i in range(st_id, ed_id + 1):
      doc_len = int(ends[i]) - (int(ends[i - 1]) if i > 0 else 0)
      data_lens.append(doc_len)

    # Subtract the offset into the first document
    st_offset = int(ends[st_id - 1]) if st_id > 0 else 0
    st_offset = st - st_offset
    data_lens[0] -= st_offset

    # Reassemble with EOS injection (exact antllm logic).
    # Pre-allocate buffer filled with EOS (doubles as padding).
    result = np.full(seq_length, self._eos_token_id, dtype=np.int32)
    data_offset = 0
    offset = 0
    for i, doc_len in enumerate(data_lens):
      if self._add_cls and offset < seq_length:
        result[offset] = self._cls_token_id
        offset += 1
      if offset >= seq_length:
        break
      n = min(doc_len, seq_length - offset)
      result[offset : offset + n] = raw_data[data_offset : data_offset + n]
      data_offset += n
      offset += n
      if offset < seq_length:
        # EOS at document boundary (already filled by np.full for trailing)
        offset += 1

    return {"text": result}

  def __getstate__(self):
    return {
        "dataset": self._dataset,
        "seq_length": self._seq_length,
        "eos_token_id": self._eos_token_id,
        "cls_token_id": self._cls_token_id,
        "add_cls": self._add_cls,
        "drop_last": self._drop_last,
    }

  def __setstate__(self, state):
    self.__init__(**state)


class LazyMapDataSource(grain.RandomAccessDataSource):
  """Grain data source replicating antllm ``fetch_data_mapping`` (no bin_index).

  Uses a pre-generated ``index_mapping`` array to reorder documents before
  the sliding-window cut + EOS injection.
  """

  def __init__(
      self,
      dataset: LazyIndexedDataset,
      seq_length: int,
      eos_token_id: int,
      index_mapping: np.ndarray,
      cls_token_id: int | None = None,
      add_cls: bool = False,
      drop_last: bool = True,
  ):
    self._dataset = dataset
    self._seq_length = seq_length
    self._eos_token_id = eos_token_id
    self._cls_token_id = cls_token_id
    self._add_cls = add_cls
    self._drop_last = drop_last
    self._index_mapping = index_mapping
    self._ends = dataset.ends

    # Compute mapping_ends from reordered document sizes
    mapping_lens = dataset.sizes[index_mapping]
    self._mapping_ends = np.cumsum(mapping_lens)
    total_mapped = int(self._mapping_ends[-1]) if len(self._mapping_ends) > 0 else 0
    if drop_last:
      self._num_samples = total_mapped // seq_length
    else:
      self._num_samples = -(-total_mapped // seq_length)

  def __len__(self) -> int:
    return self._num_samples

  def __getitem__(self, index: int):
    seq_length = self._seq_length
    st = index * seq_length
    ed = st + seq_length

    st_id = bisect_right(self._mapping_ends, st)
    ed_id = min(bisect_right(self._mapping_ends, ed), len(self._mapping_ends) - 1)

    # antllm fetch_data_mapping (non-bin_index path): st_offset = 0 (line 1241)
    index_list = list(range(st_id, ed_id + 1))
    st_offset = 0

    result = np.full(seq_length, self._eos_token_id, dtype=np.int32)
    offset = 0
    for idx in index_list:
      if self._add_cls and offset < seq_length:
        result[offset] = self._cls_token_id
        offset += 1
      if offset >= seq_length:
        break
      ori_idx = int(self._index_mapping[idx])
      curr_st = int(self._ends[ori_idx - 1]) if ori_idx > 0 else 0
      curr_ed = int(self._ends[ori_idx])
      curr_len = curr_ed - curr_st - st_offset
      n = min(curr_len, seq_length - offset)
      file_st = curr_st + st_offset
      segment = self._dataset.read_range(file_st, file_st + n)
      result[offset : offset + n] = segment
      offset += n
      st_offset = 0
      if offset < seq_length:
        offset += 1  # EOS (already filled by np.full)

    return {"text": result}

  def __getstate__(self):
    return {
        "dataset": self._dataset,
        "seq_length": self._seq_length,
        "eos_token_id": self._eos_token_id,
        "index_mapping": self._index_mapping,
        "cls_token_id": self._cls_token_id,
        "add_cls": self._add_cls,
        "drop_last": self._drop_last,
    }

  def __setstate__(self, state):
    self.__init__(**state)


class LazyPackDataSource(grain.RandomAccessDataSource):
  """Grain data source replicating antllm ``fetch_data_mapping`` with bin_index.

  Packs multiple document fragments into fixed-length samples using a
  pre-generated or runtime-built bin index.  antllm's pack mode always
  sets ``index_mapping = None`` (lazy_loader.py L912), so bin_index stores
  physical doc indices directly — no indirect mapping needed.

  Two construction modes:
    - **File-based**: ``bin_index_path`` points to a directory containing
      ``index_offset.bin`` + ``lens.npy`` (pre-generated offline).
    - **In-memory**: ``bin_index_data = (bin_entries, bin_lens)`` from
      runtime BFD packing (see ``_lazy_bfd_packing.build_bin_index``).
  """

  def __init__(
      self,
      dataset: LazyIndexedDataset,
      seq_length: int,
      eos_token_id: int,
      bin_index_path: str | None = None,
      bin_index_data: tuple[np.ndarray, np.ndarray] | None = None,
      cls_token_id: int | None = None,
      add_cls: bool = False,
  ):
    if bin_index_path is None and bin_index_data is None:
      raise ValueError("Either bin_index_path or bin_index_data must be provided")

    self._dataset = dataset
    self._seq_length = seq_length
    self._eos_token_id = eos_token_id
    self._cls_token_id = cls_token_id
    self._add_cls = add_cls
    self._bin_index_path = bin_index_path
    self._ends = dataset.ends
    self._bin_data_fd = None
    self._bin_entries = None  # in-memory mode

    if bin_index_data is not None:
      # In-memory mode (runtime BFD)
      bin_entries, bin_lens = bin_index_data
      self._bin_entries = np.asarray(bin_entries, dtype=np.int32)
      if self._bin_entries.ndim == 1 and self._bin_entries.size == 0:
        self._bin_entries = self._bin_entries.reshape(0, 2)
      self._bin_ends = np.cumsum(bin_lens)
      self._num_samples = len(self._bin_ends)
    else:
      # File-based mode (pre-generated)
      bin_lens = np.load(os.path.join(bin_index_path, "lens.npy"), mmap_mode="r")
      self._bin_ends = np.cumsum(bin_lens)
      self._num_samples = len(self._bin_ends)
      bin_data_path = os.path.join(bin_index_path, "index_offset.bin")
      self._bin_data_fd = os.open(bin_data_path, os.O_RDONLY)

  def __len__(self) -> int:
    return self._num_samples

  def _read_bin_index(self, start: int, end: int) -> np.ndarray:
    """Read (doc_idx, offset) pairs for a bin.

    ``start`` and ``end`` are in int32 element counts (not pairs).
    """
    if self._bin_entries is not None:
      # In-memory mode: bin_entries is (N, 2), start/end are element counts
      return self._bin_entries[start // 2 : end // 2]
    # File-based mode
    count = end - start
    byte_offset = start * 4  # int32
    nbytes = count * 4
    data = os.pread(self._bin_data_fd, nbytes, byte_offset)
    return np.frombuffer(data, dtype=np.int32).reshape(-1, 2)

  def __getitem__(self, index: int):
    seq_length = self._seq_length
    bin_start = 0 if index == 0 else int(self._bin_ends[index - 1])
    bin_end = int(self._bin_ends[index])
    index_list = self._read_bin_index(bin_start, bin_end)

    result = np.full(seq_length, self._eos_token_id, dtype=np.int32)
    offset = 0
    for doc_idx, st_offset in index_list:
      doc_idx = int(doc_idx)
      st_offset = int(st_offset)
      if self._add_cls and offset < seq_length:
        result[offset] = self._cls_token_id
        offset += 1
      if offset >= seq_length:
        break
      curr_st = int(self._ends[doc_idx - 1]) if doc_idx > 0 else 0
      curr_ed = int(self._ends[doc_idx])
      curr_len = curr_ed - curr_st - st_offset
      n = min(curr_len, seq_length - offset)
      file_st = curr_st + st_offset
      segment = self._dataset.read_range(file_st, file_st + n)
      result[offset : offset + n] = segment
      offset += n
      if offset < seq_length:
        offset += 1  # EOS (already filled by np.full)

    return {"text": result}

  def close(self):
    if self._bin_data_fd is not None:
      os.close(self._bin_data_fd)
      self._bin_data_fd = None

  def __del__(self):
    self.close()

  def __getstate__(self):
    state = {
        "dataset": self._dataset,
        "seq_length": self._seq_length,
        "eos_token_id": self._eos_token_id,
        "cls_token_id": self._cls_token_id,
        "add_cls": self._add_cls,
    }
    if self._bin_entries is not None:
      # In-memory mode: serialize the arrays
      state["bin_index_data"] = (self._bin_entries, np.diff(self._bin_ends, prepend=0))
    else:
      state["bin_index_path"] = self._bin_index_path
    return state

  def __setstate__(self, state):
    self.__init__(**state)


class MultiShardLazyDataSource(grain.RandomAccessDataSource):
  """Concatenates multiple .lazy shards, replicating antllm AllSampleLazyLoader.

  Routes ``global_index`` to the correct shard via ``bisect_right(cnt_ends)``.
  Supports optional per-epoch online shuffle (``loader_online_shuffle``).
  """

  def __init__(
      self,
      scatter_dir: str,
      mode: str,
      seq_length: int,
      eos_token_id: int,
      data_type: str = "text",
      cls_token_id: int | None = None,
      add_cls: bool = False,
      drop_last: bool = True,
      index_mapping_path: str | None = None,
      bin_index_base_path: str | None = None,
      loader_online_shuffle: bool = False,
      loader_seed: int = 1234,
      loader_scatter: int = -1,
      process_index: int = 0,
      num_epochs: int = 1,
      bfd_pack_sort_by_lens: bool = False,
      pack_divisible_by: int = -1,
  ):
    self._scatter_dir = scatter_dir
    self._mode = mode
    self._seq_length = seq_length
    self._eos_token_id = eos_token_id
    self._data_type = data_type
    self._cls_token_id = cls_token_id
    self._add_cls = add_cls
    self._drop_last = drop_last
    self._index_mapping_path = index_mapping_path
    self._bin_index_base_path = bin_index_base_path
    self._loader_online_shuffle = loader_online_shuffle
    self._loader_seed = loader_seed
    self._loader_scatter = loader_scatter
    self._process_index = process_index
    self._bfd_pack_sort_by_lens = bfd_pack_sort_by_lens
    self._pack_divisible_by = pack_divisible_by
    # ---------------------------------------------------------------
    # Epoch handling
    # ---------------------------------------------------------------
    # Grain's MapDataset iterates indices 0..len()-1, then StopIteration.
    # Grain's .repeat(N) does NOT help: it calls __getitem__(idx % original_len)
    # internally, so the source never sees idx >= original_len and our epoch
    # detection (epoch = idx // total_samples) would always compute epoch=0.
    #
    # Instead we inflate __len__ to total_samples * num_epochs.  Grain then
    # iterates 0..total_samples*num_epochs-1, and __getitem__ naturally sees
    # indices that span multiple epochs:
    #   epoch 0: idx 0 .. total_samples-1
    #   epoch 1: idx total_samples .. 2*total_samples-1
    #   ...
    # This lets _set_epoch() fire at epoch boundaries to re-shuffle when
    # loader_online_shuffle is enabled.
    #
    # For num_epochs=None (infinite), we use sys.maxsize as a practical upper
    # bound — large enough that training will never exhaust it.
    # ---------------------------------------------------------------
    if num_epochs is None:
      self._num_epochs = sys.maxsize
    else:
      self._num_epochs = max(num_epochs, 1)

    self._loaders = self._build_loaders()

    # Filter out empty loaders (same as antllm)
    non_empty = [loader for loader in self._loaders if len(loader) > 0]
    if non_empty:
      self._loaders = non_empty

    sample_cnts = [len(loader) for loader in self._loaders]
    self._cnt_ends = np.cumsum(sample_cnts)
    self._total_samples = int(self._cnt_ends[-1]) if len(self._cnt_ends) > 0 else 0

    if self._loader_online_shuffle:
      self._current_epoch = -1
      self._shuffled_index = None

    logger.info(
        "MultiShardLazyDataSource: scatter_dir=%s, mode=%s, shards=%d, total_samples=%d, num_epochs=%s",
        scatter_dir,
        mode,
        len(self._loaders),
        self._total_samples,
        self._num_epochs,
    )

  def _filter_shards(self, lazy_dirs: list) -> list:
    """Filter .lazy dirs by scatter group, replicating antllm shard assignment."""
    if self._loader_scatter > 0:
      # Positive: each rank reads 1 shard (antllm get_scatter_path)
      scatter_id = self._process_index % self._loader_scatter
      if scatter_id >= len(lazy_dirs):
        raise ValueError(
            f"scatter_id {scatter_id} >= shard count {len(lazy_dirs)} "
            f"(loader_scatter={self._loader_scatter}, rank={self._process_index})"
        )
      return [lazy_dirs[scatter_id]]
    elif self._loader_scatter < -1:
      # Negative merge: each rank reads total/|scatter| shards
      abs_scatter = abs(self._loader_scatter)
      scatter_size = len(lazy_dirs) // abs_scatter
      if scatter_size == 0:
        raise ValueError(f"shard count {len(lazy_dirs)} < |loader_scatter| {abs_scatter}")
      rank = self._process_index % abs_scatter
      return lazy_dirs[rank * scatter_size : (rank + 1) * scatter_size]
    else:
      # -1 or 0: read all shards
      return lazy_dirs

  def _build_loaders(self) -> list:
    """Discover .lazy subdirectories and create per-shard data sources."""
    scatter_path = Path(self._scatter_dir)
    if not scatter_path.is_dir():
      raise FileNotFoundError(f"Scatter directory not found: {self._scatter_dir}")

    lazy_dirs = sorted(scatter_path.glob("*.lazy"), key=lambda p: int(p.stem))
    if not lazy_dirs:
      # Handle hash subdirectory structure: .scatter/{hash}/*.lazy
      sub_dirs = [d for d in scatter_path.iterdir() if d.is_dir() and not d.name.endswith(".lazy")]
      if len(sub_dirs) == 1:
        lazy_dirs = sorted(sub_dirs[0].glob("*.lazy"), key=lambda p: int(p.stem))
        if lazy_dirs:
          logger.info("Found .lazy dirs inside subdirectory: %s", sub_dirs[0].name)
      if not lazy_dirs:
        raise FileNotFoundError(f"No .lazy directories found in {self._scatter_dir}")

    lazy_dirs = self._filter_shards(lazy_dirs)

    ds_name = scatter_path.stem.replace(".scatter", "")
    num_shards = len(lazy_dirs)
    logger.info(
        "Loading %d shards from %s (mode=%s) ...",
        num_shards,
        self._scatter_dir,
        self._mode,
    )
    max_workers = min(8, num_shards)
    loaders = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
      futures = {pool.submit(self._build_one_loader, d, ds_name): idx for idx, d in enumerate(lazy_dirs)}
      for done_count, future in enumerate(as_completed(futures), 1):
        loaders.append((futures[future], future.result()))
        if done_count % max(num_shards // 10, 1) == 0 or done_count == num_shards:
          logger.info("  Shard loading: %d/%d (%.0f%%)", done_count, num_shards, 100.0 * done_count / num_shards)
    # Restore original order
    loaders.sort(key=lambda x: x[0])
    return [loader for _, loader in loaders]

  def _build_one_loader(self, lazy_dir: Path, ds_name: str):
    """Build a single shard data source (called in parallel by _build_loaders)."""
    dataset = LazyIndexedDataset(str(lazy_dir), self._data_type)
    if dataset.total_tokens == 0:
      return _EmptySource()

    if self._mode == "sliding_window":
      return LazySlidingWindowDataSource(
          dataset=dataset,
          seq_length=self._seq_length,
          eos_token_id=self._eos_token_id,
          cls_token_id=self._cls_token_id,
          add_cls=self._add_cls,
          drop_last=self._drop_last,
      )
    elif self._mode == "map":
      index_mapping = self._load_index_mapping(ds_name, lazy_dir.name)
      if index_mapping is not None:
        return LazyMapDataSource(
            dataset=dataset,
            seq_length=self._seq_length,
            eos_token_id=self._eos_token_id,
            index_mapping=index_mapping,
            cls_token_id=self._cls_token_id,
            add_cls=self._add_cls,
            drop_last=self._drop_last,
        )
      # Fallback to sliding_window when no index_mapping found
      return LazySlidingWindowDataSource(
          dataset=dataset,
          seq_length=self._seq_length,
          eos_token_id=self._eos_token_id,
          cls_token_id=self._cls_token_id,
          add_cls=self._add_cls,
          drop_last=self._drop_last,
      )
    elif self._mode == "pack":
      bin_path = self._resolve_bin_index_path(ds_name, lazy_dir.name)
      if bin_path is not None:
        # Pre-generated bin_index (antllm pregen_index_mapping_type="pack")
        return LazyPackDataSource(
            dataset=dataset,
            seq_length=self._seq_length,
            eos_token_id=self._eos_token_id,
            bin_index_path=bin_path,
            cls_token_id=self._cls_token_id,
            add_cls=self._add_cls,
        )
      # Auto-generate BFD bin index at runtime (antllm lazy_loader.py:886-888)
      from maxtext.input_pipeline._lazy_bfd_packing import build_bin_index  # pylint: disable=import-outside-toplevel

      bin_entries, bin_lens = build_bin_index(
          doc_lens=dataset.sizes,
          seq_length=self._seq_length,
          sort_by_lens=self._bfd_pack_sort_by_lens,
          pack_divisible_by=self._pack_divisible_by,
      )
      return LazyPackDataSource(
          dataset=dataset,
          seq_length=self._seq_length,
          eos_token_id=self._eos_token_id,
          bin_index_data=(bin_entries, bin_lens),
          cls_token_id=self._cls_token_id,
          add_cls=self._add_cls,
      )
    else:
      raise ValueError(f"Unknown loader mode: {self._mode}")

  def _load_index_mapping(self, ds_name: str, lazy_name: str) -> np.ndarray | None:
    if self._index_mapping_path is None:
      return None
    path = os.path.join(self._index_mapping_path, f"{ds_name}.scatter/{lazy_name}/index_mapping.npy")
    try:
      return np.load(path, allow_pickle=False, mmap_mode="r")
    except FileNotFoundError:
      return None

  def _resolve_bin_index_path(self, ds_name: str, lazy_name: str) -> str | None:
    if self._bin_index_base_path is None:
      return None
    path = os.path.join(self._bin_index_base_path, f"{ds_name}.scatter/{lazy_name}")
    if os.path.isfile(os.path.join(path, "lens.npy")):
      return path
    return None

  def _set_epoch(self, epoch: int):
    """Build shuffled index for the given epoch."""
    logger.info(
        "Online shuffle: building epoch %d index (%d samples, seed=%d) ...",
        epoch,
        self._total_samples,
        self._loader_seed + epoch,
    )
    self._shuffled_index = np.arange(self._total_samples)
    rng = np.random.RandomState(self._loader_seed + epoch)
    rng.shuffle(self._shuffled_index)
    self._current_epoch = epoch

  def set_num_epochs(self, num_epochs: int):
    """Set the number of epochs, inflating __len__ accordingly.

    Called by lazy_data_processing after the source is fully built and the
    total sample count is known, so num_epochs can be auto-computed from
    training steps.
    """
    self._num_epochs = num_epochs

  def __len__(self) -> int:
    # Return inflated length so Grain sees total_samples * num_epochs indices.
    # This avoids using Grain .repeat() which mods indices back to [0, len),
    # preventing our epoch-based shuffle rotation from ever triggering.
    return self._total_samples * self._num_epochs

  def __getitem__(self, global_index: int):
    if self._loader_online_shuffle:
      epoch = global_index // self._total_samples
      if epoch != self._current_epoch:
        self._set_epoch(epoch)
      global_index = int(self._shuffled_index[global_index % self._total_samples])
    else:
      global_index = global_index % self._total_samples

    loader_index = bisect_right(self._cnt_ends, global_index)
    local_index = global_index - (int(self._cnt_ends[loader_index - 1]) if loader_index > 0 else 0)
    return self._loaders[loader_index][local_index]

  def __getstate__(self):
    return {
        "scatter_dir": self._scatter_dir,
        "mode": self._mode,
        "seq_length": self._seq_length,
        "eos_token_id": self._eos_token_id,
        "data_type": self._data_type,
        "cls_token_id": self._cls_token_id,
        "add_cls": self._add_cls,
        "drop_last": self._drop_last,
        "index_mapping_path": self._index_mapping_path,
        "bin_index_base_path": self._bin_index_base_path,
        "loader_online_shuffle": self._loader_online_shuffle,
        "loader_seed": self._loader_seed,
        "loader_scatter": self._loader_scatter,
        "process_index": self._process_index,
        "num_epochs": self._num_epochs,
    }

  def __setstate__(self, state):
    self.__init__(**state)


class _EmptySource(grain.RandomAccessDataSource):
  """Placeholder for empty shards."""

  def __len__(self):
    return 0

  def __getitem__(self, index):
    raise IndexError("Empty source")


def get_lazy_path(path: str) -> str:
  """Convert a scatter shard path to the .lazy directory path (antllm convention)."""
  return os.path.splitext(path)[0] + ".lazy"
