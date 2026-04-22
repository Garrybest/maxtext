"""BFD (Best-Fit Decreasing) bin packing for lazy data, replicating antllm build_bin_index.

Packs variable-length documents into fixed-size bins (seq_length), minimizing
wasted padding. Uses a segment tree for O(log N) best-fit bin lookup.

Reference: antllm/data/lazy_loader.py:962-1057 (SampleLazyLoader.build_bin_index)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Min segment tree (replaces antllm's `from segment_tree import SegmentTree`)
# ---------------------------------------------------------------------------


class _SegmentTree:
  """Min segment tree for tracking available bin space.

  Supports:
    - ``query(left, right)`` → minimum value in index range [left, right]
    - ``update(index, value)`` → set element at index to value

  antllm uses ``tree.query(packing_len, seq_length - 1, 'min')`` to find the
  smallest available space >= packing_len.  The tree is indexed by *space
  value* (1..seq_length-1), and each element stores either ``seq_length``
  (meaning "no bin has this exact space") or the space value itself (meaning
  "at least one bin has this space").
  """

  def __init__(self, values: list[int]):
    n = len(values)
    self._n = n
    self._tree = [0] * (2 * n)
    # Build leaves
    for i in range(n):
      self._tree[n + i] = values[i]
    # Build internal nodes bottom-up
    for i in range(n - 1, 0, -1):
      self._tree[i] = min(self._tree[2 * i], self._tree[2 * i + 1])

  def update(self, index: int, value: int):
    """Set element at ``index`` to ``value``."""
    pos = index + self._n
    self._tree[pos] = value
    pos >>= 1
    while pos >= 1:
      self._tree[pos] = min(self._tree[2 * pos], self._tree[2 * pos + 1])
      pos >>= 1

  def query(self, left: int, right: int) -> int:
    """Return min value in [left, right] inclusive."""
    result = self._tree[left + self._n]  # initialize with left element
    l = left + self._n
    r = right + self._n + 1
    while l < r:
      if l & 1:
        result = min(result, self._tree[l])
        l += 1
      if r & 1:
        r -= 1
        result = min(result, self._tree[r])
      l >>= 1
      r >>= 1
    return result


# ---------------------------------------------------------------------------
# BFD bin packing
# ---------------------------------------------------------------------------


def build_bin_index(
    doc_lens: np.ndarray,
    seq_length: int,
    sort_by_lens: bool = False,
    pack_divisible_by: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
  """Build BFD bin index, replicating antllm ``SampleLazyLoader.build_bin_index``.

  Args:
    doc_lens: Per-document token counts (1-D int array).
    seq_length: Target sequence length (bin capacity).
    sort_by_lens: If True, process docs in descending order of
        ``len % seq_length`` (classic BFD heuristic).  Matches antllm
        ``bfd_pack_sort_by_lens``.
    pack_divisible_by: If > 0, round up packing lengths to the nearest
        multiple.  Matches antllm ``pack_sample_length_divisible_by``.

  Returns:
    ``(bin_entries, bin_lens)`` where:
      - ``bin_entries``: int32 array of shape ``(total_pairs, 2)``,
        each row is ``(doc_idx, token_offset)``.
      - ``bin_lens``: int32 array of shape ``(num_bins,)``,
        each element is ``num_entries_in_bin * 2`` (matching antllm's
        ``lens.npy`` convention: count of int32 elements, not pairs).
  """
  doc_lens = np.asarray(doc_lens, dtype=np.int64)
  num_docs = len(doc_lens)

  if num_docs == 0:
    return np.array([], dtype=np.int32).reshape(0, 2), np.array([], dtype=np.int32)

  # Iteration order
  if sort_by_lens:
    sorted_index = np.argsort(-(doc_lens % seq_length))
  else:
    sorted_index = np.arange(num_docs)

  # Parallel lists per bin instead of list[tuple] to reduce Python object
  # overhead (~28B per int vs ~56B per tuple).
  bin_doc_ids: dict[int, list[int]] = {}
  bin_offsets: dict[int, list[int]] = {}

  # space2bins[space] = list of bin_ids that have exactly `space` free tokens
  space2bins: dict[int, list[int]] = {s: [] for s in range(1, seq_length)}

  # Segment tree: indexed 0..seq_length-1, element i = min available space
  # for bins with space == i.  Initial value = seq_length means "no bin".
  tree = _SegmentTree([seq_length] * seq_length)

  next_bin_id = 0
  log_interval = max(num_docs // 20, 1)  # log every 5%

  for progress_cnt, i in enumerate(sorted_index):
    doc_len = int(doc_lens[i])
    curr_bin_id = next_bin_id

    # --- Full-length chunks ---
    if doc_len >= seq_length:
      num_full = doc_len // seq_length
      for chunk in range(num_full):
        bin_doc_ids[curr_bin_id] = [int(i)]
        bin_offsets[curr_bin_id] = [chunk * seq_length]
        curr_bin_id += 1
      next_bin_id = max(next_bin_id, curr_bin_id)

    # --- Remainder packing ---
    remainder = doc_len % seq_length
    if remainder == 0:
      continue

    packing_len = remainder
    if packing_len + 1 < seq_length:
      packing_len += 1  # EOS token

    # Optional alignment
    if pack_divisible_by > 0 and packing_len % pack_divisible_by != 0:
      packing_len += pack_divisible_by - (packing_len % pack_divisible_by)

    # Best-fit search
    bfd_space = tree.query(packing_len, seq_length - 1)

    if bfd_space == seq_length:
      # No bin fits → create new bin
      found_bin = curr_bin_id if curr_bin_id == next_bin_id else next_bin_id
      next_bin_id = found_bin + 1
      offset = doc_len - remainder if doc_len >= seq_length else 0
      bin_doc_ids[found_bin] = [int(i)]
      bin_offsets[found_bin] = [offset]
    else:
      # Found existing bin with enough space
      found_bin = space2bins[bfd_space].pop()
      if not space2bins[bfd_space]:
        tree.update(bfd_space, seq_length)  # mark space as unavailable
      offset = doc_len - remainder if doc_len >= seq_length else 0
      bin_doc_ids[found_bin].append(int(i))
      bin_offsets[found_bin].append(offset)

    # Update remaining space
    new_space = bfd_space - packing_len if bfd_space != seq_length else seq_length - packing_len
    if new_space > 0:
      tree.update(new_space, new_space)
      space2bins[new_space].append(found_bin)

    if progress_cnt % log_interval == 0 and progress_cnt > 0:
      logger.info(
          "  build_bin_index: %d/%d docs (%.0f%%), %d bins so far",
          progress_cnt,
          num_docs,
          100.0 * progress_cnt / num_docs,
          next_bin_id,
      )

  # --- Convert to flat arrays matching index_offset.bin + lens.npy format ---
  # Pre-allocate numpy output and write directly from dicts, avoiding
  # intermediate list copies.
  num_bins = len(bin_doc_ids)
  total_entries = sum(len(v) for v in bin_doc_ids.values())

  if total_entries > 0:
    bin_entries = np.empty((total_entries, 2), dtype=np.int32)
    bin_lens = np.empty(num_bins, dtype=np.int32)
    cursor = 0
    for bid in range(num_bins):
      docs = bin_doc_ids[bid]
      offs = bin_offsets[bid]
      n = len(docs)
      bin_lens[bid] = n * 2
      bin_entries[cursor : cursor + n, 0] = docs
      bin_entries[cursor : cursor + n, 1] = offs
      cursor += n
  else:
    bin_entries = np.array([], dtype=np.int32).reshape(0, 2)
    bin_lens = np.array([], dtype=np.int32)
  del bin_doc_ids, bin_offsets

  logger.info(
      "BFD packing: %d docs → %d bins (seq_length=%d, sort=%s, divisible_by=%d)",
      num_docs,
      num_bins,
      seq_length,
      sort_by_lens,
      pack_divisible_by,
  )

  return bin_entries, bin_lens
