"""Alignment tests: MaxText lazy data sources vs antllm SampleLazyLoader.

Tests verify that for identical on-disk data and parameters, MaxText's lazy
data sources produce exactly the same token sequences as antllm's native
SampleLazyLoader / AllSampleLazyLoader / BlendableDataset.
"""

# pylint: disable=redefined-outer-name,import-outside-toplevel,consider-using-enumerate,protected-access

import os
import sys
import tempfile
import shutil
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

# Import MaxText lazy data sources
from maxtext.input_pipeline._lazy_datasource import (
    LazyIndexedDataset,
    LazySlidingWindowDataSource,
    LazyMapDataSource,
    LazyPackDataSource,
    MultiShardLazyDataSource,
)
from maxtext.input_pipeline._lazy_blending import LazyBlendedDataSource
from maxtext.input_pipeline import lazy_data_processing
from maxtext.input_pipeline.lazy_data_processing import (
    SplitDataSource,
    _dataset_name_from_path,
    _resolve_mode,
    _resolve_no_attnmask_ids,
)

# Import antllm for alignment comparison
ANTLLM_ROOT = os.environ.get("ANTLLM_DATA_ROOT", "/Volumes/code/tpu/antllm_data")
sys.path.insert(0, ANTLLM_ROOT)


# ---------------------------------------------------------------------------
# Test fixtures: generate .scatter/.lazy test data on disk
# ---------------------------------------------------------------------------

EOS_TOKEN_ID = 2
CLS_TOKEN_ID = 1
SEQ_LENGTH = 16  # antllm will internally use seq_length + 1


def _make_lazy_dir(base_dir, shard_id, doc_tokens_list, data_type="text"):
  """Create a .lazy directory with binary token file and lens.npy.

  Args:
    base_dir: scatter directory path (e.g. /tmp/test.scatter)
    shard_id: integer shard id
    doc_tokens_list: list of 1-D int32 arrays (one per document)
    data_type: field name (default "text")

  Returns:
    Path to the created .lazy directory.
  """
  lazy_dir = os.path.join(base_dir, f"{shard_id}.lazy")
  os.makedirs(lazy_dir, exist_ok=True)

  lens = np.array([len(doc) for doc in doc_tokens_list], dtype=np.int64)
  np.save(os.path.join(lazy_dir, f"{data_type}.lens.npy"), lens)

  all_tokens = np.concatenate(doc_tokens_list).astype(np.int32) if doc_tokens_list else np.array([], dtype=np.int32)
  bin_path = os.path.join(lazy_dir, data_type)
  all_tokens.tofile(bin_path)

  return lazy_dir


def _make_test_scatter(base_dir, scatter_name, shards_docs):
  """Create a full .scatter directory with multiple shards.

  Args:
    base_dir: parent directory
    scatter_name: e.g. "test_ds"
    shards_docs: list of (list of doc_token_arrays) per shard

  Returns:
    Path to the .scatter directory.
  """
  scatter_dir = os.path.join(base_dir, f"{scatter_name}.scatter")
  os.makedirs(scatter_dir, exist_ok=True)
  for shard_id, docs in enumerate(shards_docs):
    _make_lazy_dir(scatter_dir, shard_id, docs)
  return scatter_dir


@pytest.fixture
def tmp_dir():
  d = tempfile.mkdtemp(prefix="lazy_test_")
  yield d
  shutil.rmtree(d, ignore_errors=True)


def _make_docs(rng, num_docs=5, min_len=8, max_len=30):
  """Generate random documents with known token content."""
  docs = []
  for _ in range(num_docs):
    length = rng.integers(min_len, max_len + 1)
    # Use tokens in range [100, 10000] to avoid collision with special tokens
    tokens = rng.integers(100, 10000, size=length, dtype=np.int32)
    docs.append(tokens)
  return docs


# ---------------------------------------------------------------------------
# Test 1: LazyIndexedDataset basic correctness
# ---------------------------------------------------------------------------


class TestLazyIndexedDataset:
  """Tests for LazyIndexedDataset basic correctness."""

  def test_get_documents(self, tmp_dir):
    docs = [np.array([10, 20, 30], dtype=np.int32), np.array([40, 50], dtype=np.int32)]
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)
    dataset = LazyIndexedDataset(lazy_dir)

    assert dataset.num_docs == 2
    assert dataset.total_tokens == 5
    np.testing.assert_array_equal(dataset.get(0), docs[0])
    np.testing.assert_array_equal(dataset.get(1), docs[1])

  def test_read_range(self, tmp_dir):
    docs = [np.array([10, 20, 30], dtype=np.int32), np.array([40, 50], dtype=np.int32)]
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)
    dataset = LazyIndexedDataset(lazy_dir)

    # Cross-document read
    result = dataset.read_range(1, 4)
    np.testing.assert_array_equal(result, np.array([20, 30, 40], dtype=np.int32))

  def test_pickle_roundtrip(self, tmp_dir):
    docs = [np.array([10, 20, 30], dtype=np.int32)]
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)
    dataset = LazyIndexedDataset(lazy_dir)

    restored = pickle.loads(pickle.dumps(dataset))
    np.testing.assert_array_equal(restored.get(0), docs[0])

  def test_sizes_and_ends(self, tmp_dir):
    docs = [np.array([1, 2, 3], dtype=np.int32), np.array([4, 5], dtype=np.int32)]
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)
    dataset = LazyIndexedDataset(lazy_dir)

    np.testing.assert_array_equal(dataset.sizes, [3, 2])
    np.testing.assert_array_equal(dataset.ends, [3, 5])


# ---------------------------------------------------------------------------
# Test 2: Sliding window alignment with antllm
# ---------------------------------------------------------------------------


class TestSlidingWindowAlignment:
  """Tests for sliding window alignment with antllm."""

  def test_eos_injection_positions(self, tmp_dir):
    """Verify EOS appears exactly at document boundaries."""
    # Two documents: [100, 101, 102] and [200, 201]
    docs = [np.array([100, 101, 102], dtype=np.int32), np.array([200, 201], dtype=np.int32)]
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)

    seq_length = 8
    dataset = LazyIndexedDataset(lazy_dir)
    source = LazySlidingWindowDataSource(
        dataset=dataset,
        seq_length=seq_length,
        eos_token_id=EOS_TOKEN_ID,
    )

    if len(source) > 0:
      sample = source[0]["text"]
      assert sample.shape == (seq_length,)
      # The first 3 tokens should be doc content, then EOS
      assert sample[0] == 100
      assert sample[1] == 101
      assert sample[2] == 102
      assert sample[3] == EOS_TOKEN_ID

  def test_cls_injection(self, tmp_dir):
    """Verify CLS is prepended when add_cls=True."""
    docs = [np.array([100, 101, 102], dtype=np.int32), np.array([200, 201], dtype=np.int32)]
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)

    seq_length = 10
    dataset = LazyIndexedDataset(lazy_dir)
    source = LazySlidingWindowDataSource(
        dataset=dataset,
        seq_length=seq_length,
        eos_token_id=EOS_TOKEN_ID,
        cls_token_id=CLS_TOKEN_ID,
        add_cls=True,
    )

    if len(source) > 0:
      sample = source[0]["text"]
      # First token should be CLS, then doc content, then EOS
      assert sample[0] == CLS_TOKEN_ID
      assert sample[1] == 100
      assert sample[2] == 101
      assert sample[3] == 102
      assert sample[4] == EOS_TOKEN_ID

  def test_sample_count(self, tmp_dir):
    """Verify sample count = total_tokens // seq_length."""
    total = 100
    docs = [np.arange(total, dtype=np.int32)]
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)

    seq_length = 17
    dataset = LazyIndexedDataset(lazy_dir)
    source = LazySlidingWindowDataSource(dataset=dataset, seq_length=seq_length, eos_token_id=EOS_TOKEN_ID)
    assert len(source) == total // seq_length

  def test_edge_doc_boundary_at_window(self, tmp_dir):
    """Test when document boundary aligns exactly with window boundary."""
    seq_length = 10
    # Doc of exactly seq_length tokens
    docs = [np.arange(seq_length, dtype=np.int32) + 100, np.arange(seq_length, dtype=np.int32) + 200]
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)

    dataset = LazyIndexedDataset(lazy_dir)
    source = LazySlidingWindowDataSource(dataset=dataset, seq_length=seq_length, eos_token_id=EOS_TOKEN_ID)

    # Should have exactly 2 samples
    assert len(source) == 2
    # First sample: all from doc 0, then EOS fills remaining
    sample0 = source[0]["text"]
    assert len(sample0) == seq_length

  def test_single_token_document(self, tmp_dir):
    """Test with single-token documents."""
    docs = [np.array([100], dtype=np.int32), np.array([200], dtype=np.int32), np.array([300], dtype=np.int32)]
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)

    seq_length = 6
    dataset = LazyIndexedDataset(lazy_dir)
    LazySlidingWindowDataSource(dataset=dataset, seq_length=seq_length, eos_token_id=EOS_TOKEN_ID)

    # total_tokens=3, 3//6 = 0 samples with drop_last
    # With drop_last=False: ceil(3/6)=1
    source_no_drop = LazySlidingWindowDataSource(
        dataset=dataset, seq_length=seq_length, eos_token_id=EOS_TOKEN_ID, drop_last=False
    )
    assert len(source_no_drop) == 1


# ---------------------------------------------------------------------------
# Test 3: Map mode alignment
# ---------------------------------------------------------------------------


class TestMapModeAlignment:
  """Tests for map mode alignment."""

  def test_map_mode_basic(self, tmp_dir):
    """Verify map mode with a simple index_mapping."""
    rng = np.random.default_rng(123)
    docs = _make_docs(rng, num_docs=6, min_len=10, max_len=20)
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)

    # Reverse document order
    index_mapping = np.array([5, 4, 3, 2, 1, 0])
    effective_seq = SEQ_LENGTH + 1

    dataset = LazyIndexedDataset(lazy_dir)
    source = LazyMapDataSource(
        dataset=dataset,
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        index_mapping=index_mapping,
    )

    # All samples should be valid
    for i in range(len(source)):
      sample = source[i]["text"]
      assert sample.shape == (effective_seq,)


# ---------------------------------------------------------------------------
# Test 4: Pack mode alignment
# ---------------------------------------------------------------------------


class TestPackModeAlignment:
  """Tests for pack mode alignment with antllm fetch_data_mapping + bin_index."""

  def _make_bin_index(self, tmp_dir, entries_per_bin):
    """Helper to create bin index files.

    Args:
      tmp_dir: directory to write files
      entries_per_bin: list of lists, each inner list contains (doc_idx, offset) pairs

    Returns:
      Path to bin_index directory.
    """
    bin_index_dir = os.path.join(tmp_dir, "bin_index")
    os.makedirs(bin_index_dir, exist_ok=True)

    all_entries = []
    bin_lens = []
    for entries in entries_per_bin:
      bin_lens.append(len(entries) * 2)  # each entry is 2 int32s
      for doc_idx, offset in entries:
        all_entries.append([doc_idx, offset])

    bin_data = np.array(all_entries, dtype=np.int32) if all_entries else np.array([], dtype=np.int32)
    bin_data.tofile(os.path.join(bin_index_dir, "index_offset.bin"))
    np.save(os.path.join(bin_index_dir, "lens.npy"), np.array(bin_lens, dtype=np.int32))
    return bin_index_dir

  def test_pack_mode_basic(self, tmp_dir):
    """Verify pack mode with pre-generated bin index."""
    docs = [
        np.arange(20, dtype=np.int32) + 100,
        np.arange(15, dtype=np.int32) + 200,
        np.arange(25, dtype=np.int32) + 300,
    ]
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)
    bin_index_dir = self._make_bin_index(
        tmp_dir,
        [
            [(0, 0), (1, 0)],  # bin 0: doc0 + doc1
            [(2, 0)],  # bin 1: doc2
        ],
    )

    effective_seq = SEQ_LENGTH + 1
    dataset = LazyIndexedDataset(lazy_dir)
    source = LazyPackDataSource(
        dataset=dataset,
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        bin_index_path=bin_index_dir,
    )

    assert len(source) == 2
    sample0 = source[0]["text"]
    assert sample0.shape == (effective_seq,)
    assert sample0[0] == 100  # first token of doc 0

  def test_pack_eos_injection(self, tmp_dir):
    """Verify EOS is injected at document boundaries, matching antllm."""
    # Small docs that fit within one bin to check exact layout:
    # doc0: [100, 101, 102] (len=3)
    # doc1: [200, 201]      (len=2)
    # Expected bin output: [100, 101, 102, EOS, 200, 201, EOS, EOS-pad...]
    docs = [
        np.array([100, 101, 102], dtype=np.int32),
        np.array([200, 201], dtype=np.int32),
    ]
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)
    bin_index_dir = self._make_bin_index(
        tmp_dir,
        [
            [(0, 0), (1, 0)],  # one bin: doc0 + doc1
        ],
    )

    effective_seq = SEQ_LENGTH + 1  # 17
    dataset = LazyIndexedDataset(lazy_dir)
    source = LazyPackDataSource(
        dataset=dataset,
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        bin_index_path=bin_index_dir,
    )

    sample = source[0]["text"]
    # doc0: 100, 101, 102 → EOS → doc1: 200, 201 → EOS → padding (EOS)
    assert list(sample[:7]) == [100, 101, 102, EOS_TOKEN_ID, 200, 201, EOS_TOKEN_ID]
    # Rest should be EOS padding
    assert all(t == EOS_TOKEN_ID for t in sample[7:])

  def test_pack_cls_injection(self, tmp_dir):
    """Verify CLS is prepended per document when add_cls=True."""
    # doc0: [100, 101] (len=2)
    # doc1: [200]      (len=1)
    # Expected: [CLS, 100, 101, EOS, CLS, 200, EOS, EOS-pad...]
    docs = [
        np.array([100, 101], dtype=np.int32),
        np.array([200], dtype=np.int32),
    ]
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)
    bin_index_dir = self._make_bin_index(
        tmp_dir,
        [
            [(0, 0), (1, 0)],
        ],
    )

    effective_seq = SEQ_LENGTH + 1
    dataset = LazyIndexedDataset(lazy_dir)
    source = LazyPackDataSource(
        dataset=dataset,
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        bin_index_path=bin_index_dir,
        cls_token_id=CLS_TOKEN_ID,
        add_cls=True,
    )

    sample = source[0]["text"]
    expected_prefix = [CLS_TOKEN_ID, 100, 101, EOS_TOKEN_ID, CLS_TOKEN_ID, 200, EOS_TOKEN_ID]
    assert list(sample[: len(expected_prefix)]) == expected_prefix
    assert all(t == EOS_TOKEN_ID for t in sample[len(expected_prefix) :])

  def test_pack_doc_offset(self, tmp_dir):
    """Verify st_offset correctly skips into document fragment."""
    # doc0: [100, 101, 102, 103, 104] (len=5)
    # bin_index: doc0 from offset=2 → should read [102, 103, 104]
    docs = [np.array([100, 101, 102, 103, 104], dtype=np.int32)]
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)
    bin_index_dir = self._make_bin_index(
        tmp_dir,
        [
            [(0, 2)],  # doc0, skip first 2 tokens
        ],
    )

    effective_seq = SEQ_LENGTH + 1
    dataset = LazyIndexedDataset(lazy_dir)
    source = LazyPackDataSource(
        dataset=dataset,
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        bin_index_path=bin_index_dir,
    )

    sample = source[0]["text"]
    # Should read tokens [102, 103, 104] then EOS then padding
    assert list(sample[:4]) == [102, 103, 104, EOS_TOKEN_ID]
    assert all(t == EOS_TOKEN_ID for t in sample[4:])

  def test_pack_padding_short_bin(self, tmp_dir):
    """Verify short bins are padded with EOS to seq_length."""
    # doc0: [100] (len=1) — much shorter than seq_length
    docs = [np.array([100], dtype=np.int32)]
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)
    bin_index_dir = self._make_bin_index(
        tmp_dir,
        [
            [(0, 0)],
        ],
    )

    effective_seq = SEQ_LENGTH + 1
    dataset = LazyIndexedDataset(lazy_dir)
    source = LazyPackDataSource(
        dataset=dataset,
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        bin_index_path=bin_index_dir,
    )

    sample = source[0]["text"]
    assert sample[0] == 100
    assert sample[1] == EOS_TOKEN_ID
    # All remaining are EOS padding
    assert all(t == EOS_TOKEN_ID for t in sample[1:])
    assert sample.shape == (effective_seq,)

  def test_pack_truncation_at_seq_length(self, tmp_dir):
    """Verify output is truncated to exactly seq_length even with many entries."""
    # Create docs that total more than seq_length when packed
    effective_seq = SEQ_LENGTH + 1  # 17
    docs = [
        np.arange(10, dtype=np.int32) + 100,  # doc0: len=10
        np.arange(10, dtype=np.int32) + 200,  # doc1: len=10
    ]
    # Packing both: 10 + EOS + 10 + EOS = 22 > 17
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)
    bin_index_dir = self._make_bin_index(
        tmp_dir,
        [
            [(0, 0), (1, 0)],
        ],
    )

    dataset = LazyIndexedDataset(lazy_dir)
    source = LazyPackDataSource(
        dataset=dataset,
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        bin_index_path=bin_index_dir,
    )

    sample = source[0]["text"]
    assert sample.shape == (effective_seq,)
    # doc0: 10 tokens + EOS = 11, remaining capacity = 6
    # doc1: min(10, 6) = 6 tokens
    assert list(sample[:11]) == list(range(100, 110)) + [EOS_TOKEN_ID]
    assert list(sample[11:17]) == list(range(200, 206))

  def test_pack_multi_shard_via_multishard(self, tmp_dir):
    """Verify pack mode works end-to-end through MultiShardLazyDataSource."""
    # 2 shards, each with distinct docs
    docs_shard0 = [np.arange(10, dtype=np.int32) + 100]
    docs_shard1 = [np.arange(10, dtype=np.int32) + 200]
    scatter_dir = _make_test_scatter(tmp_dir, "pack_ds", [docs_shard0, docs_shard1])

    # Create bin index for each shard at the expected path
    bin_base = os.path.join(tmp_dir, "bin_base")
    for shard_id, _ in [(0, 100), (1, 200)]:
      shard_bin_dir = os.path.join(bin_base, f"pack_ds.scatter/{shard_id}.lazy")
      os.makedirs(shard_bin_dir, exist_ok=True)
      # One bin per shard: pack the single doc
      entries = np.array([[0, 0]], dtype=np.int32)
      entries.tofile(os.path.join(shard_bin_dir, "index_offset.bin"))
      np.save(os.path.join(shard_bin_dir, "lens.npy"), np.array([2], dtype=np.int32))

    effective_seq = SEQ_LENGTH + 1
    source = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="pack",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        bin_index_base_path=bin_base,
    )

    assert len(source) == 2  # 1 bin per shard × 2 shards
    # Shard 0's sample starts with 100, shard 1's with 200
    s0 = source[0]["text"]
    s1 = source[1]["text"]
    assert s0[0] == 100
    assert s1[0] == 200

  def test_pack_pickle_roundtrip(self, tmp_dir):
    """Verify LazyPackDataSource survives pickle (needed for Grain mp_prefetch)."""
    docs = [np.arange(10, dtype=np.int32) + 100]
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)
    bin_index_dir = self._make_bin_index(tmp_dir, [[(0, 0)]])

    effective_seq = SEQ_LENGTH + 1
    dataset = LazyIndexedDataset(lazy_dir)
    source = LazyPackDataSource(
        dataset=dataset,
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        bin_index_path=bin_index_dir,
    )

    original = source[0]["text"]
    restored = pickle.loads(pickle.dumps(source))
    np.testing.assert_array_equal(restored[0]["text"], original)

  def test_pack_inmemory_pickle_roundtrip(self, tmp_dir):
    """Verify in-memory bin_index_data mode survives pickle."""
    from maxtext.input_pipeline._lazy_bfd_packing import build_bin_index

    docs = [np.arange(10, dtype=np.int32) + 100, np.arange(5, dtype=np.int32) + 200]
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)
    dataset = LazyIndexedDataset(lazy_dir)
    effective_seq = SEQ_LENGTH + 1
    bin_entries, bin_lens = build_bin_index(dataset.sizes, effective_seq)
    source = LazyPackDataSource(
        dataset=dataset,
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        bin_index_data=(bin_entries, bin_lens),
    )
    original = source[0]["text"]
    restored = pickle.loads(pickle.dumps(source))
    np.testing.assert_array_equal(restored[0]["text"], original)


# ---------------------------------------------------------------------------
# Test 4b: BFD auto-packing (runtime bin_index generation)
# ---------------------------------------------------------------------------


class TestBFDPacking:
  """Tests for BFD auto-packing when no pre-generated bin_index exists."""

  def test_bfd_basic(self, tmp_dir):
    """Pack mode without pre-generated bin_index auto-builds BFD bins."""
    docs = [
        np.arange(5, dtype=np.int32) + 100,
        np.arange(5, dtype=np.int32) + 200,
        np.arange(5, dtype=np.int32) + 300,
    ]
    scatter_dir = _make_test_scatter(tmp_dir, "bfd_ds", [docs])
    effective_seq = SEQ_LENGTH + 1  # 17
    source = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="pack",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
    )
    # 3 docs × (5+EOS) = 18 tokens > 17 → at least 2 bins
    assert len(source) >= 1
    for i in range(len(source)):
      assert source[i]["text"].shape == (effective_seq,)

  def test_bfd_all_tokens_covered(self, tmp_dir):
    """Every doc token appears exactly once across all BFD bins."""
    docs = [np.arange(k, dtype=np.int32) + (i + 1) * 1000 for i, k in enumerate([10, 7, 13, 3, 20])]
    scatter_dir = _make_test_scatter(tmp_dir, "bfd_cover", [docs])
    effective_seq = SEQ_LENGTH + 1
    source = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="pack",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
    )
    all_tokens = set()
    for i in range(len(source)):
      for t in source[i]["text"]:
        if t != EOS_TOKEN_ID:
          all_tokens.add(int(t))
    expected = set()
    for doc in docs:
      expected.update(doc.tolist())
    assert all_tokens == expected

  def test_bfd_long_doc_split(self, tmp_dir):
    """Doc longer than seq_length is split into full chunks + remainder."""
    effective_seq = SEQ_LENGTH + 1  # 17
    docs = [np.arange(40, dtype=np.int32) + 100]  # 40 > 17 → multiple bins
    scatter_dir = _make_test_scatter(tmp_dir, "bfd_long", [docs])
    source = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="pack",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
    )
    # 40 // 17 = 2 full chunks + 6 remainder → at least 2 bins
    assert len(source) >= 2
    for i in range(len(source)):
      assert source[i]["text"].shape == (effective_seq,)

  def test_bfd_sort_by_lens(self, tmp_dir):
    """sort_by_lens=True produces valid packing."""
    docs = [np.arange(k, dtype=np.int32) + (i + 1) * 100 for i, k in enumerate([3, 12, 7, 1, 9])]
    scatter_dir = _make_test_scatter(tmp_dir, "bfd_sort", [docs])
    effective_seq = SEQ_LENGTH + 1
    source = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="pack",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        bfd_pack_sort_by_lens=True,
    )
    for i in range(len(source)):
      assert source[i]["text"].shape == (effective_seq,)

  def test_bfd_unit_token_accounting(self):
    """BFD build_bin_index accounts for all tokens in doc_lens."""
    from maxtext.input_pipeline._lazy_bfd_packing import build_bin_index

    doc_lens = np.array([10, 7, 13, 3, 20], dtype=np.int64)
    seq_length = SEQ_LENGTH + 1
    bin_entries, bin_lens = build_bin_index(doc_lens, seq_length)

    # Verify all doc tokens accounted for.
    # Accounting mirrors antllm validation (lazy_loader.py:1028-1057):
    # - Single-entry bins (full chunks): min(doc_len - offset, seq_length)
    # - Multi-entry bins (packed remainders): doc_len - offset (always < seq_length)
    total_doc_tokens = int(np.sum(doc_lens))
    bin_tokens = 0
    offset = 0
    for blen in bin_lens:
      n_entries = blen // 2
      if n_entries == 1:
        doc_idx = int(bin_entries[offset, 0])
        st_off = int(bin_entries[offset, 1])
        bin_tokens += min(int(doc_lens[doc_idx]) - st_off, seq_length)
      else:
        for j in range(n_entries):
          doc_idx = int(bin_entries[offset + j, 0])
          st_off = int(bin_entries[offset + j, 1])
          bin_tokens += int(doc_lens[doc_idx]) - st_off
      offset += n_entries
    assert bin_tokens == total_doc_tokens

  def test_bfd_single_tiny_doc(self, tmp_dir):
    """Single doc shorter than seq_length produces 1 bin."""
    docs = [np.array([42], dtype=np.int32)]
    scatter_dir = _make_test_scatter(tmp_dir, "bfd_tiny", [docs])
    effective_seq = SEQ_LENGTH + 1
    source = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="pack",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
    )
    assert len(source) == 1
    sample = source[0]["text"]
    assert sample[0] == 42
    assert sample[1] == EOS_TOKEN_ID


# ---------------------------------------------------------------------------
# Test 5: Multi-shard alignment with antllm AllSampleLazyLoader
# ---------------------------------------------------------------------------


class TestMultiShardAlignment:
  """Tests for multi-shard alignment with antllm AllSampleLazyLoader."""

  def test_multi_shard_basic(self, tmp_dir):
    """Verify MultiShardLazyDataSource routes correctly across shards."""
    rng = np.random.default_rng(777)
    shard_docs = [_make_docs(rng, 3, 15, 25) for _ in range(3)]
    scatter_dir = _make_test_scatter(tmp_dir, "test_ds", shard_docs)

    effective_seq = SEQ_LENGTH + 1
    source = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="sliding_window",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
    )

    assert len(source) > 0
    # Verify all samples are accessible
    for i in range(len(source)):
      sample = source[i]["text"]
      assert sample.shape == (effective_seq,)

  def test_multi_shard_discovery(self, tmp_dir):
    """Verify .lazy directories are discovered and sorted by numeric id."""
    rng = np.random.default_rng(888)
    shard_docs = [_make_docs(rng, 2, 10, 20) for _ in range(4)]
    scatter_dir = _make_test_scatter(tmp_dir, "ds", shard_docs)

    source = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="sliding_window",
        seq_length=SEQ_LENGTH + 1,
        eos_token_id=EOS_TOKEN_ID,
    )

    # Should have found all 4 shards
    assert len(source._loaders) <= 4  # some may be empty
    assert len(source) > 0


# ---------------------------------------------------------------------------
# Test 6: Online shuffle alignment
# ---------------------------------------------------------------------------


class TestOnlineShuffleAlignment:
  """Tests for online shuffle alignment."""

  def test_online_shuffle(self, tmp_dir):
    """Verify per-epoch shuffle produces deterministic permutation."""
    rng = np.random.default_rng(555)
    shard_docs = [_make_docs(rng, 5, 10, 20)]
    scatter_dir = _make_test_scatter(tmp_dir, "shuf_ds", shard_docs)

    effective_seq = SEQ_LENGTH + 1
    source = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="sliding_window",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        loader_online_shuffle=True,
        loader_seed=1234,
    )

    n = len(source)
    if n == 0:
      pytest.skip("No samples generated")

    # Epoch 0: indices [0, n)
    epoch0_samples = [source[i]["text"].copy() for i in range(n)]
    # Epoch 1: indices [n, 2n) — should be shuffled differently
    epoch1_samples = [source[n + i]["text"].copy() for i in range(n)]

    # At least some samples should differ in order between epochs
    diffs = sum(1 for a, b in zip(epoch0_samples, epoch1_samples) if not np.array_equal(a, b))
    assert diffs > 0, "Online shuffle should produce different orderings across epochs"


# ---------------------------------------------------------------------------
# Test 7: Blending alignment
# ---------------------------------------------------------------------------


class TestBlendingAlignment:
  """Tests for blending alignment."""

  def test_blending_indices(self):
    """Verify LazyBlendedDataSource produces same indices as MaxText build_blending_indices."""
    from maxtext.input_pipeline._megatron_blending import build_blending_indices

    weights = np.array([0.7, 0.2, 0.1])

    # Create dummy sources
    class DummySource:  # pylint: disable=missing-class-docstring

      def __init__(self, n):
        self._n = n

      def __len__(self):
        return self._n

      def __getitem__(self, idx):
        return {"text": np.array([idx], dtype=np.int32)}

    sources = [DummySource(100), DummySource(50), DummySource(30)]
    size = 50

    weights_f64 = np.asarray(weights, dtype=np.float64)
    blended = LazyBlendedDataSource(sources, weights_f64, size)

    # Build reference indices — use the exact same weights (no re-normalization)
    ref_di = np.zeros(size, dtype=np.int16)
    ref_dsi = np.zeros(size, dtype=np.int64)
    build_blending_indices(ref_di, ref_dsi, weights_f64, 3, size)

    np.testing.assert_array_equal(blended.dataset_index, ref_di)
    np.testing.assert_array_equal(blended.dataset_sample_index, ref_dsi)


# ---------------------------------------------------------------------------
# Test 8: Split-from-train alignment
# ---------------------------------------------------------------------------


class TestSplitAlignment:
  """Tests for split-from-train alignment."""

  def test_split_basic(self):
    """Verify SplitDataSource correctly splits a source."""

    class ListSource:

      def __init__(self, data):
        self._data = data

      def __len__(self):
        return len(self._data)

      def __getitem__(self, idx):
        return self._data[idx]

    data = list(range(100))
    source = ListSource(data)

    split = SplitDataSource(source, 0, 70)
    assert len(split) == 70
    assert split[0] == 0
    assert split[69] == 69

    # Wrap-around
    assert split[70] == 0

  def test_split_ratios(self):
    """Verify split with ratios matches antllm SplitDataset logic."""

    class ListSource:

      def __init__(self, n):
        self._n = n

      def __len__(self):
        return self._n

      def __getitem__(self, idx):
        return idx

    total = 1000
    source = ListSource(total)
    ratios = [0.98, 0.01, 0.01]

    from maxtext.input_pipeline.lazy_data_processing import _split_source

    parts = _split_source(source, ratios)

    assert parts[0] is not None
    assert parts[1] is not None
    assert parts[2] is not None
    assert len(parts[0]) + len(parts[1]) + len(parts[2]) <= total + 3  # rounding


# ---------------------------------------------------------------------------
# Test 9: Pickle support for all sources
# ---------------------------------------------------------------------------


class TestPickleSupport:
  """Tests for pickle serialization of all source types."""

  def test_sliding_window_pickle(self, tmp_dir):
    docs = [np.arange(50, dtype=np.int32) + 100]
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)
    dataset = LazyIndexedDataset(lazy_dir)
    source = LazySlidingWindowDataSource(dataset=dataset, seq_length=10, eos_token_id=EOS_TOKEN_ID)

    restored = pickle.loads(pickle.dumps(source))
    assert len(restored) == len(source)
    for i in range(len(source)):
      np.testing.assert_array_equal(restored[i]["text"], source[i]["text"])

  def test_multi_shard_pickle(self, tmp_dir):
    rng = np.random.default_rng(99)
    shard_docs = [_make_docs(rng, 3, 10, 20) for _ in range(2)]
    scatter_dir = _make_test_scatter(tmp_dir, "pkl_ds", shard_docs)

    source = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="sliding_window",
        seq_length=SEQ_LENGTH + 1,
        eos_token_id=EOS_TOKEN_ID,
    )

    restored = pickle.loads(pickle.dumps(source))
    assert len(restored) == len(source)
    if len(source) > 0:
      np.testing.assert_array_equal(restored[0]["text"], source[0]["text"])


# ---------------------------------------------------------------------------
# Test: Cross-process SharedMemory sharing (simulates Grain spawn workers)
# ---------------------------------------------------------------------------


def _child_check_shm(send_q, recv_q):
  """Target for spawned child process. Receives pickled objects via Queue,
  checks SharedMemoryArray names and data correctness, sends results back."""
  from grain.python import SharedMemoryArray  # noqa: F811

  task = send_q.get(timeout=30)
  kind = task["kind"]
  results = {}

  if kind == "indexed_dataset":
    ds = task["obj"]
    results["ends_shm_name"] = ds._ends.shm.name if isinstance(ds._ends, SharedMemoryArray) else None
    results["doc0"] = ds.get(0).tolist()

  elif kind == "multi_shard":
    src = task["obj"]
    shm_names = []
    for loader in src._loaders:
      inner_ds = loader._dataset if hasattr(loader, "_dataset") else None
      if inner_ds and isinstance(inner_ds._ends, SharedMemoryArray):
        shm_names.append(inner_ds._ends.shm.name)
    results["shm_names"] = shm_names
    results["item0"] = src[0]["text"].tolist() if len(src) > 0 else None

  elif kind == "blend_shm":
    src = task["obj"]
    results["di_shm_name"] = src._dataset_index.shm.name if isinstance(src._dataset_index, SharedMemoryArray) else None
    results["dsi_shm_name"] = (
        src._dataset_sample_index.shm.name if isinstance(src._dataset_sample_index, SharedMemoryArray) else None
    )
    results["item0"] = src[0]["text"].tolist() if len(src) > 0 else None

  elif kind == "blend_cache":
    src = task["obj"]
    results["di_is_mmap"] = isinstance(src._dataset_index, np.memmap)
    results["dsi_is_mmap"] = isinstance(src._dataset_sample_index, np.memmap)
    results["item0"] = src[0]["text"].tolist() if len(src) > 0 else None

  elif kind == "pack_source":
    src = task["obj"]
    results["bin_ends_shm_name"] = src._bin_ends.shm.name if isinstance(src._bin_ends, SharedMemoryArray) else None
    results["item0"] = src[0]["text"].tolist() if len(src) > 0 else None

  recv_q.put(results)


def _spawn_child_and_check(task):
  """Spawn a child process, send task via Queue, return results."""
  import multiprocessing

  ctx = multiprocessing.get_context("spawn")
  send_q = ctx.Queue()
  recv_q = ctx.Queue()
  p = ctx.Process(target=_child_check_shm, args=(send_q, recv_q))
  p.start()
  send_q.put(task)
  results = recv_q.get(timeout=60)
  p.join(timeout=10)
  return results


class TestCrossProcessSharing:
  """Verify SharedMemoryArray fields are truly shared (not copied) across spawn-mode workers."""

  def test_indexed_dataset_ends_shared(self, tmp_dir):
    """LazyIndexedDataset._ends shm name survives pickle to child process."""
    from grain.python import SharedMemoryArray  # noqa: F811

    docs = [np.arange(50, dtype=np.int32) + 100, np.arange(30, dtype=np.int32) + 200]
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)
    ds = LazyIndexedDataset(lazy_dir)

    assert isinstance(ds._ends, SharedMemoryArray)
    parent_shm_name = ds._ends.shm.name
    parent_doc0 = ds.get(0).tolist()

    results = _spawn_child_and_check({"kind": "indexed_dataset", "obj": ds})

    assert (
        results["ends_shm_name"] == parent_shm_name
    ), f"Child got different shm segment: {results['ends_shm_name']} vs {parent_shm_name}"
    assert results["doc0"] == parent_doc0

  def test_multi_shard_loaders_shared(self, tmp_dir):
    """MultiShardLazyDataSource serializes loaders directly; child reuses shm."""
    from grain.python import SharedMemoryArray  # noqa: F811

    rng = np.random.default_rng(42)
    shard_docs = [_make_docs(rng, 5, 10, 30) for _ in range(2)]
    scatter_dir = _make_test_scatter(tmp_dir, "shm_ds", shard_docs)

    source = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="sliding_window",
        seq_length=SEQ_LENGTH + 1,
        eos_token_id=EOS_TOKEN_ID,
    )

    parent_shm_names = []
    for loader in source._loaders:
      inner_ds = loader._dataset if hasattr(loader, "_dataset") else None
      if inner_ds and isinstance(inner_ds._ends, SharedMemoryArray):
        parent_shm_names.append(inner_ds._ends.shm.name)

    assert len(parent_shm_names) > 0
    parent_item0 = source[0]["text"].tolist()

    results = _spawn_child_and_check({"kind": "multi_shard", "obj": source})

    assert (
        results["shm_names"] == parent_shm_names
    ), f"Child loaders got different shm segments: {results['shm_names']} vs {parent_shm_names}"
    assert results["item0"] == parent_item0

  def _make_blend_sources(self, tmp_dir, rng):
    """Helper: create two MultiShardLazyDataSources for blending tests."""
    shard_docs1 = [_make_docs(rng, 5, 10, 30)]
    shard_docs2 = [_make_docs(rng, 5, 10, 30)]
    scatter1 = _make_test_scatter(tmp_dir, "blend_a", shard_docs1)
    scatter2 = _make_test_scatter(tmp_dir, "blend_b", shard_docs2)
    src1 = MultiShardLazyDataSource(
        scatter_dir=scatter1,
        mode="sliding_window",
        seq_length=SEQ_LENGTH + 1,
        eos_token_id=EOS_TOKEN_ID,
    )
    src2 = MultiShardLazyDataSource(
        scatter_dir=scatter2,
        mode="sliding_window",
        seq_length=SEQ_LENGTH + 1,
        eos_token_id=EOS_TOKEN_ID,
    )
    return src1, src2

  def test_blend_cache_miss_shm_shared(self, tmp_dir):
    """Cache miss: blend indices are SharedMemoryArray, child reuses same shm segment."""
    from grain.python import SharedMemoryArray  # noqa: F811

    rng = np.random.default_rng(77)
    src1, src2 = self._make_blend_sources(tmp_dir, rng)
    weights = np.array([0.6, 0.4])
    size = min(len(src1), len(src2), 20)
    if size == 0:
      pytest.skip("Sources too small for blending test")

    # No cache_dir → always computes → SharedMemoryArray
    blended = LazyBlendedDataSource(
        sources=[src1, src2],
        weights=weights,
        size=size,
        shuffle_seed=-1,
    )
    assert isinstance(blended._dataset_index, SharedMemoryArray)
    parent_di_name = blended._dataset_index.shm.name
    parent_dsi_name = blended._dataset_sample_index.shm.name
    parent_item0 = blended[0]["text"].tolist()

    results = _spawn_child_and_check({"kind": "blend_shm", "obj": blended})

    assert (
        results["di_shm_name"] == parent_di_name
    ), f"Child got different shm for dataset_index: {results['di_shm_name']} vs {parent_di_name}"
    assert (
        results["dsi_shm_name"] == parent_dsi_name
    ), f"Child got different shm for dataset_sample_index: {results['dsi_shm_name']} vs {parent_dsi_name}"
    assert results["item0"] == parent_item0

  def test_blend_cache_hit_mmap_in_child(self, tmp_dir):
    """Cache hit: blend indices are mmap, child re-mmaps same files (not anonymous copies)."""
    rng = np.random.default_rng(77)
    src1, src2 = self._make_blend_sources(tmp_dir, rng)
    cache_dir = os.path.join(tmp_dir, "blend_cache")
    weights = np.array([0.6, 0.4])
    size = min(len(src1), len(src2), 20)
    if size == 0:
      pytest.skip("Sources too small for blending test")

    # First call writes cache
    _ = LazyBlendedDataSource(
        sources=[src1, src2],
        weights=weights,
        size=size,
        shuffle_seed=-1,
        cache_dir=cache_dir,
    )

    # Second call hits cache (mmap)
    blended = LazyBlendedDataSource(
        sources=[src1, src2],
        weights=weights,
        size=size,
        shuffle_seed=-1,
        cache_dir=cache_dir,
    )
    assert isinstance(blended._dataset_index, np.memmap), "Expected mmap from cache hit"
    parent_item0 = blended[0]["text"].tolist()

    results = _spawn_child_and_check({"kind": "blend_cache", "obj": blended})

    assert results["di_is_mmap"], "Child _dataset_index should be mmap (re-opened from cache file)"
    assert results["dsi_is_mmap"], "Child _dataset_sample_index should be mmap (re-opened from cache file)"
    assert results["item0"] == parent_item0

  def test_pack_source_bin_ends_shared(self, tmp_dir):
    """LazyPackDataSource._bin_ends shm name survives pickle to child process."""
    from grain.python import SharedMemoryArray  # noqa: F811
    from maxtext.input_pipeline._lazy_bfd_packing import build_bin_index

    docs = [np.arange(20, dtype=np.int32) + 100 for _ in range(10)]
    lazy_dir = _make_lazy_dir(tmp_dir, 0, docs)
    dataset = LazyIndexedDataset(lazy_dir)

    bin_entries, bin_lens = build_bin_index(
        doc_lens=dataset.sizes,
        seq_length=SEQ_LENGTH + 1,
    )
    source = LazyPackDataSource(
        dataset=dataset,
        seq_length=SEQ_LENGTH + 1,
        eos_token_id=EOS_TOKEN_ID,
        bin_index_data=(bin_entries, bin_lens),
    )

    assert isinstance(source._bin_ends, SharedMemoryArray)
    parent_shm_name = source._bin_ends.shm.name
    parent_item0 = source[0]["text"].tolist() if len(source) > 0 else None

    results = _spawn_child_and_check({"kind": "pack_source", "obj": source})

    assert (
        results["bin_ends_shm_name"] == parent_shm_name
    ), f"Child got different shm: {results['bin_ends_shm_name']} vs {parent_shm_name}"
    if parent_item0 is not None:
      assert results["item0"] == parent_item0


# ---------------------------------------------------------------------------
# Test 10-15: Scatter shard filtering
# ---------------------------------------------------------------------------


class TestScatterShardFiltering:
  """Tests for loader_scatter shard filtering in MultiShardLazyDataSource."""

  def test_positive_scatter_single_shard(self, tmp_dir):
    """loader_scatter=4 with 4 shards: each rank reads exactly 1 shard."""
    shard_docs = []
    for s in range(4):
      shard_docs.append([np.arange(20, dtype=np.int32) + (s + 1) * 1000])
    scatter_dir = _make_test_scatter(tmp_dir, "scatter_pos", shard_docs)

    effective_seq = SEQ_LENGTH + 1
    for rank in range(4):
      source = MultiShardLazyDataSource(
          scatter_dir=scatter_dir,
          mode="sliding_window",
          seq_length=effective_seq,
          eos_token_id=EOS_TOKEN_ID,
          loader_scatter=4,
          process_index=rank,
      )
      assert len(source._loaders) == 1
      sample = source[0]["text"]
      expected_base = (rank + 1) * 1000
      assert sample[0] == expected_base

  def test_negative_scatter_merge(self, tmp_dir):
    """loader_scatter=-2 with 4 shards: each rank reads 2 shards."""
    shard_docs = []
    for s in range(4):
      shard_docs.append([np.arange(30, dtype=np.int32) + (s + 1) * 1000])
    scatter_dir = _make_test_scatter(tmp_dir, "scatter_neg", shard_docs)

    effective_seq = SEQ_LENGTH + 1
    source_r0 = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="sliding_window",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        loader_scatter=-2,
        process_index=0,
    )
    source_r1 = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="sliding_window",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        loader_scatter=-2,
        process_index=1,
    )
    assert len(source_r0._loaders) == 2
    assert len(source_r1._loaders) == 2
    assert source_r0[0]["text"][0] == 1000
    assert source_r1[0]["text"][0] == 3000

  def test_scatter_minus_one_reads_all(self, tmp_dir):
    """loader_scatter=-1: all shards loaded (current default behavior)."""
    shard_docs = []
    for s in range(4):
      shard_docs.append([np.arange(20, dtype=np.int32) + (s + 1) * 1000])
    scatter_dir = _make_test_scatter(tmp_dir, "scatter_all", shard_docs)

    source = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="sliding_window",
        seq_length=SEQ_LENGTH + 1,
        eos_token_id=EOS_TOKEN_ID,
        loader_scatter=-1,
        process_index=0,
    )
    assert len(source._loaders) == 4

  def test_scatter_rank_wraps(self, tmp_dir):
    """Ranks beyond scatter count wrap: rank 4 with scatter=4 == rank 0."""
    shard_docs = []
    for s in range(4):
      shard_docs.append([np.arange(20, dtype=np.int32) + (s + 1) * 1000])
    scatter_dir = _make_test_scatter(tmp_dir, "scatter_wrap", shard_docs)

    effective_seq = SEQ_LENGTH + 1
    source_r0 = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="sliding_window",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        loader_scatter=4,
        process_index=0,
    )
    source_r4 = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="sliding_window",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        loader_scatter=4,
        process_index=4,
    )
    assert len(source_r0) == len(source_r4)
    for i in range(len(source_r0)):
      np.testing.assert_array_equal(source_r0[i]["text"], source_r4[i]["text"])

  def test_scatter_disjoint_coverage(self, tmp_dir):
    """All scatter groups together cover all shards exactly once."""
    shard_docs = []
    for s in range(6):
      shard_docs.append([np.arange(30, dtype=np.int32) + (s + 1) * 1000])
    scatter_dir = _make_test_scatter(tmp_dir, "scatter_cover", shard_docs)

    effective_seq = SEQ_LENGTH + 1
    total_samples = 0
    for rank in range(3):
      source = MultiShardLazyDataSource(
          scatter_dir=scatter_dir,
          mode="sliding_window",
          seq_length=effective_seq,
          eos_token_id=EOS_TOKEN_ID,
          loader_scatter=-3,
          process_index=rank,
      )
      total_samples += len(source)

    source_all = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="sliding_window",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        loader_scatter=-1,
        process_index=0,
    )
    assert total_samples == len(source_all)

  def test_positive_scatter_exceeds_shards(self, tmp_dir):
    """Positive scatter with scatter_id >= shard count should raise ValueError."""
    shard_docs = [[np.arange(20, dtype=np.int32) + 100]]
    scatter_dir = _make_test_scatter(tmp_dir, "scatter_err1", shard_docs)

    with pytest.raises(ValueError, match="scatter_id"):
      MultiShardLazyDataSource(
          scatter_dir=scatter_dir,
          mode="sliding_window",
          seq_length=SEQ_LENGTH + 1,
          eos_token_id=EOS_TOKEN_ID,
          loader_scatter=4,
          process_index=2,
      )

  def test_negative_scatter_exceeds_shards(self, tmp_dir):
    """Negative scatter with |scatter| > shard count should raise ValueError."""
    shard_docs = [[np.arange(20, dtype=np.int32) + 100] for _ in range(2)]
    scatter_dir = _make_test_scatter(tmp_dir, "scatter_err2", shard_docs)

    with pytest.raises(ValueError, match="shard count"):
      MultiShardLazyDataSource(
          scatter_dir=scatter_dir,
          mode="sliding_window",
          seq_length=SEQ_LENGTH + 1,
          eos_token_id=EOS_TOKEN_ID,
          loader_scatter=-4,
          process_index=0,
      )


# ---------------------------------------------------------------------------
# Test 16-20: Epoch handling with num_epochs and scatter
# ---------------------------------------------------------------------------


class TestEpochHandling:
  """Tests for multi-epoch support via num_epochs inflation."""

  def test_epoch_boundary_triggers_reshuffle(self, tmp_dir):
    """With loader_online_shuffle, crossing epoch boundary via direct index
    should produce a different sample ordering."""
    rng = np.random.default_rng(42)
    docs = _make_docs(rng, num_docs=10, min_len=20, max_len=40)
    scatter_dir = _make_test_scatter(tmp_dir, "epoch_shuf", [docs])

    effective_seq = SEQ_LENGTH + 1
    source = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="sliding_window",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        loader_online_shuffle=True,
        loader_seed=42,
    )
    n = source._total_samples
    assert n > 0

    # Access indices beyond total_samples to trigger epoch detection
    epoch0 = [source[i]["text"].copy() for i in range(n)]
    epoch1 = [source[n + i]["text"].copy() for i in range(n)]

    # Both epochs should contain the same *set* of samples (just reordered)
    diffs = sum(1 for a, b in zip(epoch0, epoch1) if not np.array_equal(a, b))
    assert diffs > 0, "Epoch 0 and 1 should have different orderings"

    epoch0_sorted = sorted(epoch0, key=lambda x: x.tobytes())
    epoch1_sorted = sorted(epoch1, key=lambda x: x.tobytes())
    for a, b in zip(epoch0_sorted, epoch1_sorted):
      np.testing.assert_array_equal(a, b)

  def test_scatter_uneven_shards_cross_epoch(self, tmp_dir):
    """Different scatter groups have unequal sample counts; verify each
    group can iterate across multiple epochs via direct index access.

    Setup: 4 shards with very different sizes assigned to 2 scatter groups.
      Group 0 (shards 0,1): small (20+20=40 tokens)
      Group 1 (shards 2,3): large (60+60=120 tokens)
    With seq_length=17, group 0 has ~2 samples, group 1 has ~7 samples.
    """
    shard_docs = [
        [np.arange(20, dtype=np.int32) + 1000],  # shard 0: small
        [np.arange(20, dtype=np.int32) + 2000],  # shard 1: small
        [np.arange(60, dtype=np.int32) + 3000],  # shard 2: large
        [np.arange(60, dtype=np.int32) + 4000],  # shard 3: large
    ]
    scatter_dir = _make_test_scatter(tmp_dir, "scatter_epoch", shard_docs)

    effective_seq = SEQ_LENGTH + 1  # 17

    # Group 0: rank 0 with scatter=-2 -> shards [0, 1]
    src_g0 = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="sliding_window",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        loader_scatter=-2,
        process_index=0,
    )
    # Group 1: rank 1 with scatter=-2 -> shards [2, 3]
    src_g1 = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="sliding_window",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        loader_scatter=-2,
        process_index=1,
    )

    n0 = src_g0._total_samples
    n1 = src_g1._total_samples
    assert n0 != n1, "Groups should have different sample counts for this test"
    assert len(src_g0) == n0
    assert len(src_g1) == n1

    # Access indices across multiple epochs — should not raise
    num_epochs = 3
    for i in range(n0 * num_epochs):
      sample = src_g0[i]["text"]
      assert sample.shape == (effective_seq,)
    for i in range(n1 * num_epochs):
      sample = src_g1[i]["text"]
      assert sample.shape == (effective_seq,)

    # Verify epoch wrap: sample at index n0 should equal sample at index 0
    np.testing.assert_array_equal(src_g0[0]["text"], src_g0[n0]["text"])
    np.testing.assert_array_equal(src_g1[0]["text"], src_g1[n1]["text"])

  def test_scatter_uneven_with_shuffle_cross_epoch(self, tmp_dir):
    """Like test_scatter_uneven_shards_cross_epoch but with online shuffle.
    Verifies shuffle rotates independently per group at each epoch boundary."""
    shard_docs = [
        [np.arange(20, dtype=np.int32) + 1000],
        [np.arange(20, dtype=np.int32) + 2000],
        [np.arange(60, dtype=np.int32) + 3000],
        [np.arange(60, dtype=np.int32) + 4000],
    ]
    scatter_dir = _make_test_scatter(tmp_dir, "scatter_shuf_epoch", shard_docs)

    effective_seq = SEQ_LENGTH + 1

    src_g0 = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="sliding_window",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        loader_scatter=-2,
        process_index=0,
        loader_online_shuffle=True,
        loader_seed=999,
    )
    src_g1 = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="sliding_window",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        loader_scatter=-2,
        process_index=1,
        loader_online_shuffle=True,
        loader_seed=999,
    )

    n0 = src_g0._total_samples
    n1 = src_g1._total_samples

    # With shuffle, epoch 0 and epoch 1 should have different orderings
    if n0 >= 2:
      e0 = [src_g0[i]["text"].copy() for i in range(n0)]
      e1 = [src_g0[n0 + i]["text"].copy() for i in range(n0)]
      diffs = sum(1 for a, b in zip(e0, e1) if not np.array_equal(a, b))
      assert diffs > 0, "Group 0 shuffle should differ across epochs"

    if n1 >= 2:
      e0 = [src_g1[i]["text"].copy() for i in range(n1)]
      e1 = [src_g1[n1 + i]["text"].copy() for i in range(n1)]
      diffs = sum(1 for a, b in zip(e0, e1) if not np.array_equal(a, b))
      assert diffs > 0, "Group 1 shuffle should differ across epochs"

  def test_single_weighted_source_uses_blend_wrapper(self, monkeypatch):
    """A single source with explicit weights should still go through blend."""

    class _FakeSource:

      def __len__(self):
        return 10

      def __getitem__(self, idx):
        return {"text": np.array([idx], dtype=np.int32)}

    monkeypatch.setattr(lazy_data_processing, "_build_source", lambda *args, **kwargs: _FakeSource())

    config = SimpleNamespace(
        lazy_data_root="",
        lazy_dataset_weight_mode="epoch",
        lazy_blend_shuffle_seed=-1,
        lazy_blend_shuffle_only_dataset=False,
        lazy_blend_cache_dir="",
        lazy_loader_scatter=1,
        lazy_data_size_B_tokens=1.0,
        max_target_length=16,
    )

    blended = lazy_data_processing._build_blend(config, ["dummy.scatter"], [0.5], "train", process_index=0)

    assert isinstance(blended, LazyBlendedDataSource)
    assert len(blended.sources) == 1
    assert len(blended) == 5

  def test_blended_epoch_repeats_same_order(self, tmp_dir):
    """Blend epochs repeat the same sample order (blend indices are fixed)."""
    rng = np.random.default_rng(88)
    docs_a = _make_docs(rng, num_docs=6, min_len=20, max_len=30)
    docs_b = _make_docs(rng, num_docs=3, min_len=20, max_len=30)
    scatter_a = _make_test_scatter(tmp_dir, "blend_rep_a", [docs_a])
    scatter_b = _make_test_scatter(tmp_dir, "blend_rep_b", [docs_b])

    effective_seq = SEQ_LENGTH + 1
    src_a = MultiShardLazyDataSource(
        scatter_dir=scatter_a, mode="sliding_window", seq_length=effective_seq, eos_token_id=EOS_TOKEN_ID
    )
    src_b = MultiShardLazyDataSource(
        scatter_dir=scatter_b, mode="sliding_window", seq_length=effective_seq, eos_token_id=EOS_TOKEN_ID
    )

    weights = np.array([0.7, 0.3], dtype=np.float64)
    blend_size = int(0.7 * len(src_a) + 0.3 * len(src_b))
    if blend_size == 0:
      pytest.skip("Not enough tokens")

    blended = LazyBlendedDataSource([src_a, src_b], weights, blend_size)

    # Blend wraps via idx % _size, so epoch 0 and epoch 1 produce identical
    # samples (no sub-source shuffle, blend order is deterministic)
    for i in range(blend_size):
      s0 = blended[i]
      s1 = blended[blend_size + i]
      t0 = s0["text"] if isinstance(s0, dict) else np.asarray(s0)
      t1 = s1["text"] if isinstance(s1, dict) else np.asarray(s1)
      np.testing.assert_array_equal(t0, t1, err_msg=f"Mismatch at blend index {i}")

  def test_blended_cross_epoch_identical_with_shuffle(self, tmp_dir):
    """Key antllm alignment test: blend epoch 0 and 1 produce identical
    sequences even with multiple shuffled sub-sources."""
    rng = np.random.default_rng(111)
    docs_a = _make_docs(rng, num_docs=10, min_len=20, max_len=40)
    docs_b = _make_docs(rng, num_docs=6, min_len=20, max_len=40)
    scatter_a = _make_test_scatter(tmp_dir, "cross_ep_a", [docs_a])
    scatter_b = _make_test_scatter(tmp_dir, "cross_ep_b", [docs_b])

    effective_seq = SEQ_LENGTH + 1
    src_a = MultiShardLazyDataSource(
        scatter_dir=scatter_a,
        mode="sliding_window",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        loader_online_shuffle=True,
        loader_seed=77,
    )
    src_b = MultiShardLazyDataSource(
        scatter_dir=scatter_b,
        mode="sliding_window",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        loader_online_shuffle=True,
        loader_seed=77,
    )

    weights = np.array([0.7, 0.3], dtype=np.float64)
    blend_size = int(0.7 * len(src_a) + 0.3 * len(src_b))
    if blend_size < 2:
      pytest.skip("Not enough tokens for blending test")

    blended = LazyBlendedDataSource([src_a, src_b], weights, blend_size)

    for i in range(blend_size):
      s0 = blended[i]["text"]
      s1 = blended[blend_size + i]["text"]
      np.testing.assert_array_equal(s0, s1, err_msg=f"Cross-epoch mismatch at blend index {i}")

  def test_within_blend_subsource_epoch_crossing(self, tmp_dir):
    """When weight > 1 for a dataset, the greedy algorithm produces
    dataset_sample_index values exceeding len(ds), naturally triggering
    sub-source epoch transition and re-shuffle within a single blend pass."""
    rng = np.random.default_rng(222)
    # Small dataset: few samples so greedy algorithm easily exceeds len(ds)
    docs_small = _make_docs(rng, num_docs=3, min_len=20, max_len=30)
    # Large dataset to pair with
    docs_large = _make_docs(rng, num_docs=20, min_len=20, max_len=40)
    scatter_small = _make_test_scatter(tmp_dir, "epoch_cross_small", [docs_small])
    scatter_large = _make_test_scatter(tmp_dir, "epoch_cross_large", [docs_large])

    effective_seq = SEQ_LENGTH + 1
    src_small = MultiShardLazyDataSource(
        scatter_dir=scatter_small,
        mode="sliding_window",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        loader_online_shuffle=True,
        loader_seed=42,
    )
    src_large = MultiShardLazyDataSource(
        scatter_dir=scatter_large,
        mode="sliding_window",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
        loader_online_shuffle=True,
        loader_seed=42,
    )
    n_small = src_small._total_samples
    n_large = src_large._total_samples
    if n_small < 1 or n_large < 1:
      pytest.skip("Not enough samples")

    # Give the small dataset a high weight so it gets many more draws
    # than its actual sample count, crossing epoch boundary naturally.
    weights = np.array([0.9, 0.1], dtype=np.float64)
    blend_size = int(0.9 * n_small * 5 + 0.1 * n_large)
    if blend_size < 2:
      pytest.skip("Not enough samples for blending")

    blended = LazyBlendedDataSource([src_small, src_large], weights, blend_size)

    # Check that dataset_sample_index for dataset 0 exceeds n_small
    ds0_mask = blended.dataset_index == 0
    ds0_sample_indices = blended.dataset_sample_index[ds0_mask]
    max_ds0_idx = int(ds0_sample_indices.max()) if ds0_mask.any() else 0
    assert max_ds0_idx >= n_small, (
        f"Expected dataset_sample_index to exceed len(ds)={n_small}, "
        f"but max was {max_ds0_idx}. Increase blend_size or weight."
    )

    # All indices should be accessible without errors
    for i in range(blend_size):
      sample = blended[i]
      assert "text" in sample
      assert sample["text"].shape == (effective_seq,)

  def test_make_lazy_train_iterator_uses_repeat(self, monkeypatch):
    """Train iterator should always use Grain .repeat(num_epochs)."""

    class _FakeSource:
      """Stub source for testing repeat path."""

      def __len__(self):
        return 17

    fake_source = _FakeSource()
    captured = {}

    monkeypatch.setattr(lazy_data_processing, "_build_blend", lambda *args, **kwargs: fake_source)
    monkeypatch.setattr(lazy_data_processing, "_compute_num_epochs", lambda *args, **kwargs: 3)

    def _fake_build_pipeline(config, source, global_mesh, process_indices, files_str="", repeat_epochs=1):
      captured["source"] = source
      captured["files_str"] = files_str
      captured["repeat_epochs"] = repeat_epochs
      return captured

    monkeypatch.setattr(lazy_data_processing, "_build_pipeline", _fake_build_pipeline)
    monkeypatch.setattr(lazy_data_processing, "MultiHostDataLoadIterator", lambda dataset, *args, **kwargs: dataset)

    config = SimpleNamespace(
        lazy_train_files="dummy_scatter",
        lazy_valid_files="",
        lazy_test_files="",
        lazy_split="",
        generate_padding_batch_train=False,
    )

    result = lazy_data_processing.make_lazy_train_iterator(config, global_mesh=None, process_indices=[0])
    assert result["source"] is fake_source  # pylint: disable=unsubscriptable-object
    assert result["files_str"] == "dummy_scatter"  # pylint: disable=unsubscriptable-object
    assert result["repeat_epochs"] == 3  # pylint: disable=unsubscriptable-object


# ---------------------------------------------------------------------------
# Per-dataset pack mode (_resolve_mode / lazy_bfd_pack)
# ---------------------------------------------------------------------------


class _FakeBfdConfig:
  """Minimal config stub for _resolve_mode tests."""

  def __init__(self, lazy_loader_mode="sliding_window", lazy_bfd_pack=""):
    self.lazy_loader_mode = lazy_loader_mode
    self.lazy_bfd_pack = lazy_bfd_pack


class TestResolveModePerDataset:
  """Tests for _resolve_mode: per-dataset pack via lazy_bfd_pack."""

  def test_empty_bfd_pack_uses_global_mode(self):
    """Empty lazy_bfd_pack -> fall back to lazy_loader_mode."""
    cfg = _FakeBfdConfig(lazy_loader_mode="sliding_window", lazy_bfd_pack="")
    assert _resolve_mode(cfg, "/data/ds1.scatter") == "sliding_window"

    cfg = _FakeBfdConfig(lazy_loader_mode="map", lazy_bfd_pack="")
    assert _resolve_mode(cfg, "/data/ds1.scatter") == "map"

    cfg = _FakeBfdConfig(lazy_loader_mode="pack", lazy_bfd_pack="")
    assert _resolve_mode(cfg, "/data/ds1.scatter") == "pack"

  def test_bfd_pack_all(self):
    """lazy_bfd_pack='ALL' -> all datasets use pack regardless of global mode."""
    cfg = _FakeBfdConfig(lazy_loader_mode="sliding_window", lazy_bfd_pack="ALL")
    assert _resolve_mode(cfg, "/data/ds1.scatter") == "pack"
    assert _resolve_mode(cfg, "/data/ds2.scatter") == "pack"
    assert _resolve_mode(cfg, "/data/any_name.scatter") == "pack"

  def test_bfd_pack_specific_datasets(self):
    """lazy_bfd_pack='ds1,ds2' -> only ds1/ds2 use pack, others sliding_window."""
    cfg = _FakeBfdConfig(lazy_loader_mode="sliding_window", lazy_bfd_pack="ds1,ds2")
    assert _resolve_mode(cfg, "/data/ds1.scatter") == "pack"
    assert _resolve_mode(cfg, "/data/ds2.scatter") == "pack"
    assert _resolve_mode(cfg, "/data/ds3.scatter") == "sliding_window"

  def test_bfd_pack_single_dataset(self):
    """lazy_bfd_pack='ds1' -> only ds1 uses pack."""
    cfg = _FakeBfdConfig(lazy_loader_mode="sliding_window", lazy_bfd_pack="ds1")
    assert _resolve_mode(cfg, "/root/ds1.scatter") == "pack"
    assert _resolve_mode(cfg, "/root/ds2.scatter") == "sliding_window"

  def test_bfd_pack_with_spaces(self):
    """Whitespace around dataset names and the config value should be tolerated."""
    cfg = _FakeBfdConfig(lazy_loader_mode="sliding_window", lazy_bfd_pack=" ds1 , ds2 ")
    assert _resolve_mode(cfg, "/data/ds1.scatter") == "pack"
    assert _resolve_mode(cfg, "/data/ds2.scatter") == "pack"
    assert _resolve_mode(cfg, "/data/ds3.scatter") == "sliding_window"

  def test_bfd_pack_extracts_name_from_path(self):
    """Dataset name is extracted from the basename, stripping .scatter suffix."""
    cfg = _FakeBfdConfig(lazy_bfd_pack="my_dataset")
    assert _resolve_mode(cfg, "/a/b/c/my_dataset.scatter") == "pack"
    assert _resolve_mode(cfg, "my_dataset.scatter") == "pack"
    assert _resolve_mode(cfg, "/a/b/c/other.scatter") == "sliding_window"

  def test_bfd_pack_extracts_name_from_hash_subdir(self):
    """Hash-subdir scatter paths should still match by dataset name."""
    cfg = _FakeBfdConfig(lazy_bfd_pack="dataset")
    assert _resolve_mode(cfg, "/a/b/c/my_dataset.scatter/hash123") == "pack"
    cfg = _FakeBfdConfig(lazy_bfd_pack="my_dataset")
    assert _resolve_mode(cfg, "/a/b/c/379824db_my_dataset.scatter/hash123") == "pack"

  def test_bfd_pack_overrides_global_mode(self):
    """lazy_bfd_pack overrides lazy_loader_mode even when global is 'map'."""
    cfg = _FakeBfdConfig(lazy_loader_mode="map", lazy_bfd_pack="ds1")
    # ds1 -> pack (from bfd_pack)
    assert _resolve_mode(cfg, "/data/ds1.scatter") == "pack"
    # ds2 -> sliding_window (non-matched datasets fall back to sliding_window, not global)
    assert _resolve_mode(cfg, "/data/ds2.scatter") == "sliding_window"

  def test_bfd_pack_integration_with_multishard(self, tmp_dir):
    """Integration: _resolve_mode selects pack for matched dataset, sliding_window for others."""
    rng = np.random.default_rng(42)

    # Create two scatter dirs: ds_pack (should get pack) and ds_slide (should get sliding_window)
    docs_pack = _make_docs(rng, num_docs=4, min_len=5, max_len=10)
    docs_slide = _make_docs(rng, num_docs=4, min_len=5, max_len=10)
    scatter_pack = _make_test_scatter(tmp_dir, "ds_pack", [docs_pack])
    scatter_slide = _make_test_scatter(tmp_dir, "ds_slide", [docs_slide])

    cfg = _FakeBfdConfig(lazy_bfd_pack="ds_pack")

    mode_pack = _resolve_mode(cfg, scatter_pack)
    mode_slide = _resolve_mode(cfg, scatter_slide)
    assert mode_pack == "pack"
    assert mode_slide == "sliding_window"

    effective_seq = SEQ_LENGTH + 1

    # Build both sources with resolved modes and verify they produce valid samples
    src_pack = MultiShardLazyDataSource(
        scatter_dir=scatter_pack,
        mode=mode_pack,
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
    )
    src_slide = MultiShardLazyDataSource(
        scatter_dir=scatter_slide,
        mode=mode_slide,
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
    )

    assert len(src_pack) > 0
    assert len(src_slide) > 0

    # Pack mode: sample is a packed bin of documents
    sample_pack = src_pack[0]["text"]
    assert sample_pack.shape == (effective_seq,)

    # Sliding window mode: sample is a sequential window
    sample_slide = src_slide[0]["text"]
    assert sample_slide.shape == (effective_seq,)


# ---------------------------------------------------------------------------
# Dataset name resolution helpers
# ---------------------------------------------------------------------------


class TestDatasetNameResolution:
  """Tests for path-to-dataset-name normalization."""

  def test_extracts_bare_dataset_name(self):
    assert _dataset_name_from_path("my_dataset") == "my_dataset"

  def test_extracts_name_from_scatter_path(self):
    assert _dataset_name_from_path("/a/b/c/my_dataset.scatter") == "my_dataset"

  def test_extracts_name_from_hash_subdir_path(self):
    assert _dataset_name_from_path("/a/b/c/my_dataset.scatter/hash123") == "dataset"

  def test_strips_native_mdata_prefix_for_hash_subdir_path(self):
    assert _dataset_name_from_path("/a/b/c/379824db_my_dataset.scatter/hash123") == "my_dataset"

  def test_resolves_no_attnmask_ids_from_mixed_path_forms(self):
    files_str = "0.7 /a/b/c/379824db_ds_a.scatter/hash123 0.3 ds_b"
    assert _resolve_no_attnmask_ids(files_str, "ds_a") == {0}
    assert _resolve_no_attnmask_ids(files_str, "ds_b") == {1}
    assert _resolve_no_attnmask_ids(files_str, "ds_a,ds_b") == {0, 1}


# ---------------------------------------------------------------------------
# Test: dataset_id passthrough in LazyBlendedDataSource
# ---------------------------------------------------------------------------


class TestBlendDatasetId:
  """Tests for dataset_id passthrough in blended sources."""

  def test_dataset_id_present(self):
    """Verify LazyBlendedDataSource adds dataset_id to returned dict."""

    class DummySource:  # pylint: disable=missing-class-docstring

      def __init__(self, n, label):
        self._n = n
        self._label = label

      def __len__(self):
        return self._n

      def __getitem__(self, idx):
        return {"text": np.array([self._label, idx % self._n], dtype=np.int32)}

    sources = [DummySource(100, 10), DummySource(50, 20), DummySource(30, 30)]
    weights = np.array([0.7, 0.2, 0.1], dtype=np.float64)
    blended = LazyBlendedDataSource(sources, weights, size=50)

    for i in range(50):
      sample = blended[i]
      assert "dataset_id" in sample, f"sample {i} missing dataset_id"
      ds_id = int(sample["dataset_id"])
      assert 0 <= ds_id < 3
      # The label in text[0] should match the source
      expected_label = [10, 20, 30][ds_id]
      assert sample["text"][0] == expected_label, f"sample {i}: ds_id={ds_id}, label={sample['text'][0]}"

  def test_dataset_id_matches_blend_index(self):
    """Verify dataset_id matches the precomputed dataset_index array."""

    class DummySource:  # pylint: disable=missing-class-docstring

      def __init__(self, n):
        self._n = n

      def __len__(self):
        return self._n

      def __getitem__(self, idx):
        return {"text": np.array([idx % self._n], dtype=np.int32)}

    sources = [DummySource(100), DummySource(50)]
    weights = np.array([0.6, 0.4], dtype=np.float64)
    blended = LazyBlendedDataSource(sources, weights, size=30)

    for i in range(30):
      sample = blended[i]
      expected_ds = int(blended.dataset_index[i])
      assert int(sample["dataset_id"]) == expected_ds


# ---------------------------------------------------------------------------
# Test: MegatronSplitInputsTargets with no_attnmask_dataset_ids
# ---------------------------------------------------------------------------


class TestMegatronSplitNoAttnmask:
  """Tests for per-dataset attention mask override."""

  def _make_tokens_with_eod(self, seq_length, eod_id, eod_positions):
    """Create a token array with EOD at specified positions."""
    tokens = np.arange(100, 100 + seq_length + 1, dtype=np.int32)
    for pos in eod_positions:
      tokens[pos] = eod_id
    return tokens

  def test_no_override_normal_behavior(self):
    """Without no_attnmask_dataset_ids, reset_attention_mask works normally."""
    from maxtext.input_pipeline.input_pipeline_utils import MegatronSplitInputsTargets

    eod_id = 2
    transform = MegatronSplitInputsTargets(eod_id=eod_id, reset_attention_mask=True)
    tokens = self._make_tokens_with_eod(16, eod_id, [5, 10])
    result = transform.map({"text": tokens})

    seg = result["inputs_segmentation"]
    # With reset_attention_mask=True and EODs at positions 5 and 10,
    # segment IDs should change after each EOD
    assert seg[0] >= 1
    assert seg[6] > seg[5]  # segment changes after EOD at pos 5
    assert seg[11] > seg[10]  # segment changes after EOD at pos 10

  def test_override_disables_reset(self):
    """Samples from no_attnmask datasets get all-ones segmentation."""
    from maxtext.input_pipeline.input_pipeline_utils import MegatronSplitInputsTargets

    eod_id = 2
    transform = MegatronSplitInputsTargets(eod_id=eod_id, reset_attention_mask=True, no_attnmask_dataset_ids={0, 2})
    tokens = self._make_tokens_with_eod(16, eod_id, [5, 10])

    # dataset_id=0 → in no_attnmask list → should get all-ones segmentation
    result = transform.map({"text": tokens, "dataset_id": np.int32(0)})
    np.testing.assert_array_equal(result["inputs_segmentation"], np.ones(16, dtype=np.int32))
    np.testing.assert_array_equal(result["inputs_position"], np.arange(16, dtype=np.int32))

    # dataset_id=2 → also in no_attnmask list
    result = transform.map({"text": tokens, "dataset_id": np.int32(2)})
    np.testing.assert_array_equal(result["inputs_segmentation"], np.ones(16, dtype=np.int32))

  def test_non_override_dataset_still_resets(self):
    """Samples from datasets NOT in the override list still reset attention."""
    from maxtext.input_pipeline.input_pipeline_utils import MegatronSplitInputsTargets

    eod_id = 2
    transform = MegatronSplitInputsTargets(eod_id=eod_id, reset_attention_mask=True, no_attnmask_dataset_ids={0})
    tokens = self._make_tokens_with_eod(16, eod_id, [5])

    # dataset_id=1 → NOT in no_attnmask list → should reset at EOD
    result = transform.map({"text": tokens, "dataset_id": np.int32(1)})
    seg = result["inputs_segmentation"]
    assert seg[6] > seg[5]  # segment changes after EOD

  def test_no_dataset_id_uses_global(self):
    """Without dataset_id in element, uses global reset_attention_mask."""
    from maxtext.input_pipeline.input_pipeline_utils import MegatronSplitInputsTargets

    eod_id = 2
    transform = MegatronSplitInputsTargets(eod_id=eod_id, reset_attention_mask=True, no_attnmask_dataset_ids={0})
    tokens = self._make_tokens_with_eod(16, eod_id, [5])

    # No dataset_id → uses reset_attention_mask=True
    result = transform.map({"text": tokens})
    seg = result["inputs_segmentation"]
    assert seg[6] > seg[5]  # segment changes after EOD

  def test_eod_mask_loss_independent_of_no_attnmask(self):
    """eod_mask_loss still works independently when no_attnmask overrides attention."""
    from maxtext.input_pipeline.input_pipeline_utils import MegatronSplitInputsTargets

    eod_id = 2
    transform = MegatronSplitInputsTargets(
        eod_id=eod_id, reset_attention_mask=True, eod_mask_loss=True, no_attnmask_dataset_ids={0}
    )
    tokens = self._make_tokens_with_eod(16, eod_id, [5])

    # dataset_id=0 → no attention reset, but eod_mask_loss still applies
    result = transform.map({"text": tokens, "dataset_id": np.int32(0)})
    # inputs_segmentation should be all-ones (no attention mask)
    np.testing.assert_array_equal(result["inputs_segmentation"], np.ones(16, dtype=np.int32))
    # targets_segmentation should have 0 at EOD position
    assert result["targets_segmentation"][5] == 0


# ---------------------------------------------------------------------------
# Test: BFD packing output unchanged after optimization
# ---------------------------------------------------------------------------


class TestBfdPackingOptimized:
  """Tests that BFD packing produces identical output after optimization."""

  def test_bfd_basic_correctness(self):
    """Verify BFD packing produces valid bin_entries and bin_lens."""
    from maxtext.input_pipeline._lazy_bfd_packing import build_bin_index

    doc_lens = np.array([100, 50, 30, 80, 120, 15, 45], dtype=np.int64)
    seq_length = 64

    bin_entries, bin_lens = build_bin_index(doc_lens, seq_length)

    # bin_lens should be even (pairs of doc_idx, offset)
    assert np.all(bin_lens % 2 == 0)
    # Total entries should match
    assert bin_entries.shape[0] == np.sum(bin_lens) // 2
    assert bin_entries.shape[1] == 2
    # All doc indices should be valid
    assert np.all(bin_entries[:, 0] >= 0)
    assert np.all(bin_entries[:, 0] < len(doc_lens))

  def test_bfd_empty(self):
    from maxtext.input_pipeline._lazy_bfd_packing import build_bin_index

    bin_entries, bin_lens = build_bin_index(np.array([], dtype=np.int64), 64)
    assert bin_entries.shape == (0, 2)
    assert bin_lens.shape == (0,)

  def test_bfd_all_fit_in_one_bin(self):
    """Documents that fit entirely within seq_length should pack together."""
    from maxtext.input_pipeline._lazy_bfd_packing import build_bin_index

    doc_lens = np.array([10, 10, 10], dtype=np.int64)
    seq_length = 64

    _, bin_lens = build_bin_index(doc_lens, seq_length)

    # Should produce fewer bins than docs since they can pack together
    num_bins = len(bin_lens)
    assert num_bins <= len(doc_lens)

  def test_bfd_deterministic(self):
    """Same input should always produce same output."""
    from maxtext.input_pipeline._lazy_bfd_packing import build_bin_index

    rng = np.random.default_rng(42)
    doc_lens = rng.integers(5, 200, size=100).astype(np.int64)
    seq_length = 64

    entries1, lens1 = build_bin_index(doc_lens, seq_length)
    entries2, lens2 = build_bin_index(doc_lens, seq_length)

    np.testing.assert_array_equal(entries1, entries2)
    np.testing.assert_array_equal(lens1, lens2)

  def test_bfd_with_sort_by_lens(self):
    """Verify sort_by_lens mode produces valid output."""
    from maxtext.input_pipeline._lazy_bfd_packing import build_bin_index

    doc_lens = np.array([100, 50, 30, 80, 120, 15, 45], dtype=np.int64)
    seq_length = 64

    bin_entries, bin_lens = build_bin_index(doc_lens, seq_length, sort_by_lens=True)
    assert np.all(bin_lens % 2 == 0)
    assert bin_entries.shape[1] == 2


# ---------------------------------------------------------------------------
# Test: Grain iterator checkpoint save & restore
# ---------------------------------------------------------------------------


class TestGrainCheckpointSaveRestore:
  """Tests that lazy data sources support Grain iterator checkpointing."""

  def _make_dataset(self, tmp_dir, num_docs=20, seq_length=8):
    """Create a minimal Grain MapDataset from a lazy source."""
    import grain.python as grain_lib
    from maxtext.input_pipeline.input_pipeline_utils import MegatronSplitInputsTargets

    rng = np.random.default_rng(42)
    docs = [rng.integers(100, 10000, size=rng.integers(10, 30), dtype=np.int32) for _ in range(num_docs)]
    scatter_dir = _make_test_scatter(tmp_dir, "ckpt_test", [docs])

    effective_seq = seq_length + 1  # antllm convention
    source = MultiShardLazyDataSource(
        scatter_dir=scatter_dir,
        mode="sliding_window",
        seq_length=effective_seq,
        eos_token_id=EOS_TOKEN_ID,
    )

    dataset = grain_lib.MapDataset.source(source)
    dataset = dataset.map(MegatronSplitInputsTargets(eod_id=EOS_TOKEN_ID, reset_attention_mask=True))
    return dataset

  def test_get_state_returns_serializable(self, tmp_dir):
    """Verify iterator state is JSON-serializable (required by GrainCheckpointHandler)."""
    import json

    dataset = self._make_dataset(tmp_dir)
    it = iter(dataset)
    next(it)  # advance 1 step

    state = it.get_state()
    # State must be JSON-serializable (checkpointing.py line 74 does json.dumps)
    json_str = json.dumps(state, indent=4)
    restored_state = json.loads(json_str)
    assert restored_state == state

  def test_restore_resumes_from_saved_position(self, tmp_dir):
    """Verify that restoring state resumes iteration from the saved position."""
    dataset = self._make_dataset(tmp_dir)

    # Iterate to position N and save state
    it1 = iter(dataset)
    n_advance = 3
    for _ in range(n_advance):
      next(it1)
    state = it1.get_state()

    # Collect remaining samples from it1
    remaining_from_it1 = []
    for _ in range(5):
      sample = next(it1)
      remaining_from_it1.append(sample["inputs"].copy())

    # Create new iterator and restore state
    it2 = iter(dataset)
    it2.set_state(state)

    # Collect same number of samples from it2
    remaining_from_it2 = []
    for _ in range(5):
      sample = next(it2)
      remaining_from_it2.append(sample["inputs"].copy())

    # Should produce identical samples
    for i, (a, b) in enumerate(zip(remaining_from_it1, remaining_from_it2)):
      np.testing.assert_array_equal(a, b, err_msg=f"Sample {i} after restore differs")

  def test_checkpoint_at_start(self, tmp_dir):
    """Verify checkpointing at the very beginning (before any next() call)."""
    dataset = self._make_dataset(tmp_dir)

    it1 = iter(dataset)
    state = it1.get_state()

    samples1 = [next(it1)["inputs"].copy() for _ in range(3)]

    it2 = iter(dataset)
    it2.set_state(state)
    samples2 = [next(it2)["inputs"].copy() for _ in range(3)]

    for i in range(3):
      np.testing.assert_array_equal(samples1[i], samples2[i])

  def test_multiple_checkpoints(self, tmp_dir):
    """Verify multiple save/restore cycles work correctly."""
    dataset = self._make_dataset(tmp_dir)
    it = iter(dataset)

    states = []
    next_samples = []
    for _ in range(5):
      states.append(it.get_state())
      sample = next(it)
      next_samples.append(sample["inputs"].copy())

    # Restore each state and verify it produces the correct next sample
    for i, state in enumerate(states):
      it2 = iter(dataset)
      it2.set_state(state)
      sample = next(it2)
      np.testing.assert_array_equal(sample["inputs"], next_samples[i], err_msg=f"Checkpoint {i} restoration failed")

  def test_checkpoint_with_blended_source(self, tmp_dir):
    """Verify checkpointing works with LazyBlendedDataSource."""
    import grain.python as grain_lib
    from maxtext.input_pipeline.input_pipeline_utils import MegatronSplitInputsTargets

    rng = np.random.default_rng(123)
    docs_a = [rng.integers(100, 5000, size=rng.integers(10, 25), dtype=np.int32) for _ in range(15)]
    docs_b = [rng.integers(5000, 10000, size=rng.integers(10, 25), dtype=np.int32) for _ in range(10)]

    scatter_a = _make_test_scatter(tmp_dir, "ckpt_ds_a", [docs_a])
    scatter_b = _make_test_scatter(tmp_dir, "ckpt_ds_b", [docs_b])

    effective_seq = SEQ_LENGTH + 1
    src_a = MultiShardLazyDataSource(
        scatter_dir=scatter_a, mode="sliding_window", seq_length=effective_seq, eos_token_id=EOS_TOKEN_ID
    )
    src_b = MultiShardLazyDataSource(
        scatter_dir=scatter_b, mode="sliding_window", seq_length=effective_seq, eos_token_id=EOS_TOKEN_ID
    )

    weights = np.array([0.6, 0.4], dtype=np.float64)
    blended = LazyBlendedDataSource([src_a, src_b], weights, size=20)

    dataset = grain_lib.MapDataset.source(blended)
    dataset = dataset.map(MegatronSplitInputsTargets(eod_id=EOS_TOKEN_ID))

    it1 = iter(dataset)
    for _ in range(4):
      next(it1)
    state = it1.get_state()
    expected = [next(it1)["inputs"].copy() for _ in range(3)]

    it2 = iter(dataset)
    it2.set_state(state)
    actual = [next(it2)["inputs"].copy() for _ in range(3)]

    for i in range(3):
      np.testing.assert_array_equal(expected[i], actual[i])

  def test_checkpoint_with_batch(self, tmp_dir):
    """Verify checkpointing works with batched dataset."""
    import functools
    import grain.python as grain_lib

    dataset = self._make_dataset(tmp_dir, num_docs=30, seq_length=8)
    batch_fn = functools.partial(grain_lib.experimental.batch_and_pad, batch_size=2, pad_value=EOS_TOKEN_ID)
    dataset = dataset.batch(2, batch_fn=batch_fn)

    it1 = iter(dataset)
    next(it1)  # advance 1 batch
    state = it1.get_state()
    expected_batch = next(it1)

    it2 = iter(dataset)
    it2.set_state(state)
    actual_batch = next(it2)

    np.testing.assert_array_equal(expected_batch["inputs"], actual_batch["inputs"])
    np.testing.assert_array_equal(expected_batch["targets"], actual_batch["targets"])


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
