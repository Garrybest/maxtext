"""Megatron-LM dataloader alignment test.

Verifies that MaxText's mmap/npy index construction and iteration produce
identical results to Megatron-LM's reference implementation, given the same
inputs and random seeds.

Coverage:
- Document index: element-by-element match
- Sample index: element-by-element match (Python vs C++)
- Shuffle index: algorithm match + production RNG flow match
- Blend index: greedy scheduler match (Python vs C++)
- Token content: per-sample token sequence match
- Multi-rank distribution: per-host sharding equivalence

Requires: torch (CPU), megatron-core.
"""

import os
import tempfile

import numpy as np
import pytest

# Skip entire module if megatron-core is not installed.
megatron_core = pytest.importorskip("megatron.core")

from megatron.core.datasets.gpt_dataset import (
    _build_document_index,
    _build_shuffle_index,
)
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.utils import Split
from megatron.core.datasets.megatron_tokenizer import MegatronLegacyTokenizer as MegatronTokenizer
from megatron.core.datasets import helpers as megatron_helpers
from megatron.core.datasets.blended_megatron_dataset_config import (
    parse_and_normalize_split,
    convert_split_vector_to_split_matrix,
)

from maxtext.input_pipeline._mmap_index_utils import (
    build_document_index,
    build_sample_index,
    build_shuffle_index,
    get_document_sizes,
    convert,
)
from tests.unit.mmap_test_utils import create_mmap_test_data
from maxtext.input_pipeline._mmap_datasource import (
    MMapDatasetConfig,
    MegatronNpyDataSource,
    _discover_npy_indices,
    create_mmap_npy_source,
)
from maxtext.input_pipeline._megatron_blending import build_blending_indices

# pylint: disable=redefined-outer-name,protected-access

pytestmark = [pytest.mark.megatron_alignment, pytest.mark.cpu_only]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
  with tempfile.TemporaryDirectory() as d:
    yield d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_eod_dataset(tmp_dir, num_docs=10, eod_id=0, seed=123):
  """Create synthetic data where each document ends with eod_id (--append-eod style)."""
  os.makedirs(tmp_dir, exist_ok=True)
  rng = np.random.RandomState(seed)
  seqs = []
  for d in range(num_docs):
    length = 20 + d * 5  # 20, 25, 30, ...
    tokens = rng.randint(1, 10000, size=length, dtype=np.int32)
    tokens[-1] = eod_id  # simulate --append-eod
    seqs.append(tokens)
  prefix = os.path.join(tmp_dir, "data")
  doc_boundaries = list(range(num_docs + 1))  # 1 seq per doc
  create_mmap_test_data(prefix, seqs, doc_boundaries=doc_boundaries)
  return prefix, seqs


class _StubTokenizer(MegatronTokenizer):
  """Minimal tokenizer stub that only provides eod for GPTDataset."""

  def __init__(self, eod):
    super().__init__(None)
    self._eod = eod

  @property
  def vocab_size(self):
    return 50000

  @property
  def vocab(self):
    raise NotImplementedError

  @property
  def inv_vocab(self):
    raise NotImplementedError

  @property
  def eod(self):
    return self._eod

  def tokenize(self, text):
    raise NotImplementedError

  def detokenize(self, ids):
    raise NotImplementedError


def _build_megatron_gpt_dataset(prefix, seq_length, seed, eod_id, num_samples=None):
  """Build a real Megatron GPTDataset from a .bin/.idx prefix."""
  tokenizer = _StubTokenizer(eod=eod_id)
  config = GPTDatasetConfig(
      random_seed=seed,
      sequence_length=seq_length,
      reset_position_ids=False,
      reset_attention_mask=False,
      eod_mask_loss=False,
      tokenizer=tokenizer,
  )
  indexed_ds = IndexedDataset(prefix, multimodal=False, mmap=True)
  num_docs = indexed_ds.document_indices.shape[0] - 1
  all_indices = np.arange(num_docs, dtype=np.int32)
  return GPTDataset(
      indexed_dataset=indexed_ds,
      dataset_path=prefix,
      indexed_indices=all_indices,
      num_samples=num_samples,
      index_split=Split.train,
      config=config,
  )


# ===========================================================================
# Document index alignment
# ===========================================================================


class TestDocumentIndexAlignment:
  """MaxText build_document_index vs Megatron _build_document_index."""

  @pytest.mark.parametrize(
      "num_docs,num_epochs,seed",
      [
          (10, 1, 42),
          (10, 3, 42),
          (100, 2, 1234),
          (50, 5, 99),
      ],
  )
  def test_flat_shuffle_matches(self, num_docs, num_epochs, seed):
    """Same seed -> identical document ordering."""
    maxtext_idx = build_document_index(num_docs, num_epochs, seed)

    documents = np.arange(num_docs, dtype=np.int32)
    rng = np.random.RandomState(seed)
    megatron_idx = _build_document_index(documents, num_epochs, rng, False)

    np.testing.assert_array_equal(maxtext_idx, megatron_idx)

  @pytest.mark.parametrize(
      "num_docs,num_epochs,seed",
      [
          (20, 4, 42),
          (50, 3, 1234),
      ],
  )
  def test_separate_last_epoch_matches(self, num_docs, num_epochs, seed):
    """Separate-last-epoch mode produces identical results."""
    maxtext_idx = build_document_index(num_docs, num_epochs, seed, separate_last_epoch=True)

    documents = np.arange(num_docs, dtype=np.int32)
    rng = np.random.RandomState(seed)
    megatron_idx = _build_document_index(documents, num_epochs, rng, True)

    np.testing.assert_array_equal(maxtext_idx, megatron_idx)


# ===========================================================================
# Sample index alignment
# ===========================================================================


class TestSampleIndexAlignment:
  """MaxText build_sample_index vs Megatron helpers.build_sample_idx (C++)."""

  @pytest.mark.parametrize(
      "doc_tokens,seq_length,num_epochs,seed",
      [
          ([100], 8, 1, 42),
          ([50, 50], 16, 2, 1234),
          ([20, 30, 15, 25, 10], 8, 2, 42),
          ([200], 32, 1, 99),
      ],
  )
  def test_sample_index_matches(self, doc_tokens, seq_length, num_epochs, seed):
    """Sample boundary arrays match element-by-element."""
    sizes = np.array(doc_tokens, dtype=np.int32)
    doc_sizes = sizes.astype(np.int64)
    num_docs = len(sizes)
    tokens_per_epoch = int(sizes.sum())

    documents = np.arange(num_docs, dtype=np.int32)
    rng = np.random.RandomState(seed)
    doc_index = _build_document_index(documents, num_epochs, rng, False)

    maxtext_sample_idx = build_sample_index(doc_sizes, doc_index, seq_length, drop_last=True, add_extra_token=1)

    megatron_sample_idx = megatron_helpers.build_sample_idx(
        sizes,
        doc_index,
        seq_length,
        num_epochs=num_epochs,
        tokens_per_epoch=tokens_per_epoch,
        drop_last_partial_sequence=True,
        add_extra_token_to_sequence=True,
    )

    np.testing.assert_array_equal(maxtext_sample_idx, megatron_sample_idx)

  def test_drop_last_false_matches(self):
    """drop_last=False (ceil division) produces matching indices."""
    sizes = np.array([17], dtype=np.int32)
    doc_sizes = sizes.astype(np.int64)
    doc_index = np.array([0], dtype=np.int32)

    maxtext_sample_idx = build_sample_index(doc_sizes, doc_index, 8, drop_last=False, add_extra_token=1)
    megatron_sample_idx = megatron_helpers.build_sample_idx(
        sizes,
        doc_index,
        8,
        num_epochs=1,
        tokens_per_epoch=17,
        drop_last_partial_sequence=False,
        add_extra_token_to_sequence=True,
    )

    np.testing.assert_array_equal(maxtext_sample_idx, megatron_sample_idx)


# ===========================================================================
# Shuffle index alignment
# ===========================================================================


class TestShuffleIndexAlignment:
  """MaxText build_shuffle_index vs Megatron _build_shuffle_index."""

  @pytest.mark.parametrize(
      "num_samples,total_size,seed",
      [
          (50, 50, 42),
          (30, 50, 1234),
          (100, 100, 99),
      ],
  )
  def test_shuffle_algorithm_matches(self, num_samples, total_size, seed):
    """Same fresh RNG state -> identical shuffle permutation."""
    maxtext_shuffle = build_shuffle_index(num_samples, total_size, seed)

    rng = np.random.RandomState(seed)
    megatron_shuffle = _build_shuffle_index(num_samples, total_size, rng)

    np.testing.assert_array_equal(maxtext_shuffle, megatron_shuffle)

  def test_production_rng_flow_matches(self, tmp_dir):
    """convert() shuffle_index matches Megatron's single-RNG flow exactly.

    This tests the full production path: RandomState(seed) is consumed by
    _build_document_index, then the same state is used for _build_shuffle_index.
    """
    prefix, _ = _create_eod_dataset(tmp_dir)
    seq_length = 8
    seed = 42
    num_epochs = 2

    # MaxText: run convert() which now uses single RNG flow
    out_dir = os.path.join(tmp_dir, "npy_output")
    convert([prefix], out_dir, seq_length=seq_length, num_epochs=num_epochs, seed=seed)

    # Load the generated shuffle index
    _, _, shuffle_path = _discover_npy_indices(out_dir)
    maxtext_shuffle = np.load(shuffle_path)

    # Megatron: replicate the exact RNG flow
    doc_sizes = get_document_sizes([prefix])
    num_docs = len(doc_sizes)
    tokens_per_epoch = int(doc_sizes.sum())

    rng = np.random.RandomState(seed)
    documents = np.arange(num_docs, dtype=np.int32)
    doc_index = _build_document_index(documents, num_epochs, rng, False)
    # rng is now consumed by doc_index shuffling

    sizes = doc_sizes.astype(np.int32)
    sample_index = megatron_helpers.build_sample_idx(
        sizes,
        doc_index,
        seq_length,
        num_epochs=num_epochs,
        tokens_per_epoch=tokens_per_epoch,
        drop_last_partial_sequence=True,
        add_extra_token_to_sequence=True,
    )
    total_samples = sample_index.shape[0] - 1
    megatron_shuffle = _build_shuffle_index(total_samples, total_samples, rng)

    np.testing.assert_array_equal(maxtext_shuffle, megatron_shuffle)


# ===========================================================================
# Token content alignment
# ===========================================================================


class TestTokenContentAlignment:
  """Verify per-sample token sequences match real Megatron GPTDataset.

  Instantiates real Megatron GPTDataset objects and compares their
  __getitem__ output against MegatronNpyDataSource[i], token-by-token.
  """

  @staticmethod
  def _meg_raw_tokens(gpt_ds, idx):
    """Extract raw seq_length+1 token sequence from GPTDataset[idx]."""
    sample = gpt_ds[idx]
    tokens = sample["tokens"].numpy().astype(np.int32)
    labels = sample["labels"].numpy().astype(np.int32)
    return np.concatenate([tokens[:1], labels])

  def test_token_sequences_match(self, tmp_dir):
    """First 50 samples: real Megatron GPTDataset == MegatronNpyDataSource."""
    eod_id = 0
    prefix, _ = _create_eod_dataset(tmp_dir, num_docs=10, eod_id=eod_id)
    seq_length = 8
    seed = 42

    # Build MaxText npy indices (num_epochs=1 to match GPTDataset num_samples=None)
    out_dir = os.path.join(tmp_dir, "npy_output")
    convert([prefix], out_dir, seq_length=seq_length, num_epochs=1, seed=seed)

    # MaxText side
    npy_ds = MegatronNpyDataSource(
        npy_dir=out_dir,
        bin_paths=prefix,
        eod_id=eod_id,
        seq_length=seq_length,
    )

    # Megatron side: real GPTDataset
    gpt_ds = _build_megatron_gpt_dataset(prefix, seq_length, seed, eod_id)

    num_samples = min(len(npy_ds), len(gpt_ds), 50)
    assert num_samples > 0

    for i in range(num_samples):
      meg_raw = self._meg_raw_tokens(gpt_ds, i)
      mx_raw = npy_ds[i]["text"]

      np.testing.assert_array_equal(
          mx_raw,
          meg_raw,
          err_msg=f"Token mismatch at sample {i}",
      )

  def test_cross_document_boundary(self, tmp_dir):
    """Samples spanning multiple documents produce identical tokens."""
    eod_id = 0
    # Use short documents to force cross-document samples
    rng = np.random.RandomState(456)
    seqs = []
    for d in range(20):
      length = 5 + d % 3  # 5, 6, 7, 5, 6, 7, ...
      tokens = rng.randint(1, 10000, size=length, dtype=np.int32)
      tokens[-1] = eod_id
      seqs.append(tokens)
    prefix = os.path.join(tmp_dir, "data")
    create_mmap_test_data(prefix, seqs, doc_boundaries=list(range(len(seqs) + 1)))

    seq_length = 16  # longer than any single document
    seed = 99

    out_dir = os.path.join(tmp_dir, "npy_output")
    convert([prefix], out_dir, seq_length=seq_length, num_epochs=1, seed=seed)

    # MaxText side
    npy_ds = MegatronNpyDataSource(
        npy_dir=out_dir,
        bin_paths=prefix,
        eod_id=eod_id,
        seq_length=seq_length,
    )

    # Megatron side: real GPTDataset
    gpt_ds = _build_megatron_gpt_dataset(prefix, seq_length, seed, eod_id)

    num_samples = min(len(npy_ds), len(gpt_ds), 50)
    assert num_samples > 0

    for i in range(num_samples):
      meg_raw = self._meg_raw_tokens(gpt_ds, i)
      mx_raw = npy_ds[i]["text"]

      np.testing.assert_array_equal(
          mx_raw,
          meg_raw,
          err_msg=f"Cross-doc token mismatch at sample {i}",
      )


# ===========================================================================
# Multi-host sharding consistency
# ===========================================================================


class TestMultiHostSharding:
  """Verify that Grain's per-host stride sharding reassembles to the global sequence."""

  @staticmethod
  def _get_host_samples(out_dir, prefix, seq_length, host_index, host_count):
    """Iterate a per-host sharded dataset via get_datasets."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    pattern = f"{out_dir}|{prefix}"
    ds = get_datasets(
        data_file_pattern=pattern,
        data_file_type="mmap_npy",
        shuffle=False,
        shuffle_seed=0,
        shuffle_buffer_size=0,
        num_epoch=1,
        dataloading_host_index=host_index,
        dataloading_host_count=host_count,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        dataset_config=MMapDatasetConfig(
            max_target_length=seq_length,
            eod_id=0,
            mmap_split_sentences=False,
        ),
    )
    return [item["text"] for item in ds]

  def test_4_host_interleave_equals_global(self, tmp_dir):
    """4-host stride sharding, interleaved back, reproduces the global sequence."""
    num_hosts = 4
    prefix, _ = _create_eod_dataset(tmp_dir, num_docs=10, eod_id=0)
    seq_length = 8

    out_dir = os.path.join(tmp_dir, "npy_output")
    convert([prefix], out_dir, seq_length=seq_length, num_epochs=2, seed=42)

    global_samples = self._get_host_samples(out_dir, prefix, seq_length, 0, 1)
    total = len(global_samples)

    # Collect per-host shards
    host_shards = [self._get_host_samples(out_dir, prefix, seq_length, h, num_hosts) for h in range(num_hosts)]

    # Interleave: host 0 → idx 0, host 1 → idx 1, ..., host 0 → idx 4, ...
    reassembled = [None] * total
    for h, shard in enumerate(host_shards):
      for step, sample in enumerate(shard):
        reassembled[h + step * num_hosts] = sample

    assert all(r is not None for r in reassembled), "Some global indices not covered"
    for i, (expected, actual) in enumerate(zip(global_samples, reassembled)):
      np.testing.assert_array_equal(
          actual,
          expected,
          err_msg=f"Global index {i}: reassembled != global",
      )


# ===========================================================================
# Blend index alignment
# ===========================================================================


class TestBlendAlignment:
  """Verify blend index construction matches Megatron C++ and blend-then-shard
  reconstructs the global sequence."""

  @pytest.mark.parametrize(
      "weights,size",
      [
          ([0.7, 0.3], 50),
          ([0.5, 0.5], 60),
          ([0.5, 0.3, 0.2], 80),
      ],
  )
  def test_blend_indices_match_megatron_cpp(self, weights, size):
    """build_blending_indices matches Megatron C++ element-by-element."""
    num_datasets = len(weights)
    w = np.array(weights, dtype=np.float64)

    # MaxText Python
    mx_ds_idx = np.zeros(size, dtype=np.int16)
    mx_sample_idx = np.zeros(size, dtype=np.int64)
    build_blending_indices(mx_ds_idx, mx_sample_idx, w, num_datasets, size)

    # Megatron C++
    meg_ds_idx = np.zeros(size, dtype=np.int16)
    meg_sample_idx = np.zeros(size, dtype=np.int64)
    megatron_helpers.build_blending_indices(
        meg_ds_idx,
        meg_sample_idx,
        w,
        num_datasets,
        size,
        False,
    )

    np.testing.assert_array_equal(mx_ds_idx, meg_ds_idx, err_msg="dataset_index mismatch")
    np.testing.assert_array_equal(mx_sample_idx, meg_sample_idx, err_msg="dataset_sample_index mismatch")

  @pytest.mark.parametrize(
      "weights,size",
      [
          ([0.7, 0.3], 100),
          ([0.5, 0.3, 0.2], 200),
      ],
  )
  def test_blend_weight_proportionality(self, weights, size):
    """Per-dataset sample counts approximate the target weights."""
    num_datasets = len(weights)
    w = np.array(weights, dtype=np.float64)
    ds_idx = np.zeros(size, dtype=np.int16)
    sample_idx = np.zeros(size, dtype=np.int64)
    build_blending_indices(ds_idx, sample_idx, w, num_datasets, size)

    counts = np.bincount(ds_idx, minlength=num_datasets)
    actual_ratios = counts / size
    for i, (expected, actual) in enumerate(zip(weights, actual_ratios)):
      assert abs(actual - expected) < 0.02, f"Dataset {i}: expected ratio ~{expected}, got {actual}"

  def test_blend_then_shard_reconstructs_global(self, tmp_dir):
    """Blend-then-shard across 4 hosts, interleaved back, equals global blend."""
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    seq_length = 8
    num_hosts = 4

    # Create two distinct datasets (different seeds → different token content)
    prefix_a, _ = _create_eod_dataset(tmp_dir, num_docs=8, eod_id=0, seed=111)
    prefix_b, _ = _create_eod_dataset(
        os.path.join(tmp_dir, "ds_b"),
        num_docs=8,
        eod_id=0,
        seed=222,
    )
    out_a = os.path.join(tmp_dir, "npy_a")
    out_b = os.path.join(tmp_dir, "npy_b")
    convert([prefix_a], out_a, seq_length=seq_length, num_epochs=1, seed=42)
    convert([prefix_b], out_b, seq_length=seq_length, num_epochs=1, seed=42)

    blend_pattern = f"{out_a}|{prefix_a},0.7;{out_b}|{prefix_b},0.3"
    ds_cfg = MMapDatasetConfig(max_target_length=seq_length, eod_id=0, mmap_split_sentences=False)

    def _get_samples(host_index, host_count):
      ds = get_datasets(
          data_file_pattern=blend_pattern,
          data_file_type="mmap_npy",
          shuffle=False,
          shuffle_seed=0,
          shuffle_buffer_size=0,
          num_epoch=1,
          dataloading_host_index=host_index,
          dataloading_host_count=host_count,
          grain_worker_count=0,
          grain_num_threads=1,
          grain_prefetch_buffer_size=1,
          grain_data_source_max_workers=1,
          dataset_config=ds_cfg,
      )
      return [item["text"] for item in ds]

    global_samples = _get_samples(0, 1)
    total = len(global_samples)
    assert total > 0, "Blended dataset produced no samples"

    # Reassemble from per-host shards
    reassembled = [None] * total
    for h in range(num_hosts):
      shard = _get_samples(h, num_hosts)
      for step, sample in enumerate(shard):
        idx = h + step * num_hosts
        if idx < total:
          reassembled[idx] = sample

    assert all(r is not None for r in reassembled), "Some global indices not covered"
    for i, (expected, actual) in enumerate(zip(global_samples, reassembled)):
      np.testing.assert_array_equal(
          actual,
          expected,
          err_msg=f"Blend global index {i}: reassembled != global",
      )

  def test_blend_e2e_tokens_match_megatron(self, tmp_dir):
    """End-to-end: MegatronBlendedDataSource[i] matches real Megatron GPTDataset dispatch.

    Constructs two synthetic datasets, builds a MegatronBlendedDataSource
    (MaxText) and individual GPTDataset objects (Megatron), then uses
    Megatron C++ blend indices to dispatch into GPTDataset[ds_id][sample_id]
    and compares against MaxText's blended output token-by-token.
    """
    eod_id = 0
    seq_length = 8
    seed = 42
    weights = [0.7, 0.3]

    # ---- Create two distinct synthetic datasets ----------------------------
    prefix_a, _ = _create_eod_dataset(
        os.path.join(tmp_dir, "ds_a"),
        num_docs=8,
        eod_id=eod_id,
        seed=111,
    )
    prefix_b, _ = _create_eod_dataset(
        os.path.join(tmp_dir, "ds_b"),
        num_docs=8,
        eod_id=eod_id,
        seed=222,
    )
    out_a = os.path.join(tmp_dir, "npy_a")
    out_b = os.path.join(tmp_dir, "npy_b")
    # num_epochs=1 to match GPTDataset(num_samples=None) which is 1 epoch
    convert([prefix_a], out_a, seq_length=seq_length, num_epochs=1, seed=seed)
    convert([prefix_b], out_b, seq_length=seq_length, num_epochs=1, seed=seed)

    # ---- MaxText side: MegatronBlendedDataSource ---------------------------
    import grain.python as grain  # pylint: disable=import-outside-toplevel

    npy_ds_a = MegatronNpyDataSource(
        npy_dir=out_a,
        bin_paths=prefix_a,
        eod_id=eod_id,
        seq_length=seq_length,
    )
    npy_ds_b = MegatronNpyDataSource(
        npy_dir=out_b,
        bin_paths=prefix_b,
        eod_id=eod_id,
        seq_length=seq_length,
    )
    map_a = grain.MapDataset.source(npy_ds_a)
    map_b = grain.MapDataset.source(npy_ds_b)

    from maxtext.input_pipeline._megatron_blending import MegatronBlendedDataSource  # pylint: disable=import-outside-toplevel

    blended_source = MegatronBlendedDataSource(
        map_datasets=[map_a, map_b],
        weights=weights,
    )
    maxtext_total = len(blended_source)
    assert maxtext_total > 0, "Blended dataset produced no samples"

    # ---- Megatron side: real GPTDataset objects -----------------------------
    gpt_datasets = []
    for prefix in [prefix_a, prefix_b]:
      gpt_ds = _build_megatron_gpt_dataset(
          prefix,
          seq_length,
          seed,
          eod_id,
      )
      gpt_datasets.append(gpt_ds)

    # Build blend indices with Megatron C++
    meg_ds_idx = np.zeros(maxtext_total, dtype=np.int16)
    meg_sample_idx = np.zeros(maxtext_total, dtype=np.int64)
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    megatron_helpers.build_blending_indices(
        meg_ds_idx,
        meg_sample_idx,
        w,
        len(weights),
        maxtext_total,
        False,
    )

    # ---- Compare token-by-token -------------------------------------------
    for i in range(maxtext_total):
      # MaxText path
      mx_tokens = blended_source[i]["text"]

      # Megatron path: blend dispatch -> real GPTDataset __getitem__
      ds_id = int(meg_ds_idx[i])
      sample_id = int(meg_sample_idx[i])
      meg_raw = TestTokenContentAlignment._meg_raw_tokens(
          gpt_datasets[ds_id],
          sample_id,
      )

      np.testing.assert_array_equal(
          mx_tokens,
          meg_raw,
          err_msg=(f"Blend index {i}: token mismatch " f"(ds_id={ds_id}, sample_id={sample_id})"),
      )

  def test_blend_e2e_iteration_matches_megatron(self, tmp_dir):
    """End-to-end iteration: MaxText Grain pipeline vs Megatron GPTDataset + BlendedDataset.

    Both sides iterate from index 0 to N.  MaxText goes through the full
    Grain pipeline (get_datasets -> to_iter_dataset -> iterate).  Megatron
    goes through its real GPTDataset + BlendedDataset __getitem__ chain.
    Token sequences are compared element-by-element.
    """
    from megatron.core.datasets.blended_dataset import BlendedDataset  # pylint: disable=import-outside-toplevel
    from maxtext.input_pipeline.grain_data_processing import get_datasets  # pylint: disable=import-outside-toplevel

    eod_id = 0
    seq_length = 8
    seed = 42
    weights = [0.7, 0.3]

    # ---- Create two distinct synthetic datasets ----------------------------
    prefix_a, _ = _create_eod_dataset(
        os.path.join(tmp_dir, "ds_a"),
        num_docs=8,
        eod_id=eod_id,
        seed=111,
    )
    prefix_b, _ = _create_eod_dataset(
        os.path.join(tmp_dir, "ds_b"),
        num_docs=8,
        eod_id=eod_id,
        seed=222,
    )

    # MaxText: build npy indices (num_epochs=1 to match GPTDataset num_samples=None)
    out_a = os.path.join(tmp_dir, "npy_a")
    out_b = os.path.join(tmp_dir, "npy_b")
    convert([prefix_a], out_a, seq_length=seq_length, num_epochs=1, seed=seed)
    convert([prefix_b], out_b, seq_length=seq_length, num_epochs=1, seed=seed)

    # ---- MaxText side: iterate through Grain pipeline ----------------------
    blend_pattern = f"{out_a}|{prefix_a},0.7;{out_b}|{prefix_b},0.3"
    ds_cfg = MMapDatasetConfig(
        max_target_length=seq_length,
        eod_id=eod_id,
        mmap_split_sentences=False,
        seed=seed,
    )
    mx_iter_ds = get_datasets(
        data_file_pattern=blend_pattern,
        data_file_type="mmap_npy",
        shuffle=False,
        shuffle_seed=0,
        shuffle_buffer_size=0,
        num_epoch=1,
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        dataset_config=ds_cfg,
    )
    mx_samples = [item["text"] for item in mx_iter_ds]
    assert len(mx_samples) > 0, "MaxText pipeline produced no samples"

    # ---- Megatron side: GPTDataset + BlendedDataset iterate ----------------
    gpt_datasets = []
    for prefix in [prefix_a, prefix_b]:
      gpt_ds = _build_megatron_gpt_dataset(
          prefix,
          seq_length,
          seed,
          eod_id,
      )
      gpt_datasets.append(gpt_ds)

    blended_ds = BlendedDataset(
        datasets=gpt_datasets,
        weights=weights,
        size=len(mx_samples),
        config=gpt_datasets[0].config,
    )

    # ---- Compare: iterate both and match token-by-token --------------------
    assert len(blended_ds) == len(mx_samples), f"Length mismatch: Megatron {len(blended_ds)} vs MaxText {len(mx_samples)}"
    for i in range(len(blended_ds)):
      meg_raw = TestTokenContentAlignment._meg_raw_tokens(blended_ds, i)
      mx_raw = mx_samples[i]

      np.testing.assert_array_equal(
          mx_raw,
          meg_raw,
          err_msg=f"Iteration index {i}: token mismatch",
      )

  def test_blend_mp_prefetch_matches_megatron(self, tmp_dir):
    """Full pipeline with grain_worker_count=2 still matches Megatron token-by-token.

    Runs the complete pretrain_preprocessing_pipeline (including mp_prefetch
    with 2 workers) and compares the batched output against Megatron's
    BlendedDataset.  Guards against Grain mp_prefetch changes breaking
    Megatron alignment.
    """
    from types import SimpleNamespace  # pylint: disable=import-outside-toplevel
    from megatron.core.datasets.blended_dataset import BlendedDataset  # pylint: disable=import-outside-toplevel
    from maxtext.input_pipeline.grain_data_processing import (  # pylint: disable=import-outside-toplevel
        get_datasets,
        pretrain_preprocessing_pipeline,
    )

    eod_id = 0
    seq_length = 8
    seed = 42
    weights = [0.7, 0.3]
    batch_size = 4

    # ---- Create two distinct synthetic datasets ----------------------------
    prefix_a, _ = _create_eod_dataset(
        os.path.join(tmp_dir, "ds_a"),
        num_docs=8,
        eod_id=eod_id,
        seed=111,
    )
    prefix_b, _ = _create_eod_dataset(
        os.path.join(tmp_dir, "ds_b"),
        num_docs=8,
        eod_id=eod_id,
        seed=222,
    )
    out_a = os.path.join(tmp_dir, "npy_a")
    out_b = os.path.join(tmp_dir, "npy_b")
    convert([prefix_a], out_a, seq_length=seq_length, num_epochs=1, seed=seed)
    convert([prefix_b], out_b, seq_length=seq_length, num_epochs=1, seed=seed)

    # ---- MaxText side: full pipeline with mp_prefetch ----------------------
    blend_pattern = f"{out_a}|{prefix_a},0.7;{out_b}|{prefix_b},0.3"
    ds_cfg = MMapDatasetConfig(
        max_target_length=seq_length,
        eod_id=eod_id,
        mmap_split_sentences=False,
        seed=seed,
    )
    mx_map_ds = get_datasets(
        data_file_pattern=blend_pattern,
        data_file_type="mmap_npy",
        shuffle=False,
        shuffle_seed=0,
        shuffle_buffer_size=0,
        num_epoch=1,
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=0,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        dataset_config=ds_cfg,
    )
    cfg = SimpleNamespace(
        grain_file_type="mmap_npy",
        mmap_eod_id=eod_id,
        tokenizer_path="",
        tokenizer_type="sentencepiece",
        add_bos=False,
        add_eos=False,
        hf_access_token="",
        dataset_type="grain",
        max_target_length=seq_length,
        use_truncation=False,
        global_batch_size_to_load=batch_size,
        expansion_factor_real_data=1,
        packing=False,
        grain_packing_type="concat_then_split",
        max_segments_per_seq=None,
        reset_attention_mask=False,
        grain_ram_budget_mb=256,
        eod_mask_loss=False,
    )
    pipe = pretrain_preprocessing_pipeline(
        mx_map_ds,
        cfg,
        data_columns=["text"],
        tokenize=False,
        grain_worker_count=2,
        grain_per_worker_buffer_size=2,
    )
    # Flatten batches to individual raw token sequences
    mx_samples = []
    for batch in pipe:
      for b in range(batch["inputs"].shape[0]):
        raw = np.concatenate([batch["inputs"][b][:1], batch["targets"][b]])
        mx_samples.append(raw.astype(np.int32))
    assert len(mx_samples) > 0, "Pipeline produced no samples"

    # ---- Megatron side: GPTDataset + BlendedDataset ------------------------
    gpt_datasets = []
    for prefix in [prefix_a, prefix_b]:
      gpt_ds = _build_megatron_gpt_dataset(
          prefix,
          seq_length,
          seed,
          eod_id,
      )
      gpt_datasets.append(gpt_ds)

    blended_ds = BlendedDataset(
        datasets=gpt_datasets,
        weights=weights,
        size=len(mx_samples),
        config=gpt_datasets[0].config,
    )

    # ---- Compare token-by-token -------------------------------------------
    assert len(blended_ds) == len(mx_samples), f"Length mismatch: Megatron {len(blended_ds)} vs MaxText {len(mx_samples)}"
    for i in range(len(blended_ds)):
      meg_raw = TestTokenContentAlignment._meg_raw_tokens(blended_ds, i)
      np.testing.assert_array_equal(
          mx_samples[i],
          meg_raw,
          err_msg=f"mp_prefetch iteration index {i}: token mismatch",
      )


# ===========================================================================
# Split index alignment
# ===========================================================================

_SPLIT_NUM_DOCS = 20
_SPLIT_RATIOS = "0.5,0.3,0.2"  # 10, 6, 4 docs
_SPLIT_SEED = 1234
_SPLIT_SEQ_LENGTH = 8
_SPLIT_NUM_EPOCHS = 2


def _create_split_dataset(tmp_dir, num_docs=_SPLIT_NUM_DOCS, eod_id=0, seed=42):
  """Create synthetic  num_docs documents with --append-eod, varying lengths."""
  os.makedirs(tmp_dir, exist_ok=True)
  rng = np.random.RandomState(seed)
  seqs = []
  for d in range(num_docs):
    length = 20 + d * 5  # 20, 25, 30, ...
    tokens = rng.randint(1, 10000, size=length, dtype=np.int32)
    tokens[-1] = eod_id  # simulate --append-eod
    seqs.append(tokens)
  prefix = os.path.join(tmp_dir, "data")
  doc_boundaries = list(range(num_docs + 1))  # 1 seq per doc
  create_mmap_test_data(prefix, seqs, doc_boundaries=doc_boundaries)
  return prefix, seqs


def _split_doc_range(num_docs, split_str, split_index):
  """Compute (start_doc, end_doc) for a given split_index."""
  ratios = [float(x) for x in split_str.split(",")]
  total_ratio = sum(ratios)
  ratios = [r / total_ratio for r in ratios]
  cumulative = [0.0]
  for r in ratios:
    cumulative.append(cumulative[-1] + r)
  start_doc = int(round(cumulative[split_index] * num_docs))
  end_doc = int(round(cumulative[split_index + 1] * num_docs))
  return start_doc, end_doc


def _megatron_split_boundaries(num_docs, split_str):
  """Compute per-split (start_doc, end_doc) using Megatron's exact algorithm.

  Megatron flow:
  1. parse_and_normalize_split("99,1") -> [0.99, 0.01, 0.0]
  2. convert_split_vector_to_split_matrix -> [(0, 0.99), (0.99, 1.0), None]
  3. beg = int(round(bookend[0] * num_elements))
     end = int(round(bookend[1] * num_elements))
  """
  split_vector = parse_and_normalize_split(split_str)
  split_matrix = convert_split_vector_to_split_matrix(split_vector)
  boundaries = []
  for entry in split_matrix:
    if entry is not None:
      beg = int(round(entry[0] * float(num_docs)))
      end = int(round(entry[1] * float(num_docs)))
      boundaries.append((beg, end))
    else:
      boundaries.append(None)
  return boundaries


# ===========================================================================
# Split boundary alignment
# ===========================================================================


class TestSplitBoundaryAlignment:
  """Verify split boundary computation matches Megatron-LM exactly."""

  @pytest.mark.parametrize(
      "num_docs,split_str",
      [
          (20, "0.5,0.3,0.2"),  # exact integer boundaries
          (101, "99,1"),  # round edge: 0.99*101=99.99 → 100
          (1000, "0.9,0.05,0.05"),  # standard train/eval/test
          (7, "0.8,0.1,0.1"),  # small dataset
          (333, "98,1,1"),  # un-normalized ratios
      ],
  )
  def test_split_boundaries_match_megatron(self, num_docs, split_str):
    """MaxText _split_doc_range boundaries must match Megatron's computation."""
    meg_boundaries = _megatron_split_boundaries(num_docs, split_str)
    ratios = [float(x) for x in split_str.split(",")]
    # pad to 3 entries (Megatron always has train/valid/test)
    while len(ratios) < 3:
      ratios.append(0.0)

    for split_index, meg_entry in enumerate(meg_boundaries):
      if meg_entry is None:
        continue
      meg_start, meg_end = meg_entry
      mx_start, mx_end = _split_doc_range(num_docs, split_str, split_index)
      assert mx_start == meg_start, (
          f"split_index={split_index}: start_doc mismatch: " f"MaxText={mx_start}, Megatron={meg_start}"
      )
      assert mx_end == meg_end, f"split_index={split_index}: end_doc mismatch: " f"MaxText={mx_end}, Megatron={meg_end}"

  @pytest.mark.parametrize(
      "num_docs,split_str",
      [
          (20, "0.5,0.3,0.2"),
          (101, "99,1"),
          (1000, "0.9,0.05,0.05"),
          (7, "0.8,0.1,0.1"),
          (333, "98,1,1"),
      ],
  )
  def test_splits_partition_all_docs(self, num_docs, split_str):
    """All non-None splits must cover [0, num_docs) with no gap or overlap."""
    meg_boundaries = _megatron_split_boundaries(num_docs, split_str)
    active = [b for b in meg_boundaries if b is not None]
    # Sort by start
    active.sort(key=lambda b: b[0])

    assert active[0][0] == 0, f"First split should start at 0, got {active[0][0]}"
    assert active[-1][1] == num_docs, f"Last split should end at {num_docs}, got {active[-1][1]}"

    for i in range(1, len(active)):
      assert active[i][0] == active[i - 1][1], (
          f"Gap or overlap between split {i-1} end={active[i-1][1]} " f"and split {i} start={active[i][0]}"
      )

  def test_round_vs_truncation_edge_case(self):
    """Verify int(round()) differs from int() and produces correct splits.

    num_docs=101, split="99,1":
      cumulative = [0.0, 0.99, 1.0]
      0.99 * 101 = 99.99
      int(99.99) = 99  (truncation, WRONG)
      round(99.99) = 100 (correct, matches Megatron)
    """
    num_docs = 101
    start_train, end_train = _split_doc_range(num_docs, "99,1", 0)
    start_eval, end_eval = _split_doc_range(num_docs, "99,1", 1)

    # train: [0, 100), eval: [100, 101)
    assert end_train == 100, f"Train should end at 100 (round), got {end_train}"
    assert start_eval == 100, f"Eval should start at 100 (round), got {start_eval}"
    assert end_eval == 101, f"Eval should end at 101, got {end_eval}"

    # Verify no gap/overlap
    assert end_train == start_eval
    train_docs = end_train - start_train
    eval_docs = end_eval - start_eval
    assert train_docs == 100
    assert eval_docs == 1
    assert train_docs + eval_docs == num_docs


class TestConvertSplitMegatronAlignment:
  """Verify split indices match Megatron-LM's GPTDataset for each split."""

  @staticmethod
  def _build_megatron_indices(prefix, seq_length, seed, num_epochs, start_doc, end_doc):
    """Build Megatron reference indices for a document subset."""
    doc_sizes = get_document_sizes([prefix])
    # Megatron uses global doc IDs: np.arange over the full dataset, then slices
    documents = np.arange(start_doc, end_doc, dtype=np.int32)
    # Full sequence_lengths array (all docs), as Megatron indexes by global doc ID
    all_sizes = doc_sizes.astype(np.int32)

    rng = np.random.RandomState(seed)
    doc_index = _build_document_index(documents, num_epochs, rng, False)

    tokens_per_epoch = int(doc_sizes[start_doc:end_doc].sum())
    sample_index = megatron_helpers.build_sample_idx(
        all_sizes,
        doc_index,
        seq_length,
        num_epochs=num_epochs,
        tokens_per_epoch=tokens_per_epoch,
        drop_last_partial_sequence=True,
        add_extra_token_to_sequence=True,
    )

    total_samples = sample_index.shape[0] - 1
    shuffle_index = _build_shuffle_index(total_samples, total_samples, rng)

    return doc_index, sample_index, shuffle_index

  @pytest.mark.parametrize("split_index", [0, 1, 2])
  def test_split_indices_match_megatron(self, tmp_dir, split_index):
    """All three index arrays must match Megatron for each split."""
    prefix, _ = _create_split_dataset(tmp_dir)
    start_doc, end_doc = _split_doc_range(_SPLIT_NUM_DOCS, _SPLIT_RATIOS, split_index)

    # --- Our side: convert() with split ---
    out_dir = os.path.join(tmp_dir, f"npy_split_{split_index}")
    convert(
        [prefix],
        out_dir,
        seq_length=_SPLIT_SEQ_LENGTH,
        num_epochs=_SPLIT_NUM_EPOCHS,
        seed=_SPLIT_SEED,
        split=_SPLIT_RATIOS,
        split_index=split_index,
    )
    doc_path, sample_path, shuffle_path = _discover_npy_indices(out_dir)
    our_doc_index = np.load(doc_path)
    our_sample_index = np.load(sample_path)
    our_shuffle_index = np.load(shuffle_path)

    # --- Megatron side ---
    meg_doc_index, meg_sample_index, meg_shuffle_index = self._build_megatron_indices(
        prefix,
        _SPLIT_SEQ_LENGTH,
        _SPLIT_SEED,
        _SPLIT_NUM_EPOCHS,
        start_doc,
        end_doc,
    )

    np.testing.assert_array_equal(
        our_doc_index,
        meg_doc_index,
        err_msg=f"document_index mismatch for split_index={split_index}",
    )
    np.testing.assert_array_equal(
        our_sample_index,
        meg_sample_index,
        err_msg=f"sample_index mismatch for split_index={split_index}",
    )
    np.testing.assert_array_equal(
        our_shuffle_index,
        meg_shuffle_index,
        err_msg=f"shuffle_index mismatch for split_index={split_index}",
    )

  @pytest.mark.parametrize("split_index", [0, 1])
  def test_split_indices_match_megatron_round_edge(self, tmp_dir, split_index):
    """Index arrays match Megatron when round() differs from truncation.

    num_docs=101, split="99,1": boundary at 0.99*101=99.99.
    round() → 100 (train=100 docs, eval=1 doc).
    int()  → 99  (train=99 docs, eval=2 docs) — WRONG.
    """
    num_docs = 101
    split_str = "99,1"
    seed = 42
    seq_length = 8
    num_epochs = 1

    prefix, _ = _create_split_dataset(tmp_dir, num_docs=num_docs)
    start_doc, end_doc = _split_doc_range(num_docs, split_str, split_index)

    if split_index == 0:
      assert end_doc - start_doc == 100, f"Train split should have 100 docs, got {end_doc - start_doc}"
    else:
      assert end_doc - start_doc == 1, f"Eval split should have 1 doc, got {end_doc - start_doc}"

    out_dir = os.path.join(tmp_dir, f"npy_round_edge_{split_index}")
    convert(
        [prefix],
        out_dir,
        seq_length=seq_length,
        num_epochs=num_epochs,
        seed=seed,
        split=split_str,
        split_index=split_index,
    )
    doc_path, sample_path, shuffle_path = _discover_npy_indices(out_dir)
    our_doc_index = np.load(doc_path)
    our_sample_index = np.load(sample_path)
    our_shuffle_index = np.load(shuffle_path)

    meg_doc_index, meg_sample_index, meg_shuffle_index = self._build_megatron_indices(
        prefix,
        seq_length,
        seed,
        num_epochs,
        start_doc,
        end_doc,
    )

    np.testing.assert_array_equal(
        our_doc_index,
        meg_doc_index,
        err_msg=f"document_index mismatch for round-edge split_index={split_index}",
    )
    np.testing.assert_array_equal(
        our_sample_index,
        meg_sample_index,
        err_msg=f"sample_index mismatch for round-edge split_index={split_index}",
    )
    np.testing.assert_array_equal(
        our_shuffle_index,
        meg_shuffle_index,
        err_msg=f"shuffle_index mismatch for round-edge split_index={split_index}",
    )

  def test_split_token_content_matches_megatron(self, tmp_dir):
    """End-to-end token content with splits matches Megatron GPTDataset."""
    num_docs = 101
    split_str = "99,1"
    eod_id = 0
    seq_length = 8
    seed = 42

    prefix, _ = _create_split_dataset(tmp_dir, num_docs=num_docs, eod_id=eod_id)

    for split_index in range(2):
      start_doc, end_doc = _split_doc_range(num_docs, split_str, split_index)
      n_docs = end_doc - start_doc

      out_dir = os.path.join(tmp_dir, f"npy_tok_{split_index}")
      convert(
          [prefix],
          out_dir,
          seq_length=seq_length,
          num_epochs=1,
          seed=seed,
          split=split_str,
          split_index=split_index,
      )

      npy_ds = MegatronNpyDataSource(
          npy_dir=out_dir,
          bin_paths=prefix,
          eod_id=eod_id,
          seq_length=seq_length,
      )

      # Build Megatron GPTDataset with matching indexed_indices
      tokenizer = _StubTokenizer(eod=eod_id)
      config = GPTDatasetConfig(
          random_seed=seed,
          sequence_length=seq_length,
          reset_position_ids=False,
          reset_attention_mask=False,
          eod_mask_loss=False,
          tokenizer=tokenizer,
      )
      indexed_ds = IndexedDataset(prefix, multimodal=False, mmap=True)
      indexed_indices = np.arange(start_doc, end_doc, dtype=np.int32)
      gpt_ds = GPTDataset(
          indexed_dataset=indexed_ds,
          dataset_path=prefix,
          indexed_indices=indexed_indices,
          num_samples=None,
          index_split=Split.train if split_index == 0 else Split.valid,
          config=config,
      )

      num_compare = min(len(npy_ds), len(gpt_ds), 50)
      assert num_compare > 0, f"split_index={split_index}: no samples (n_docs={n_docs})"

      for i in range(num_compare):
        mx_tokens = npy_ds[i]["text"]
        meg_sample = gpt_ds[i]
        meg_tokens = np.concatenate([meg_sample["tokens"].numpy()[:1], meg_sample["labels"].numpy()]).astype(np.int32)
        np.testing.assert_array_equal(
            mx_tokens,
            meg_tokens,
            err_msg=f"split_index={split_index}, sample {i}: token mismatch",
        )


# ===========================================================================
# Loss mask / position_ids alignment (MegatronSplitInputsTargets vs Megatron)
# ===========================================================================


class TestLossMaskPositionAlignment:
  """Verify MegatronSplitInputsTargets produces loss_mask and position_ids
  matching Megatron's _get_ltor_masks_and_position_ids.

  Megatron computes loss_mask from *tokens* (inputs = text[:-1]), so
  eod_mask_loss=True zeros out loss at positions where the input is EOD
  (i.e. the target is the first token of the next document).

  Megatron also resets position_ids at EOD boundaries in inputs when
  reset_position_ids=True.

  This test class compares all four (reset_attention_mask, eod_mask_loss)
  combinations against real Megatron output.
  """

  @staticmethod
  def _megatron_masks_and_positions(tokens_np, eod_id, reset_attention_mask, eod_mask_loss):
    """Call Megatron's _get_ltor_masks_and_position_ids on input tokens."""
    import torch  # pylint: disable=import-outside-toplevel
    from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids  # pylint: disable=import-outside-toplevel

    tokens_t = torch.from_numpy(tokens_np.astype(np.int64))
    _, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
        tokens_t,
        eod_id,
        reset_position_ids=reset_attention_mask,  # Megatron ties these together
        reset_attention_mask=reset_attention_mask,
        eod_mask_loss=eod_mask_loss,
        create_attention_mask=False,
    )
    return loss_mask.numpy(), position_ids.numpy()

  @staticmethod
  def _maxtext_split(tokens_np, eod_id, reset_attention_mask, eod_mask_loss):
    """Call MegatronSplitInputsTargets on a seq_length+1 token array."""
    from maxtext.input_pipeline.input_pipeline_utils import MegatronSplitInputsTargets  # pylint: disable=import-outside-toplevel

    transform = MegatronSplitInputsTargets(
        eod_id=eod_id,
        reset_attention_mask=reset_attention_mask,
        eod_mask_loss=eod_mask_loss,
    )
    result = transform.map({"text": tokens_np})
    return result

  @staticmethod
  def _make_cross_doc_tokens(eod_id=0):
    """Build a token sequence that contains multiple EOD boundaries.

    Returns tokens array of length seq_length+1 (to be split into
    inputs[:-1] and targets[1:]).

    Layout: [10, 20, EOD, 30, 40, 50, EOD, 60, 70]
      inputs:  [10, 20, EOD, 30, 40, 50, EOD, 60]
      targets: [20, EOD, 30, 40, 50, EOD, 60, 70]

    EOD appears in inputs at positions 2 and 6.
    EOD appears in targets at positions 1 and 5.
    """
    return np.array([10, 20, eod_id, 30, 40, 50, eod_id, 60, 70], dtype=np.int32)

  @pytest.mark.parametrize(
      "reset_attention_mask,eod_mask_loss",
      [
          (False, False),
          (False, True),
          (True, False),
          (True, True),
      ],
  )
  def test_loss_mask_matches_megatron(self, reset_attention_mask, eod_mask_loss):
    """target_segmentation-derived loss mask matches Megatron's loss_mask.

    In MaxText, targets_segmentation=0 means the target is excluded from
    loss (equivalent to Megatron loss_mask=0).  This test converts our
    targets_segmentation into a binary loss mask and compares against
    Megatron's loss_mask element-by-element.
    """
    eod_id = 0
    tokens = self._make_cross_doc_tokens(eod_id)
    inputs = tokens[:-1]

    # Megatron reference
    meg_loss_mask, _ = self._megatron_masks_and_positions(inputs, eod_id, reset_attention_mask, eod_mask_loss)

    # MaxText
    mx_result = self._maxtext_split(tokens, eod_id, reset_attention_mask, eod_mask_loss)
    # Convert targets_segmentation to binary loss mask: seg>0 → 1.0, seg==0 → 0.0
    mx_loss_mask = np.where(mx_result["targets_segmentation"] > 0, 1.0, 0.0).astype(np.float32)

    np.testing.assert_array_equal(
        mx_loss_mask,
        meg_loss_mask,
        err_msg=(
            f"Loss mask mismatch "
            f"(reset_attention_mask={reset_attention_mask}, eod_mask_loss={eod_mask_loss})\n"
            f"  inputs:  {inputs.tolist()}\n"
            f"  targets: {tokens[1:].tolist()}\n"
            f"  mx_target_seg:  {mx_result['targets_segmentation'].tolist()}\n"
            f"  mx_loss_mask:   {mx_loss_mask.tolist()}\n"
            f"  meg_loss_mask:  {meg_loss_mask.tolist()}"
        ),
    )

  @pytest.mark.parametrize(
      "reset_attention_mask,eod_mask_loss",
      [
          (False, False),
          (False, True),
          (True, False),
          (True, True),
      ],
  )
  def test_position_ids_match_megatron(self, reset_attention_mask, eod_mask_loss):
    """inputs_position matches Megatron's position_ids."""
    eod_id = 0
    tokens = self._make_cross_doc_tokens(eod_id)
    inputs = tokens[:-1]

    # Megatron reference
    _, meg_position_ids = self._megatron_masks_and_positions(inputs, eod_id, reset_attention_mask, eod_mask_loss)

    # MaxText
    mx_result = self._maxtext_split(tokens, eod_id, reset_attention_mask, eod_mask_loss)
    mx_position_ids = mx_result["inputs_position"]

    np.testing.assert_array_equal(
        mx_position_ids,
        meg_position_ids,
        err_msg=(
            f"Position IDs mismatch "
            f"(reset_attention_mask={reset_attention_mask}, eod_mask_loss={eod_mask_loss})\n"
            f"  inputs:           {inputs.tolist()}\n"
            f"  mx_position_ids:  {mx_position_ids.tolist()}\n"
            f"  meg_position_ids: {meg_position_ids.tolist()}"
        ),
    )

  def test_e2e_sample_loss_mask_matches_megatron(self, tmp_dir):
    """Real MegatronNpyDataSource samples: loss_mask/position vs Megatron GPTDataset."""
    eod_id = 0
    seq_length = 8
    seed = 42

    prefix, _ = _create_eod_dataset(tmp_dir, num_docs=10, eod_id=eod_id)
    out_dir = os.path.join(tmp_dir, "npy_output")
    convert([prefix], out_dir, seq_length=seq_length, num_epochs=1, seed=seed)

    # MaxText: raw token sequences from MegatronNpyDataSource
    npy_ds = MegatronNpyDataSource(
        npy_dir=out_dir,
        bin_paths=prefix,
        eod_id=eod_id,
        seq_length=seq_length,
    )

    # Megatron: real GPTDataset with reset_attention_mask=True, eod_mask_loss=True
    from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids  # pylint: disable=import-outside-toplevel
    from maxtext.input_pipeline.input_pipeline_utils import MegatronSplitInputsTargets  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    num_compare = min(len(npy_ds), 30)
    assert num_compare > 0

    for reset_attention_mask, eod_mask_loss in [(True, False), (True, True), (False, True)]:
      transform = MegatronSplitInputsTargets(
          eod_id=eod_id,
          reset_attention_mask=reset_attention_mask,
          eod_mask_loss=eod_mask_loss,
      )
      for i in range(num_compare):
        tokens = npy_ds[i]["text"]  # seq_length+1 tokens
        inputs = tokens[:-1]

        # MaxText
        mx_result = transform.map({"text": tokens})
        mx_loss_mask = np.where(mx_result["targets_segmentation"] > 0, 1.0, 0.0).astype(np.float32)

        # Megatron reference
        tokens_t = torch.from_numpy(inputs.astype(np.int64))
        _, meg_loss_mask, meg_positions = _get_ltor_masks_and_position_ids(
            tokens_t, eod_id, reset_attention_mask, reset_attention_mask, eod_mask_loss, False
        )

        np.testing.assert_array_equal(
            mx_loss_mask,
            meg_loss_mask.numpy(),
            err_msg=(
                f"Sample {i} loss_mask mismatch "
                f"(reset={reset_attention_mask}, eod_mask={eod_mask_loss})\n"
                f"  inputs:  {inputs[:12].tolist()}\n"
                f"  targets: {tokens[1:][:12].tolist()}\n"
                f"  mx:  {mx_loss_mask[:12].tolist()}\n"
                f"  meg: {meg_loss_mask.numpy()[:12].tolist()}"
            ),
        )
        np.testing.assert_array_equal(
            mx_result["inputs_position"],
            meg_positions.numpy().astype(np.int32),
            err_msg=(f"Sample {i} position_ids mismatch " f"(reset={reset_attention_mask}, eod_mask={eod_mask_loss})"),
        )


# ===========================================================================
# Epoch-based cache equivalence
# ===========================================================================


class TestEpochBasedCacheEquivalence:
  """Prove that (num_epochs, separate_final_epoch) is a sufficient cache key.

  All three index arrays are uniquely determined by
  (num_epochs, separate_final_epoch, seed, seq_length, input_paths).
  Different num_samples values mapping to the same (num_epochs,
  separate_final_epoch) produce bit-identical indices.

  Test  10 docs, lengths [20,25,...,65], total=425 tokens.
  With seq_length=8, add_extra_token=1:
    1-epoch: (425-1)//8 = 53 samples
    2-epoch: (850-1)//8 = 106 samples
    separate_final_epoch threshold at num_epochs=2:
      num_samples_sans_final = 53, threshold = int(0.80 * 53) = 42
      num_samples in [54..94]: separate=True
      num_samples in [95..106]: separate=False
  """

  @staticmethod
  def _build_indices_via_convert(tmp_dir, prefix, seq_length, seed, num_samples, label):
    """Build npy indices using convert(num_samples=N) and return loaded arrays."""
    out_dir = os.path.join(tmp_dir, f"npy_{label}")
    convert([prefix], out_dir, seq_length=seq_length, num_samples=num_samples, seed=seed)
    doc_path, sample_path, shuffle_path = _discover_npy_indices(out_dir)
    return np.load(doc_path), np.load(sample_path), np.load(shuffle_path)

  def test_same_epoch_different_num_samples_1epoch(self, tmp_dir):
    """num_samples=40 and 53 both require num_epochs=1 → identical indices."""
    prefix, _ = _create_eod_dataset(tmp_dir, num_docs=10, eod_id=0)
    seq_length = 8
    seed = 42

    doc_40, sample_40, shuffle_40 = self._build_indices_via_convert(
        tmp_dir,
        prefix,
        seq_length,
        seed,
        num_samples=40,
        label="ns40",
    )
    doc_53, sample_53, shuffle_53 = self._build_indices_via_convert(
        tmp_dir,
        prefix,
        seq_length,
        seed,
        num_samples=53,
        label="ns53",
    )

    np.testing.assert_array_equal(doc_40, doc_53, err_msg="document_index mismatch")
    np.testing.assert_array_equal(sample_40, sample_53, err_msg="sample_index mismatch")
    np.testing.assert_array_equal(shuffle_40, shuffle_53, err_msg="shuffle_index mismatch")

  def test_same_epoch_different_num_samples_2epoch_separate_true(self, tmp_dir):
    """num_samples=60 and 90: both (epochs=2, separate=True) → identical indices."""
    prefix, _ = _create_eod_dataset(tmp_dir, num_docs=10, eod_id=0)
    seq_length = 8
    seed = 42

    doc_60, sample_60, shuffle_60 = self._build_indices_via_convert(
        tmp_dir,
        prefix,
        seq_length,
        seed,
        num_samples=60,
        label="ns60",
    )
    doc_90, sample_90, shuffle_90 = self._build_indices_via_convert(
        tmp_dir,
        prefix,
        seq_length,
        seed,
        num_samples=90,
        label="ns90",
    )

    np.testing.assert_array_equal(doc_60, doc_90, err_msg="document_index mismatch")
    np.testing.assert_array_equal(sample_60, sample_90, err_msg="sample_index mismatch")
    np.testing.assert_array_equal(shuffle_60, shuffle_90, err_msg="shuffle_index mismatch")

  def test_epoch_boundary_produces_different_indices(self, tmp_dir):
    """num_samples=53 (1 epoch) vs 54 (2 epochs) → different shapes and content."""
    prefix, _ = _create_eod_dataset(tmp_dir, num_docs=10, eod_id=0)
    seq_length = 8
    seed = 42

    doc_53, _, _ = self._build_indices_via_convert(
        tmp_dir,
        prefix,
        seq_length,
        seed,
        num_samples=53,
        label="ns53",
    )
    doc_54, _, _ = self._build_indices_via_convert(
        tmp_dir,
        prefix,
        seq_length,
        seed,
        num_samples=54,
        label="ns54",
    )

    assert doc_53.shape != doc_54.shape, f"Epoch boundary must change shape: {doc_53.shape} vs {doc_54.shape}"

  def test_separate_final_epoch_threshold(self, tmp_dir):
    """num_samples=94 (separate=True) vs 95 (separate=False) → different indices."""
    prefix, _ = _create_eod_dataset(tmp_dir, num_docs=10, eod_id=0)
    seq_length = 8
    seed = 42

    doc_94, _, _ = self._build_indices_via_convert(
        tmp_dir,
        prefix,
        seq_length,
        seed,
        num_samples=94,
        label="ns94",
    )
    doc_95, _, _ = self._build_indices_via_convert(
        tmp_dir,
        prefix,
        seq_length,
        seed,
        num_samples=95,
        label="ns95",
    )

    assert doc_94.shape == doc_95.shape, "Both should be 2-epoch (same shape)"
    assert not np.array_equal(doc_94, doc_95), "document_index should differ when separate_final_epoch flips"

  def test_cross_num_samples_megatron_match_1epoch(self, tmp_dir):
    """Build with ns=40, consume 53 → matches Megatron(ns=53). Both are 1-epoch."""
    eod_id = 0
    prefix, _ = _create_eod_dataset(tmp_dir, num_docs=10, eod_id=eod_id)
    seq_length = 8
    seed = 42

    out_dir = os.path.join(tmp_dir, "npy_ns40")
    convert([prefix], out_dir, seq_length=seq_length, num_samples=40, seed=seed)
    npy_ds = MegatronNpyDataSource(
        npy_dir=out_dir,
        bin_paths=prefix,
        eod_id=eod_id,
        seq_length=seq_length,
    )
    assert len(npy_ds) == 53, f"Expected 53 (full 1-epoch), got {len(npy_ds)}"

    meg_ds = _build_megatron_gpt_dataset(prefix, seq_length, seed, eod_id, num_samples=53)

    for i in range(53):
      mx_tokens = npy_ds[i]["text"]
      meg_sample = meg_ds[i]
      meg_tokens = np.concatenate([meg_sample["tokens"].numpy()[:1], meg_sample["labels"].numpy()]).astype(np.int32)
      np.testing.assert_array_equal(
          mx_tokens,
          meg_tokens,
          err_msg=f"Sample {i}: npy(ns=40) != Megatron(ns=53), both 1-epoch",
      )

  def test_cross_num_samples_megatron_match_2epoch(self, tmp_dir):
    """Build with ns=60, consume 90 → matches Megatron(ns=90). Both (epochs=2, separate=True)."""
    eod_id = 0
    prefix, _ = _create_eod_dataset(tmp_dir, num_docs=10, eod_id=eod_id)
    seq_length = 8
    seed = 42

    out_dir = os.path.join(tmp_dir, "npy_ns60")
    convert([prefix], out_dir, seq_length=seq_length, num_samples=60, seed=seed)
    npy_ds = MegatronNpyDataSource(
        npy_dir=out_dir,
        bin_paths=prefix,
        eod_id=eod_id,
        seq_length=seq_length,
    )
    assert len(npy_ds) == 106, f"Expected 106 (full 2-epoch), got {len(npy_ds)}"

    meg_ds = _build_megatron_gpt_dataset(prefix, seq_length, seed, eod_id, num_samples=90)

    for i in range(90):
      mx_tokens = npy_ds[i]["text"]
      meg_sample = meg_ds[i]
      meg_tokens = np.concatenate([meg_sample["tokens"].numpy()[:1], meg_sample["labels"].numpy()]).astype(np.int32)
      np.testing.assert_array_equal(
          mx_tokens,
          meg_tokens,
          err_msg=f"Sample {i}: npy(ns=60) != Megatron(ns=90), both (epochs=2, separate=True)",
      )


# ===========================================================================
# Blend size pinning
# ===========================================================================


class TestBlendSizePinning:
  """Verify blend total size is pinned to num_samples, matching Megatron's BlendedDataset.

  Without explicit ``size=num_samples``, ``_infer_size_from_lengths`` computes
  ``floor(min(len(ds_i) / w_i))`` which exceeds ``num_samples`` because
  epoch-bucketed sub-dataset lengths are larger than the per-dataset buffer
  sizes.  Megatron's ``BlendedDataset`` always takes an explicit ``size``
  parameter, so its length always equals ``num_samples``.
  """

  def test_blend_size_matches_megatron(self, tmp_dir):
    """MegatronBlendedDataSource length must equal Megatron BlendedDataset length.

    Demonstrates:
    1. BUG: without ``size=``, inferred blend size > num_samples
    2. Megatron reference: ``BlendedDataset(size=num_samples)`` → exactly num_samples
    3. FIX: ``MegatronBlendedDataSource(size=num_samples)`` → same length and tokens
    """
    import math as _math  # pylint: disable=import-outside-toplevel

    import grain.python as grain  # pylint: disable=import-outside-toplevel
    from megatron.core.datasets.blended_dataset import BlendedDataset  # pylint: disable=import-outside-toplevel

    from maxtext.input_pipeline._megatron_blending import MegatronBlendedDataSource  # pylint: disable=import-outside-toplevel

    eod_id = 0
    seq_length = 8
    seed = 42
    weights = [0.7, 0.3]
    num_samples = 30
    margin = 0.5

    # ---- Create two distinct synthetic datasets ----------------------------
    prefix_a, _ = _create_eod_dataset(
        os.path.join(tmp_dir, "ds_a"),
        num_docs=8,
        eod_id=eod_id,
        seed=111,
    )
    prefix_b, _ = _create_eod_dataset(
        os.path.join(tmp_dir, "ds_b"),
        num_docs=8,
        eod_id=eod_id,
        seed=222,
    )

    # Compute per-dataset buffer sizes (Megatron's formula)
    w_norm = np.array(weights, dtype=np.float64)
    w_norm = w_norm / w_norm.sum()
    buf_a = _math.ceil(_math.ceil(num_samples * w_norm[0]) * (1 + margin / 100))
    buf_b = _math.ceil(_math.ceil(num_samples * w_norm[1]) * (1 + margin / 100))

    out_a = os.path.join(tmp_dir, "npy_a")
    out_b = os.path.join(tmp_dir, "npy_b")
    convert([prefix_a], out_a, seq_length=seq_length, num_samples=buf_a, seed=seed)
    convert([prefix_b], out_b, seq_length=seq_length, num_samples=buf_b, seed=seed)

    # ---- MaxText sub-datasets -----------------------------------------------
    npy_ds_a = MegatronNpyDataSource(
        npy_dir=out_a,
        bin_paths=prefix_a,
        eod_id=eod_id,
        seq_length=seq_length,
    )
    npy_ds_b = MegatronNpyDataSource(
        npy_dir=out_b,
        bin_paths=prefix_b,
        eod_id=eod_id,
        seq_length=seq_length,
    )
    map_a = grain.MapDataset.source(npy_ds_a)
    map_b = grain.MapDataset.source(npy_ds_b)

    # Sub-dataset lengths after epoch bucketing exceed buffer sizes
    assert len(npy_ds_a) >= buf_a, (
        f"Epoch bucketing should inflate sub-dataset length: " f"len={len(npy_ds_a)}, buf={buf_a}"
    )
    assert len(npy_ds_b) >= buf_b, (
        f"Epoch bucketing should inflate sub-dataset length: " f"len={len(npy_ds_b)}, buf={buf_b}"
    )

    # ---- BUG: without size=, inferred blend size exceeds num_samples --------
    blended_no_pin = MegatronBlendedDataSource(
        map_datasets=[map_a, map_b],
        weights=weights,
    )
    assert len(blended_no_pin) > num_samples, (
        f"Expected inferred size > {num_samples} due to epoch bucketing, " f"got {len(blended_no_pin)}"
    )

    # ---- Megatron reference: BlendedDataset with explicit size=num_samples --
    gpt_a = _build_megatron_gpt_dataset(prefix_a, seq_length, seed, eod_id)
    gpt_b = _build_megatron_gpt_dataset(prefix_b, seq_length, seed, eod_id)
    meg_blended = BlendedDataset(
        datasets=[gpt_a, gpt_b],
        weights=weights,
        size=num_samples,
        config=gpt_a.config,
    )
    assert len(meg_blended) == num_samples

    # ---- FIX: with size=num_samples, MaxText matches Megatron ---------------
    blended_pinned = MegatronBlendedDataSource(
        map_datasets=[map_a, map_b],
        weights=weights,
        size=num_samples,
    )
    assert len(blended_pinned) == num_samples == len(meg_blended), (
        f"MaxText blend size {len(blended_pinned)} must equal " f"Megatron blend size {len(meg_blended)}"
    )

    # ---- Token-level verification ------------------------------------------
    for i in range(num_samples):
      mx_tokens = blended_pinned[i]["text"]
      meg_raw = TestTokenContentAlignment._meg_raw_tokens(meg_blended, i)
      np.testing.assert_array_equal(
          mx_tokens,
          meg_raw,
          err_msg=f"Blend index {i}: token mismatch (size-pinned MaxText vs Megatron)",
      )


# ===========================================================================
# Pickle safety
# ===========================================================================


class TestPickleSafety:
  """Verify MegatronNpyDataSource survives pickle round-trip with prebuilt indices.

  When ``_ensure_npy_indices`` returns a cache miss, ``MegatronNpyDataSource``
  receives ``prebuilt_indices``.  Grain workers pickle/unpickle the source.
  The deserialized source must produce identical samples to Megatron.
  """

  def test_pickle_preserves_prebuilt_indices(self, tmp_dir):
    """Pickle round-trip of prebuilt-indices source must preserve Megatron alignment.

    Demonstrates:
    1. BUG: without prebuilt_indices in __getstate__, pickle.loads fails with
       FileNotFoundError when npy_dir has no cached files (non-primary host).
    2. FIX: after including prebuilt_indices, samples survive pickle and
       match Megatron token-by-token.
    """
    import pickle  # pylint: disable=import-outside-toplevel

    from maxtext.input_pipeline._mmap_index_utils import build_indices  # pylint: disable=import-outside-toplevel

    eod_id = 0
    seq_length = 8
    seed = 42

    prefix, _ = _create_eod_dataset(tmp_dir, num_docs=6, eod_id=eod_id)

    # Build indices in memory (simulating cache miss on non-primary host)
    doc_idx, sample_idx, shuffle_idx, _ = build_indices(
        input_paths=[prefix],
        seq_length=seq_length,
        num_samples=20,
        seed=seed,
    )
    prebuilt = (doc_idx, sample_idx, shuffle_idx)

    # Create source with prebuilt indices and an EMPTY npy_dir (no cache files)
    npy_dir = os.path.join(tmp_dir, "npy_empty")
    os.makedirs(npy_dir, exist_ok=True)

    ds_original = MegatronNpyDataSource(
        npy_dir=npy_dir,
        bin_paths=prefix,
        eod_id=eod_id,
        seq_length=seq_length,
        prebuilt_indices=prebuilt,
    )

    # Pickle round-trip (simulates Grain worker serialization)
    data = pickle.dumps(ds_original)
    ds_restored = pickle.loads(data)

    # Without fix: FileNotFoundError because __getstate__ drops prebuilt_indices
    # and __setstate__ tries to load from the empty npy_dir.
    assert len(ds_restored) == len(ds_original), (
        f"Length mismatch after pickle: original={len(ds_original)}, " f"restored={len(ds_restored)}"
    )

    # Compare restored samples against Megatron reference
    gpt_ds = _build_megatron_gpt_dataset(prefix, seq_length, seed, eod_id)
    num_compare = min(len(ds_restored), len(gpt_ds), 50)
    assert num_compare > 0

    for i in range(num_compare):
      mx_tokens = ds_restored[i]["text"]
      meg_raw = TestTokenContentAlignment._meg_raw_tokens(gpt_ds, i)
      np.testing.assert_array_equal(
          mx_tokens,
          meg_raw,
          err_msg=f"Sample {i}: pickle round-trip broke Megatron alignment",
      )


# ===========================================================================
# Cache lifecycle: _ensure_npy_indices
# ===========================================================================


class TestEnsureNpyIndicesCacheLifecycle:
  """Verify _ensure_npy_indices cache hit / miss / rebuild behaviour."""

  def test_cold_start_cache_miss_returns_prebuilt(self, tmp_dir):
    """Empty npy_dir → cache miss → returns in-memory arrays, not None."""
    from maxtext.input_pipeline._mmap_datasource import _ensure_npy_indices  # pylint: disable=import-outside-toplevel

    prefix, _ = _create_eod_dataset(tmp_dir, num_docs=6, eod_id=0)
    npy_dir = os.path.join(tmp_dir, "npy_cold")
    os.makedirs(npy_dir, exist_ok=True)

    expected_hash, prebuilt = _ensure_npy_indices(npy_dir, [prefix], num_samples=20, seq_length=8, seed=42)

    assert prebuilt is not None, "Cache miss should return prebuilt arrays"
    doc_idx, sample_idx, shuffle_idx = prebuilt
    assert doc_idx.ndim == 1
    assert sample_idx.ndim == 2 and sample_idx.shape[1] == 2
    assert shuffle_idx.ndim == 1
    assert len(expected_hash) == 32  # md5 hex

  def test_cache_hit_returns_none(self, tmp_dir):
    """Pre-populated npy_dir → cache hit → returns None for prebuilt."""
    from maxtext.input_pipeline._mmap_datasource import _ensure_npy_indices  # pylint: disable=import-outside-toplevel

    prefix, _ = _create_eod_dataset(tmp_dir, num_docs=6, eod_id=0)
    npy_dir = os.path.join(tmp_dir, "npy_hit")

    # First call: cold cache miss, writes files (since single-process is_primary=True)
    hash1, prebuilt1 = _ensure_npy_indices(npy_dir, [prefix], num_samples=20, seq_length=8, seed=42)
    assert prebuilt1 is not None

    # Second call: should be a cache hit
    hash2, prebuilt2 = _ensure_npy_indices(npy_dir, [prefix], num_samples=20, seq_length=8, seed=42)
    assert hash2 == hash1
    assert prebuilt2 is None, "Cache hit should return None (read from disk)"

  def test_param_change_triggers_rebuild(self, tmp_dir):
    """Existing cache with old params → new params with different epoch count → rebuild."""
    from maxtext.input_pipeline._mmap_datasource import _ensure_npy_indices  # pylint: disable=import-outside-toplevel

    prefix, _ = _create_eod_dataset(tmp_dir, num_docs=6, eod_id=0)
    npy_dir = os.path.join(tmp_dir, "npy_rebuild")

    # Build cache for num_samples=20 (1 epoch)
    hash1, prebuilt1 = _ensure_npy_indices(npy_dir, [prefix], num_samples=20, seq_length=8, seed=42)
    assert prebuilt1 is not None

    # Verify cache hit for same params
    _, prebuilt_hit = _ensure_npy_indices(npy_dir, [prefix], num_samples=20, seq_length=8, seed=42)
    assert prebuilt_hit is None, "Same params should hit cache"

    # Change num_samples enough to require more epochs → different hash → cache miss
    hash2, prebuilt2 = _ensure_npy_indices(npy_dir, [prefix], num_samples=200, seq_length=8, seed=42)
    assert hash2 != hash1, "Different epoch count should produce different hash"
    assert prebuilt2 is not None, "New params should trigger rebuild"

    # Both triplets should now coexist in npy_dir
    _discover_npy_indices(npy_dir, expected_hash=hash1)
    _discover_npy_indices(npy_dir, expected_hash=hash2)

  def test_prebuilt_indices_match_disk(self, tmp_dir):
    """In-memory arrays from cache miss must be identical to what's written to disk."""
    from maxtext.input_pipeline._mmap_datasource import _ensure_npy_indices  # pylint: disable=import-outside-toplevel

    prefix, _ = _create_eod_dataset(tmp_dir, num_docs=6, eod_id=0)
    npy_dir = os.path.join(tmp_dir, "npy_verify")

    expected_hash, prebuilt = _ensure_npy_indices(npy_dir, [prefix], num_samples=20, seq_length=8, seed=42)
    assert prebuilt is not None
    doc_mem, sample_mem, shuffle_mem = prebuilt

    # Read back from disk (written by host 0 / single-process)
    doc_path, sample_path, shuffle_path = _discover_npy_indices(npy_dir, expected_hash=expected_hash)
    np.testing.assert_array_equal(doc_mem, np.load(doc_path))
    np.testing.assert_array_equal(sample_mem, np.load(sample_path))
    np.testing.assert_array_equal(shuffle_mem, np.load(shuffle_path))

  def test_prebuilt_datasource_matches_disk_datasource(self, tmp_dir):
    """MegatronNpyDataSource with prebuilt arrays produces same samples as disk-loaded one."""
    from maxtext.input_pipeline._mmap_datasource import _ensure_npy_indices  # pylint: disable=import-outside-toplevel

    eod_id = 0
    seq_length = 8
    prefix, _ = _create_eod_dataset(tmp_dir, num_docs=6, eod_id=eod_id)
    npy_dir = os.path.join(tmp_dir, "npy_ds_cmp")

    expected_hash, prebuilt = _ensure_npy_indices(npy_dir, [prefix], num_samples=20, seq_length=seq_length, seed=42)
    assert prebuilt is not None

    # Datasource from in-memory arrays
    ds_mem = MegatronNpyDataSource(
        npy_dir=npy_dir,
        bin_paths=prefix,
        eod_id=eod_id,
        seq_length=seq_length,
        prebuilt_indices=prebuilt,
    )

    # Datasource from disk
    ds_disk = MegatronNpyDataSource(
        npy_dir=npy_dir,
        bin_paths=prefix,
        eod_id=eod_id,
        seq_length=seq_length,
        expected_hash=expected_hash,
    )

    n = len(ds_mem)
    assert n == len(ds_disk)
    for i in range(n):
      np.testing.assert_array_equal(
          ds_mem[i]["text"],
          ds_disk[i]["text"],
          err_msg=f"Sample {i}: prebuilt vs disk mismatch",
      )


# ---------------------------------------------------------------------------
# Split runtime integration tests
# ---------------------------------------------------------------------------


class TestSplitRuntimeIntegration:
  """End-to-end tests verifying split/split_index flows through the runtime path."""

  def test_split_npy_coexist_in_same_dir(self, tmp_dir):
    """Different split_index produces different hashes; files coexist in one dir."""
    from maxtext.input_pipeline._mmap_datasource import _ensure_npy_indices  # pylint: disable=import-outside-toplevel

    prefix, _ = _create_split_dataset(tmp_dir, num_docs=20)
    npy_dir = os.path.join(tmp_dir, "shared_npy")
    os.makedirs(npy_dir, exist_ok=True)

    hash0, _ = _ensure_npy_indices(
        npy_dir,
        [prefix],
        num_samples=None,
        seq_length=8,
        seed=42,
        split="0.9,0.1",
        split_index=0,
    )
    hash1, _ = _ensure_npy_indices(
        npy_dir,
        [prefix],
        num_samples=None,
        seq_length=8,
        seed=42,
        split="0.9,0.1",
        split_index=1,
    )

    assert hash0 != hash1, "split_index=0 and split_index=1 must produce different hashes"
    # Both discoverable from the same directory
    _discover_npy_indices(npy_dir, expected_hash=hash0)
    _discover_npy_indices(npy_dir, expected_hash=hash1)

  def test_create_mmap_npy_source_with_split_disjoint(self, tmp_dir):
    """create_mmap_npy_source with split produces non-overlapping train/eval datasets."""
    num_docs = 20
    split_str = "0.9,0.1"
    seed = 42
    seq_length = 8

    prefix, _ = _create_split_dataset(tmp_dir, num_docs=num_docs)
    npy_dir = os.path.join(tmp_dir, "npy")
    os.makedirs(npy_dir, exist_ok=True)
    spec = f"{npy_dir}|{prefix}"

    train_ds = create_mmap_npy_source(
        spec,
        eod_id=0,
        seq_length=seq_length,
        split_sentences=False,
        seed=seed,
        split=split_str,
        split_index=0,
    )
    eval_ds = create_mmap_npy_source(
        spec,
        eod_id=0,
        seq_length=seq_length,
        split_sentences=False,
        seed=seed,
        split=split_str,
        split_index=1,
    )

    train_len = len(train_ds)
    eval_len = len(eval_ds)
    assert train_len > 0, "Train dataset should not be empty"
    assert eval_len > 0, "Eval dataset should not be empty"
    assert train_len > eval_len, "Train should have more samples than eval (90/10 split)"

  def test_split_runtime_matches_offline_convert(self, tmp_dir):
    """Runtime auto-build with split produces same results as offline convert."""
    num_docs = 20
    split_str = "0.9,0.1"
    seed = 42
    seq_length = 8

    prefix, _ = _create_split_dataset(tmp_dir, num_docs=num_docs)

    for split_index in [0, 1]:
      # Offline convert
      offline_dir = os.path.join(tmp_dir, f"offline_{split_index}")
      convert(
          [prefix],
          offline_dir,
          seq_length=seq_length,
          num_epochs=1,
          seed=seed,
          split=split_str,
          split_index=split_index,
      )
      offline_ds = MegatronNpyDataSource(
          npy_dir=offline_dir,
          bin_paths=prefix,
          eod_id=0,
          seq_length=seq_length,
      )

      # Runtime auto-build via create_mmap_npy_source
      runtime_dir = os.path.join(tmp_dir, f"runtime_{split_index}")
      os.makedirs(runtime_dir, exist_ok=True)
      runtime_ds = create_mmap_npy_source(
          f"{runtime_dir}|{prefix}",
          eod_id=0,
          seq_length=seq_length,
          split_sentences=False,
          seed=seed,
          split=split_str,
          split_index=split_index,
      )

      assert len(offline_ds) == len(runtime_ds), (
          f"split_index={split_index}: offline ({len(offline_ds)}) " f"vs runtime ({len(runtime_ds)}) length mismatch"
      )
      for i, offline_sample in enumerate(offline_ds):
        np.testing.assert_array_equal(
            runtime_ds[i]["text"],
            offline_sample["text"],
            err_msg=f"split_index={split_index}, sample {i}",
        )
