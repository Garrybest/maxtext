"""Tests for GenerateDocSegmentIds with reset_attention_mask flag."""

import numpy as np
import numpy.testing as npt
import pytest

from maxtext.input_pipeline.input_pipeline_utils import GenerateDocSegmentIds

pytestmark = pytest.mark.cpu_only


# ---------------------------------------------------------------------------
# reset_attention_mask=True (default / existing behaviour)
# ---------------------------------------------------------------------------
class TestGenerateDocSegmentIdsResetTrue:
  """Tests for the default mode where attention resets at document boundaries.

  EOD belongs to the preceding document for attention purposes:
  - EOD keeps the preceding document's segment ID
  - EOD continues the preceding document's position counter
  - New segment starts AFTER EOD
  """

  def _make_transform(self, eod_id: int = 0) -> GenerateDocSegmentIds:
    return GenerateDocSegmentIds(eod_id=eod_id, reset_attention_mask=True)

  def test_single_doc_no_eod(self):
    """No EOD tokens -> single segment, sequential positions."""
    transform = self._make_transform(eod_id=0)
    element = {"tokens": np.array([10, 20, 30, 40], dtype=np.int32)}
    result = transform.map(element)

    npt.assert_array_equal(result["tokens_segmentation"], [1, 1, 1, 1])
    npt.assert_array_equal(result["tokens_position"], [0, 1, 2, 3])

  def test_multi_doc_with_eod(self):
    """Multiple documents separated by EOD tokens.

    EOD belongs to preceding doc: [10, 20, EOD, 30, 40, EOD, 50]
    - tokens[0:3] (10,20,EOD) -> seg 1, pos [0,1,2]
    - tokens[3:6] (30,40,EOD) -> seg 2, pos [0,1,2]
    - tokens[6]   (50)        -> seg 3, pos [0]
    """
    transform = self._make_transform(eod_id=0)
    element = {"tokens": np.array([10, 20, 0, 30, 40, 0, 50], dtype=np.int32)}
    result = transform.map(element)

    npt.assert_array_equal(result["tokens_segmentation"], [1, 1, 1, 2, 2, 2, 3])
    npt.assert_array_equal(result["tokens_position"], [0, 1, 2, 0, 1, 2, 0])

  def test_multiple_columns(self):
    """Transform applies to all columns in the element dict.

    [10, EOD, 20] -> seg [1, 1, 2], pos [0, 1, 0]
    """
    transform = self._make_transform(eod_id=0)
    element = {
        "input_ids": np.array([10, 0, 20], dtype=np.int32),
        "target_ids": np.array([10, 0, 20], dtype=np.int32),
    }
    result = transform.map(element)

    for col in ("input_ids", "target_ids"):
      npt.assert_array_equal(result[f"{col}_segmentation"], [1, 1, 2])
      npt.assert_array_equal(result[f"{col}_position"], [0, 1, 0])


# ---------------------------------------------------------------------------
# reset_attention_mask=False (new behaviour — cross-document attention)
# ---------------------------------------------------------------------------
class TestGenerateDocSegmentIdsResetFalse:
  """Tests for the mode where cross-document attention is allowed."""

  def _make_transform(self, eod_id: int = 0) -> GenerateDocSegmentIds:
    return GenerateDocSegmentIds(eod_id=eod_id, reset_attention_mask=False)

  def test_single_doc_no_eod(self):
    """No EOD tokens -> same result as reset=True (no boundaries to differ)."""
    transform = self._make_transform(eod_id=0)
    element = {"tokens": np.array([10, 20, 30, 40], dtype=np.int32)}
    result = transform.map(element)

    npt.assert_array_equal(result["tokens_segmentation"], [1, 1, 1, 1])
    npt.assert_array_equal(result["tokens_position"], [0, 1, 2, 3])

  def test_multi_doc_uniform_seg(self):
    """eod_mask_loss=False (default): all tokens including EOD share seg=1."""
    transform = self._make_transform(eod_id=0)
    element = {"tokens": np.array([10, 20, 0, 30, 40, 0, 50], dtype=np.int32)}
    result = transform.map(element)

    npt.assert_array_equal(result["tokens_segmentation"], [1, 1, 1, 1, 1, 1, 1])
    npt.assert_array_equal(result["tokens_position"], [0, 1, 2, 3, 4, 5, 6])

  def test_multi_doc_eod_mask_loss(self):
    """eod_mask_loss=True: EOD tokens get seg=0, non-EOD get seg=1."""
    transform = GenerateDocSegmentIds(eod_id=0, reset_attention_mask=False, eod_mask_loss=True)
    element = {"tokens": np.array([10, 20, 0, 30, 40, 0, 50], dtype=np.int32)}
    result = transform.map(element)

    npt.assert_array_equal(result["tokens_segmentation"], [1, 1, 0, 1, 1, 0, 1])
    npt.assert_array_equal(result["tokens_position"], [0, 1, 2, 3, 4, 5, 6])

  def test_eod_at_boundaries(self):
    """EOD tokens at start and end — eod_mask_loss=False keeps seg=1."""
    transform = self._make_transform(eod_id=0)
    element = {"tokens": np.array([0, 10, 20, 0], dtype=np.int32)}
    result = transform.map(element)

    npt.assert_array_equal(result["tokens_segmentation"], [1, 1, 1, 1])
    npt.assert_array_equal(result["tokens_position"], [0, 1, 2, 3])

  def test_multiple_columns(self):
    """Transform applies to all columns in the element dict."""
    transform = self._make_transform(eod_id=0)
    element = {
        "input_ids": np.array([10, 0, 20], dtype=np.int32),
        "target_ids": np.array([10, 0, 20], dtype=np.int32),
    }
    result = transform.map(element)

    for col in ("input_ids", "target_ids"):
      npt.assert_array_equal(result[f"{col}_segmentation"], [1, 1, 1])
      npt.assert_array_equal(result[f"{col}_position"], [0, 1, 2])
