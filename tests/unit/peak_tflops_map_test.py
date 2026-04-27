# Copyright 2023-2026 Google LLC
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

"""Tests for peak_tflops_map.py"""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from maxtext.utils.peak_tflops_map import _effective_compute_dtype, get_peak_tflops_per_device


def _cfg(*, peak=0.0, dtype="bfloat16", quantization=""):
  """Build a minimal config SimpleNamespace with the fields the helpers read."""
  return SimpleNamespace(peak_tflops_per_device=peak, dtype=dtype, quantization=quantization)


class EffectiveComputeDtypeTest(unittest.TestCase):
  """Tests for _effective_compute_dtype()."""

  def test_default_bfloat16_maps_to_bf16(self):
    self.assertEqual(_effective_compute_dtype(_cfg()), "bf16")

  def test_explicit_float32_passthrough(self):
    # float32 isn't in any peak table — passthrough so lookup misses and warns.
    self.assertEqual(_effective_compute_dtype(_cfg(dtype="float32")), "float32")

  def test_fp8_quantization_variants_map_to_fp8(self):
    # Source of truth for these variant names: layers/quantizations.py::_get_quant_config.
    for q in ("fp8", "aqt_fp8", "aqt_fp8_full", "fp8_blockwise", "nanoo_fp8"):
      with self.subTest(quantization=q):
        self.assertEqual(_effective_compute_dtype(_cfg(quantization=q)), "fp8")

  def test_te_prefix_maps_to_fp8(self):
    # TransformerEngine quantization names start with "te_" and are FP8 on NV GPU.
    self.assertEqual(_effective_compute_dtype(_cfg(quantization="te_gemm")), "fp8")

  def test_quantization_dominates_over_dtype(self):
    # Even with dtype=bfloat16 (the typical FP8 setup keeps activations in bf16),
    # quantization=fp8 must select the fp8 peak.
    self.assertEqual(_effective_compute_dtype(_cfg(dtype="bfloat16", quantization="fp8")), "fp8")

  def test_int8_passthrough(self):
    # int8 isn't in the peak table; passthrough so lookup will miss and warn.
    self.assertEqual(_effective_compute_dtype(_cfg(quantization="int8")), "int8")

  def test_dtype_as_numpy_dtype_object(self):
    # pyconfig coerces YAML "bfloat16" into a numpy.dtype object before we ever
    # see it. Helper must handle that — calling .lower() directly would
    # AttributeError. Regression for a real PR48 CI failure.
    self.assertEqual(_effective_compute_dtype(_cfg(dtype=np.dtype("float32"))), "float32")


class GetPeakTflopsPerDeviceTest(unittest.TestCase):
  """Tests for get_peak_tflops_per_device()."""

  @mock.patch("maxtext.utils.peak_tflops_map.jax.process_index", return_value=0)
  @mock.patch(
      "maxtext.utils.peak_tflops_map.jax.devices",
      return_value=[SimpleNamespace(device_kind="TPU v5 lite")],
  )
  def test_override_wins_over_auto_detect(self, _mock_devs, _mock_pi):
    result = get_peak_tflops_per_device(_cfg(peak=500.0))
    self.assertEqual(result, 500.0)

  @mock.patch("maxtext.utils.peak_tflops_map.jax.process_index", return_value=0)
  @mock.patch(
      "maxtext.utils.peak_tflops_map.jax.devices",
      return_value=[SimpleNamespace(device_kind="TPU v6 lite")],
  )
  def test_auto_detect_v6e_bf16(self, _mock_devs, _mock_pi):
    self.assertEqual(get_peak_tflops_per_device(_cfg()), 918.0)

  @mock.patch("maxtext.utils.peak_tflops_map.jax.process_index", return_value=0)
  @mock.patch(
      "maxtext.utils.peak_tflops_map.jax.devices",
      return_value=[SimpleNamespace(device_kind="TPU v5p")],
  )
  def test_auto_detect_v5p_long_form(self, _mock_devs, _mock_pi):
    # Both "TPU v5" and "TPU v5p" map to 459.0 because the exact device_kind
    # JAX returns on v5p hardware hasn't been verified yet.
    # aot_identical_test.py uses "TPU v5p"; megablox/common.py's example
    # comment uses "TPU v5". Locking in both keys prevents MFU regressions
    # whichever one JAX actually emits.
    self.assertEqual(get_peak_tflops_per_device(_cfg()), 459.0)

  @mock.patch("maxtext.utils.peak_tflops_map.jax.process_index", return_value=0)
  @mock.patch(
      "maxtext.utils.peak_tflops_map.jax.devices",
      return_value=[SimpleNamespace(device_kind="TPU7x")],
  )
  def test_auto_detect_tpu7x_bf16(self, _mock_devs, _mock_pi):
    # v7x's device_kind breaks the "TPU v{N} [lite]" pattern: it's "TPU7x"
    # (no "v", no space). Verified on a real v7x pod.
    #
    # Expected 1153.5 = 2307 (per chip, per Google Cloud docs) / 2 chiplets.
    # MFU's numerator is per-JAX-device TFLOPs/sec, so the denominator must
    # also be per-JAX-device — using 2307 directly would understate MFU ~2×.
    self.assertEqual(get_peak_tflops_per_device(_cfg()), 1153.5)

  @mock.patch("maxtext.utils.peak_tflops_map.jax.process_index", return_value=0)
  @mock.patch(
      "maxtext.utils.peak_tflops_map.jax.devices",
      return_value=[SimpleNamespace(device_kind="TPU7x")],
  )
  def test_auto_detect_tpu7x_fp8(self, _mock_devs, _mock_pi):
    # v7x fp8 chip-level peak = 4614 (per Google Cloud docs), divided by
    # 2 chiplets per chip = 2307 per JAX device. fp8 is 2× bf16 on v7x's MXU.
    self.assertEqual(get_peak_tflops_per_device(_cfg(quantization="fp8")), 2307.0)

  @mock.patch("maxtext.utils.peak_tflops_map.logging.warning")
  @mock.patch("maxtext.utils.peak_tflops_map.jax.process_index", return_value=0)
  @mock.patch(
      "maxtext.utils.peak_tflops_map.jax.devices",
      return_value=[SimpleNamespace(device_kind="TPU v6 lite")],
  )
  def test_v6e_with_fp8_returns_zero_and_warns(self, _mock_devs, _mock_pi, mock_warn):
    # v6e has no fp8 entry — peak lookup misses on the dtype level, MFU is
    # suppressed, and the lead host emits one warning.
    self.assertEqual(get_peak_tflops_per_device(_cfg(quantization="fp8")), 0.0)
    mock_warn.assert_called_once()
    msg = " ".join(str(a) for a in mock_warn.call_args[0])
    self.assertIn("TPU v6 lite", msg)
    self.assertIn("fp8", msg)

  @mock.patch("maxtext.utils.peak_tflops_map.logging.warning")
  @mock.patch("maxtext.utils.peak_tflops_map.jax.process_index", return_value=0)
  @mock.patch(
      "maxtext.utils.peak_tflops_map.jax.devices",
      return_value=[SimpleNamespace(device_kind="TPU v99 foo")],
  )
  def test_unknown_chip_on_lead_host_returns_zero_and_warns(self, _mock_devs, _mock_pi, mock_warn):
    self.assertEqual(get_peak_tflops_per_device(_cfg()), 0.0)
    mock_warn.assert_called_once()
    self.assertIn("TPU v99 foo", " ".join(str(a) for a in mock_warn.call_args[0]))

  @mock.patch("maxtext.utils.peak_tflops_map.logging.warning")
  @mock.patch("maxtext.utils.peak_tflops_map.jax.process_index", return_value=1)
  @mock.patch(
      "maxtext.utils.peak_tflops_map.jax.devices",
      return_value=[SimpleNamespace(device_kind="TPU v99 foo")],
  )
  def test_unknown_chip_on_non_lead_host_returns_zero_silently(self, _mock_devs, _mock_pi, mock_warn):
    self.assertEqual(get_peak_tflops_per_device(_cfg()), 0.0)
    mock_warn.assert_not_called()


if __name__ == "__main__":
  unittest.main()
