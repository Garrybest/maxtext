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

"""Peak TFLOPs per device, keyed by (jax device_kind, effective compute dtype)."""

import jax
from absl import logging

# Dense peak TFLOPs per JAX device, keyed by device_kind then compute dtype.
# Source: Google Cloud public docs. Extend when new chips ship.
#
# Keys are jax.devices()[0].device_kind — the exact string JAX returns at
# runtime, which is NOT always the marketing name (e.g. v7x's key is
# "TPU7x", not "TPU v7x"). v5p has two keys because the real device_kind
# on v5p hardware hasn't been verified yet; tests/integration/aot_identical_test.py
# uses "TPU v5p" while src/maxtext/kernels/megablox/common.py references
# "TPU v5". Both map to 459.0 so whichever JAX returns, MFU fires.
#
# TPU7x note: 1 chip = 2 JAX devices, so the per-chip peaks from Google
# Cloud docs (https://docs.cloud.google.com/tpu/docs/tpu7x) — bf16 = 2307,
# fp8 = 4614 — are halved to per-JAX-device peaks below. fp8 is 2× bf16
# (standard v7x MXU behavior at fp8 precision).
#
# Unlisted (chip, dtype) pairs return 0.0, which suppresses perf/mfu via
# the `peak > 0` guard in metric_logger.record_train_metrics.
_CHIP_TO_PEAK_TFLOPS: dict[str, dict[str, float]] = {
    "TPU v5 lite": {"bf16": 197.0},  # v5e
    "TPU v5": {"bf16": 459.0},  # v5p (candidate 1)
    "TPU v5p": {"bf16": 459.0},  # v5p (candidate 2)
    "TPU v6 lite": {"bf16": 918.0},  # v6e / Trillium — verified in src/maxtext/utils/max_utils.py:506
    # v7x / Ironwood — chip-level 2307 (bf16) / 4614 (fp8), divided by 2
    # chiplets to get per-JAX-device. Verified on a real v7x pod.
    "TPU7x": {"bf16": 1153.5, "fp8": 2307.0},
}

# config.quantization values that route matmuls through FP8.
# Source of truth: src/maxtext/layers/quantizations.py::_get_quant_config.
# `te_*` (TransformerEngine, NV GPU) is matched separately by prefix.
_FP8_QUANTIZATIONS = frozenset({"fp8", "aqt_fp8", "aqt_fp8_full", "fp8_blockwise", "nanoo_fp8"})


def _effective_compute_dtype(config) -> str:
  """Infer the matmul compute dtype for peak-table lookup.

  config.dtype is the activation dtype (typically bfloat16) and does NOT
  change when FP8 quantization is enabled — FP8 routes through
  config.quantization, which controls the matmul path. So when
  quantization is set, that wins; otherwise fall back to dtype.
  """
  q = (getattr(config, "quantization", "") or "").lower()
  if q in _FP8_QUANTIZATIONS or q.startswith("te_"):
    return "fp8"
  if q == "":
    # pyconfig coerces YAML "bfloat16" into numpy.dtype(...); str() normalizes
    # both that and plain strings to the dtype name ("bfloat16", "float32", ...).
    dtype = str(getattr(config, "dtype", "") or "").lower()
    if dtype in ("bfloat16", "bf16"):
      return "bf16"
    return dtype
  return q  # int8 / intmp / unknown — won't match any peak entry.


def get_peak_tflops_per_device(config) -> float:
  """Return peak TFLOPs/device; config override wins, else auto-detect.

  Returns 0.0 (MFU emission skipped) when either the device_kind or the
  effective compute dtype is not in the peak table and no override is set.
  In the skip case, the lead host (process 0) emits one WARNING so the
  message isn't duplicated per host in multi-host jobs.
  """
  if config.peak_tflops_per_device > 0:
    return config.peak_tflops_per_device

  kind = jax.devices()[0].device_kind
  dtype = _effective_compute_dtype(config)
  peak = _CHIP_TO_PEAK_TFLOPS.get(kind, {}).get(dtype, 0.0)

  if peak == 0.0 and jax.process_index() == 0:
    logging.warning(
        "perf/mfu skipped: no peak TFLOPs entry for device_kind=%r, "
        "compute_dtype=%r. Set peak_tflops_per_device config to enable "
        "MFU emission.",
        kind,
        dtype,
    )
  return peak
