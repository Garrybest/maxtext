# Copyright 2025 Google LLC
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

"""Unit tests for blockwise FP8 quantization and matmul kernels.

These tests exercise native FP8 Pallas kernels and must run on TPU hardware.
They do NOT use interpret=False since CPU does not support native FP8 dot_general.
"""

import types
import unittest
import unittest.mock
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

import qwix
from maxtext.layers.quantizations import BlockwiseFp8Provider
from maxtext.layers.quantizations import get_quantization_rule, get_qt_provider

from maxtext.kernels.megablox.blockwise_fp8 import (
    _align_tile_tpu,
    quantize_blockwise_1d,
    quantize_blockwise_2d,
    blockwise_fp8_matmul_1dx2d,
    blockwise_fp8_matmul_1dx1d,
    fused_blockwise_fp8_matmul_1dx2d,
    fused_blockwise_fp8_matmul_1dx1d,
    BlockwiseFp8DotGeneralOp,
)


def _nrmse(ref, result):
  """Compute normalized RMSE between ref and result."""
  diff = ref - result.astype(jnp.float32)
  rmse = jnp.sqrt(jnp.mean(diff**2))
  rms_ref = jnp.sqrt(jnp.mean(ref**2))
  return float(rmse / (rms_ref + 1e-6))


class QuantizeBlockwise1DTest(unittest.TestCase):
  """Tests for 1D blockwise quantization."""

  def test_shapes_k_axis(self):
    """Output shapes: qx same as x, scale reduced by block_size on K axis."""
    M, K = 4096, 2048
    x = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=jnp.bfloat16)
    qx, sx = quantize_blockwise_1d(x, block_size=128, axis=-1, interpret=False)
    self.assertEqual(qx.shape, (M, K))
    self.assertEqual(qx.dtype, jnp.float8_e4m3fn)
    self.assertEqual(sx.shape, (M, K // 128))
    self.assertEqual(sx.dtype, jnp.float32)

  def test_shapes_m_axis(self):
    """Output shapes for M-axis (column-independent) quantization."""
    M, K = 4096, 2048
    x = jax.random.normal(jax.random.PRNGKey(1), (M, K), dtype=jnp.bfloat16)
    qx, sx = quantize_blockwise_1d(x, block_size=128, axis=0, interpret=False)
    self.assertEqual(qx.shape, (M, K))
    self.assertEqual(qx.dtype, jnp.float8_e4m3fn)
    self.assertEqual(sx.shape, (M // 128, K))
    self.assertEqual(sx.dtype, jnp.float32)

  def test_dequantize_error_k_axis(self):
    """Reconstruction error within FP8 tolerance for K-axis."""
    M, K = 4096, 2048
    x = jax.random.normal(jax.random.PRNGKey(2), (M, K), dtype=jnp.bfloat16)
    qx, sx = quantize_blockwise_1d(x, block_size=128, axis=-1, interpret=False)

    # Dequantize: for each 128-block, multiply qx by scale
    qx_f32 = qx.astype(jnp.float32)
    # sx: [M, K//128], repeat along K to get [M, K]
    sx_expanded = jnp.repeat(sx, 128, axis=1)
    x_reconstructed = qx_f32 * sx_expanded

    # FP8 e4m3 has ~2^-3 relative precision, so 15% tolerance is reasonable
    x_f32 = x.astype(jnp.float32)
    rel_error = jnp.abs(x_f32 - x_reconstructed) / (jnp.abs(x_f32) + 1e-6)
    mean_rel_error = jnp.mean(rel_error)
    self.assertLess(float(mean_rel_error), 0.15, f"Mean relative error {float(mean_rel_error):.4f} too high")


class QuantizeBlockwise2DTest(unittest.TestCase):
  """Tests for 2D blockwise quantization."""

  def test_shapes(self):
    """Output shapes for 2D blockwise quantization."""
    K, N = 2048, 7168
    x = jax.random.normal(jax.random.PRNGKey(3), (K, N), dtype=jnp.bfloat16)
    qx, sx = quantize_blockwise_2d(x, block_size=128)
    self.assertEqual(qx.shape, (K, N))
    self.assertEqual(qx.dtype, jnp.float8_e4m3fn)
    self.assertEqual(sx.shape, (K // 128, N // 128))

  def test_dequantize_error(self):
    """Reconstruction error within FP8 tolerance for 2D."""
    K, N = 2048, 7168
    x = jax.random.normal(jax.random.PRNGKey(4), (K, N), dtype=jnp.bfloat16)
    qx, sx = quantize_blockwise_2d(x, block_size=128)

    # Dequantize: for each 128x128 block, multiply qx by scalar scale
    qx_f32 = qx.astype(jnp.float32)
    sx_expanded = jnp.repeat(jnp.repeat(sx, 128, axis=0), 128, axis=1)
    x_reconstructed = qx_f32 * sx_expanded

    x_f32 = x.astype(jnp.float32)
    rel_error = jnp.abs(x_f32 - x_reconstructed) / (jnp.abs(x_f32) + 1e-6)
    mean_rel_error = jnp.mean(rel_error)
    self.assertLess(float(mean_rel_error), 0.15)


class Matmul1Dx2DTest(unittest.TestCase):
  """Tests for 1D x 2D blockwise FP8 matmul."""

  def test_correctness(self):
    """Compare against bf16 reference matmul."""
    M, K, N = 4096, 2048, 7168
    key1, key2 = jax.random.split(jax.random.PRNGKey(5))
    lhs = jax.random.normal(key1, (M, K), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (K, N), dtype=jnp.bfloat16)

    # Reference
    ref = jnp.matmul(lhs.astype(jnp.float32), rhs.astype(jnp.float32))

    # Quantize
    qlhs, slhs = quantize_blockwise_1d(lhs, block_size=128, axis=-1, interpret=False)
    qrhs, srhs = quantize_blockwise_2d(rhs, block_size=128)

    # FP8 matmul
    result = blockwise_fp8_matmul_1dx2d(qlhs, slhs, qrhs, srhs, block_size=128, interpret=False)

    # Check normalized RMSE (robust to near-zero elements unlike per-element relative error)
    diff = ref - result.astype(jnp.float32)
    rmse = jnp.sqrt(jnp.mean(diff**2))
    rms_ref = jnp.sqrt(jnp.mean(ref**2))
    normalized_rmse = rmse / (rms_ref + 1e-6)
    self.assertLess(float(normalized_rmse), 0.05, f"1Dx2D matmul normalized RMSE {float(normalized_rmse):.4f} too high")


class Matmul1Dx1DTest(unittest.TestCase):
  """Tests for 1D x 1D blockwise FP8 matmul (wgrad)."""

  def test_correctness(self):
    """Compare against bf16 reference for lhs^T @ rhs."""
    M, H, D = 4096, 2048, 7168
    key1, key2 = jax.random.split(jax.random.PRNGKey(6))
    lhs = jax.random.normal(key1, (M, H), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (M, D), dtype=jnp.bfloat16)

    # Reference: lhs^T @ rhs
    ref = jnp.matmul(lhs.astype(jnp.float32).T, rhs.astype(jnp.float32))

    # Quantize along M-axis (axis=0)
    qlhs, slhs = quantize_blockwise_1d(lhs, block_size=128, axis=0, interpret=False)
    qrhs, srhs = quantize_blockwise_1d(rhs, block_size=128, axis=0, interpret=False)

    # FP8 matmul
    result = blockwise_fp8_matmul_1dx1d(qlhs, slhs, qrhs, srhs, block_size=128, interpret=False)

    # Check normalized RMSE (robust to near-zero elements unlike per-element relative error)
    diff = ref - result.astype(jnp.float32)
    rmse = jnp.sqrt(jnp.mean(diff**2))
    rms_ref = jnp.sqrt(jnp.mean(ref**2))
    normalized_rmse = rmse / (rms_ref + 1e-6)
    self.assertLess(float(normalized_rmse), 0.05, f"1Dx1D matmul normalized RMSE {float(normalized_rmse):.4f} too high")


class DotGeneralOpForwardTest(unittest.TestCase):
  """Tests for BlockwiseFp8DotGeneralOp forward pass."""

  def test_forward_basic(self):
    """BlockwiseFp8DotGeneralOp forward vs jax.lax.dot_general."""
    M, K, N = 4096, 2048, 7168
    key1, key2 = jax.random.split(jax.random.PRNGKey(7))
    lhs = jax.random.normal(key1, (M, K), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (K, N), dtype=jnp.bfloat16)

    dimension_numbers = (((1,), (0,)), ((), ()))

    # Reference
    ref = jax.lax.dot_general(lhs, rhs, dimension_numbers, preferred_element_type=jnp.float32)

    # FP8 op
    op = BlockwiseFp8DotGeneralOp(block_size=128)
    op.interpret = False
    result = op(lhs, rhs, dimension_numbers)

    # Check normalized RMSE (robust to near-zero elements unlike per-element relative error)
    diff = ref - result.astype(jnp.float32)
    rmse = jnp.sqrt(jnp.mean(diff**2))
    rms_ref = jnp.sqrt(jnp.mean(ref**2))
    normalized_rmse = rmse / (rms_ref + 1e-6)
    self.assertLess(float(normalized_rmse), 0.05, f"Forward normalized RMSE {float(normalized_rmse):.4f} too high")


class DotGeneralOpBackwardTest(unittest.TestCase):
  """Tests for BlockwiseFp8DotGeneralOp backward pass."""

  def test_backward_basic(self):
    """jax.grad through BlockwiseFp8DotGeneralOp vs bf16 reference."""
    M, K, N = 4096, 2048, 7168
    key1, key2 = jax.random.split(jax.random.PRNGKey(8))
    lhs = jax.random.normal(key1, (M, K), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (K, N), dtype=jnp.bfloat16)

    dimension_numbers = (((1,), (0,)), ((), ()))

    # Reference gradient
    def ref_fn(l, r):
      return jnp.sum(jax.lax.dot_general(l, r, dimension_numbers, preferred_element_type=jnp.float32))

    ref_grads = jax.grad(ref_fn, argnums=(0, 1))(lhs, rhs)

    # FP8 gradient
    op = BlockwiseFp8DotGeneralOp(block_size=128)
    op.interpret = False

    def fp8_fn(l, r):
      return jnp.sum(op(l, r, dimension_numbers))

    fp8_grads = jax.grad(fp8_fn, argnums=(0, 1))(lhs, rhs)

    # Check lhs gradient (normalized RMSE, 20% tolerance for gradients —
    # larger matrices accumulate more FP8 quantization error in backward)
    lhs_diff = ref_grads[0].astype(jnp.float32) - fp8_grads[0].astype(jnp.float32)
    lhs_rmse = jnp.sqrt(jnp.mean(lhs_diff**2))
    lhs_rms_ref = jnp.sqrt(jnp.mean(ref_grads[0].astype(jnp.float32) ** 2))
    lhs_nrmse = lhs_rmse / (lhs_rms_ref + 1e-6)
    self.assertLess(float(lhs_nrmse), 0.2, f"LHS grad normalized RMSE {float(lhs_nrmse):.4f} too high")

    # Check rhs gradient
    rhs_diff = ref_grads[1].astype(jnp.float32) - fp8_grads[1].astype(jnp.float32)
    rhs_rmse = jnp.sqrt(jnp.mean(rhs_diff**2))
    rhs_rms_ref = jnp.sqrt(jnp.mean(ref_grads[1].astype(jnp.float32) ** 2))
    rhs_nrmse = rhs_rmse / (rhs_rms_ref + 1e-6)
    self.assertLess(float(rhs_nrmse), 0.2, f"RHS grad normalized RMSE {float(rhs_nrmse):.4f} too high")


class PaddingTest(unittest.TestCase):
  """Tests for inputs not divisible by block_size."""

  def test_non_divisible_dims(self):
    """Dims not divisible by 128 are padded correctly."""
    M, K, N = 4001, 2003, 7111
    key1, key2 = jax.random.split(jax.random.PRNGKey(9))
    lhs = jax.random.normal(key1, (M, K), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (K, N), dtype=jnp.bfloat16)

    dimension_numbers = (((1,), (0,)), ((), ()))

    ref = jax.lax.dot_general(lhs, rhs, dimension_numbers, preferred_element_type=jnp.float32)

    op = BlockwiseFp8DotGeneralOp(block_size=128)
    op.interpret = False
    result = op(lhs, rhs, dimension_numbers)

    # Shape must match
    self.assertEqual(result.shape, ref.shape)

    # Check normalized RMSE (robust to near-zero elements)
    diff = ref - result.astype(jnp.float32)
    rmse = jnp.sqrt(jnp.mean(diff**2))
    rms_ref = jnp.sqrt(jnp.mean(ref**2))
    normalized_rmse = rmse / (rms_ref + 1e-6)
    self.assertLess(float(normalized_rmse), 0.05)


class MultiDimContractionTest(unittest.TestCase):
  """Tests for multi-dimensional contraction."""

  def test_batched_matmul(self):
    """dimension_numbers with batch dimensions."""
    B, M, K, N = 2, 4096, 2048, 7168
    key1, key2 = jax.random.split(jax.random.PRNGKey(10))
    lhs = jax.random.normal(key1, (B, M, K), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (B, K, N), dtype=jnp.bfloat16)

    # Batch on dim 0, contract K on dim 2 of lhs, dim 1 of rhs
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))

    ref = jax.lax.dot_general(lhs, rhs, dimension_numbers, preferred_element_type=jnp.float32)

    op = BlockwiseFp8DotGeneralOp(block_size=128)
    op.interpret = False
    result = op(lhs, rhs, dimension_numbers)

    self.assertEqual(result.shape, ref.shape)

    # Check normalized RMSE (robust to near-zero elements)
    diff = ref - result.astype(jnp.float32)
    rmse = jnp.sqrt(jnp.mean(diff**2))
    rms_ref = jnp.sqrt(jnp.mean(ref**2))
    normalized_rmse = rmse / (rms_ref + 1e-6)
    self.assertLess(float(normalized_rmse), 0.05)


class FusedMatmul1Dx2DTest(unittest.TestCase):
  """Tests for fused quantize + matmul 1D x 2D."""

  def test_correctness(self):
    """Fused 1dx2d output vs bf16 reference matmul."""
    M, K, N = 4096, 2048, 7168
    key1, key2 = jax.random.split(jax.random.PRNGKey(20))
    lhs = jax.random.normal(key1, (M, K), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (K, N), dtype=jnp.bfloat16)

    ref = jnp.matmul(lhs.astype(jnp.float32), rhs.astype(jnp.float32))

    result = fused_blockwise_fp8_matmul_1dx2d(lhs, rhs, block_size=128, interpret=False)

    diff = ref - result.astype(jnp.float32)
    rmse = jnp.sqrt(jnp.mean(diff**2))
    rms_ref = jnp.sqrt(jnp.mean(ref**2))
    normalized_rmse = rmse / (rms_ref + 1e-6)
    self.assertLess(float(normalized_rmse), 0.05, f"Fused 1Dx2D normalized RMSE {float(normalized_rmse):.4f} too high")


class FusedMatmul1Dx1DTest(unittest.TestCase):
  """Tests for fused quantize + matmul 1D x 1D (wgrad)."""

  def test_correctness(self):
    """Fused 1dx1d output vs bf16 reference (lhs^T @ rhs)."""
    M, H, D = 4096, 2048, 7168
    key1, key2 = jax.random.split(jax.random.PRNGKey(21))
    lhs = jax.random.normal(key1, (M, H), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (M, D), dtype=jnp.bfloat16)

    ref = jnp.matmul(lhs.astype(jnp.float32).T, rhs.astype(jnp.float32))

    result = fused_blockwise_fp8_matmul_1dx1d(lhs, rhs, block_size=128, interpret=False)

    diff = ref - result.astype(jnp.float32)
    rmse = jnp.sqrt(jnp.mean(diff**2))
    rms_ref = jnp.sqrt(jnp.mean(ref**2))
    normalized_rmse = rmse / (rms_ref + 1e-6)
    self.assertLess(float(normalized_rmse), 0.05, f"Fused 1Dx1D normalized RMSE {float(normalized_rmse):.4f} too high")


class FusedVsNonFusedTest(unittest.TestCase):
  """Tests that fused and non-fused produce nearly identical results."""

  def test_1dx2d_consistency(self):
    """Fused vs non-fused 1dx2d should match closely."""
    M, K, N = 4096, 2048, 7168
    key1, key2 = jax.random.split(jax.random.PRNGKey(22))
    lhs = jax.random.normal(key1, (M, K), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (K, N), dtype=jnp.bfloat16)

    # Non-fused path
    qlhs, slhs = quantize_blockwise_1d(lhs, block_size=128, axis=-1, interpret=False)
    qrhs, srhs = quantize_blockwise_2d(rhs, block_size=128)
    non_fused = blockwise_fp8_matmul_1dx2d(qlhs, slhs, qrhs, srhs, block_size=128, interpret=False)

    # Fused path
    fused = fused_blockwise_fp8_matmul_1dx2d(lhs, rhs, block_size=128, interpret=False)

    diff = non_fused.astype(jnp.float32) - fused.astype(jnp.float32)
    rmse = jnp.sqrt(jnp.mean(diff**2))
    rms_ref = jnp.sqrt(jnp.mean(non_fused.astype(jnp.float32) ** 2))
    normalized_rmse = rmse / (rms_ref + 1e-6)
    self.assertLess(
        float(normalized_rmse), 0.001, f"Fused vs non-fused 1Dx2D NRMSE {float(normalized_rmse):.6f} too high"
    )


class DotGeneralOpFusedForwardTest(unittest.TestCase):
  """Tests for BlockwiseFp8DotGeneralOp with use_fused=True forward pass."""

  def test_forward(self):
    """BlockwiseFp8DotGeneralOp(use_fused=True) forward vs dot_general ref."""
    M, K, N = 4096, 2048, 7168
    key1, key2 = jax.random.split(jax.random.PRNGKey(23))
    lhs = jax.random.normal(key1, (M, K), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (K, N), dtype=jnp.bfloat16)

    dimension_numbers = (((1,), (0,)), ((), ()))

    ref = jax.lax.dot_general(lhs, rhs, dimension_numbers, preferred_element_type=jnp.float32)

    op = BlockwiseFp8DotGeneralOp(block_size=128, use_fused=True)
    op.interpret = False
    result = op(lhs, rhs, dimension_numbers)

    diff = ref - result.astype(jnp.float32)
    rmse = jnp.sqrt(jnp.mean(diff**2))
    rms_ref = jnp.sqrt(jnp.mean(ref**2))
    normalized_rmse = rmse / (rms_ref + 1e-6)
    self.assertLess(float(normalized_rmse), 0.05, f"Fused forward NRMSE {float(normalized_rmse):.4f} too high")


class DotGeneralOpFusedBackwardTest(unittest.TestCase):
  """Tests for BlockwiseFp8DotGeneralOp with use_fused=True backward pass."""

  def _run_backward(self, cache_rhs):
    """Helper: jax.grad through fused Op vs bf16 reference."""
    M, K, N = 4096, 2048, 7168
    key1, key2 = jax.random.split(jax.random.PRNGKey(24))
    lhs = jax.random.normal(key1, (M, K), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (K, N), dtype=jnp.bfloat16)

    dimension_numbers = (((1,), (0,)), ((), ()))

    # Reference gradient
    def ref_fn(l, r):
      return jnp.sum(jax.lax.dot_general(l, r, dimension_numbers, preferred_element_type=jnp.float32))

    ref_grads = jax.grad(ref_fn, argnums=(0, 1))(lhs, rhs)

    # Fused FP8 gradient
    op = BlockwiseFp8DotGeneralOp(block_size=128, use_fused=True, cache_rhs=cache_rhs)
    op.interpret = False

    def fp8_fn(l, r):
      return jnp.sum(op(l, r, dimension_numbers))

    fp8_grads = jax.grad(fp8_fn, argnums=(0, 1))(lhs, rhs)

    # Check lhs gradient
    lhs_diff = ref_grads[0].astype(jnp.float32) - fp8_grads[0].astype(jnp.float32)
    lhs_rmse = jnp.sqrt(jnp.mean(lhs_diff**2))
    lhs_rms_ref = jnp.sqrt(jnp.mean(ref_grads[0].astype(jnp.float32) ** 2))
    lhs_nrmse = lhs_rmse / (lhs_rms_ref + 1e-6)
    tag = f"cache_rhs={cache_rhs}"
    self.assertLess(float(lhs_nrmse), 0.2, f"Fused LHS grad NRMSE {float(lhs_nrmse):.4f} too high ({tag})")

    # Check rhs gradient
    rhs_diff = ref_grads[1].astype(jnp.float32) - fp8_grads[1].astype(jnp.float32)
    rhs_rmse = jnp.sqrt(jnp.mean(rhs_diff**2))
    rhs_rms_ref = jnp.sqrt(jnp.mean(ref_grads[1].astype(jnp.float32) ** 2))
    rhs_nrmse = rhs_rmse / (rhs_rms_ref + 1e-6)
    self.assertLess(float(rhs_nrmse), 0.2, f"Fused RHS grad NRMSE {float(rhs_nrmse):.4f} too high ({tag})")

  def test_backward_cache_rhs_true(self):
    """Backward with cache_rhs=True (pre-quantized RHS cached)."""
    self._run_backward(cache_rhs=True)

  def test_backward_cache_rhs_false(self):
    """Backward with cache_rhs=False (kernel re-quantizes RHS)."""
    self._run_backward(cache_rhs=False)


class ConfigIntegrationTest(unittest.TestCase):
  """Tests for config integration."""

  def test_rule_creation(self):
    """Verify fp8_blockwise rule and provider can be created."""
    config = MagicMock()
    config.quantization = "fp8_blockwise"
    config.fp8_format = "e4m3"
    config.weight_quantization_calibration_method = "absmax"
    config.act_quantization_calibration_method = "absmax"
    config.bwd_quantization_calibration_method = "absmax"

    rule = get_quantization_rule(config)
    self.assertIsNotNone(rule)
    self.assertEqual(rule.weight_qtype, jnp.float8_e4m3fn)
    self.assertEqual(rule.act_qtype, jnp.float8_e4m3fn)
    self.assertEqual(rule.bwd_qtype, jnp.float8_e4m3fn)
    self.assertEqual(rule.tile_size, 128)

    provider = get_qt_provider(config)
    self.assertIsNotNone(provider)


class FusedPreQuantizedRhsTest(unittest.TestCase):
  """Tests for fused matmul with pre-quantized RHS."""

  def test_correctness(self):
    """Pre-quantized RHS path vs bf16 reference matmul."""
    M, K, N = 4096, 2048, 7168
    key1, key2 = jax.random.split(jax.random.PRNGKey(30))
    lhs = jax.random.normal(key1, (M, K), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (K, N), dtype=jnp.bfloat16)

    ref = jnp.matmul(lhs.astype(jnp.float32), rhs.astype(jnp.float32))

    # Pre-quantize RHS
    qrhs, srhs = quantize_blockwise_2d(rhs, block_size=128)

    result = fused_blockwise_fp8_matmul_1dx2d(
        lhs, qrhs, block_size=128, interpret=False, rhs_quantized=True, rhs_scale=srhs
    )

    nrmse = _nrmse(ref, result)
    self.assertLess(nrmse, 0.05, f"Pre-quantized RHS NRMSE {nrmse:.4f} too high")

  def test_consistency_with_fused(self):
    """Pre-quantized RHS path vs regular fused path should be close."""
    M, K, N = 4096, 2048, 7168
    key1, key2 = jax.random.split(jax.random.PRNGKey(31))
    lhs = jax.random.normal(key1, (M, K), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (K, N), dtype=jnp.bfloat16)

    # Regular fused (quantizes both LHS and RHS inside kernel)
    fused = fused_blockwise_fp8_matmul_1dx2d(lhs, rhs, block_size=128, interpret=False)

    # Pre-quantized RHS
    qrhs, srhs = quantize_blockwise_2d(rhs, block_size=128)
    preq = fused_blockwise_fp8_matmul_1dx2d(
        lhs, qrhs, block_size=128, interpret=False, rhs_quantized=True, rhs_scale=srhs
    )

    nrmse = _nrmse(fused.astype(jnp.float32), preq)
    self.assertLess(nrmse, 0.001, f"Pre-quantized vs fused NRMSE {nrmse:.6f} too high")

  def test_transpose_for_dgrad(self):
    """Pre-quantized RHS transpose correctness for dgrad: g @ rhs^T."""
    M, K, N = 4096, 2048, 7168
    key1, key2 = jax.random.split(jax.random.PRNGKey(32))
    g = jax.random.normal(key1, (M, N), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (K, N), dtype=jnp.bfloat16)

    ref = jnp.matmul(g.astype(jnp.float32), rhs.astype(jnp.float32).T)

    # Pre-quantize RHS, then transpose for dgrad
    qrhs, srhs = quantize_blockwise_2d(rhs, block_size=128)
    result = fused_blockwise_fp8_matmul_1dx2d(
        g,
        jnp.swapaxes(qrhs, -2, -1),
        block_size=128,
        interpret=False,
        rhs_quantized=True,
        rhs_scale=jnp.swapaxes(srhs, -2, -1),
    )

    nrmse = _nrmse(ref, result)
    self.assertLess(nrmse, 0.05, f"Transpose dgrad NRMSE {nrmse:.4f} too high")


class BlockwiseFp8ProviderTest(unittest.TestCase):
  """Tests for BlockwiseFp8Provider shard_map vs fallback correctness.

  Requires TPU with >= 2 devices. The shard_map path wraps
  BlockwiseFp8DotGeneralOp(use_fused=True) in jax.shard_map, while the
  fallback path calls BlockwiseFp8DotGeneralOp(use_fused=False, use_pallas=False)
  directly. Tests verify both paths produce aligned results against each other
  and against a bf16 reference.
  """

  def setUp(self):
    n_devices = len(jax.devices())
    if n_devices < 2:
      self.skipTest("Need >= 2 devices for shard_map tests")

    self.n_devices = n_devices
    self.mesh = Mesh(np.array(jax.devices()).reshape(n_devices), ("expert",))
    self.rules = [
        ("activation_batch", "expert"),
        ("activation_norm_length", None),
        ("activation_embed", None),
        ("embed", "expert"),
        ("mlp", None),
    ]
    self.input_axes = ("activation_batch", "activation_norm_length", "activation_embed")
    self.kernel_axes = ("embed", "mlp")
    self.module_axis = (-1,)
    self.dimension_numbers = (((2,), (0,)), ((), ()))

    key1, key2 = jax.random.split(jax.random.PRNGKey(42))
    self.lhs = jax.random.normal(key1, (n_devices, 4096, 2048), dtype=jnp.bfloat16)
    self.rhs = jax.random.normal(key2, (2048, 7168), dtype=jnp.bfloat16)

  def _call_provider(self, provider, lhs, rhs, dimension_numbers, input_axes, kernel_axes, module_axis):
    """Call provider.dot_general with mocked Flax module introspection."""
    mock_module = types.SimpleNamespace(
        kernel_axes=kernel_axes,
        input_activation_axes=input_axes,
        axis=module_axis,
    )
    rule = qwix.QtRule(
        module_path=".*",
        weight_qtype=jnp.float8_e4m3fn,
        tile_size=128,
        op_names=("dot_general",),
    )
    with (
        unittest.mock.patch.object(provider, "_get_current_rule_and_op_id", return_value=(rule, "test_op")),
        unittest.mock.patch("qwix._src.flax_util.get_current_module", return_value=mock_module),
    ):
      return provider.dot_general(lhs, rhs, dimension_numbers)

  def _make_sharded_provider(self):
    """Create provider that uses the shard_map path."""
    rule = qwix.QtRule(
        module_path=".*",
        weight_qtype=jnp.float8_e4m3fn,
        tile_size=128,
        op_names=("dot_general",),
    )
    return BlockwiseFp8Provider(rules=[rule], mesh=self.mesh, logical_axis_rules=self.rules)

  def _make_fallback_provider(self):
    """Create provider that uses the fallback path (mesh=None)."""
    rule = qwix.QtRule(
        module_path=".*",
        weight_qtype=jnp.float8_e4m3fn,
        tile_size=128,
        op_names=("dot_general",),
    )
    return BlockwiseFp8Provider(rules=[rule], mesh=None, logical_axis_rules=None)

  def test_forward_sharded_vs_fallback(self):
    """Shard_map path vs fallback path forward pass: NRMSE < 5%."""
    sharded = self._call_provider(
        self._make_sharded_provider(),
        self.lhs,
        self.rhs,
        self.dimension_numbers,
        self.input_axes,
        self.kernel_axes,
        self.module_axis,
    )
    fallback = self._call_provider(
        self._make_fallback_provider(),
        self.lhs,
        self.rhs,
        self.dimension_numbers,
        self.input_axes,
        self.kernel_axes,
        self.module_axis,
    )

    nrmse = _nrmse(sharded.astype(jnp.float32), fallback)
    self.assertLess(nrmse, 0.05, f"Sharded vs fallback forward NRMSE {nrmse:.4f} too high")

  def test_forward_vs_bf16(self):
    """Both FP8 paths vs bf16 reference forward: NRMSE < 5%."""
    ref = jax.lax.dot_general(self.lhs, self.rhs, self.dimension_numbers, preferred_element_type=jnp.float32)

    sharded = self._call_provider(
        self._make_sharded_provider(),
        self.lhs,
        self.rhs,
        self.dimension_numbers,
        self.input_axes,
        self.kernel_axes,
        self.module_axis,
    )
    fallback = self._call_provider(
        self._make_fallback_provider(),
        self.lhs,
        self.rhs,
        self.dimension_numbers,
        self.input_axes,
        self.kernel_axes,
        self.module_axis,
    )

    sharded_nrmse = _nrmse(ref, sharded)
    self.assertLess(sharded_nrmse, 0.05, f"Sharded vs bf16 NRMSE {sharded_nrmse:.4f} too high")

    fallback_nrmse = _nrmse(ref, fallback)
    self.assertLess(fallback_nrmse, 0.05, f"Fallback vs bf16 NRMSE {fallback_nrmse:.4f} too high")

  def test_backward_sharded_vs_fallback(self):
    """Shard_map vs fallback backward pass: gradient NRMSE < 10%."""
    sharded_provider = self._make_sharded_provider()
    fallback_provider = self._make_fallback_provider()

    def sharded_fn(l, r):
      return jnp.sum(
          self._call_provider(
              sharded_provider, l, r, self.dimension_numbers, self.input_axes, self.kernel_axes, self.module_axis
          )
      )

    def fallback_fn(l, r):
      return jnp.sum(
          self._call_provider(
              fallback_provider, l, r, self.dimension_numbers, self.input_axes, self.kernel_axes, self.module_axis
          )
      )

    sharded_grads = jax.grad(sharded_fn, argnums=(0, 1))(self.lhs, self.rhs)
    fallback_grads = jax.grad(fallback_fn, argnums=(0, 1))(self.lhs, self.rhs)

    lhs_nrmse = _nrmse(sharded_grads[0].astype(jnp.float32), fallback_grads[0])
    self.assertLess(lhs_nrmse, 0.10, f"LHS grad sharded vs fallback NRMSE {lhs_nrmse:.4f} too high")

    rhs_nrmse = _nrmse(sharded_grads[1].astype(jnp.float32), fallback_grads[1])
    self.assertLess(rhs_nrmse, 0.10, f"RHS grad sharded vs fallback NRMSE {rhs_nrmse:.4f} too high")

  def test_backward_vs_bf16(self):
    """Both FP8 paths' gradients vs bf16 reference: NRMSE < 20%.

    FP8 backward passes quantize gradients too, so error accumulates
    beyond the forward-only 5% threshold. 20% tolerance verifies
    gradients remain in the right ballpark without false failures.
    """

    def ref_fn(l, r):
      return jnp.sum(jax.lax.dot_general(l, r, self.dimension_numbers, preferred_element_type=jnp.float32))

    ref_grads = jax.grad(ref_fn, argnums=(0, 1))(self.lhs, self.rhs)

    sharded_provider = self._make_sharded_provider()
    fallback_provider = self._make_fallback_provider()

    def sharded_fn(l, r):
      return jnp.sum(
          self._call_provider(
              sharded_provider, l, r, self.dimension_numbers, self.input_axes, self.kernel_axes, self.module_axis
          )
      )

    def fallback_fn(l, r):
      return jnp.sum(
          self._call_provider(
              fallback_provider, l, r, self.dimension_numbers, self.input_axes, self.kernel_axes, self.module_axis
          )
      )

    sharded_grads = jax.grad(sharded_fn, argnums=(0, 1))(self.lhs, self.rhs)
    fallback_grads = jax.grad(fallback_fn, argnums=(0, 1))(self.lhs, self.rhs)

    # Sharded vs bf16
    lhs_s = _nrmse(ref_grads[0].astype(jnp.float32), sharded_grads[0])
    self.assertLess(lhs_s, 0.20, f"Sharded LHS grad vs bf16 NRMSE {lhs_s:.4f} too high")
    rhs_s = _nrmse(ref_grads[1].astype(jnp.float32), sharded_grads[1])
    self.assertLess(rhs_s, 0.20, f"Sharded RHS grad vs bf16 NRMSE {rhs_s:.4f} too high")

    # Fallback vs bf16
    lhs_f = _nrmse(ref_grads[0].astype(jnp.float32), fallback_grads[0])
    self.assertLess(lhs_f, 0.20, f"Fallback LHS grad vs bf16 NRMSE {lhs_f:.4f} too high")
    rhs_f = _nrmse(ref_grads[1].astype(jnp.float32), fallback_grads[1])
    self.assertLess(rhs_f, 0.20, f"Fallback RHS grad vs bf16 NRMSE {rhs_f:.4f} too high")


class AlignTileTpuTest(unittest.TestCase):
  """Unit tests for _align_tile_tpu helper."""

  def test_dim_equals_tile(self):
    # tile == dim -> return dim (always valid, even if not 128-aligned)
    self.assertEqual(_align_tile_tpu(64, 512), 64)

  def test_standard_case(self):
    # D=1024, td=512 -> 512 (divides 1024, is 128-aligned)
    self.assertEqual(_align_tile_tpu(1024, 512), 512)

  def test_d640_td512(self):
    # The failing case: D=640, td=512 -> must return 128 (not 320)
    result = _align_tile_tpu(640, 512)
    self.assertEqual(result % 128, 0)
    self.assertEqual(640 % result, 0)
    self.assertEqual(result, 128)

  def test_d896_td512(self):
    # D=896=7*128, td=512 -> 128 (896%256!=0, 896%384!=0, 896%512!=0)
    result = _align_tile_tpu(896, 512)
    self.assertEqual(result % 128, 0)
    self.assertEqual(896 % result, 0)

  def test_d768_td512(self):
    # D=768=6*128, td=512 -> 384 (768%512!=0, 768%384=0, 384%128=0)
    result = _align_tile_tpu(768, 512)
    self.assertEqual(result, 384)

  def test_large_dim(self):
    # D=7168, td=512 -> 512 (7168%512=0)
    self.assertEqual(_align_tile_tpu(7168, 512), 512)


class NonStandardDimTest(unittest.TestCase):
  """Test 1dx1d kernels with non-standard dimensions that trigger alignment."""

  def test_fused_1dx1d_d640(self):
    M, H, D = 4096, 2048, 640
    key1, key2 = jax.random.split(jax.random.PRNGKey(99))
    lhs = jax.random.normal(key1, (M, H), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (M, D), dtype=jnp.bfloat16)
    ref = jnp.matmul(lhs.astype(jnp.float32).T, rhs.astype(jnp.float32))
    result = fused_blockwise_fp8_matmul_1dx1d(lhs, rhs)
    nrmse = _nrmse(ref, result.astype(jnp.float32))
    self.assertLess(nrmse, 0.05, f"D=640 fused NRMSE {nrmse:.4f}")

  def test_fused_1dx1d_h640(self):
    M, H, D = 4096, 640, 2048
    key1, key2 = jax.random.split(jax.random.PRNGKey(100))
    lhs = jax.random.normal(key1, (M, H), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (M, D), dtype=jnp.bfloat16)
    ref = jnp.matmul(lhs.astype(jnp.float32).T, rhs.astype(jnp.float32))
    result = fused_blockwise_fp8_matmul_1dx1d(lhs, rhs)
    nrmse = _nrmse(ref, result.astype(jnp.float32))
    self.assertLess(nrmse, 0.05, f"H=640 fused NRMSE {nrmse:.4f}")

  def test_nonfused_1dx1d_d640(self):
    M, H, D = 4096, 2048, 640
    key1, key2 = jax.random.split(jax.random.PRNGKey(101))
    lhs = jax.random.normal(key1, (M, H), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (M, D), dtype=jnp.bfloat16)
    ref = jnp.matmul(lhs.astype(jnp.float32).T, rhs.astype(jnp.float32))
    qlhs, slhs = quantize_blockwise_1d(lhs, jnp.float8_e4m3fn, axis=0)
    qrhs, srhs = quantize_blockwise_1d(rhs, jnp.float8_e4m3fn, axis=0)
    result = blockwise_fp8_matmul_1dx1d(qlhs, slhs, qrhs, srhs)
    nrmse = _nrmse(ref, result.astype(jnp.float32))
    self.assertLess(nrmse, 0.05, f"D=640 non-fused NRMSE {nrmse:.4f}")


class WgradTest(unittest.TestCase):
  """Tests for wgrad (weight gradient) computation through BlockwiseFp8DotGeneralOp.

  The wgrad is computed using jax.vjp, which invokes the custom_vjp backward
  pass that uses fused_blockwise_fp8_matmul_1dx1d internally.
  """

  def test_wgrad_correctness(self):
    """Verify wgrad (weight gradient) computation via jax.vjp."""
    M, K, N = 4096, 2048, 7168
    key1, key2 = jax.random.split(jax.random.PRNGKey(40))
    lhs = jax.random.normal(key1, (M, K), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (K, N), dtype=jnp.bfloat16)

    dimension_numbers = (((1,), (0,)), ((), ()))

    # Reference weight gradient: lhs^T @ output_grad
    out_grad = jax.random.normal(key1, (M, N), dtype=jnp.bfloat16)
    ref_wgrad = jnp.matmul(lhs.astype(jnp.float32).T, out_grad.astype(jnp.float32))

    # FP8 wgrad via jax.vjp through BlockwiseFp8DotGeneralOp
    op = BlockwiseFp8DotGeneralOp(block_size=128, use_fused=True, cache_rhs=True)
    op.interpret = False

    def fwd_fn(r):
      return jnp.sum(op(lhs, r, dimension_numbers) * out_grad)

    # Compute gradient w.r.t. rhs (weight)
    _, vjp_fn = jax.vjp(fwd_fn, rhs)
    (wgrad_fp8,) = vjp_fn(jnp.array(1.0, dtype=jnp.bfloat16))
    wgrad_fp8 = wgrad_fp8.astype(jnp.float32)

    nrmse = _nrmse(ref_wgrad, wgrad_fp8)
    self.assertLess(nrmse, 0.20, f"Wgrad NRMSE {nrmse:.4f} too high (expected < 0.20)")

  def test_wgrad_cache_rhs_false(self):
    """Verify wgrad with cache_rhs=False."""
    M, K, N = 4096, 2048, 7168
    key1, key2 = jax.random.split(jax.random.PRNGKey(41))
    lhs = jax.random.normal(key1, (M, K), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (K, N), dtype=jnp.bfloat16)

    dimension_numbers = (((1,), (0,)), ((), ()))
    out_grad = jax.random.normal(key1, (M, N), dtype=jnp.bfloat16)

    ref_wgrad = jnp.matmul(lhs.astype(jnp.float32).T, out_grad.astype(jnp.float32))

    # FP8 wgrad with cache_rhs=False
    op = BlockwiseFp8DotGeneralOp(block_size=128, use_fused=True, cache_rhs=False)
    op.interpret = False

    def fwd_fn(r):
      return jnp.sum(op(lhs, r, dimension_numbers) * out_grad)

    _, vjp_fn = jax.vjp(fwd_fn, rhs)
    (wgrad_fp8,) = vjp_fn(jnp.array(1.0, dtype=jnp.bfloat16))
    wgrad_fp8 = wgrad_fp8.astype(jnp.float32)

    nrmse = _nrmse(ref_wgrad, wgrad_fp8)
    self.assertLess(nrmse, 0.20, f"Wgrad (cache_rhs=False) NRMSE {nrmse:.4f} too high")

  def test_wgrad_vs_bf16_reference(self):
    """Compare FP8 wgrad against bf16 reference gradient."""
    M, K, N = 2048, 1024, 4096
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(42), 3)
    lhs = jax.random.normal(key1, (M, K), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (K, N), dtype=jnp.bfloat16)
    out_grad = jax.random.normal(key3, (M, N), dtype=jnp.bfloat16)

    dimension_numbers = (((1,), (0,)), ((), ()))

    # Reference gradient using bf16 dot_general
    def ref_fn(l, r):
      return jnp.sum(jax.lax.dot_general(l, r, dimension_numbers, preferred_element_type=jnp.float32) * out_grad)

    ref_grads = jax.grad(ref_fn, argnums=(0, 1))(lhs, rhs)
    ref_wgrad = ref_grads[1].astype(jnp.float32)

    # FP8 gradient
    op = BlockwiseFp8DotGeneralOp(block_size=128, use_fused=True, cache_rhs=True)
    op.interpret = False

    def fp8_fn(l, r):
      return jnp.sum(op(l, r, dimension_numbers) * out_grad)

    fp8_grads = jax.grad(fp8_fn, argnums=(0, 1))(lhs, rhs)
    fp8_wgrad = fp8_grads[1].astype(jnp.float32)

    nrmse = _nrmse(ref_wgrad, fp8_wgrad)
    self.assertLess(nrmse, 0.20, f"FP8 wgrad vs bf16 reference NRMSE {nrmse:.4f} too high")

  def test_wgrad_padding(self):
    """Test wgrad with non-128-aligned dimensions."""
    M, K, N = 4001, 2003, 7111  # Prime numbers, not divisible by 128
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(43), 3)
    lhs = jax.random.normal(key1, (M, K), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (K, N), dtype=jnp.bfloat16)
    out_grad = jax.random.normal(key3, (M, N), dtype=jnp.bfloat16)

    dimension_numbers = (((1,), (0,)), ((), ()))

    ref_wgrad = jnp.matmul(lhs.astype(jnp.float32).T, out_grad.astype(jnp.float32))

    op = BlockwiseFp8DotGeneralOp(block_size=128, use_fused=True, cache_rhs=True)
    op.interpret = False

    def fwd_fn(r):
      return jnp.sum(op(lhs, r, dimension_numbers) * out_grad)

    _, vjp_fn = jax.vjp(fwd_fn, rhs)
    (wgrad_fp8,) = vjp_fn(jnp.array(1.0, dtype=jnp.bfloat16))
    wgrad_fp8 = wgrad_fp8.astype(jnp.float32)

    self.assertEqual(wgrad_fp8.shape, ref_wgrad.shape, f"Wgrad shape mismatch: {wgrad_fp8.shape} vs {ref_wgrad.shape}")

    nrmse = _nrmse(ref_wgrad, wgrad_fp8)
    self.assertLess(nrmse, 0.20, f"Wgrad with padding NRMSE {nrmse:.4f} too high")

  def test_wgrad_multi_contract_dim(self):
    """Test wgrad with batch dimensions."""
    B, M, K, N = 2, 4096, 2048, 7168
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(44), 3)
    lhs = jax.random.normal(key1, (B, M, K), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (B, K, N), dtype=jnp.bfloat16)  # rhs also has batch dim
    out_grad = jax.random.normal(key3, (B, M, N), dtype=jnp.bfloat16)

    # Batch on dim 0 for both, contract K on dim 2 of lhs and dim 1 of rhs
    # lhs: (B, M, K), rhs: (B, K, N) -> output: (B, M, N)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))

    def ref_fn(l, r):
      result = jax.lax.dot_general(l, r, dimension_numbers, preferred_element_type=jnp.float32)
      return jnp.sum(result * out_grad.astype(jnp.float32))

    ref_grads = jax.grad(ref_fn, argnums=(0, 1))(lhs, rhs)
    ref_wgrad = ref_grads[1].astype(jnp.float32)

    op = BlockwiseFp8DotGeneralOp(block_size=128, use_fused=True, cache_rhs=True)
    op.interpret = False

    def fp8_fn(l, r):
      return jnp.sum(op(l, r, dimension_numbers) * out_grad)

    fp8_grads = jax.grad(fp8_fn, argnums=(0, 1))(lhs, rhs)
    fp8_wgrad = fp8_grads[1].astype(jnp.float32)

    nrmse = _nrmse(ref_wgrad, fp8_wgrad)
    self.assertLess(nrmse, 0.20, f"Wgrad multi-contract dim NRMSE {nrmse:.4f} too high")


if __name__ == "__main__":
  unittest.main()
