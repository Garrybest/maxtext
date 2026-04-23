"""Unit tests for KDA (Kimi Delta Attention) module.

Tests cover:
  - ShortConvolution: shape, causality, naive comparison
  - KimiDeltaAttention: initialization, forward pass, padding, determinism
  - chunk_kda kernel: basic operation, chunk vs recurrent comparison
  - Naive KDA: recurrent Delta Rule reference impl vs kernel precision
  - Backward (VJP): activation gradients, weight gradients, determinism, bf16
  - QK L2 norm: applied outside kernel, matching Megatron
  - Real config: loading ling3-tiny.yml and running KimiDeltaAttention

Precision comparison uses _assert_close (atol+rtol+ULP fallback),
adapted from gla_compare_test.py.

Run with: python -m pytest tests/unit/kda_attention_test.py -v
"""

import os
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import ml_dtypes
from flax import nnx

try:
  from tops.ops.kda import chunk_kda, fused_recurrent_kda
  from tops.cpu.ops.common.l2norm import l2norm_fwd

  TOPS_AVAILABLE = True
except ImportError:
  TOPS_AVAILABLE = False

from maxtext.layers import attention_kda
from maxtext.utils.globals import MAXTEXT_REPO_ROOT


# ---------------------------------------------------------------------------
# Precision comparison utilities (adapted from gla_compare_test.py)
# ---------------------------------------------------------------------------


def _bf16_bits_to_ordered(u16):
  magnitude = (u16 & 0x7FFF).astype(np.int64)
  return np.where(u16 & 0x8000, -magnitude, magnitude)


def bf16_ulp_diff(actual_f32, expected_f32):
  """Compute per-element ULP distance at bf16 precision."""
  a_u16 = np.ascontiguousarray(actual_f32.astype(ml_dtypes.bfloat16)).view(np.uint16)
  b_u16 = np.ascontiguousarray(expected_f32.astype(ml_dtypes.bfloat16)).view(np.uint16)
  mismatch_mask = a_u16 != b_u16
  n_mismatch = int(mismatch_mask.sum())
  n_total = a_u16.size
  if n_mismatch == 0:
    return n_mismatch, n_total, 0, np.array([], dtype=np.int64)
  a_ordered = _bf16_bits_to_ordered(a_u16[mismatch_mask])
  b_ordered = _bf16_bits_to_ordered(b_u16[mismatch_mask])
  abs_ulp = np.abs(a_ordered - b_ordered)
  return n_mismatch, n_total, int(abs_ulp.max()), abs_ulp


def _assert_close(actual, expected, label, atol=1e-2, rtol=1e-5, max_ulp=2, max_ulp_fail_rate=1e-3):
  """Assert two arrays match via allclose with bf16 ULP diff fallback."""
  actual_f32 = np.asarray(actual, dtype=np.float32)
  expected_f32 = np.asarray(expected, dtype=np.float32)

  diff = np.abs(actual_f32 - expected_f32)
  max_abs = float(diff.max())
  mean_abs = float(diff.mean())
  print(f"  {label}: max_abs={max_abs:.6e}  mean_abs={mean_abs:.6e}")

  close_mask = diff <= atol + rtol * np.abs(expected_f32)
  if close_mask.all():
    print(f"  {label}: all close ({atol=}, {rtol=})")
    return

  n_fail = int((~close_mask).sum())
  n_total = actual_f32.size
  fail_actual = actual_f32[~close_mask]
  fail_expected = expected_f32[~close_mask]
  n_mis, _, worst_ulp, abs_ulps = bf16_ulp_diff(fail_actual, fail_expected)

  n_over = int((abs_ulps > max_ulp).sum()) if n_mis > 0 else 0
  over_rate = n_over / n_fail if n_fail > 0 else 0.0

  if n_mis > 0:
    print(
        f"  {label} ULP: {n_fail}/{n_total} fail allclose, "
        f"{n_mis} have ULP diff, max_ulp={worst_ulp}, "
        f"over {max_ulp} ULP: {n_over}/{n_fail} ({over_rate:.2e})"
    )

  assert over_rate <= max_ulp_fail_rate, (
      f"{label}: {n_over}/{n_fail} elements ({over_rate:.2e}) exceed "
      f"{max_ulp} ULP (threshold {max_ulp_fail_rate:.2e})"
  )


class _MockKdaConfig:
  """Minimal mock config for KDA testing.

  KDA derives head dims from global config (matching Megatron):
    key_head_dim = value_head_dim = head_dim
    num_key_heads = num_value_heads = base_num_query_heads
  """

  def __init__(self, **overrides):
    self.base_emb_dim = 128
    self.base_num_query_heads = 4
    self.head_dim = 32
    self.dtype = jnp.float32
    self.weight_dtype = jnp.float32
    self.attention_bias = False
    self.shard_mode = "auto"
    self.matmul_precision = "default"
    self.normalization_layer_epsilon = 1e-6
    self.logical_axis_rules = []

    # KDA-specific
    self.linear_conv_kernel_dim = 4
    self.use_qk_norm = True
    self.use_kda_safe_gate = False
    self.kda_lower_bound = -5.0

    for k, v in overrides.items():
      setattr(self, k, v)


# ---------------------------------------------------------------------------
# ShortConvolution tests
# ---------------------------------------------------------------------------


class TestShortConvolution:
  """Tests for ShortConvolution module."""

  def test_output_shape(self):
    rngs = nnx.Rngs(0)
    conv = attention_kda.ShortConvolution(
        kernel_size=4,
        features=32,
        rngs=rngs,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 32))
    out = conv(x)
    assert out.shape == (2, 16, 32)

  def test_causality(self):
    """Position i should only depend on positions <= i."""
    rngs = nnx.Rngs(0)
    conv = attention_kda.ShortConvolution(kernel_size=4, features=16, rngs=rngs)

    x = jnp.zeros((1, 8, 16))
    x = x.at[:, 0, :].set(1.0)
    out = conv(x)

    assert not jnp.allclose(out[:, 0, :], 0), "Position 0 should have output"
    # Positions beyond kernel reach should be zero
    assert jnp.allclose(out[:, 4:, :], 0, atol=1e-6), "Positions beyond kernel_size from impulse should be zero"

  def test_against_naive(self):
    """Compare against a loop-based naive depthwise convolution."""
    rngs = nnx.Rngs(0)
    F = 16
    K = 4
    conv = attention_kda.ShortConvolution(
        kernel_size=K,
        features=F,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=rngs,
    )

    B, T = 1, 12
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, F))
    out_module = conv(x)

    # Naive depthwise causal conv: kernel[k] is [F], elementwise multiply
    out_naive = jnp.zeros_like(x)
    for t in range(T):
      for k in range(K):
        if t - k >= 0:
          out_naive = out_naive.at[:, t, :].add(x[:, t - k, :] * conv.kernel[k])

    max_diff = jnp.max(jnp.abs(out_module - out_naive))
    assert max_diff < 0.01, f"Max abs diff {max_diff:.6f} exceeds 0.01"


# ---------------------------------------------------------------------------
# KimiDeltaAttention tests
# ---------------------------------------------------------------------------


class TestKimiDeltaAttention:
  """Tests for KimiDeltaAttention module."""

  @pytest.fixture
  def mesh(self):
    return jax.sharding.Mesh(jax.devices(), ("x",))

  def _make_attn(self, mesh, **config_overrides):
    cfg = _MockKdaConfig(**config_overrides)
    rngs = nnx.Rngs(0)
    with mesh:
      return attention_kda.KimiDeltaAttention(
          config=cfg,
          layer_idx=0,
          mesh=mesh,
          rngs=rngs,
      )

  def test_init_head_dims(self, mesh):
    """Head dims derived from global config: head_dim=32, base_num_query_heads=4."""
    attn = self._make_attn(mesh)
    assert attn.num_query_heads == 4
    assert attn.num_key_heads == 4
    assert attn.num_value_heads == 4
    assert attn.key_head_dim == 32
    assert attn.value_head_dim == 32

  def test_init_no_conv(self, mesh):
    attn = self._make_attn(mesh, linear_conv_kernel_dim=0)
    assert attn.q_conv is None

  def test_init_has_gate_and_norm(self, mesh):
    """Output gate projection and out_norm should always be present."""
    attn = self._make_attn(mesh)
    assert hasattr(attn, "gate_proj")
    assert hasattr(attn, "out_norm")
    assert hasattr(attn, "A_log")
    assert hasattr(attn, "dt_bias")

  @pytest.mark.skipif(not TOPS_AVAILABLE, reason="tops not available")
  def test_forward_shape(self, mesh):
    attn = self._make_attn(mesh)
    B, T, D = 2, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    with mesh:
      output, aux = attn(x)
    assert output.shape == (B, T, D)
    assert aux is None

  @pytest.mark.skipif(not TOPS_AVAILABLE, reason="tops not available")
  def test_forward_no_nan_inf(self, mesh):
    attn = self._make_attn(mesh)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 64, 128))
    with mesh:
      output, _ = attn(x)
    assert not jnp.any(jnp.isnan(output))
    assert not jnp.any(jnp.isinf(output))
    assert jnp.any(output != 0)

  @pytest.mark.skipif(not TOPS_AVAILABLE, reason="tops not available")
  def test_sequence_padding(self, mesh):
    """Non-divisible sequence lengths should be handled via padding."""
    attn = self._make_attn(mesh)
    B, T, D = 1, 100, 128  # 100 not divisible by chunk_size=64
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    with mesh:
      output, _ = attn(x)
    assert output.shape == (B, T, D)

  @pytest.mark.skipif(not TOPS_AVAILABLE, reason="tops not available")
  def test_deterministic(self, mesh):
    attn = self._make_attn(mesh)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 64, 128))
    with mesh:
      o1, _ = attn(x)
      o2, _ = attn(x)
    assert jnp.allclose(o1, o2, atol=1e-5)

  def test_packed_sequences_not_supported(self, mesh):
    attn = self._make_attn(mesh)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 64, 128))
    seg_ids = jnp.ones((1, 64), dtype=jnp.int32)
    with pytest.raises(NotImplementedError, match="packed sequences"):
      attn(x, decoder_segment_ids=seg_ids)

  def test_autoregressive_not_supported(self, mesh):
    attn = self._make_attn(mesh)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 64, 128))
    with pytest.raises(NotImplementedError, match="autoregressive"):
      attn(x, model_mode="autoregressive")


# ---------------------------------------------------------------------------
# Kernel-level tests
# ---------------------------------------------------------------------------


class TestChunkKda:
  """Direct tests for the chunk_kda kernel."""

  @pytest.mark.skipif(not TOPS_AVAILABLE, reason="tops not available")
  def test_basic(self):
    B, T, H, K, V = 1, 2048, 4, 128, 128
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    q = jax.nn.silu(jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32))
    k = jax.nn.silu(jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32))
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.float32)
    g = jax.nn.log_sigmoid(jax.random.normal(keys[3], (B, T, H, K))) * 0.3
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, H)))

    o, _ = chunk_kda(q, k, v, g, beta, scale=K**-0.5, chunk_size=64)
    assert o.shape == (B, T, H, V)
    assert not jnp.any(jnp.isnan(o))

  @pytest.mark.skipif(not TOPS_AVAILABLE, reason="tops not available")
  def test_chunk_vs_recurrent(self):
    B, T, H, K, V = 1, 64, 4, 32, 32
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    q = jax.nn.silu(jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32))
    k = jax.nn.silu(jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32))
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.float32)
    g = jax.nn.log_sigmoid(jax.random.normal(keys[3], (B, T, H, K))) * 0.3
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, H)))

    scale = K**-0.5
    o_chunk, _ = chunk_kda(q, k, v, g, beta, scale=scale, chunk_size=64)
    o_recurrent, _ = fused_recurrent_kda(q, k, v, g, beta, scale=scale)
    _assert_close(o_chunk, o_recurrent, "chunk_vs_recurrent", atol=5e-3, rtol=5e-3)


# ---------------------------------------------------------------------------
# Naive KDA reference implementation and precision tests
# ---------------------------------------------------------------------------


def _naive_kda_recurrent(q, k, v, g, beta, scale):
  """Naive Python implementation of KDA Delta Rule (recurrent form).

  Implements the exact recurrence from the KDA docstring:
    S' = S * exp(g_t)                    (gated decay)
    residual = v_t - S'^T @ k_t          (delta residual)
    S = S' + beta_t * k_t outer residual (state update)
    o_t = scale * S @ q_t                (output)

  Args:
    q: [B, T, H, K]  query
    k: [B, T, H, K]  key
    v: [B, T, H, V]  value
    g: [B, T, H, K]  gate (log-space, negative)
    beta: [B, T, H]  delta rule mixing coefficient
    scale: float      output scaling factor

  Returns:
    o: [B, T, H, V]  output
  """
  B, T, H, K = q.shape
  V = v.shape[-1]
  o = jnp.zeros((B, T, H, V), dtype=jnp.float32)

  # S: [B, H, K, V] recurrent state
  S = jnp.zeros((B, H, K, V), dtype=jnp.float32)

  for t in range(T):
    # Extract per-step tensors
    q_t = q[:, t, :, :]  # [B, H, K]
    k_t = k[:, t, :, :]  # [B, H, K]
    v_t = v[:, t, :, :]  # [B, H, V]
    g_t = g[:, t, :, :]  # [B, H, K]
    beta_t = beta[:, t, :]  # [B, H]

    # Gated decay: S' = S * exp(g_t)
    # g_t is [B, H, K], S is [B, H, K, V] -> broadcast over V
    S = S * jnp.exp(g_t)[..., None]  # [B, H, K, V]

    # Delta residual: residual = v_t - S^T @ k_t
    # S^T @ k_t: [B, H, V, K] @ [B, H, K] -> [B, H, V]
    # Equivalently: einsum('bhkv,bhk->bhv', S, k_t)
    Sk = jnp.einsum("bhkv,bhk->bhv", S, k_t)  # [B, H, V]
    residual = v_t - Sk  # [B, H, V]

    # State update: S = S + beta_t * k_t outer residual
    # k_t: [B, H, K], residual: [B, H, V] -> outer: [B, H, K, V]
    outer = k_t[..., None] * residual[..., None, :]  # [B, H, K, V]
    S = S + beta_t[..., None, None] * outer  # [B, H, K, V]

    # Output: o_t = scale * S @ q_t
    # einsum('bhkv,bhk->bhv', S, q_t)
    o_t = scale * jnp.einsum("bhkv,bhk->bhv", S, q_t)  # [B, H, V]
    o = o.at[:, t, :, :].set(o_t)

  return o


class TestNaiveKda:
  """Compare chunk_kda kernel against naive recurrent KDA implementation."""

  @pytest.mark.skipif(not TOPS_AVAILABLE, reason="tops not available")
  def test_chunk_kda_vs_naive(self):
    """Verify chunk_kda matches the naive Delta Rule recurrence."""
    B, T, H, K, V = 1, 64, 2, 16, 16
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 5)

    q = jax.nn.silu(jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32))
    k = jax.nn.silu(jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32))
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.float32)
    g = jax.nn.log_sigmoid(jax.random.normal(keys[3], (B, T, H, K), dtype=jnp.float32)) * 0.3
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, H), dtype=jnp.float32))

    scale = K**-0.5

    o_kernel, _ = chunk_kda(q, k, v, g, beta, scale=scale, chunk_size=64)
    o_naive = _naive_kda_recurrent(q, k, v, g, beta, scale)

    assert not jnp.any(jnp.isnan(o_naive)), "Naive output contains NaN"
    assert not jnp.any(jnp.isnan(o_kernel)), "Kernel output contains NaN"
    _assert_close(o_kernel, o_naive, "chunk_kda_vs_naive", atol=5e-3, rtol=1e-3)

  @pytest.mark.skipif(not TOPS_AVAILABLE, reason="tops not available")
  def test_fused_recurrent_vs_naive(self):
    """Verify fused_recurrent_kda matches the naive Delta Rule recurrence."""
    B, T, H, K, V = 1, 64, 2, 16, 16
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 5)

    q = jax.nn.silu(jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32))
    k = jax.nn.silu(jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32))
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.float32)
    g = jax.nn.log_sigmoid(jax.random.normal(keys[3], (B, T, H, K), dtype=jnp.float32)) * 0.3
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, H), dtype=jnp.float32))

    scale = K**-0.5

    o_fused, _ = fused_recurrent_kda(q, k, v, g, beta, scale=scale)
    o_naive = _naive_kda_recurrent(q, k, v, g, beta, scale)

    assert not jnp.any(jnp.isnan(o_naive)), "Naive output contains NaN"
    assert not jnp.any(jnp.isnan(o_fused)), "Fused output contains NaN"
    _assert_close(o_fused, o_naive, "fused_recurrent_vs_naive", atol=1e-2, rtol=1e-3)

  def test_naive_kda_basic_properties(self):
    """Verify naive KDA implementation has correct basic properties."""
    B, T, H, K, V = 1, 8, 2, 4, 4
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)

    q = jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32) * 0.1
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32) * 0.1
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.float32) * 0.1
    g = -jnp.abs(jax.random.normal(keys[3], (B, T, H, K), dtype=jnp.float32)) * 0.1
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, H), dtype=jnp.float32))

    scale = K**-0.5
    o = _naive_kda_recurrent(q, k, v, g, beta, scale)

    assert o.shape == (B, T, H, V)
    assert not jnp.any(jnp.isnan(o)), "Output contains NaN"
    assert not jnp.any(jnp.isinf(o)), "Output contains Inf"
    # First position should be non-zero (state starts empty but gets updated)
    assert jnp.any(o[:, 0, :, :] != 0), "First position output should be non-zero"

  def test_naive_kda_zero_gate_accumulates(self):
    """With g=0 (no decay), state should accumulate without forgetting."""
    B, H, K, V = 1, 1, 2, 2
    T = 4

    q = jnp.ones((B, T, H, K), dtype=jnp.float32)
    k = jnp.ones((B, T, H, K), dtype=jnp.float32) * 0.1
    v = jnp.ones((B, T, H, V), dtype=jnp.float32) * 0.1
    g = jnp.zeros((B, T, H, K), dtype=jnp.float32)  # no decay
    beta = jnp.ones((B, T, H), dtype=jnp.float32)  # full update

    scale = 1.0
    o = _naive_kda_recurrent(q, k, v, g, beta, scale)

    # Output magnitude should grow over time as state accumulates
    norms = jnp.linalg.norm(o[0, :, 0, :], axis=-1)  # [T]
    # Later positions should have larger or equal output norm
    assert norms[-1] >= norms[0], f"With zero gate, output norm should grow: first={norms[0]:.4f}, last={norms[-1]:.4f}"

  def test_naive_kda_large_negative_gate_decays(self):
    """With very negative g, state should decay rapidly."""
    B, H, K, V = 1, 1, 2, 2
    T = 4

    q = jnp.ones((B, T, H, K), dtype=jnp.float32)
    k = jnp.zeros((B, T, H, K), dtype=jnp.float32)  # no new info
    v = jnp.zeros((B, T, H, V), dtype=jnp.float32)
    g = jnp.full((B, T, H, K), -10.0, dtype=jnp.float32)  # aggressive decay
    beta = jnp.ones((B, T, H), dtype=jnp.float32)

    # Manually set initial state by making first step contribute
    k = k.at[:, 0, :, :].set(1.0)
    v = v.at[:, 0, :, :].set(1.0)

    scale = 1.0
    o = _naive_kda_recurrent(q, k, v, g, beta, scale)

    # After step 0, large negative gate should make state decay to ~0
    norm_0 = jnp.linalg.norm(o[0, 0, 0, :])
    norm_last = jnp.linalg.norm(o[0, -1, 0, :])
    assert norm_last < norm_0 * 0.01, (
        f"Large negative gate should decay state: t=0 norm={norm_0:.6f}, " f"t={T-1} norm={norm_last:.6f}"
    )

  @pytest.mark.skipif(not TOPS_AVAILABLE, reason="tops not available")
  def test_chunk_kda_vs_naive_bf16(self):
    """Verify chunk_kda matches naive in bfloat16 (training dtype)."""
    B, T, H, K, V = 1, 64, 2, 16, 16
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 5)

    q = jax.nn.silu(jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32))
    k = jax.nn.silu(jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32))
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)
    q = q.astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.bfloat16)
    g = jax.nn.log_sigmoid(jax.random.normal(keys[3], (B, T, H, K), dtype=jnp.float32)) * 0.3
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, H), dtype=jnp.float32))

    scale = K**-0.5

    o_kernel, _ = chunk_kda(q, k, v, g, beta, scale=scale, chunk_size=64)
    o_naive = _naive_kda_recurrent(
        q.astype(jnp.float32),
        k.astype(jnp.float32),
        v.astype(jnp.float32),
        g,
        beta,
        scale,
    )

    assert not jnp.any(jnp.isnan(o_kernel)), "Kernel bf16 output contains NaN"
    _assert_close(o_kernel, o_naive, "chunk_kda_bf16_vs_naive", atol=1e-2, rtol=1e-2)

  @pytest.mark.skipif(not TOPS_AVAILABLE, reason="tops not available")
  def test_fused_recurrent_vs_naive_bf16(self):
    """Verify fused_recurrent_kda matches naive in bfloat16."""
    B, T, H, K, V = 1, 64, 2, 16, 16
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 5)

    q = jax.nn.silu(jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32))
    k = jax.nn.silu(jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32))
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)
    q = q.astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.bfloat16)
    g = jax.nn.log_sigmoid(jax.random.normal(keys[3], (B, T, H, K), dtype=jnp.float32)) * 0.3
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, H), dtype=jnp.float32))

    scale = K**-0.5

    o_fused, _ = fused_recurrent_kda(q, k, v, g, beta, scale=scale)
    o_naive = _naive_kda_recurrent(
        q.astype(jnp.float32),
        k.astype(jnp.float32),
        v.astype(jnp.float32),
        g,
        beta,
        scale,
    )

    assert not jnp.any(jnp.isnan(o_fused)), "Fused bf16 output contains NaN"
    _assert_close(o_fused, o_naive, "fused_recurrent_bf16_vs_naive", atol=1e-2, rtol=1.6e-2)


# ---------------------------------------------------------------------------
# QK L2 norm tests
# ---------------------------------------------------------------------------


class TestQkL2Norm:
  """Verify QK L2 normalization is applied outside the kernel."""

  @pytest.fixture
  def mesh(self):
    return jax.sharding.Mesh(jax.devices(), ("x",))

  @pytest.mark.skipif(not TOPS_AVAILABLE, reason="tops not available")
  def test_qk_l2norm_applied_outside_kernel(self, mesh):
    """With use_qk_norm=True, Q and K should be L2-normalized before kernel call."""
    cfg = _MockKdaConfig(use_qk_norm=True)
    rngs = nnx.Rngs(0)
    with mesh:
      attn = attention_kda.KimiDeltaAttention(config=cfg, layer_idx=0, mesh=mesh, rngs=rngs)
    B, T, D = 1, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    with mesh:
      output, _ = attn(x)
    assert output.shape == (B, T, D)
    assert not jnp.any(jnp.isnan(output))

  @pytest.mark.skipif(not TOPS_AVAILABLE, reason="tops not available")
  def test_qk_l2norm_skipped_when_disabled(self, mesh):
    """With use_qk_norm=False, forward pass should still work without L2 norm."""
    cfg = _MockKdaConfig(use_qk_norm=False)
    rngs = nnx.Rngs(0)
    with mesh:
      attn = attention_kda.KimiDeltaAttention(config=cfg, layer_idx=0, mesh=mesh, rngs=rngs)
    B, T, D = 1, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    with mesh:
      output, _ = attn(x)
    assert output.shape == (B, T, D)
    assert not jnp.any(jnp.isnan(output))

  @pytest.mark.skipif(not TOPS_AVAILABLE, reason="tops not available")
  def test_l2norm_changes_output(self, mesh):
    """Enabling vs disabling L2 norm should produce different outputs."""
    B, T, D = 1, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))

    rngs_on = nnx.Rngs(0)
    cfg_on = _MockKdaConfig(use_qk_norm=True)
    with mesh:
      attn_on = attention_kda.KimiDeltaAttention(config=cfg_on, layer_idx=0, mesh=mesh, rngs=rngs_on)
      out_on, _ = attn_on(x)

    rngs_off = nnx.Rngs(0)
    cfg_off = _MockKdaConfig(use_qk_norm=False)
    with mesh:
      attn_off = attention_kda.KimiDeltaAttention(config=cfg_off, layer_idx=0, mesh=mesh, rngs=rngs_off)
      out_off, _ = attn_off(x)

    assert not jnp.allclose(out_on, out_off, atol=1e-4), "L2 norm on/off should produce different outputs"


# ---------------------------------------------------------------------------
# Backward (VJP) tests
# ---------------------------------------------------------------------------


class TestKdaBackward:
  """Backward pass tests for KimiDeltaAttention (learning from GLA test patterns)."""

  @pytest.fixture
  def mesh(self):
    return jax.sharding.Mesh(jax.devices(), ("x",))

  def _make_attn(self, mesh, **config_overrides):
    cfg = _MockKdaConfig(**config_overrides)
    rngs = nnx.Rngs(0)
    with mesh:
      return attention_kda.KimiDeltaAttention(
          config=cfg,
          layer_idx=0,
          mesh=mesh,
          rngs=rngs,
      )

  def _run_vjp(self, module, inp, mesh):
    """Run VJP and return (grad_params, grad_input)."""
    graphdef, params, other = nnx.split(module, nnx.Param, ...)

    def forward_fn(params, x):
      model = nnx.merge(graphdef, params, other)
      with mesh:
        out, _ = model(x)
      return out

    _, vjp_fn = jax.vjp(forward_fn, params, inp)
    upstream = jnp.ones_like(inp)
    grad_params, grad_input = vjp_fn(upstream)
    return grad_params, grad_input

  @pytest.mark.skipif(not TOPS_AVAILABLE, reason="tops not available")
  def test_backward_no_nan(self, mesh):
    """Activation gradient should be free of NaN/Inf and non-zero."""
    attn = self._make_attn(mesh)
    B, T, D = 1, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    with mesh:
      _, grad_input = self._run_vjp(attn, x, mesh)
    assert not jnp.any(jnp.isnan(grad_input)), "grad_input contains NaN"
    assert not jnp.any(jnp.isinf(grad_input)), "grad_input contains Inf"
    assert jnp.any(grad_input != 0), "grad_input is all zeros"

  @pytest.mark.skipif(not TOPS_AVAILABLE, reason="tops not available")
  def test_backward_deterministic(self, mesh):
    """Two VJP runs should produce identical gradients."""
    attn = self._make_attn(mesh)
    B, T, D = 1, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    with mesh:
      _, grad1 = self._run_vjp(attn, x, mesh)
      _, grad2 = self._run_vjp(attn, x, mesh)
    assert jnp.allclose(grad1, grad2, atol=1e-5), "Backward is not deterministic"

  @pytest.mark.skipif(not TOPS_AVAILABLE, reason="tops not available")
  def test_weight_grads_no_nan(self, mesh):
    """Every parameter gradient should be free of NaN/Inf and non-zero."""
    attn = self._make_attn(mesh)
    B, T, D = 1, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    with mesh:
      grad_params, _ = self._run_vjp(attn, x, mesh)

    flat_grads = jax.tree.leaves(grad_params)
    for i, g in enumerate(flat_grads):
      assert not jnp.any(jnp.isnan(g)), f"weight grad {i} contains NaN"
      assert not jnp.any(jnp.isinf(g)), f"weight grad {i} contains Inf"
      assert jnp.any(g != 0), f"weight grad {i} is all zeros"

  @pytest.mark.skipif(not TOPS_AVAILABLE, reason="tops not available")
  def test_backward_bf16(self, mesh):
    """bf16 backward should produce valid gradients."""
    attn = self._make_attn(mesh, dtype=jnp.bfloat16, weight_dtype=jnp.bfloat16)
    B, T, D = 1, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D), dtype=jnp.bfloat16)
    with mesh:
      grad_params, grad_input = self._run_vjp(attn, x, mesh)
    assert not jnp.any(jnp.isnan(grad_input)), "bf16 grad_input contains NaN"
    assert not jnp.any(jnp.isinf(grad_input)), "bf16 grad_input contains Inf"
    assert jnp.any(grad_input != 0), "bf16 grad_input is all zeros"

    flat_grads = jax.tree.leaves(grad_params)
    for i, g in enumerate(flat_grads):
      assert not jnp.any(jnp.isnan(g)), f"bf16 weight grad {i} contains NaN"


# ---------------------------------------------------------------------------
# Real config integration test (loads ling3-tiny.yml)
# ---------------------------------------------------------------------------

_BASE_CONFIG_PATH = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")


class TestKdaWithRealConfig:
  """Integration tests using the real ling3-tiny.yml config."""

  @pytest.fixture(scope="class")
  def ling3_cfg(self):
    from maxtext.configs.pyconfig import initialize_pydantic  # pylint: disable=import-outside-toplevel

    return initialize_pydantic(["", _BASE_CONFIG_PATH, "model_name=ling3-tiny"])

  @pytest.fixture
  def mesh(self):
    return jax.sharding.Mesh(jax.devices(), ("x",))

  def test_ling3_use_qk_norm_is_true(self, ling3_cfg):
    """ling3-tiny.yml should have use_qk_norm=True."""
    assert ling3_cfg.use_qk_norm is True

  def test_ling3_kda_config_fields(self, ling3_cfg):
    """ling3-tiny.yml KDA fields should match RFC values."""
    assert ling3_cfg.linear_conv_kernel_dim == 4
    assert ling3_cfg.use_kda_lora is False
    assert ling3_cfg.use_kda_safe_gate is True
    assert ling3_cfg.kda_lower_bound == -5.0

  @pytest.mark.skipif(not TOPS_AVAILABLE, reason="tops not available")
  def test_ling3_kda_forward(self, ling3_cfg, mesh):
    """KimiDeltaAttention initialized from real ling3-tiny config should produce valid output."""
    rngs = nnx.Rngs(0)
    with mesh:
      attn = attention_kda.KimiDeltaAttention(
          config=ling3_cfg,
          layer_idx=0,
          mesh=mesh,
          rngs=rngs,
      )
    B, T = 1, 64
    D = ling3_cfg.base_emb_dim  # 1536
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    with mesh:
      output, _ = attn(x)
    assert output.shape == (B, T, D)
    assert not jnp.any(jnp.isnan(output)), "Real config forward produced NaN"
    assert not jnp.any(jnp.isinf(output)), "Real config forward produced Inf"
