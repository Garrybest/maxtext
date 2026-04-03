#!/usr/bin/env python3
"""End-to-end test for BailingMoeV2LinearAttention against Megatron dump.

Instantiates the BailingMoeV2LinearAttention module with checkpoint weights,
runs forward and backward passes with Megatron input, and verifies:
  1. Forward output matches Megatron reference within bf16 tolerance.
  2. Backward activation gradient matches Megatron reference (allclose).
  3. Backward weight gradients match Megatron reference (allclose + ULP diff).

Environment variables:
  DUMP_DIR  (required) - Megatron dump base directory
  CKPT_PATH (optional) - Orbax checkpoint path

Run directly:
  DUMP_DIR=/path/to/dump python3 tests/unit/gla_compare_test.py -v
  or
  DUMP_DIR=/path/to/dump python3 tests/unit/gla_compare_test.py GlaCompareTest.test_forward_bf16 -v

Or via unittest:
  DUMP_DIR=/path/to/dump python3 -m unittest tests.unit.gla_compare_test -v
"""

import os
import types
import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import numpy as np
import ml_dtypes
import orbax.checkpoint as ocp

from argus import list_tensors, load_tensor

from maxtext.common.common_types import MODEL_MODE_TRAIN, ShardMode
from maxtext.layers.attention_gla import BailingMoeV2LinearAttention

# -- Defaults -----------------------------------------------------------------
DEFAULT_CKPT_PATH = "/models/gpu-ckpt-ling2.5/AL_MODEL_HF20E256_ORBAX_MTP/0/items/"

# Model hyperparameters (from al_model.yml)
NUM_HEADS = 16
NUM_KV_HEADS = 16
HEAD_DIM = 128
EMB_DIM = 2048
NUM_LAYERS = 20
GROUP_NORM_SIZE = 4
PARTIAL_ROTARY_FACTOR = 0.5
EPSILON = 1.0e-6
ROPE_MIN_TIMESCALE = 1
ROPE_MAX_TIMESCALE = 10000
LAYER_IDX = 0

# Orbax checkpoint key paths for layer 0 GLA weights
_PREFIX = (
    "params",
    "params",
    "decoder",
    "dense_layers_0",
    "self_attention",
    "attention",
)
PARAM_MAP = {
    "query_key_value.kernel": _PREFIX + ("query_key_value", "kernel"),
    "dense.kernel": _PREFIX + ("dense", "kernel"),
    "g_proj.kernel": _PREFIX + ("g_proj", "kernel"),
    "query_layernorm.scale": _PREFIX + ("query_layernorm", "scale"),
    "key_layernorm.scale": _PREFIX + ("key_layernorm", "scale"),
    "g_norm.scale": _PREFIX + ("g_norm", "scale"),
}

# -- Forward tensor name variants (CI dumps append _0 suffix) -----------------
ATTN_INPUT_NAMES = [
    "layer_0_lightning_attn_input",
    "layer_0_lightning_attn_input_0",
]
ATTN_OUTPUT_NAMES = [
    "layer_0_lightning_attn_output",
    "layer_0_lightning_attn_output_0",
]

# -- Backward tensor name variants --------------------------------------------
# Upstream gradient d_loss/d(attn_output): Argus naming {name}_output_{i}_grad
ATTN_OUTPUT_GRAD_NAMES = [
    "layer_0_lightning_attn_output_0_grad",
    "layer_0_lightning_attn_output_grad",
    "layer_0_lightning_attn_output_grad_0",
]
# Reference gradient d_loss/d(attn_input): Argus naming {name}_input_grad_{i}
ATTN_INPUT_GRAD_NAMES = [
    "layer_0_lightning_attn_input_grad_0",
    "layer_0_lightning_attn_input_grad",
]

# -- Weight gradient keys (Megatron naming, layer 0) ---------------------------
_MEG_PREFIX = "module.module.decoder.layers.0.self_attention"
WEIGHT_GRAD_MAP = {
    "query_key_value.kernel": f"{_MEG_PREFIX}.linear_qkv.weight",
    "dense.kernel": f"{_MEG_PREFIX}.linear_proj.weight",
    "g_proj.kernel": f"{_MEG_PREFIX}.linear_gate.weight",
    "query_layernorm.scale": f"{_MEG_PREFIX}.q_layernorm.weight",
    "key_layernorm.scale": f"{_MEG_PREFIX}.k_layernorm.weight",
    "g_norm.scale": f"{_MEG_PREFIX}.pre_gate_norm.weight",
}


# -- Helpers -------------------------------------------------------------------


def find_dump_dir(base):
  """Locate the first rank dir under the first step dir (sorted numerically)."""

  def _numeric_key(name, prefix):
    return int(name[len(prefix) :])

  step_dirs = [d for d in os.listdir(base) if d.startswith("step_")]
  if not step_dirs:
    raise ValueError(f"No step_* dirs in {base}")
  step_dirs.sort(key=lambda d: _numeric_key(d, "step_"))
  step_path = os.path.join(base, step_dirs[0])

  rank_dirs = [d for d in os.listdir(step_path) if d.startswith("rank_")]
  if not rank_dirs:
    raise ValueError(f"No rank_* dirs in {step_path}")
  rank_dirs.sort(key=lambda d: _numeric_key(d, "rank_"))
  return os.path.join(step_path, rank_dirs[0])


def resolve_tensor_name(dump_dir, candidates, category="forward"):
  """Find the first matching tensor name from a list of candidates."""
  available = list_tensors(dump_dir, category=category)
  for name in candidates:
    if name in available or f"{category}:{name}" in available:
      return name
  raise ValueError(f"None of {candidates} found in dump (category={category}).\n" f"  Available tensors: {available}")


def load_ckpt_param(ckpt, key_path):
  """Navigate nested checkpoint dict and return param as numpy array."""
  node = ckpt
  for key in key_path:
    node = node[key]
  return np.array(node)


def make_config():
  """Create a minimal config namespace for BailingMoeV2LinearAttention."""
  return types.SimpleNamespace(
      base_num_query_heads=NUM_HEADS,
      base_num_kv_heads=NUM_KV_HEADS,
      head_dim=HEAD_DIM,
      base_emb_dim=EMB_DIM,
      base_num_decoder_layers=NUM_LAYERS,
      dtype=jnp.bfloat16,
      weight_dtype=jnp.bfloat16,
      attention_bias=False,
      shard_mode=ShardMode.AUTO,
      matmul_precision="default",
      use_qk_norm=True,
      normalization_layer_epsilon=EPSILON,
      rope_min_timescale=ROPE_MIN_TIMESCALE,
      rope_max_timescale=ROPE_MAX_TIMESCALE,
      partial_rotary_factor=PARTIAL_ROTARY_FACTOR,
      group_norm_size=GROUP_NORM_SIZE,
      use_linear_silu=False,
      logical_axis_rules=[],
  )


def make_mesh():
  """Create a single-axis mesh with all available devices."""
  devices = jax.devices()
  return Mesh(np.array(devices).reshape(-1), ("devices",))


def load_weights_into_module(module, ckpt):
  """Load checkpoint weights into BailingMoeV2LinearAttention module."""
  for attr_path, ckpt_path in PARAM_MAP.items():
    parts = attr_path.split(".")
    obj = module
    for part in parts[:-1]:
      obj = getattr(obj, part)
    param = getattr(obj, parts[-1])
    param.value = jnp.array(load_ckpt_param(ckpt, ckpt_path))


def megatron_grad_to_maxtext(meg_grad, target_shape):
  """Reshape Megatron weight gradient to MaxText kernel layout.

  Megatron uses PyTorch convention [out_features, in_features] (2D).
  MaxText DenseGeneral uses [in_features, ...out_shape] or [in1, in2, out].
  The transpose + reshape to target_shape handles all cases.
  For 1D weights (layernorm scales) no reshape is needed.
  """
  if meg_grad.ndim >= 2:
    return meg_grad.T.reshape(target_shape)
  return meg_grad


def _reorder_qkv_interleaved_to_grouped(qkv, num_q_heads, num_kv_heads):
  """Reorder QKV head dim from Megatron interleaved to MaxText grouped layout.

  Megatron interleaves per KV-head group:
    [Q0..Q_{r-1}, K0, V0,  Q_r..Q_{2r-1}, K1, V1, ...]
    where r = num_q_heads // num_kv_heads

  MaxText groups all Q, K, V contiguously:
    [Q0, Q1, ..., Q_{H-1}, K0, ..., K_{Hkv-1}, V0, ..., V_{Hkv-1}]

  qkv shape: [..., num_q_heads + 2*num_kv_heads, head_dim]
  """
  *leading, _, head_dim = qkv.shape
  ratio = num_q_heads // num_kv_heads
  group_size = ratio + 2  # Q*ratio + K + V per KV-head group
  grouped = qkv.reshape(*leading, num_kv_heads, group_size, head_dim)
  q = grouped[..., :ratio, :].reshape(*leading, num_q_heads, head_dim)
  k = grouped[..., ratio : ratio + 1, :].reshape(*leading, num_kv_heads, head_dim)
  v = grouped[..., ratio + 1 :, :].reshape(*leading, num_kv_heads, head_dim)
  return np.concatenate([q, k, v], axis=-2)


def load_megatron_weight_grads(dump_dir, module):
  """Load Megatron weight gradients from dump, reshaped to MaxText layout."""
  result = {}
  available = list_tensors(dump_dir, category="grads")
  for attr_path, meg_key in WEIGHT_GRAD_MAP.items():
    if meg_key in available or f"grads:{meg_key}" in available:
      # Get target shape from module's actual parameter
      parts = attr_path.split(".")
      obj = module
      for part in parts[:-1]:
        obj = getattr(obj, part)
      target_shape = getattr(obj, parts[-1]).value.shape
      raw = np.array(load_tensor(dump_dir, f"grads:{meg_key}"), dtype=np.float32)
      grad = megatron_grad_to_maxtext(raw, target_shape)
      # QKV: reorder from Megatron interleaved (QKVQKV...) to MaxText grouped (QQQ...KKK...VVV)
      if attr_path == "query_key_value.kernel":
        grad = _reorder_qkv_interleaved_to_grouped(grad, NUM_HEADS, NUM_KV_HEADS)
      result[attr_path] = grad
  return result


def _bf16_bits_to_ordered(u16):
  """Convert bf16 uint16 bit pattern to ordered integer for ULP comparison.

  IEEE 754 sign-magnitude -> ordered integer:
    positive bf16 -> +magnitude,  negative bf16 -> -magnitude
  This ensures subtraction gives correct ULP distance across signs.
  """
  magnitude = (u16 & 0x7FFF).astype(np.int64)
  return np.where(u16 & 0x8000, -magnitude, magnitude)


def bf16_ulp_diff(actual_f32, expected_f32):
  """Compute per-element ULP distance at bf16 precision.

  Casts fp32 inputs to bf16 via ml_dtypes, bitcasts to uint16 via
  numpy view(), converts to sign-ordered integers, then subtracts
  to get true ULP distance.

  Returns (n_mismatch, n_total, max_ulp, abs_ulp_diffs_on_mismatched).
  """
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


# -- Test class ----------------------------------------------------------------


class GlaCompareTest(unittest.TestCase):
  """End-to-end test: BailingMoeV2LinearAttention vs Megatron reference."""

  @classmethod
  def setUpClass(cls):
    dump_dir_env = os.environ.get("DUMP_DIR")
    if not dump_dir_env:
      raise unittest.SkipTest("DUMP_DIR not set; skipping Megatron comparison")

    ckpt_path = os.environ.get("CKPT_PATH", DEFAULT_CKPT_PATH)

    # Resolve dump directory
    cls.dump_dir = find_dump_dir(dump_dir_env)

    # -- Build config and mesh --
    config = make_config()
    mesh = make_mesh()
    cls.config = config
    cls.mesh = mesh

    # -- Load checkpoint, build module and load weights --
    ckpt = ocp.StandardCheckpointer().restore(ckpt_path)
    rngs = nnx.Rngs(params=0)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      cls.module = BailingMoeV2LinearAttention(
          config=config,
          layer_idx=LAYER_IDX,
          mesh=mesh,
          rngs=rngs,
      )
      load_weights_into_module(cls.module, ckpt)

  # -- Helper -----------------------------------------------------------------

  def _load_input(self):
    """Load Megatron input tensor, return as jnp bf16 [batch, seq, hidden]."""
    input_name = resolve_tensor_name(self.dump_dir, ATTN_INPUT_NAMES)
    meg_input = load_tensor(self.dump_dir, f"forward:{input_name}")
    return jnp.asarray(np.swapaxes(np.array(meg_input), 0, 1), jnp.bfloat16)

  def _run_forward(self, inp_bf16):
    """Run module forward pass, return output as np fp32."""
    B, T = inp_bf16.shape[:2]
    decoder_positions = jnp.broadcast_to(jnp.arange(T)[None, :], (B, T))
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      out, _ = self.module(
          hidden_states=inp_bf16,
          decoder_positions=decoder_positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
    return np.array(out, dtype=np.float32)

  def _run_vjp(self, inp_bf16, upstream_grad_bf16):
    """Run VJP, return (grad_params, grad_input_f32)."""
    B, T = inp_bf16.shape[:2]
    decoder_positions = jnp.broadcast_to(jnp.arange(T)[None, :], (B, T))
    mesh = self.mesh
    config = self.config
    graphdef, params, other = nnx.split(self.module, nnx.Param, ...)

    def forward_fn(params, x):
      model = nnx.merge(graphdef, params, other)
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        out, _ = model(
            hidden_states=x,
            decoder_positions=decoder_positions,
            deterministic=True,
            model_mode=MODEL_MODE_TRAIN,
        )
      return out

    _, vjp_fn = jax.vjp(forward_fn, params, inp_bf16)
    grad_params, grad_input = vjp_fn(upstream_grad_bf16)
    return grad_params, np.array(grad_input, dtype=np.float32)

  def _assert_close(self, actual_f32, expected_f32, label, atol=1e-2, rtol=1e-5, max_ulp=2, max_ulp_fail_rate=1e-3):
    """Assert two fp32 arrays match via allclose, falling back to bf16 ULP diff.

    Elements that satisfy |a - b| <= atol + rtol * |b| are considered close.
    Only the remaining elements are checked for ULP distance.
    Elements exceeding max_ulp are tolerated if their ratio is below
    max_ulp_fail_rate (e.g. near-zero sign flips from accumulation order).
    """
    diff = np.abs(actual_f32 - expected_f32)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    print(f"  {label}: max_abs={max_abs:.6e}  mean_abs={mean_abs:.6e}")

    close_mask = diff <= atol + rtol * np.abs(expected_f32)
    if close_mask.all():
      print(f"all close, {rtol=} {atol=}")
      return  # all elements pass allclose

    # Only check ULP on elements that fail allclose
    n_fail = int((~close_mask).sum())
    n_total = actual_f32.size
    fail_indices = np.argwhere(~close_mask)
    fail_actual = actual_f32[~close_mask]
    fail_expected = expected_f32[~close_mask]
    n_mis, _, worst_ulp, abs_ulps = bf16_ulp_diff(fail_actual, fail_expected)

    # Count elements exceeding max_ulp threshold
    n_over = int((abs_ulps > max_ulp).sum()) if n_mis > 0 else 0
    over_rate = n_over / n_fail

    if n_mis > 0:
      worst_idx = int(np.argmax(abs_ulps))
      worst_pos = tuple(fail_indices[worst_idx])
      print(
          f"  {label} ULP: {n_fail}/{n_total} fail allclose, "
          f"{n_mis} have ULP diff, max_ulp={worst_ulp}, "
          f"over {max_ulp} ULP: {n_over}/{n_fail} ({over_rate:.2e}), "
          f"worst position: {worst_pos}, "
          f"actual={fail_actual[worst_idx]} "
          f"expected={fail_expected[worst_idx]}"
      )
    else:
      print(f"  {label} ULP: {n_fail}/{n_total} fail allclose, " f"0 have ULP diff")

    self.assertLessEqual(
        over_rate,
        max_ulp_fail_rate,
        f"{label}: {n_over}/{n_fail} elements ({over_rate:.2e}) exceed "
        f"{max_ulp} ULP (threshold {max_ulp_fail_rate:.2e})",
    )

  # -- Forward test ------------------------------------------------------------

  def test_forward_bf16(self):
    """Forward output matches Megatron within bf16 tolerance."""
    atol, rtol = 1e-2, 1e-3

    inp_bf16 = self._load_input()
    actual_f32 = self._run_forward(inp_bf16)

    output_name = resolve_tensor_name(self.dump_dir, ATTN_OUTPUT_NAMES)
    meg_output = load_tensor(self.dump_dir, f"forward:{output_name}")
    expected_f32 = np.swapaxes(np.array(meg_output), 0, 1).astype(np.float32)

    self._assert_close(actual_f32, expected_f32, "forward", atol=atol, rtol=rtol)

  # -- Backward activation gradient test ---------------------------------------

  def test_backward_bf16(self):
    """Backward d_loss/d_input matches Megatron within bf16 tolerance."""
    # For bf16, atol, rtol setting advised from https://docs.pytorch.org/docs/stable/testing.html
    atol, rtol = 1e-5, 1.6e-2

    inp_bf16 = self._load_input()

    # Load upstream gradient (d_loss/d_output)
    output_grad_name = resolve_tensor_name(self.dump_dir, ATTN_OUTPUT_GRAD_NAMES, category="backward")
    meg_output_grad = load_tensor(self.dump_dir, f"backward:{output_grad_name}")
    upstream_grad_bf16 = jnp.asarray(np.swapaxes(np.array(meg_output_grad), 0, 1), jnp.bfloat16)

    # Load expected gradient (d_loss/d_input)
    input_grad_name = resolve_tensor_name(self.dump_dir, ATTN_INPUT_GRAD_NAMES, category="backward")
    meg_input_grad = load_tensor(self.dump_dir, f"backward:{input_grad_name}")
    expected_grad_f32 = np.swapaxes(np.array(meg_input_grad), 0, 1).astype(np.float32)

    _, actual_grad_f32 = self._run_vjp(inp_bf16, upstream_grad_bf16)
    self._assert_close(actual_grad_f32, expected_grad_f32, "backward", atol=atol, rtol=rtol)

  # -- Backward weight gradient tests -----------------------------------------

  def test_weight_grads_bf16(self):
    """Weight gradients match Megatron reference within bf16 tolerance."""
    # For bf16, atol, rtol setting advised from https://docs.pytorch.org/docs/stable/testing.html
    atol, rtol = 1e-2, 1e-4

    meg_weight_grads = load_megatron_weight_grads(self.dump_dir, self.module)
    if not meg_weight_grads:
      self.skipTest("No weight gradient data found in dump")

    inp_bf16 = self._load_input()

    # Load upstream gradient (d_loss/d_output)
    output_grad_name = resolve_tensor_name(self.dump_dir, ATTN_OUTPUT_GRAD_NAMES, category="backward")
    meg_output_grad = load_tensor(self.dump_dir, f"backward:{output_grad_name}")
    upstream_grad_bf16 = jnp.asarray(np.swapaxes(np.array(meg_output_grad), 0, 1), jnp.bfloat16)

    grad_params, _ = self._run_vjp(inp_bf16, upstream_grad_bf16)

    for attr_path, meg_grad in meg_weight_grads.items():
      with self.subTest(param=attr_path):
        parts = attr_path.split(".")
        our_grad = grad_params
        for part in parts:
          our_grad = our_grad[part]
        self._assert_close(
            np.array(our_grad, dtype=np.float32),
            np.asarray(meg_grad, dtype=np.float32),
            f"weight_grad/{attr_path}",
            atol=atol,
            rtol=rtol,
        )


if __name__ == "__main__":
  unittest.main()
