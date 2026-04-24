#!/usr/bin/env python3
"""Compare MaxText blockwise FP8 Linear outputs with Megatron FP8 Linear dumps.

Both sides use blockwise FP8 quantization (per-128-element absmax scaling):
  - Megatron: TE Float8BlockScaling (--fp8-recipe blockwise, block_scaling_dim=1)
  - MaxText:  BlockwiseFp8Provider with BlockwiseFp8DotGeneralOp

This test reproduces MaxText's blockwise FP8 matmul using BlockwiseFp8DotGeneralOp,
then compares against the Megatron dump output.

Environment variables (override defaults):
  DUMP_DIR  - Megatron FP8 dump base directory
  CKPT_PATH - Orbax checkpoint path

Run directly:
  python3 tests/blockwise_fp8_compare_test.py -v
"""

import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from argus import list_tensors, load_tensor
from maxtext.kernels.megablox.blockwise_fp8 import BlockwiseFp8DotGeneralOp

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_DUMP_DIR = "/models/argus_dump/megatron/fp8_linear_dump_0323"
DEFAULT_CKPT_PATH = "/models/gpu-ckpt-ling2.5/AL_MODEL_HF20E256_ORBAX_MTP/0/items/"

# Blockwise FP8 quantization block size.
FP8_BLOCK_SIZE = 128

# ULP thresholds for blockwise FP8 cross-framework comparison.
# Both MaxText (BlockwiseFp8DotGeneralOp) and Megatron (TE Float8BlockScaling)
# use per-128-element absmax scaling with e4m3. Differences arise from:
#   1. Implementation details (block boundary handling, scale rounding)
#   2. Matmul accumulation order differences between XLA and cuBLAS
# Thresholds are set on the p99.9 ULP percentile.
#
# Standard layers (single contraction, moderate K):
#   Expected p99.9 ~ 32-64 ULP, threshold at 128.
# Multi-head up-projections (fan-out to many heads):
#   Expected p99.9 ~ 64-128 ULP, threshold at 256.
# Low-rank compressions (large K → small output, e.g. emb→q_lora_rank):
#   Quantization noise from many FP8 blocks concentrates into fewer outputs.
#   Expected p99.9 ~ 100-180 ULP, threshold at 256.
ULP_P999_THRESHOLD = 128
ULP_P999_THRESHOLD_MULTIHEAD = 256
ULP_P999_THRESHOLD_LOWRANK = 256

# FP8 blockwise quantization noise floor.
# FP8 e4m3 blockwise quantization applies a per-block scale = max(|block|) / 448.
# For values well below the block max, the quantization step exceeds the value's
# own BF16 precision, making raw BF16 ULP meaningless. We clamp the reference
# magnitude to this floor so that "noisy small values" don't inflate ULP counts.
# 2^-4 = 0.0625 is within FP8 normal range and gives ~8 ULP for typical FP8
# quantization errors at this magnitude.
FP8_NOISE_FLOOR = 0.0625

# Model config for QKV head layout reordering.
# Megatron uses per-group interleaved [Q_g0,K_g0,V_g0, Q_g1,K_g1,V_g1,...],
# MaxText uses contiguous [Q_all, K_all, V_all].
NUM_Q_HEADS = 16
NUM_KV_HEADS = 16
HEAD_DIM = 128

# ── Linear layer specs ───────────────────────────────────────────────────────
# Each entry: (dump_name, ckpt_key_path, contract_dims)
#
# dump_name:       Megatron dump tensor prefix (e.g. "layer_0_lightning_attn_linear_qkv")
# ckpt_key_path:   Orbax checkpoint key tuple for the weight kernel
# contract_dims:   number of leading kernel dims that are contraction (input) features.
#                  Matches MaxText DenseGeneral's len(axis): 1 for axis=(-1,), 2 for axis=(-2,-1).
#
# MaxText kernels have shape (in_features_shape..., out_features_shape...) where
# the first `contract_dims` dims are contracted with the input's trailing dims.
#
# Note: Megatron's "linear_q_up_proj" is actually the down-projection (H→q_lora_rank),
#       "linear_q_down_proj" is the up-projection (q_lora_rank→heads*head_dim).
#       MaxText names: wq_a = down, wq_b = up.

DENSE_LAYER_LINEARS = [
    (
        "layer_0_lightning_attn_linear_qkv",
        (
            "params",
            "params",
            "decoder",
            "dense_layers_0",
            "self_attention",
            "attention",
            "query_key_value",
            "kernel",
        ),
        1,  # kernel (emb, nh, hd), contract on emb
    ),
    (
        "layer_0_lightning_attn_linear_proj",
        (
            "params",
            "params",
            "decoder",
            "dense_layers_0",
            "self_attention",
            "attention",
            "dense",
            "kernel",
        ),
        2,  # kernel (nh, hd, emb), contract on (nh, hd)
    ),
    (
        "layer_0_lightning_attn_linear_gate",
        (
            "params",
            "params",
            "decoder",
            "dense_layers_0",
            "self_attention",
            "attention",
            "g_proj",
            "kernel",
        ),
        1,  # kernel (emb, nh, hd), contract on emb
    ),
    (
        "layer_0_dense_mlp_linear_fc2",
        ("params", "params", "decoder", "dense_layers_0", "mlp", "wo", "kernel"),
        1,  # kernel (intermediate, emb), contract on intermediate
    ),
]

MLA_LAYER_LINEARS = [
    (
        "layer_4_mla_attn_linear_q_down_proj",
        (
            "params",
            "params",
            "decoder",
            "moe_layers_3",
            "self_attention",
            "attention",
            "wq_a",
            "kernel",
        ),
        1,  # kernel (emb, q_lora_rank), contract on emb
    ),
    (
        "layer_4_mla_attn_linear_q_up_proj",
        (
            "params",
            "params",
            "decoder",
            "moe_layers_3",
            "self_attention",
            "attention",
            "wq_b",
            "kernel",
        ),
        1,  # kernel (q_lora_rank, nh, qk_hd), contract on q_lora_rank
    ),
    (
        "layer_4_mla_attn_linear_kv_down_proj",
        (
            "params",
            "params",
            "decoder",
            "moe_layers_3",
            "self_attention",
            "attention",
            "wkv_a",
            "kernel",
        ),
        1,  # kernel (emb, kv_lora_rank+rope), contract on emb
    ),
    (
        "layer_4_mla_attn_linear_kv_up_proj",
        (
            "params",
            "params",
            "decoder",
            "moe_layers_3",
            "self_attention",
            "attention",
            "wkv_b",
            "kernel",
        ),
        1,  # kernel (kv_lora_rank, nh, nope+v), contract on kv_lora_rank
    ),
    (
        "layer_4_mla_attn_linear_proj",
        (
            "params",
            "params",
            "decoder",
            "moe_layers_3",
            "self_attention",
            "attention",
            "out",
            "kernel",
        ),
        2,  # kernel (nh, hd, emb), contract on (nh, hd)
    ),
]

MOE_SHARED_EXPERT_LINEARS = [
    (
        "layer_1_shared_expert_linear_fc2",
        (
            "params",
            "params",
            "decoder",
            "moe_layers_0",
            "ALMoeBlock_0",
            "shared_experts",
            "wo",
            "kernel",
        ),
        1,  # kernel (intermediate, emb), contract on intermediate
    ),
]

ALL_LINEARS = DENSE_LAYER_LINEARS + MLA_LAYER_LINEARS + MOE_SHARED_EXPERT_LINEARS

# Representative subset covering all unique code paths:
#   dense_qkv:          contract_dims=1, deinterleave=Yes, threshold=MULTIHEAD
#   dense_proj:          contract_dims=2, deinterleave=No,  threshold=standard
#   mla_q_down:          contract_dims=1, deinterleave=No,  threshold=LOWRANK
#   mla_kv_up:           contract_dims=1, deinterleave=No,  threshold=MULTIHEAD
#   shared_expert_fc2:   contract_dims=1, deinterleave=No,  threshold=standard (MoE path)
REPRESENTATIVE_LINEARS = [
    DENSE_LAYER_LINEARS[0],  # dense_qkv
    DENSE_LAYER_LINEARS[1],  # dense_proj
    MLA_LAYER_LINEARS[0],  # mla_q_down
    MLA_LAYER_LINEARS[3],  # mla_kv_up
    MOE_SHARED_EXPERT_LINEARS[0],  # shared_expert_fc2
]


# ── Helpers ──────────────────────────────────────────────────────────────────


def find_dump_dir(base):
  """Locate the step directory containing rank_* subdirs."""
  entries = os.listdir(base)
  step_dirs = sorted(d for d in entries if d.startswith("step_"))
  if not step_dirs:
    raise ValueError(f"No step_* dirs in {base}")
  step_path = os.path.join(base, step_dirs[0])
  if not any(d.startswith("rank_") for d in os.listdir(step_path)):
    raise ValueError(f"No rank_* dirs in {step_path}")
  return step_path


def resolve_tensor_name(available, base_name, category="forward"):
  """Find tensor name, trying both bare and _0 suffix variants."""
  for suffix in ("", "_0"):
    name = f"{base_name}{suffix}"
    if name in available or f"{category}:{name}" in available:
      return name
  return None


def load_ckpt_weight(ckpt_path, key_path):
  """Load a weight tensor from Orbax checkpoint."""
  ckpt = ocp.StandardCheckpointer().restore(ckpt_path)
  node = ckpt
  for key in key_path:
    node = node[key]
  return np.array(node, dtype=np.float32)


def dense_general_fp8(inp_flat, weight, contract_dims=1):
  """Reproduce MaxText DenseGeneral forward pass with BlockwiseFp8DotGeneralOp.

  Uses BlockwiseFp8DotGeneralOp to run the real blockwise FP8 quantization
  and matmul — the same code path that MaxText uses when fp8_blockwise
  quantization is enabled via BlockwiseFp8Provider.

  We flatten to 2D (M, K) × (K, N) before the op so that:
    - All contraction dims merge into a single K
    - Multi-dim outputs are restored via reshape after the matmul

  Args:
      inp_flat: input tensor from Megatron dump, shape (..., in_features_flat)
      weight: MaxText kernel from checkpoint, shape (in_features..., out_features...)
      contract_dims: number of leading kernel dims = len(DenseGeneral.axis)
  """
  in_features_shape = weight.shape[:contract_dims]
  out_features_shape = weight.shape[contract_dims:]
  inp_bf16 = jnp.asarray(inp_flat, jnp.bfloat16)
  kernel_bf16 = jnp.asarray(weight, jnp.bfloat16)

  batch_shape = inp_bf16.shape[:-1]
  contract_size = int(np.prod(in_features_shape))
  out_size = int(np.prod(out_features_shape))

  # Flatten to 2D for the op
  M = int(np.prod(batch_shape))
  inp_2d = inp_bf16.reshape(M, contract_size)
  kernel_2d = kernel_bf16.reshape(contract_size, out_size)

  # Blockwise FP8 matmul via BlockwiseFp8DotGeneralOp
  op = BlockwiseFp8DotGeneralOp(
      block_size=FP8_BLOCK_SIZE,
      fp8_dtype=jnp.float8_e4m3fn,
      use_fused=True,
      cache_rhs=True,
  )
  dimension_numbers = (((1,), (0,)), ((), ()))
  result = op(inp_2d, kernel_2d, dimension_numbers)

  # Restore shape: (M, N_flat) → batch_shape + out_features_shape
  return result.reshape(batch_shape + out_features_shape)


def _forward_fn(inp_2d, kernel_2d):
  """Forward pass for vjp computation."""
  op = BlockwiseFp8DotGeneralOp(
      block_size=FP8_BLOCK_SIZE,
      fp8_dtype=jnp.float8_e4m3fn,
      use_fused=True,
      cache_rhs=True,
  )
  dimension_numbers = (((1,), (0,)), ((), ()))
  return op(inp_2d, kernel_2d, dimension_numbers)


def dense_general_fp8_dgrad(out_grad, weight, contract_dims=1):
  """Compute dgrad (input gradient) using jax.vjp through BlockwiseFp8DotGeneralOp.

  Uses JAX's automatic differentiation through the custom_vjp defined in
  BlockwiseFp8DotGeneralOp, which implements the fused blockwise FP8 backward
  kernels (fused_blockwise_fp8_matmul_1dx2d for dgrad, fused_blockwise_fp8_matmul_1dx1d
  for wgrad).

  Args:
      out_grad: output gradient from dump, shape (..., out_features_flat)
      weight: MaxText kernel from checkpoint, shape (in_features..., out_features...)
      contract_dims: number of leading kernel dims (in_features dims in forward)
  """
  in_features_shape = weight.shape[:contract_dims]
  out_features_shape = weight.shape[contract_dims:]
  out_grad_bf16 = jnp.asarray(out_grad, jnp.bfloat16)
  kernel_bf16 = jnp.asarray(weight, jnp.bfloat16)

  batch_shape = out_grad_bf16.shape[:-1]
  in_size = int(np.prod(in_features_shape))
  out_size = int(np.prod(out_features_shape))

  # Flatten to 2D
  M = int(np.prod(batch_shape))
  inp_2d = jnp.zeros((M, in_size), dtype=jnp.bfloat16)  # dummy input for vjp
  out_grad_2d = out_grad_bf16.reshape(M, out_size)  # (M, N)
  kernel_2d = kernel_bf16.reshape(in_size, out_size)  # (K, N)

  # Use jax.vjp to compute true backward gradient through BlockwiseFp8DotGeneralOp
  # This invokes the custom_vjp which uses fused_blockwise_fp8_matmul_1dx2d for dgrad
  _, vjp_fn = jax.vjp(lambda x: _forward_fn(x, kernel_2d), inp_2d)
  (dlhs_2d,) = vjp_fn(out_grad_2d)

  return dlhs_2d.reshape(batch_shape + in_features_shape)


def bf16_ulp_distance(a_f32, b_f32, noise_floor=FP8_NOISE_FLOOR):
  """Compute ULP distance between two arrays in BF16 precision.

  BF16 has 7 mantissa bits (vs f32's 23). One BF16 ULP = 2^16 f32 ULPs.
  Uses np.spacing (exact for normals, correct for subnormals) scaled by 2^16.

  The noise_floor clamps the reference magnitude so that values below it
  (where FP8 blockwise quantization noise dominates the signal) don't produce
  inflated ULP counts. This reflects the physical reality that FP8's absolute
  quantization error has a lower bound independent of the value's magnitude.

  Returns an array of ULP distances (float64 to avoid overflow).
  """
  diff = np.abs(a_f32.astype(np.float64) - b_f32.astype(np.float64))
  ref = np.maximum(np.abs(a_f32), np.abs(b_f32)).astype(np.float32)
  # Clamp ref to noise floor: below this, FP8 quantization error exceeds
  # the value's BF16 precision, so raw ULP is not meaningful.
  ref_clamped = np.maximum(ref, np.float32(noise_floor))
  # np.spacing gives f32 ULP; multiply by 2^16 for bf16 ULP
  ulp_size = (np.spacing(ref_clamped) * (1 << 16)).astype(np.float64)
  return diff / ulp_size


def deinterleave_megatron_qkv(output, num_q_heads, num_kv_heads, head_dim):
  """Convert Megatron's interleaved QKV to MaxText's contiguous layout.

  Megatron stores linear_qkv output as per-group interleaved:
    [Q_g0, K_g0, V_g0, Q_g1, K_g1, V_g1, ...]
  MaxText expects contiguous blocks:
    [Q_all, K_all, V_all]

  Args:
      output: tensor with last dim = (num_q_heads + 2*num_kv_heads) * head_dim
      num_q_heads: number of query heads (e.g. 16)
      num_kv_heads: number of KV heads / query groups (e.g. 16)
      head_dim: dimension per head (e.g. 128)
  """
  batch_shape = output.shape[:-1]
  ng = num_kv_heads  # num query groups
  hpg = num_q_heads // ng  # query heads per group
  group_size = (hpg + 2) * head_dim

  # Reshape into groups
  grouped = output.reshape(batch_shape + (ng, group_size))

  # Split each group: [Q_heads(hpg*hd), K_head(hd), V_head(hd)]
  q_per_group = hpg * head_dim
  Q = grouped[..., :q_per_group]
  K = grouped[..., q_per_group : q_per_group + head_dim]
  V = grouped[..., q_per_group + head_dim :]

  # Flatten each into contiguous blocks
  Q = Q.reshape(batch_shape + (num_q_heads * head_dim,))
  K = K.reshape(batch_shape + (num_kv_heads * head_dim,))
  V = V.reshape(batch_shape + (num_kv_heads * head_dim,))

  return jnp.concatenate([Q, K, V], axis=-1)


# ── Test class ───────────────────────────────────────────────────────────


class BlockwiseFp8CompareTest(unittest.TestCase):
  """Compare MaxText Linear outputs against Megatron FP8 reference dumps."""

  @classmethod
  def setUpClass(cls):
    dump_dir = os.environ.get("DUMP_DIR", DEFAULT_DUMP_DIR)
    cls.ckpt_path = os.environ.get("CKPT_PATH", DEFAULT_CKPT_PATH)

    if not os.path.isdir(dump_dir):
      raise unittest.SkipTest(f"Dump dir not found: {dump_dir}  " f"(set DUMP_DIR env var or mount the GCS volume)")

    cls.dump_dir = find_dump_dir(dump_dir)
    cls.available_fwd = list_tensors(cls.dump_dir, category="forward")
    cls.available_bwd = list_tensors(cls.dump_dir, category="backward")

    # Load checkpoint once, cache weights for all representative layers.
    ckpt = ocp.StandardCheckpointer().restore(cls.ckpt_path)
    cls._weight_cache = {}
    for _, key_path, _ in REPRESENTATIVE_LINEARS:
      node = ckpt
      for key in key_path:
        node = node[key]
      cls._weight_cache[key_path] = np.array(node, dtype=np.float32)
    del ckpt

    # Cache all dump tensors for representative layers (avoid per-test I/O).
    cls._dump_cache = {}
    for dump_name, _, _ in REPRESENTATIVE_LINEARS:
      # Forward tensors
      input_name = resolve_tensor_name(cls.available_fwd, f"{dump_name}_input")
      output_name = resolve_tensor_name(cls.available_fwd, f"{dump_name}_output")
      if input_name and output_name:
        cls._dump_cache[(dump_name, "fwd_input")] = load_tensor(cls.dump_dir, f"forward:{input_name}", dp_mode="replica")
        cls._dump_cache[(dump_name, "fwd_output")] = load_tensor(
            cls.dump_dir, f"forward:{output_name}", dp_mode="replica"
        )

      # Backward tensors
      out_grad_name = resolve_tensor_name(cls.available_bwd, f"{dump_name}_output_grad", category="backward")
      in_grad_name = resolve_tensor_name(cls.available_bwd, f"{dump_name}_input_grad", category="backward")
      if out_grad_name and in_grad_name:
        cls._dump_cache[(dump_name, "bwd_output_grad")] = load_tensor(
            cls.dump_dir, f"backward:{out_grad_name}", dp_mode="replica"
        )
        cls._dump_cache[(dump_name, "bwd_input_grad")] = load_tensor(
            cls.dump_dir, f"backward:{in_grad_name}", dp_mode="replica"
        )

  def _run_linear_compare(
      self,
      dump_name,
      ckpt_key_path,
      contract_dims,
      deinterleave_qkv_output=False,
      max_ulp_p999=ULP_P999_THRESHOLD,
      min_cosine_sim=0.99,
  ):
    """Load Megatron dump, compute MaxText linear, compare via multiple metrics.

    Comparison metrics:
      1. ULP distance (BF16): scale-invariant precision measure
      2. Absolute difference: raw numerical error (mean/std/max)
      3. Relative difference: error relative to signal magnitude
      4. Cosine similarity: per-sample directional alignment

    Args:
        max_ulp_p999: maximum allowed p99.9 ULP distance in BF16 space.
        min_cosine_sim: minimum allowed mean cosine similarity.
    """
    # Use cached dump tensors (loaded in setUpClass)
    meg_input = self._dump_cache.get((dump_name, "fwd_input"))
    meg_output = self._dump_cache.get((dump_name, "fwd_output"))
    if meg_input is None or meg_output is None:
      self.skipTest(f"Tensors for {dump_name} not found in dump")

    # Megatron [seq, batch, hidden] -> MaxText [batch, seq, hidden]
    inp_bf16 = jnp.asarray(jnp.swapaxes(jnp.array(meg_input), 0, 1), jnp.bfloat16)
    expected_bf16 = jnp.asarray(jnp.swapaxes(jnp.array(meg_output), 0, 1), jnp.bfloat16)

    # Megatron QKV uses per-group interleaved layout [Q_g0,K_g0,V_g0,...],
    # convert to MaxText's contiguous [Q_all,K_all,V_all] for comparison.
    if deinterleave_qkv_output:
      expected_bf16 = deinterleave_megatron_qkv(expected_bf16, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)

    # Load weight and compute via blockwise FP8 matmul
    weight = self._weight_cache[ckpt_key_path]
    actual = dense_general_fp8(inp_bf16, weight, contract_dims)

    # Flatten multi-dim output for comparison with Megatron's flat format
    # e.g. (batch, seq, nh, hd) -> (batch, seq, nh*hd)
    batch_shape = actual.shape[:2]  # (batch, seq)
    actual_bf16 = actual.reshape(batch_shape + (-1,))

    # ULP distance in BF16 space
    actual_f32 = np.array(actual_bf16, dtype=np.float32)
    expected_f32 = np.array(expected_bf16, dtype=np.float32)
    ulps = bf16_ulp_distance(actual_f32, expected_f32)

    p50 = np.percentile(ulps, 50)
    p90 = np.percentile(ulps, 90)
    p99 = np.percentile(ulps, 99)
    p999 = np.percentile(ulps, 99.9)

    # Absolute difference
    abs_diff = np.abs(actual_f32 - expected_f32)

    # Relative difference (clamped denominator to avoid div-by-zero)
    ref_mag = np.maximum(np.abs(expected_f32), np.float32(FP8_NOISE_FLOOR))
    rel_diff = abs_diff / ref_mag

    # Cosine similarity per sample (flatten last dim, compute per batch*seq)
    a_flat = actual_f32.reshape(-1, actual_f32.shape[-1])
    b_flat = expected_f32.reshape(-1, expected_f32.shape[-1])
    dot_prod = np.sum(a_flat * b_flat, axis=-1)
    norm_a = np.linalg.norm(a_flat, axis=-1)
    norm_b = np.linalg.norm(b_flat, axis=-1)
    cos_sim = dot_prod / np.maximum(norm_a * norm_b, 1e-12)

    print(f"\n  {dump_name}:")
    print(f"    shapes: input={inp_bf16.shape}, weight={weight.shape}, " f"output={expected_bf16.shape}")
    print(
        f"    ULP (bf16):  mean={ulps.mean():.1f}  p50={p50:.0f}  p90={p90:.0f}  "
        f"p99={p99:.0f}  p99.9={p999:.0f}  max={ulps.max():.0f}"
    )
    print(f"    abs_diff:    mean={abs_diff.mean():.6e}  std={abs_diff.std():.6e}  " f"max={abs_diff.max():.6e}")
    print(f"    rel_diff:    mean={rel_diff.mean():.6e}  std={rel_diff.std():.6e}  " f"max={rel_diff.max():.6e}")
    print(f"    cosine_sim:  mean={cos_sim.mean():.8f}  min={cos_sim.min():.8f}  " f"std={cos_sim.std():.2e}")

    self.assertLessEqual(
        p999,
        max_ulp_p999,
        f"{dump_name}: p99.9 ULP = {p999:.1f} exceeds threshold {max_ulp_p999}  "
        f"(mean={ulps.mean():.1f}, max={ulps.max():.0f})",
    )
    self.assertGreaterEqual(
        cos_sim.mean(),
        min_cosine_sim,
        f"{dump_name}: mean cosine similarity = {cos_sim.mean():.6f} " f"below threshold {min_cosine_sim}",
    )

  def _run_dgrad_compare(
      self,
      dump_name,
      ckpt_key_path,
      contract_dims,
      deinterleave_qkv_grad=False,
      max_ulp_p999=ULP_P999_THRESHOLD,
      min_cosine_sim=0.99,
  ):
    """Compare backward dgrad: input_grad = output_grad @ weight.T

    Loads output_grad and input_grad from the backward dump, computes
    dgrad using BlockwiseFp8DotGeneralOp, and compares with the dump's input_grad.
    """
    # Use cached dump tensors (loaded in setUpClass)
    meg_out_grad = self._dump_cache.get((dump_name, "bwd_output_grad"))
    meg_in_grad = self._dump_cache.get((dump_name, "bwd_input_grad"))
    if meg_out_grad is None or meg_in_grad is None:
      self.skipTest(f"Backward tensors for {dump_name} not found in dump")

    # Megatron [seq, batch, hidden] -> [batch, seq, hidden]
    out_grad_bf16 = jnp.asarray(jnp.swapaxes(jnp.array(meg_out_grad), 0, 1), jnp.bfloat16)
    expected_bf16 = jnp.asarray(jnp.swapaxes(jnp.array(meg_in_grad), 0, 1), jnp.bfloat16)

    # QKV output_grad is in Megatron's interleaved layout;
    # deinterleave to match MaxText weight's contiguous layout.
    if deinterleave_qkv_grad:
      out_grad_bf16 = deinterleave_megatron_qkv(out_grad_bf16, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)

    # Load weight and compute dgrad via FP8
    weight = self._weight_cache[ckpt_key_path]
    actual = dense_general_fp8_dgrad(out_grad_bf16, weight, contract_dims)

    # Flatten multi-dim output for comparison
    batch_shape = actual.shape[:2]
    actual_bf16 = actual.reshape(batch_shape + (-1,))

    # Metrics
    actual_f32 = np.array(actual_bf16, dtype=np.float32)
    expected_f32 = np.array(expected_bf16, dtype=np.float32)
    ulps = bf16_ulp_distance(actual_f32, expected_f32)

    p50 = np.percentile(ulps, 50)
    p90 = np.percentile(ulps, 90)
    p99 = np.percentile(ulps, 99)
    p999 = np.percentile(ulps, 99.9)

    abs_diff = np.abs(actual_f32 - expected_f32)
    ref_mag = np.maximum(np.abs(expected_f32), np.float32(FP8_NOISE_FLOOR))
    rel_diff = abs_diff / ref_mag

    a_flat = actual_f32.reshape(-1, actual_f32.shape[-1])
    b_flat = expected_f32.reshape(-1, expected_f32.shape[-1])
    dot_prod = np.sum(a_flat * b_flat, axis=-1)
    norm_a = np.linalg.norm(a_flat, axis=-1)
    norm_b = np.linalg.norm(b_flat, axis=-1)
    cos_sim_raw = dot_prod / np.maximum(norm_a * norm_b, 1e-12)

    # Filter out near-zero vectors where cosine is meaningless.
    # Backward gradients can be very small; cosine of ~0 vs ~0 is noise.
    grad_magnitude = np.maximum(norm_a, norm_b)
    magnitude_threshold = 1e-6
    nontrivial_mask = grad_magnitude > magnitude_threshold
    cos_sim = cos_sim_raw[nontrivial_mask] if nontrivial_mask.any() else cos_sim_raw
    n_skipped = int((~nontrivial_mask).sum())

    print(f"\n  {dump_name} [backward dgrad]:")
    print(f"    shapes: output_grad={out_grad_bf16.shape}, weight={weight.shape}, " f"input_grad={expected_bf16.shape}")
    print(
        f"    ULP (bf16):  mean={ulps.mean():.1f}  p50={p50:.0f}  p90={p90:.0f}  "
        f"p99={p99:.0f}  p99.9={p999:.0f}  max={ulps.max():.0f}"
    )
    print(f"    abs_diff:    mean={abs_diff.mean():.6e}  std={abs_diff.std():.6e}  " f"max={abs_diff.max():.6e}")
    print(f"    rel_diff:    mean={rel_diff.mean():.6e}  std={rel_diff.std():.6e}  " f"max={rel_diff.max():.6e}")
    print(
        f"    cosine_sim:  mean={cos_sim.mean():.8f}  min={cos_sim.min():.8f}  "
        f"std={cos_sim.std():.2e}"
        f"  (skipped {n_skipped} near-zero samples)"
        if n_skipped
        else ""
    )

    self.assertLessEqual(
        p999,
        max_ulp_p999,
        f"{dump_name} backward dgrad: p99.9 ULP = {p999:.1f} exceeds {max_ulp_p999}  "
        f"(mean={ulps.mean():.1f}, max={ulps.max():.0f})",
    )
    # Only assert cosine when ULP shows meaningful differences.
    # When p99.9 ULP < 1, values match to BF16 precision and cosine
    # of near-zero gradients is dominated by float32 rounding noise.
    if p999 >= 1 and len(cos_sim) > 0:
      self.assertGreaterEqual(
          cos_sim.mean(),
          min_cosine_sim,
          f"{dump_name} backward dgrad: mean cosine sim = {cos_sim.mean():.6f} " f"below {min_cosine_sim}",
      )

  # ── Dense layer 0 (GLA / Lightning Attention) ─────────────────────────

  def test_dense_layer0_qkv(self):
    """layer_0 lightning_attn linear_qkv: fused Q+K+V projection."""
    self._run_linear_compare(
        *DENSE_LAYER_LINEARS[0],
        deinterleave_qkv_output=True,
        max_ulp_p999=ULP_P999_THRESHOLD_MULTIHEAD,
    )

  def test_dense_layer0_proj(self):
    """layer_0 lightning_attn linear_proj: output projection."""
    self._run_linear_compare(*DENSE_LAYER_LINEARS[1])

  # ── MLA layer (Megatron layer 4 = moe_layers_3) ──────────────────────

  def test_mla_q_down_proj(self):
    """layer_4 MLA linear_q_down_proj: Q low-rank down projection."""
    self._run_linear_compare(*MLA_LAYER_LINEARS[0], max_ulp_p999=ULP_P999_THRESHOLD_LOWRANK)

  def test_mla_kv_up_proj(self):
    """layer_4 MLA linear_kv_up_proj: KV low-rank up projection."""
    self._run_linear_compare(*MLA_LAYER_LINEARS[3], max_ulp_p999=ULP_P999_THRESHOLD_MULTIHEAD)

  # ── MoE shared expert (Megatron layer 1 = moe_layers_0) ─────────────

  def test_shared_expert_fc2(self):
    """layer_1 shared_expert linear_fc2: shared expert down projection."""
    self._run_linear_compare(*MOE_SHARED_EXPERT_LINEARS[0])

  # ── Backward dgrad tests ──────────────────────────────────────────────
  # dgrad = output_grad @ weight.T
  # Contraction dim in backward = forward output features (N).
  # Layers with small N (e.g. down-projections) will have fewer FP8 blocks.

  def test_dgrad_dense_layer0_qkv(self):
    """layer_0 lightning_attn linear_qkv backward: dgrad."""
    self._run_dgrad_compare(
        *DENSE_LAYER_LINEARS[0],
        deinterleave_qkv_grad=True,
        max_ulp_p999=ULP_P999_THRESHOLD_MULTIHEAD,
    )

  def test_dgrad_dense_layer0_proj(self):
    """layer_0 lightning_attn linear_proj backward: dgrad."""
    self._run_dgrad_compare(*DENSE_LAYER_LINEARS[1], max_ulp_p999=ULP_P999_THRESHOLD_MULTIHEAD)

  def test_dgrad_mla_q_down_proj(self):
    """layer_4 MLA linear_q_down_proj backward: dgrad.
    N=256 (q_lora_rank) → only 2 FP8 blocks in contraction."""
    self._run_dgrad_compare(*MLA_LAYER_LINEARS[0], max_ulp_p999=ULP_P999_THRESHOLD_LOWRANK)

  def test_dgrad_mla_kv_up_proj(self):
    """layer_4 MLA linear_kv_up_proj backward: dgrad.
    N=4096 (nh*(nope+v)) → 32 FP8 blocks."""
    self._run_dgrad_compare(*MLA_LAYER_LINEARS[3], max_ulp_p999=ULP_P999_THRESHOLD_MULTIHEAD)

  def test_dgrad_shared_expert_fc2(self):
    """layer_1 shared_expert linear_fc2 backward: dgrad."""
    self._run_dgrad_compare(*MOE_SHARED_EXPERT_LINEARS[0])

  # ── Presence test ────────────────────────────────────────────────────

  def test_linear_tensors_present(self):
    """All expected linear tensor names exist in the forward dump."""
    expected_prefixes = [spec[0] for spec in ALL_LINEARS]
    missing = []
    for prefix in expected_prefixes:
      input_name = resolve_tensor_name(self.available_fwd, f"{prefix}_input")
      output_name = resolve_tensor_name(self.available_fwd, f"{prefix}_output")
      if input_name is None or output_name is None:
        missing.append(prefix)
    self.assertEqual(
        missing,
        [],
        f"Missing linear tensors in forward dump: {missing}",
    )

  def test_backward_linear_tensors_present(self):
    """Linear gradient tensors exist in the backward dump."""
    expected_prefixes = [spec[0] for spec in ALL_LINEARS]
    missing = []
    for prefix in expected_prefixes:
      for grad_suffix in ("input_grad", "output_grad"):
        base = f"{prefix}_{grad_suffix}"
        found = any(
            candidate in self.available_bwd or f"backward:{candidate}" in self.available_bwd
            for candidate in (base, f"{base}_0")
        )
        if not found:
          missing.append(base)
    if missing:
      print(f"\n  Missing backward tensors: {missing[:5]}...")
    # Warn but don't fail — backward may not always be available
    self.assertEqual(
        missing,
        [],
        f"Missing backward linear tensors: {missing}",
    )


if __name__ == "__main__":
  unittest.main()
