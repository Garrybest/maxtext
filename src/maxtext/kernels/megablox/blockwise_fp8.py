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

"""DeepSeek-style blockwise FP8 quantization and matmul kernels.

Implements fine-grained 128-block scaling with Pallas kernels for
quantization and matmul, supporting forward, dgrad, and wgrad passes.
"""

# pylint: disable=too-many-positional-arguments

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def _get_fp8_bounds(fp8_dtype):
  fp8_max = float(jnp.finfo(fp8_dtype).max)
  return fp8_max, -fp8_max


def _align_tile_tpu(dim, max_tile, alignment=128):
  """Find largest tile <= max_tile that divides dim, aligned for TPU Pallas.

  TPU Pallas requires the last dimension of BlockSpec shapes to be divisible
  by `alignment` (128) or equal to the full array dimension.
  """
  tile = min(max_tile, dim)
  if tile == dim:
    return tile
  # Round down to nearest multiple of alignment, then search
  tile = (tile // alignment) * alignment
  while tile > 0 and dim % tile != 0:
    tile -= alignment
  if tile <= 0:
    return dim  # Fallback to full dimension (always valid)
  return tile


# ---------------------------------------------------------------------------
# Pure JAX quantization (Pallas-free)
# ---------------------------------------------------------------------------


def _quantize_blockwise_1d_jax(x, fp8_dtype, block_size, axis):
  """1D blockwise FP8 quantization — pure JAX, no Pallas."""
  M, K = x.shape
  fp8_max, fp8_min = _get_fp8_bounds(fp8_dtype)
  if axis in (-1, 1):
    nk = K // block_size
    x_blocks = x.astype(jnp.float32).reshape(M, nk, block_size)
    absmax = jnp.max(jnp.abs(x_blocks), axis=2, keepdims=True)
    scale = jnp.where(absmax == 0, 1.0, absmax / fp8_max)
    qx = (x_blocks / scale).clip(fp8_min, fp8_max).astype(fp8_dtype)
    return qx.reshape(M, K), scale.squeeze(2)
  elif axis == 0:
    nmb = M // block_size
    x_blocks = x.astype(jnp.float32).reshape(nmb, block_size, K)
    absmax = jnp.max(jnp.abs(x_blocks), axis=1, keepdims=True)
    scale = jnp.where(absmax == 0, 1.0, absmax / fp8_max)
    qx = (x_blocks / scale).clip(fp8_min, fp8_max).astype(fp8_dtype)
    return qx.reshape(M, K), scale.squeeze(1)
  else:
    raise ValueError(f"Unsupported axis={axis}, expected 0, 1, or -1")


# ---------------------------------------------------------------------------
# Pallas quantization kernels
# ---------------------------------------------------------------------------


def quantize_blockwise_1d(
    x, fp8_dtype=jnp.float8_e4m3fn, block_size=128, axis=-1, interpret=False, tm=1024, use_pallas=True
):
  """1D blockwise FP8 quantization via Pallas.

  Args:
    x: [M, K] bf16/f32 input
    fp8_dtype: target FP8 dtype
    block_size: quantization block size (default 128)
    axis: which axis to split into blocks (-1 for K, 0 for M)
    interpret: if True, run Pallas in interpret mode (for CPU testing)
    tm: tile size on M axis (for axis=-1) or K axis (for axis=0)

  Returns:
    (qx [M, K] fp8, scale [M, K//bs] or [M//bs, K] f32)
  """
  assert x.ndim == 2, f"Expected 2D input, got shape {x.shape}"

  if not use_pallas:
    return _quantize_blockwise_1d_jax(x, fp8_dtype, block_size, axis)

  M, K = x.shape
  fp8_max, fp8_min = _get_fp8_bounds(fp8_dtype)

  if axis in (-1, 1):
    # K-axis quantization: grid over (M // tm, K // block_size)
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"
    num_k_blocks = K // block_size
    # Adjust tm to not exceed M
    actual_tm = min(tm, M)
    assert M % actual_tm == 0, f"M={M} not divisible by tm={actual_tm}"

    def _kernel_1d_k(x_ref, qx_ref, sx_ref):
      x_full = x_ref[...].astype(jnp.float32)  # [tm, K]
      # Reshape to expose block structure: [tm, nk, block_size]
      x_blocks = x_full.reshape(actual_tm, num_k_blocks, block_size)
      absmax = jnp.max(jnp.abs(x_blocks), axis=2, keepdims=True)  # [tm, nk, 1]
      scale = jnp.where(absmax == 0, 1.0, absmax / fp8_max)
      qx_blocks = (x_blocks / scale).clip(fp8_min, fp8_max).astype(fp8_dtype)
      qx_ref[...] = qx_blocks.reshape(actual_tm, K)
      sx_ref[...] = scale.squeeze(2)  # [tm, nk]

    grid = (M // actual_tm,)
    qx, sx = pl.pallas_call(
        _kernel_1d_k,
        out_shape=[
            jax.ShapeDtypeStruct((M, K), fp8_dtype),
            jax.ShapeDtypeStruct((M, num_k_blocks), jnp.float32),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((actual_tm, K), lambda mi: (mi, 0)),
            ],
            out_specs=[
                pl.BlockSpec((actual_tm, K), lambda mi: (mi, 0)),
                pl.BlockSpec((actual_tm, num_k_blocks), lambda mi: (mi, 0)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
        ),
        interpret=interpret,
    )(x)
    return qx, sx

  elif axis == 0:
    # M-axis quantization: tile both M and K to bound VMEM usage.
    assert M % block_size == 0, f"M={M} not divisible by block_size={block_size}"
    num_m_blocks = M // block_size
    tk = min(tm, K)
    assert K % tk == 0, f"K={K} not divisible by tk={tk}"
    # Tile M dimension: each tile covers tm_m_blocks * block_size rows
    tm_m_blocks = min(num_m_blocks, max(1, tm // block_size))
    while num_m_blocks % tm_m_blocks != 0:
      tm_m_blocks -= 1
    tile_m = tm_m_blocks * block_size  # rows per M-tile

    def _kernel_1d_m(x_ref, qx_ref, sx_ref):
      x_full = x_ref[...].astype(jnp.float32)  # [tile_m, tk]
      x_blocks = x_full.reshape(tm_m_blocks, block_size, tk)
      absmax = jnp.max(jnp.abs(x_blocks), axis=1, keepdims=True)  # [tm_m_blocks, 1, tk]
      scale = jnp.where(absmax == 0, 1.0, absmax / fp8_max)
      qx_blocks = (x_blocks / scale).clip(fp8_min, fp8_max).astype(fp8_dtype)
      qx_ref[...] = qx_blocks.reshape(tile_m, tk)
      sx_ref[...] = scale.squeeze(1)  # [tm_m_blocks, tk]

    grid = (num_m_blocks // tm_m_blocks, K // tk)
    qx, sx = pl.pallas_call(
        _kernel_1d_m,
        out_shape=[
            jax.ShapeDtypeStruct((M, K), fp8_dtype),
            jax.ShapeDtypeStruct((num_m_blocks, K), jnp.float32),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((tile_m, tk), lambda mi, ki: (mi, ki)),
            ],
            out_specs=[
                pl.BlockSpec((tile_m, tk), lambda mi, ki: (mi, ki)),
                pl.BlockSpec((tm_m_blocks, tk), lambda mi, ki: (mi, ki)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
        interpret=interpret,
    )(x)
    return qx, sx

  else:
    raise ValueError(f"Unsupported axis={axis}, expected 0, 1, or -1")


def quantize_blockwise_2d(x, fp8_dtype=jnp.float8_e4m3fn, block_size=128):
  """2D blockwise FP8 quantization via Pallas.

  Args:
    x: [K, N] bf16/f32 input
    fp8_dtype: target FP8 dtype
    block_size: quantization block size (default 128)

  Returns:
    (qx [K, N] fp8, scale [K//bs, N//bs] f32)
  """
  assert x.ndim == 2, f"Expected 2D input, got shape {x.shape}"
  K, N = x.shape
  assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"
  assert N % block_size == 0, f"N={N} not divisible by block_size={block_size}"
  fp8_max, fp8_min = _get_fp8_bounds(fp8_dtype)
  num_k_blocks = K // block_size
  num_n_blocks = N // block_size

  # Pure JAX quantization (avoids Pallas BlockSpec alignment issues for 2D scales)
  x_blocks = x.reshape(num_k_blocks, block_size, num_n_blocks, block_size)
  absmax = jnp.max(jnp.abs(x_blocks.astype(jnp.float32)), axis=(1, 3))
  sx = jnp.where(absmax == 0, 1.0, absmax / fp8_max)  # [nk, nn]

  # Quantize: expand scale [nk, 1, nn, 1] for broadcasting
  scale_expanded = sx[:, None, :, None]
  qx_blocks = (x_blocks.astype(jnp.float32) / scale_expanded).clip(fp8_min, fp8_max).astype(fp8_dtype)
  qx = qx_blocks.reshape(K, N)
  return qx, sx


# ---------------------------------------------------------------------------
# Pure JAX matmul kernels (Pallas-free)
# ---------------------------------------------------------------------------


def _blockwise_fp8_matmul_1dx2d_jax(qlhs, slhs, qrhs, srhs, block_size):
  """1D x 2D scaled FP8 matmul — pure JAX, no Pallas."""
  M, K = qlhs.shape
  _, N = qrhs.shape
  nk = K // block_size
  nn = N // block_size

  # Split K axis: [M, nk, bs] and [nk, bs, N]
  lhs_tiled = qlhs.reshape(M, nk, block_size)
  rhs_tiled = qrhs.reshape(nk, block_size, N)

  # Batched matmul: contract on bs, batch on nk -> [nk, M, N]
  result = jax.lax.dot_general(
      lhs_tiled,
      rhs_tiled,
      (((2,), (1,)), ((1,), (0,))),
      preferred_element_type=jnp.float32,
  )

  # Apply LHS scale: [M, nk] -> [nk, M, 1]
  result = result * slhs.T[:, :, None]

  # Apply RHS 2D scale: [nk, nn] -> reshape result to [nk, M, nn, bs],
  # multiply, reshape back
  result = result.reshape(nk, M, nn, block_size) * srhs[:, None, :, None]
  result = result.reshape(nk, M, N)

  # Sum over K blocks -> [M, N]
  return jnp.sum(result, axis=0).astype(jnp.bfloat16)


def _blockwise_fp8_matmul_1dx1d_jax(qlhs, slhs, qrhs, srhs, block_size):
  """1D x 1D scaled FP8 matmul (wgrad) — pure JAX, no Pallas."""
  M, H = qlhs.shape
  _, D = qrhs.shape
  nm = M // block_size

  # Split M axis: [nm, bs, H] and [nm, bs, D]
  lhs_tiled = qlhs.reshape(nm, block_size, H)
  rhs_tiled = qrhs.reshape(nm, block_size, D)

  # Batched dot: contract on bs (axis 1), batch on nm (axis 0) -> [nm, H, D]
  result = jax.lax.dot_general(
      lhs_tiled,
      rhs_tiled,
      (((1,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )

  # Apply scales: [nm, H, 1] * [nm, 1, D]
  result = result * slhs[:, :, None] * srhs[:, None, :]

  # Sum over M blocks -> [H, D]
  return jnp.sum(result, axis=0).astype(jnp.bfloat16)


# ---------------------------------------------------------------------------
# Pallas matmul kernels
# ---------------------------------------------------------------------------


def blockwise_fp8_matmul_1dx2d(
    qlhs, slhs, qrhs, srhs, block_size=128, tm=1024, tn=1024, n_lane_multiplier=4, interpret=False, use_pallas=True
):
  """Blockwise FP8 matmul: 1D-scaled LHS @ 2D-scaled RHS.

  Forward and dgrad: [M, K] @ [K, N] -> [M, N]

  Uses configurable N-axis tiling with n_lane_multiplier to produce wider
  MXU matmuls ([tm, bs] @ [bs, compute_tile_n]) and an inner N-loop to
  reduce grid iterations.

  Args:
    qlhs: [M, K] fp8, quantized LHS
    slhs: [M, K//bs] f32, row-independent 1D scales
    qrhs: [K, N] fp8, quantized RHS
    srhs: [K//bs, N//bs] f32, 2D blockwise scales
    block_size: quantization block size
    tm: tile size on M axis
    tn: tile size on N axis (multiple of block_size)
    n_lane_multiplier: number of block_size columns per MXU matmul
    interpret: if True, run Pallas in interpret mode
    use_pallas: if False, use pure JAX implementation

  Returns:
    [M, N] bf16 result
  """
  if not use_pallas:
    return _blockwise_fp8_matmul_1dx2d_jax(qlhs, slhs, qrhs, srhs, block_size)

  M, K = qlhs.shape
  K2, N = qrhs.shape
  assert K == K2, f"K mismatch: {K} vs {K2}"
  num_k_blocks = K // block_size
  actual_tm = min(tm, M)

  actual_tn = min(tn, N)
  while actual_tn > block_size and N % actual_tn != 0:
    actual_tn -= block_size
  assert actual_tn % block_size == 0, f"tn={actual_tn} not multiple of block_size={block_size}"
  assert N % actual_tn == 0, f"N={N} not divisible by tn={actual_tn}"
  n_blocks_n = actual_tn // block_size
  n_lane_multiplier = min(n_lane_multiplier, n_blocks_n)
  while n_blocks_n % n_lane_multiplier != 0:
    n_lane_multiplier -= 1
  compute_tile_n = block_size * n_lane_multiplier
  steps_n = actual_tn // compute_tile_n
  bs = block_size
  n_blocks_per_cell = actual_tn // bs

  # Transpose slhs for TPU 8x128 alignment: [M, nk] → [nk, M]
  slhs_T = slhs.T
  # srhs passed via scalar_prefetch into SMEM (small tensor, direct indexing)

  def _kernel(srhs_ref, qlhs_ref, slhs_ref, qrhs_ref, out_ref, acc_scratch):
    ki = pl.program_id(2)
    pid_n = pl.program_id(1)

    @pl.when(ki == 0)
    def _zero_acc():
      acc_scratch[...] = jnp.zeros_like(acc_scratch)

    # LHS scale (same for all N sub-blocks within this K iteration)
    slhs_scale = slhs_ref[pl.ds(ki, 1), :]  # [1, tm] from transposed scale
    slhs_scale = slhs_scale.reshape(actual_tm, 1)  # [tm, 1] for broadcasting

    for ni in range(steps_n):
      n_start = ni * compute_tile_n

      # Wide FP8 matmul: [tm, bs] @ [bs, compute_tile_n] -> [tm, compute_tile_n]
      partial = jax.lax.dot_general(
          qlhs_ref[...],
          qrhs_ref[:, n_start : n_start + compute_tile_n],
          (((1,), (0,)), ((), ())),
          preferred_element_type=jnp.float32,
      )
      partial = partial * slhs_scale

      # Per-block RHS scales from SMEM
      rhs_scales = []
      for si in range(n_lane_multiplier):
        global_ni = pid_n * n_blocks_per_cell + ni * n_lane_multiplier + si
        rhs_scales.append(srhs_ref[ki, global_ni])

      # Reshape-based scale application (free on TPU, no data movement)
      rhs_scale_vec = jnp.array(rhs_scales).reshape(1, n_lane_multiplier, 1)
      partial = partial.reshape(actual_tm, n_lane_multiplier, bs) * rhs_scale_vec
      partial = partial.reshape(actual_tm, compute_tile_n)

      # Direct slice write to scratch (no accumulator concatenation)
      acc_scratch[:, n_start : n_start + compute_tile_n] += partial

    @pl.when(ki == num_k_blocks - 1)
    def _store():
      out_ref[...] = acc_scratch[...].astype(jnp.bfloat16)

  grid = (M // actual_tm, N // actual_tn, num_k_blocks)
  out = pl.pallas_call(
      _kernel,
      out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=grid,
          in_specs=[
              pl.BlockSpec((actual_tm, block_size), lambda mi, ni, ki, _s: (mi, ki)),
              pl.BlockSpec((num_k_blocks, actual_tm), lambda mi, ni, ki, _s: (0, mi)),
              pl.BlockSpec((block_size, actual_tn), lambda mi, ni, ki, _s: (ki, ni)),
          ],
          out_specs=pl.BlockSpec((actual_tm, actual_tn), lambda mi, ni, ki, _s: (mi, ni)),
          scratch_shapes=[pltpu.VMEM((actual_tm, actual_tn), jnp.float32)],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
      ),
      interpret=interpret,
  )(srhs, qlhs, slhs_T, qrhs)
  return out


def blockwise_fp8_matmul_1dx1d(
    qlhs, slhs, qrhs, srhs, block_size=128, th=512, td=512, tm=1024, interpret=False, use_pallas=True
):
  """Blockwise FP8 matmul: 1D-scaled LHS^T @ 1D-scaled RHS.

  For wgrad: contracts on M axis. lhs^T @ rhs = [H, D].

  Uses configurable M-axis tiling with an inner M-loop to reduce grid
  iterations from M // block_size to M // actual_tm.

  Args:
    qlhs: [M, H] fp8, column-independent M-axis quantized
    slhs: [M//bs, H] f32, column-independent scales
    qrhs: [M, D] fp8, column-independent M-axis quantized
    srhs: [M//bs, D] f32, column-independent scales
    block_size: quantization block size
    th: tile size on H axis
    td: tile size on D axis
    tm: tile size on M axis (multiple of block_size)
    interpret: if True, run Pallas in interpret mode
    use_pallas: if False, use pure JAX implementation

  Returns:
    [H, D] bf16 result
  """
  if not use_pallas:
    return _blockwise_fp8_matmul_1dx1d_jax(qlhs, slhs, qrhs, srhs, block_size)

  M, H = qlhs.shape
  M2, D = qrhs.shape
  assert M == M2, f"M mismatch: {M} vs {M2}"
  num_m_blocks = M // block_size
  actual_th = _align_tile_tpu(H, th)
  actual_td = _align_tile_tpu(D, td)
  actual_tm = min(tm, M)
  assert actual_tm % block_size == 0, f"tm={actual_tm} not multiple of block_size={block_size}"
  assert M % actual_tm == 0, f"M={M} not divisible by tm={actual_tm}"
  steps_m = actual_tm // block_size
  num_grid_m = M // actual_tm

  def _kernel(qlhs_ref, slhs_ref, qrhs_ref, srhs_ref, out_ref, acc_scratch):
    pid_m = pl.program_id(2)

    @pl.when(pid_m == 0)
    def _zero_acc():
      acc_scratch[...] = jnp.zeros_like(acc_scratch)

    for mi in range(steps_m):
      m_start = mi * block_size
      global_mi = pid_m * steps_m + mi

      # Extract [bs, th] and [bs, td] sub-tiles
      qlhs_sub = qlhs_ref[m_start : m_start + block_size, :]
      qrhs_sub = qrhs_ref[m_start : m_start + block_size, :]

      # FP8 matmul: [bs, th]^T @ [bs, td] -> [th, td]
      partial = jax.lax.dot_general(
          qlhs_sub,
          qrhs_sub,
          (((0,), (0,)), ((), ())),
          preferred_element_type=jnp.float32,
      )

      # Scale extraction using global M-block index
      slhs_scale = slhs_ref[pl.ds(global_mi, 1), :]  # [1, th]
      slhs_scale = slhs_scale.reshape(actual_th, 1)
      srhs_scale = srhs_ref[pl.ds(global_mi, 1), :]  # [1, td]
      partial = partial * slhs_scale * srhs_scale

      acc_scratch[...] += partial

    @pl.when(pid_m == num_grid_m - 1)
    def _store():
      out_ref[...] = acc_scratch[...].astype(jnp.bfloat16)

  grid = (H // actual_th, D // actual_td, num_grid_m)
  out = pl.pallas_call(
      _kernel,
      out_shape=jax.ShapeDtypeStruct((H, D), jnp.bfloat16),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              pl.BlockSpec((actual_tm, actual_th), lambda hi, di, mi: (mi, hi)),
              pl.BlockSpec((num_m_blocks, actual_th), lambda hi, di, mi: (0, hi)),
              pl.BlockSpec((actual_tm, actual_td), lambda hi, di, mi: (mi, di)),
              pl.BlockSpec((num_m_blocks, actual_td), lambda hi, di, mi: (0, di)),
          ],
          out_specs=pl.BlockSpec((actual_th, actual_td), lambda hi, di, mi: (hi, di)),
          scratch_shapes=[pltpu.VMEM((actual_th, actual_td), jnp.float32)],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
      ),
      interpret=interpret,
  )(qlhs, slhs, qrhs, srhs)
  return out


# ---------------------------------------------------------------------------
# Fused quantize + matmul kernels
# ---------------------------------------------------------------------------


def fused_blockwise_fp8_matmul_1dx2d(
    lhs,
    rhs,
    fp8_dtype=jnp.float8_e4m3fn,
    block_size=128,
    tm=1024,
    tk=1024,
    tn=1024,
    interpret=False,
    rhs_quantized=False,
    rhs_scale=None,
    n_lane_multiplier=4,
):
  """Fused blockwise FP8 quantize + matmul: 1D-scaled LHS x 2D-scaled RHS.

  Loads bf16 data, quantizes in VMEM, and computes FP8 matmul in a single
  kernel call, eliminating intermediate HBM round-trips for quantized tensors.

  Uses multi-block tiling: loads [tm, tk] LHS and [tk, tn] RHS tiles into VMEM,
  then processes sub-blocks of size [tm, block_size] @ [block_size, block_size]
  in unrolled Python loops. This reduces grid iterations and allows the compiler
  to overlap quantization with MXU computation.

  When rhs_quantized=True, RHS is already fp8-quantized with 2D blockwise
  scales, eliminating RHS quantization from the kernel. This is useful when
  the same RHS (weight matrix) is used in both forward and backward dgrad.

  Forward and dgrad: [M, K] @ [K, N] -> [M, N]

  Args:
    lhs: [M, K] bf16/f32 input (unquantized)
    rhs: [K, N] fp8 (if rhs_quantized) or bf16/f32 (otherwise)
    fp8_dtype: target FP8 dtype
    block_size: quantization block size (default 128)
    tm: tile size on M axis
    tk: tile size on K axis (multiple of block_size)
    tn: tile size on N axis (multiple of block_size)
    interpret: if True, run Pallas in interpret mode
    rhs_quantized: if True, rhs is already fp8, rhs_scale provides 2D scales
    rhs_scale: [K//bs, N//bs] f32 scales (required when rhs_quantized=True)

  Returns:
    [M, N] bf16 result
  """
  M, K = lhs.shape
  K2, N = rhs.shape
  assert K == K2, f"K mismatch: {K} vs {K2}"
  assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"
  assert N % block_size == 0, f"N={N} not divisible by block_size={block_size}"

  fp8_max, fp8_min = _get_fp8_bounds(fp8_dtype)
  actual_tm = min(tm, M)
  actual_tk = min(tk, K)
  while actual_tk > block_size and K % actual_tk != 0:
    actual_tk -= block_size
  actual_tn = min(tn, N)
  while actual_tn > block_size and N % actual_tn != 0:
    actual_tn -= block_size
  assert actual_tk % block_size == 0, f"tk={actual_tk} not multiple of block_size={block_size}"
  assert actual_tn % block_size == 0, f"tn={actual_tn} not multiple of block_size={block_size}"
  assert M % actual_tm == 0, f"M={M} not divisible by tm={actual_tm}"
  assert K % actual_tk == 0, f"K={K} not divisible by tk={actual_tk}"
  assert N % actual_tn == 0, f"N={N} not divisible by tn={actual_tn}"

  if rhs_quantized:
    assert rhs_scale is not None, "rhs_scale required when rhs_quantized=True"
    nk, nn = K // block_size, N // block_size
    assert rhs_scale.shape == (nk, nn), f"rhs_scale shape {rhs_scale.shape} != ({nk}, {nn})"

  steps_k = actual_tk // block_size
  n_blocks_n = actual_tn // block_size
  n_lane_multiplier = min(n_lane_multiplier, n_blocks_n)
  while n_blocks_n % n_lane_multiplier != 0:
    n_lane_multiplier -= 1
  compute_tile_n = block_size * n_lane_multiplier
  steps_n = actual_tn // compute_tile_n
  num_grid_k = K // actual_tk
  bs = block_size

  def _kernel(lhs_ref, rhs_ref, out_ref, acc_scratch):
    pid_k = pl.program_id(2)

    @pl.when(pid_k == 0)
    def _zero_acc():
      acc_scratch[...] = jnp.zeros_like(acc_scratch)

    accumulators = [None] * steps_n

    for ki in range(steps_k):
      k_start = ki * bs

      # Quantize LHS [tm, 128] with 1D per-row scales
      lhs_sub = lhs_ref[:, k_start : k_start + bs].astype(jnp.float32)
      lhs_absmax = jnp.max(jnp.abs(lhs_sub), axis=1, keepdims=True)  # [tm, 1]
      lhs_scale = jnp.where(lhs_absmax == 0, 1.0, lhs_absmax / fp8_max)
      qlhs = (lhs_sub / lhs_scale).clip(fp8_min, fp8_max).astype(fp8_dtype)

      for ni in range(steps_n):
        n_start = ni * compute_tile_n

        # Quantize each [bs, bs] sub-block separately, then concat
        qrhs_parts = []
        rhs_scales = []
        for si in range(n_lane_multiplier):
          sub_start = n_start + si * bs
          rhs_sub = rhs_ref[k_start : k_start + bs, sub_start : sub_start + bs].astype(jnp.float32)
          rhs_absmax = jnp.max(jnp.abs(rhs_sub))
          rhs_s = jnp.where(rhs_absmax == 0, 1.0, rhs_absmax / fp8_max)
          qrhs_parts.append((rhs_sub / rhs_s).clip(fp8_min, fp8_max).astype(fp8_dtype))
          rhs_scales.append(rhs_s)
        qrhs_wide = jnp.concatenate(qrhs_parts, axis=1)

        # Wide FP8 matmul: [tm, bs] @ [bs, compute_tile_n] -> [tm, compute_tile_n]
        partial = jax.lax.dot_general(
            qlhs,
            qrhs_wide,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        partial = partial * lhs_scale

        # Apply per-block RHS scales (each scalar covers bs columns)
        scaled = []
        for si in range(n_lane_multiplier):
          scaled.append(partial[:, si * bs : (si + 1) * bs] * rhs_scales[si])
        partial = jnp.concatenate(scaled, axis=1)

        if ki == 0:
          accumulators[ni] = partial
        else:
          accumulators[ni] += partial

    # Concatenate N sub-blocks: steps_n x [tm, compute_tile_n] -> [tm, actual_tn]
    acc_block = jnp.concatenate(accumulators, axis=1)
    acc_scratch[...] += acc_block

    @pl.when(pid_k == num_grid_k - 1)
    def _store():
      out_ref[...] = acc_scratch[...].astype(jnp.bfloat16)

  grid = (M // actual_tm, N // actual_tn, num_grid_k)

  if rhs_quantized:

    def _kernel_preq(rhs_scale_ref, lhs_ref, rhs_ref, out_ref, acc_scratch):
      pid_k = pl.program_id(2)
      pid_n = pl.program_id(1)

      @pl.when(pid_k == 0)
      def _zero_acc():
        acc_scratch[...] = jnp.zeros_like(acc_scratch)

      accumulators = [None] * steps_n
      # Number of block_size blocks per grid cell on N axis
      n_blocks_per_cell = actual_tn // bs

      for ki in range(steps_k):
        k_start = ki * bs
        global_ki = pid_k * steps_k + ki

        # Quantize LHS [tm, 128] with 1D per-row scales (still fused)
        lhs_sub = lhs_ref[:, k_start : k_start + bs].astype(jnp.float32)
        lhs_absmax = jnp.max(jnp.abs(lhs_sub), axis=1, keepdims=True)
        lhs_scale = jnp.where(lhs_absmax == 0, 1.0, lhs_absmax / fp8_max)
        qlhs = (lhs_sub / lhs_scale).clip(fp8_min, fp8_max).astype(fp8_dtype)

        for ni in range(steps_n):
          n_start = ni * compute_tile_n

          # Load wide pre-quantized RHS [bs, compute_tile_n]
          qrhs_wide = rhs_ref[k_start : k_start + bs, n_start : n_start + compute_tile_n]

          # Look up per-block scales from SMEM
          rhs_scales = []
          for si in range(n_lane_multiplier):
            global_ni = pid_n * n_blocks_per_cell + ni * n_lane_multiplier + si
            rhs_scales.append(rhs_scale_ref[global_ki, global_ni])

          # Wide FP8 matmul + LHS scale
          partial = jax.lax.dot_general(
              qlhs,
              qrhs_wide,
              (((1,), (0,)), ((), ())),
              preferred_element_type=jnp.float32,
          )
          partial = partial * lhs_scale

          # Apply per-block RHS scales
          scaled = []
          for si in range(n_lane_multiplier):
            scaled.append(partial[:, si * bs : (si + 1) * bs] * rhs_scales[si])
          partial = jnp.concatenate(scaled, axis=1)

          if ki == 0:
            accumulators[ni] = partial
          else:
            accumulators[ni] += partial

      acc_block = jnp.concatenate(accumulators, axis=1)
      acc_scratch[...] += acc_block

      @pl.when(pid_k == num_grid_k - 1)
      def _store():
        out_ref[...] = acc_scratch[...].astype(jnp.bfloat16)

    out = pl.pallas_call(
        _kernel_preq,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=grid,
            in_specs=[
                pl.BlockSpec((actual_tm, actual_tk), lambda mi, ni, ki, _s: (mi, ki)),
                pl.BlockSpec((actual_tk, actual_tn), lambda mi, ni, ki, _s: (ki, ni)),
            ],
            out_specs=pl.BlockSpec((actual_tm, actual_tn), lambda mi, ni, ki, _s: (mi, ni)),
            scratch_shapes=[pltpu.VMEM((actual_tm, actual_tn), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
        interpret=interpret,
    )(rhs_scale, lhs, rhs)
  else:
    out = pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((actual_tm, actual_tk), lambda mi, ni, ki: (mi, ki)),
                pl.BlockSpec((actual_tk, actual_tn), lambda mi, ni, ki: (ki, ni)),
            ],
            out_specs=pl.BlockSpec((actual_tm, actual_tn), lambda mi, ni, ki: (mi, ni)),
            scratch_shapes=[pltpu.VMEM((actual_tm, actual_tn), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
        interpret=interpret,
    )(lhs, rhs)
  return out


def fused_blockwise_fp8_matmul_1dx1d(
    lhs, rhs, fp8_dtype=jnp.float8_e4m3fn, block_size=128, th=512, td=512, interpret=False
):
  """Fused blockwise FP8 quantize + matmul: 1D-scaled LHS^T x 1D-scaled RHS.

  Loads bf16 data, quantizes in VMEM, and computes FP8 wgrad in a single
  kernel call, eliminating intermediate HBM round-trips.

  For wgrad: contracts on M axis. lhs^T @ rhs = [H, D].

  Args:
    lhs: [M, H] bf16/f32 input (unquantized)
    rhs: [M, D] bf16/f32 input (unquantized)
    fp8_dtype: target FP8 dtype
    block_size: quantization block size (default 128)
    th: tile size on H axis
    td: tile size on D axis
    interpret: if True, run Pallas in interpret mode

  Returns:
    [H, D] bf16 result
  """
  M, H = lhs.shape
  M2, D = rhs.shape
  assert M == M2, f"M mismatch: {M} vs {M2}"
  assert M % block_size == 0, f"M={M} not divisible by block_size={block_size}"

  fp8_max, fp8_min = _get_fp8_bounds(fp8_dtype)
  num_m_blocks = M // block_size
  actual_th = _align_tile_tpu(H, th)
  actual_td = _align_tile_tpu(D, td)
  assert H % actual_th == 0, f"H={H} not divisible by th={actual_th}"
  assert D % actual_td == 0, f"D={D} not divisible by td={actual_td}"

  def _kernel(lhs_ref, rhs_ref, out_ref, acc_scratch):
    mi = pl.program_id(2)

    @pl.when(mi == 0)
    def _zero_acc():
      acc_scratch[...] = jnp.zeros_like(acc_scratch)

    lhs_f32 = lhs_ref[...].astype(jnp.float32)  # [128, th]
    rhs_f32 = rhs_ref[...].astype(jnp.float32)  # [128, td]

    # Quantize LHS: 1D column-independent (absmax along M/axis=0)
    lhs_absmax = jnp.max(jnp.abs(lhs_f32), axis=0, keepdims=True)  # [1, th]
    lhs_scale = jnp.where(lhs_absmax == 0, 1.0, lhs_absmax / fp8_max)
    qlhs = (lhs_f32 / lhs_scale).clip(fp8_min, fp8_max).astype(fp8_dtype)

    # Quantize RHS: 1D column-independent (absmax along M/axis=0)
    rhs_absmax = jnp.max(jnp.abs(rhs_f32), axis=0, keepdims=True)  # [1, td]
    rhs_scale = jnp.where(rhs_absmax == 0, 1.0, rhs_absmax / fp8_max)
    qrhs = (rhs_f32 / rhs_scale).clip(fp8_min, fp8_max).astype(fp8_dtype)

    # FP8 matmul: [128, th]^T @ [128, td] -> [th, td], contract axis 0
    partial = jax.lax.dot_general(
        qlhs,
        qrhs,
        (((0,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    # Apply scales: outer-product broadcast [th, 1] x [1, td]
    partial = partial * lhs_scale.reshape(actual_th, 1) * rhs_scale
    acc_scratch[...] += partial

    @pl.when(mi == num_m_blocks - 1)
    def _store():
      out_ref[...] = acc_scratch[...].astype(jnp.bfloat16)

  grid = (H // actual_th, D // actual_td, num_m_blocks)
  out = pl.pallas_call(
      _kernel,
      out_shape=jax.ShapeDtypeStruct((H, D), jnp.bfloat16),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=[
              pl.BlockSpec((block_size, actual_th), lambda hi, di, mi: (mi, hi)),
              pl.BlockSpec((block_size, actual_td), lambda hi, di, mi: (mi, di)),
          ],
          out_specs=pl.BlockSpec((actual_th, actual_td), lambda hi, di, mi: (hi, di)),
          scratch_shapes=[pltpu.VMEM((actual_th, actual_td), jnp.float32)],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
      ),
      interpret=interpret,
  )(lhs, rhs)
  return out


# ---------------------------------------------------------------------------
# Dimension number helpers
# ---------------------------------------------------------------------------


def _canonicalize_to_2d(lhs, rhs, dimension_numbers):
  """Canonicalize arbitrary dot_general args to 2D [M, K] x [K, N].

  For batch dims, we reshape to [B, M, K] x [B, K, N] and return B.
  The caller should vmap or loop over B.

  Returns:
    lhs_2d: [M, K] or [B, M, K]
    rhs_2d: [K, N] or [B, K, N]
    batch_size: int (0 if no batch dims)
    out_shape: expected output shape from original dot_general
  """
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

  lhs_all = set(range(lhs.ndim))
  rhs_all = set(range(rhs.ndim))
  lhs_free = sorted(lhs_all - set(lhs_contract) - set(lhs_batch))
  rhs_free = sorted(rhs_all - set(rhs_contract) - set(rhs_batch))

  # Compute sizes
  batch_sizes = [lhs.shape[d] for d in lhs_batch]
  batch_size = 1
  for s in batch_sizes:
    batch_size *= s
  lhs_free_size = 1
  for d in lhs_free:
    lhs_free_size *= lhs.shape[d]
  rhs_free_size = 1
  for d in rhs_free:
    rhs_free_size *= rhs.shape[d]
  contract_size = 1
  for d in lhs_contract:
    contract_size *= lhs.shape[d]

  # Expected output shape: batch_dims + lhs_free_dims + rhs_free_dims
  out_shape = list(batch_sizes)
  for d in lhs_free:
    out_shape.append(lhs.shape[d])
  for d in rhs_free:
    out_shape.append(rhs.shape[d])

  # Transpose LHS to [batch, free, contract] then reshape to [B, M, K] or [M, K]
  lhs_perm = list(lhs_batch) + list(lhs_free) + list(lhs_contract)
  lhs = jnp.transpose(lhs, lhs_perm)

  # Transpose RHS to [batch, contract, free] then reshape to [B, K, N] or [K, N]
  rhs_perm = list(rhs_batch) + list(rhs_contract) + list(rhs_free)
  rhs = jnp.transpose(rhs, rhs_perm)

  if lhs_batch:
    lhs = lhs.reshape(batch_size, lhs_free_size, contract_size)
    rhs = rhs.reshape(batch_size, contract_size, rhs_free_size)
  else:
    lhs = lhs.reshape(lhs_free_size, contract_size)
    rhs = rhs.reshape(contract_size, rhs_free_size)

  return lhs, rhs, batch_size if lhs_batch else 0, out_shape


# ---------------------------------------------------------------------------
# BlockwiseFp8DotGeneralOp with custom_vjp
# ---------------------------------------------------------------------------


class BlockwiseFp8DotGeneralOp:
  """DeepSeek-style blockwise FP8 dot_general replacement.

  Implements custom_vjp for forward and backward passes with
  fine-grained 128-block scaling.
  """

  def __init__(
      self,
      name=None,
      block_size=128,
      fp8_dtype=jnp.float8_e4m3fn,
      use_pallas=False,
      use_fused=True,
      n_lane_multiplier=4,
      cache_rhs=True,
  ):
    self.block_size = block_size
    self.fp8_dtype = fp8_dtype
    self.interpret = False
    self.use_pallas = use_pallas
    self.use_fused = use_fused
    self.n_lane_multiplier = n_lane_multiplier
    self.cache_rhs = cache_rhs

  def __call__(self, lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None):
    block_size = self.block_size
    fp8_dtype = self.fp8_dtype
    interpret = self.interpret
    use_pallas = self.use_pallas
    use_fused = self.use_fused
    n_lane_multiplier = self.n_lane_multiplier
    cache_rhs = self.cache_rhs

    # Canonicalize to 2D (or 3D with batch)
    lhs_c, rhs_c, batch_size, out_shape = _canonicalize_to_2d(lhs, rhs, dimension_numbers)
    had_batch = batch_size > 0

    # Extract dimensions as Python ints (captured in closure, not residuals)
    if had_batch:
      _, M, K = lhs_c.shape
      _, _, N = rhs_c.shape
    else:
      M, K = lhs_c.shape
      _, N = rhs_c.shape

    # Compute padding as Python ints
    m_pad = (block_size - M % block_size) % block_size
    k_pad = (block_size - K % block_size) % block_size
    n_pad = (block_size - N % block_size) % block_size

    # Pad inputs
    if had_batch:
      lhs_c = jnp.pad(lhs_c, ((0, 0), (0, m_pad), (0, k_pad)))
      rhs_c = jnp.pad(rhs_c, ((0, 0), (0, k_pad), (0, n_pad)))
    else:
      lhs_c = jnp.pad(lhs_c, ((0, m_pad), (0, k_pad)))
      rhs_c = jnp.pad(rhs_c, ((0, k_pad), (0, n_pad)))

    def _single_fwd_with_res(l, r):
      """2D forward returning (result, residuals) — all JAX arrays."""
      if use_fused:
        if cache_rhs:
          # Pre-quantize RHS, cache for backward reuse
          qr, sr = quantize_blockwise_2d(r, fp8_dtype=fp8_dtype, block_size=block_size)
          result = fused_blockwise_fp8_matmul_1dx2d(
              l,
              qr,
              fp8_dtype=fp8_dtype,
              block_size=block_size,
              interpret=interpret,
              rhs_quantized=True,
              rhs_scale=sr,
              n_lane_multiplier=n_lane_multiplier,
          )
          return result, (l, qr, sr)
        else:
          # No pre-quantize; fused kernel handles quantization
          result = fused_blockwise_fp8_matmul_1dx2d(
              l,
              r,
              fp8_dtype=fp8_dtype,
              block_size=block_size,
              interpret=interpret,
              rhs_quantized=False,
              rhs_scale=None,
              n_lane_multiplier=n_lane_multiplier,
          )
          return result, (l, r)
      else:
        ql, sl = quantize_blockwise_1d(
            l, fp8_dtype=fp8_dtype, block_size=block_size, axis=-1, interpret=interpret, use_pallas=use_pallas
        )
        qr, sr = quantize_blockwise_2d(r, fp8_dtype=fp8_dtype, block_size=block_size)
        result = blockwise_fp8_matmul_1dx2d(
            ql,
            sl,
            qr,
            sr,
            block_size=block_size,
            n_lane_multiplier=n_lane_multiplier,
            interpret=interpret,
            use_pallas=use_pallas,
        )
        return result, (l, qr, sr)

    if use_fused:
      if cache_rhs:

        def _single_bwd(lhs_saved, qr, sr, g):
          """Fused backward with pre-quantized rhs."""
          dlhs = fused_blockwise_fp8_matmul_1dx2d(
              g,
              jnp.swapaxes(qr, -2, -1),
              fp8_dtype=fp8_dtype,
              block_size=block_size,
              interpret=interpret,
              rhs_quantized=True,
              rhs_scale=jnp.swapaxes(sr, -2, -1),
              n_lane_multiplier=n_lane_multiplier,
          )
          drhs = fused_blockwise_fp8_matmul_1dx1d(
              lhs_saved, g, fp8_dtype=fp8_dtype, block_size=block_size, interpret=interpret
          )
          return dlhs, drhs

      else:

        def _single_bwd(lhs_saved, rhs_raw, g):
          """Fused backward without cached rhs — kernel re-quantizes."""
          dlhs = fused_blockwise_fp8_matmul_1dx2d(
              g,
              jnp.swapaxes(rhs_raw, -2, -1),
              fp8_dtype=fp8_dtype,
              block_size=block_size,
              interpret=interpret,
              rhs_quantized=False,
              rhs_scale=None,
              n_lane_multiplier=n_lane_multiplier,
          )
          drhs = fused_blockwise_fp8_matmul_1dx1d(
              lhs_saved, g, fp8_dtype=fp8_dtype, block_size=block_size, interpret=interpret
          )
          return dlhs, drhs

    else:

      def _single_bwd(lhs_saved, qr, sr, g):
        """2D backward returning (dlhs, drhs) — all JAX arrays."""
        # dgrad: g @ rhs^T  (1D-scaled grad x transposed 2D-scaled rhs)
        qg, sg = quantize_blockwise_1d(
            g, fp8_dtype=fp8_dtype, block_size=block_size, axis=-1, interpret=interpret, use_pallas=use_pallas
        )
        dlhs = blockwise_fp8_matmul_1dx2d(
            qg,
            sg,
            jnp.swapaxes(qr, -2, -1),
            jnp.swapaxes(sr, -2, -1),
            block_size=block_size,
            n_lane_multiplier=n_lane_multiplier,
            interpret=interpret,
            use_pallas=use_pallas,
        )
        # wgrad: lhs^T @ g  (1D-scaled on M-axis for both)
        ql, sl = quantize_blockwise_1d(
            lhs_saved, fp8_dtype=fp8_dtype, block_size=block_size, axis=0, interpret=interpret, use_pallas=use_pallas
        )
        qg2, sg2 = quantize_blockwise_1d(
            g, fp8_dtype=fp8_dtype, block_size=block_size, axis=0, interpret=interpret, use_pallas=use_pallas
        )
        drhs = blockwise_fp8_matmul_1dx1d(
            ql, sl, qg2, sg2, block_size=block_size, interpret=interpret, use_pallas=use_pallas
        )
        return dlhs, drhs

    @jax.custom_vjp
    def _dot(lhs_c, rhs_c):
      if had_batch:
        result = jax.vmap(lambda l, r: _single_fwd_with_res(l, r)[0])(lhs_c, rhs_c)
      else:
        result = _single_fwd_with_res(lhs_c, rhs_c)[0]
      return result[..., :M, :N].reshape(out_shape)

    def _fwd(lhs_c, rhs_c):
      if had_batch:
        results, residuals = jax.vmap(_single_fwd_with_res)(lhs_c, rhs_c)
      else:
        results, residuals = _single_fwd_with_res(lhs_c, rhs_c)
      return results[..., :M, :N].reshape(out_shape), residuals

    def _bwd(residuals, grad):
      if use_fused and not cache_rhs:
        lhs_saved, rhs_raw = residuals
        # Reshape grad to match padded 2D dims, then pad
        if had_batch:
          grad_r = grad.reshape(lhs_saved.shape[0], M, N)
          grad_r = jnp.pad(grad_r, ((0, 0), (0, m_pad), (0, n_pad)))
          dlhs_c, drhs_c = jax.vmap(_single_bwd)(lhs_saved, rhs_raw, grad_r)
        else:
          grad_r = grad.reshape(M, N)
          grad_r = jnp.pad(grad_r, ((0, m_pad), (0, n_pad)))
          dlhs_c, drhs_c = _single_bwd(lhs_saved, rhs_raw, grad_r)
      else:
        lhs_saved, qrhs, srhs = residuals
        # Reshape grad to match padded 2D dims, then pad
        if had_batch:
          grad_r = grad.reshape(lhs_saved.shape[0], M, N)
          grad_r = jnp.pad(grad_r, ((0, 0), (0, m_pad), (0, n_pad)))
          dlhs_c, drhs_c = jax.vmap(_single_bwd)(lhs_saved, qrhs, srhs, grad_r)
        else:
          grad_r = grad.reshape(M, N)
          grad_r = jnp.pad(grad_r, ((0, m_pad), (0, n_pad)))
          dlhs_c, drhs_c = _single_bwd(lhs_saved, qrhs, srhs, grad_r)
      # Return padded 2D gradients — JAX AD through pad + canonicalize
      # (outside _dot) handles unpadding and shape restoration automatically.
      return dlhs_c, drhs_c

    _dot.defvjp(_fwd, _bwd)
    return _dot(lhs_c, rhs_c)
