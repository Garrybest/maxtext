# Copyright 2023–2026 Google LLC
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

"""Grouped matrix multiplication operations with custom VJPs."""

# pylint: disable=too-many-positional-arguments

import functools
import dataclasses
from typing import Literal, List, Tuple
import jax
import jax.numpy as jnp
from maxtext.kernels.megablox import backend
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_kernel as tokamax_backend
import qwix
import qwix.pallas as qpl


def gmm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int, int, int, int, int, int, int] = (128, 128, 128, 128, 128, 128, 128, 128, 128),
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    lhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None,
    rhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None,
    use_qwix_quantization: bool = False,
    use_tokamax_backend: bool = False,
    weight_gather_axes: List[Tuple[str, int]] | None = None,
    input_buffer_count: tuple[int, int, int] = (2, 2, 2),
    combine_scopes: bool = False,
    # TODO(amandaliang): get rid of the qwix_rule in favor of Qwix's interception feature
    qwix_rule: qwix.QtRule | None = None,
):
  """Grouped matrix multiplication operation."""
  quantization_rule = None
  if use_qwix_quantization:
    # get_current_rule has to be called outside of the _gmm_fwd function.
    quantization_rule = qwix_rule if qwix_rule else qpl.get_current_rule("gmm")
    if quantization_rule and not isinstance(quantization_rule, qwix.QtRule):
      raise ValueError("Expect a QtRule for quantized training.")
  else:
    # Handcraft a rule that matches the AQT's behavior.
    if lhs_quantize_dtype or rhs_quantize_dtype:
      quantization_rule = qwix.QtRule(
          weight_qtype=rhs_quantize_dtype,
          weight_calibration_method="absmax",
          act_qtype=lhs_quantize_dtype,
          act_calibration_method="absmax",
      )

  gmm_fwd_bwd = lambda *args: _gmm_fwd(*args)[0]  # pylint: disable=C3001
  gmm_fwd_bwd = jax.custom_vjp(gmm_fwd_bwd, nondiff_argnums=(3, 4, 5, 6, 9, 10, 11, 12, 13))
  gmm_fwd_bwd.defvjp(_gmm_fwd, functools.partial(_gmm_bwd, lhs.dtype, rhs.dtype))
  return gmm_fwd_bwd(
      lhs,
      rhs,
      group_sizes,
      preferred_element_type,
      tiling,
      input_buffer_count,
      combine_scopes,
      group_offset,
      existing_out,
      transpose_rhs,
      interpret,
      quantization_rule,
      use_tokamax_backend,
      weight_gather_axes,
  )


def _gmm_fwd(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int, int, int, int, int, int, int] = (128, 128, 128, 128, 128, 128, 128, 128, 128),
    input_buffer_count: tuple[int, int, int] = (2, 2, 2),
    combine_scopes: bool = False,
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    quantization_rule: qwix.QtRule | None = None,
    use_tokamax_backend: bool = False,
    weight_gather_axes: List[Tuple[str, int]] | None = None,
) -> tuple[
    jnp.ndarray,
    tuple[
        jnp.ndarray | qpl.QArray,
        jnp.ndarray | qpl.QArray,
        jnp.ndarray,
        jnp.ndarray | None,
        jnp.ndarray | qpl.QArray | None,
    ],
]:
  """Forward function for GMM VJP."""
  lhs_t = None  # Pre-computed transposed FP8 activation for backward tgmm.
  tile_size = getattr(quantization_rule, "tile_size", None) if quantization_rule else None
  if tile_size:
    # Cap kernel tiling to quantization block size. tokamax quant_block_spec
    # requires eps % kernel_tile == 0 on non-reduction axes, so kernel tiles
    # must not exceed the quantization block size.
    tiling = tuple(min(t, tile_size) for t in tiling)
  if quantization_rule:
    if quantization_rule.act_qtype and not isinstance(lhs, qpl.QArray):
      if tile_size and use_tokamax_backend:
        # TE-style: quantize both orientations from BF16 source (avoids double
        # quantization error). lhs [M, K] with (1, tile_size) blocks for
        # forward gmm; lhs_t [K, M] with (1, tile_size) blocks for backward
        # tgmm, ensuring tile_size-element blocks on the tgmm reduction axis.
        lhs_bf16 = lhs
        act_cw = [] if quantization_rule.disable_channelwise_axes else [0]
        act_tiled = {1: tile_size}
        lhs = qpl.quantize(
            lhs_bf16,
            quantization_rule.act_qtype,
            channelwise_axes=act_cw,
            tiled_axes=act_tiled,
            calibration_method=quantization_rule.act_calibration_method,
            scale_dtype=jnp.float32,
        )
        lhs_t = qpl.quantize(
            lhs_bf16.swapaxes(0, 1),
            quantization_rule.act_qtype,
            channelwise_axes=act_cw,
            tiled_axes=act_tiled,
            calibration_method=quantization_rule.act_calibration_method,
            scale_dtype=jnp.float32,
        )
      else:
        lhs = qpl.quantize(
            lhs,
            quantization_rule.act_qtype,
            channelwise_axes=[] if quantization_rule.disable_channelwise_axes else ([] if tile_size else [0]),
            tiled_axes={0: tile_size, 1: tile_size} if tile_size else None,
            calibration_method=quantization_rule.act_calibration_method,
            scale_dtype=jnp.float32,
        )
    if quantization_rule.weight_qtype and not isinstance(rhs, qpl.QArray):
      rhs = qpl.quantize(
          rhs,
          quantization_rule.weight_qtype,
          # If only considering the fwd pass, we could also enable channelwise
          # axes for the group axis, i.e., [0, 1 or 2]. However, this makes the
          # bwd pass unable to reuse the scale easily.
          channelwise_axes=[]
          if quantization_rule.disable_channelwise_axes
          else ([0] if tile_size else ([1] if transpose_rhs else [2])),
          tiled_axes={1: tile_size, 2: tile_size} if tile_size else None,
          calibration_method=quantization_rule.weight_calibration_method,
          scale_dtype=jnp.float32,
      )
      # QAG is only supported for following conditions
  if use_tokamax_backend:
    if (
        quantization_rule
        and quantization_rule.bwd_qtype
        and quantization_rule.weight_calibration_method.startswith("fixed")
        and isinstance(rhs, qpl.QArray)
        and weight_gather_axes
    ):
      for axis_name, axis_idx in weight_gather_axes:
        rhs_qvalue = jax.lax.all_gather(rhs.qvalue, axis_name, axis=axis_idx, tiled=True)
        if _is_blockwise_qarray(rhs):
          rhs_scale = jax.lax.all_gather(rhs.scale, axis_name, axis=axis_idx, tiled=True)
          rhs = dataclasses.replace(rhs, qvalue=rhs_qvalue, scale=rhs_scale)
        else:
          rhs = dataclasses.replace(rhs, qvalue=rhs_qvalue)
    out = tokamax_backend.gmm(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        out_dtype=preferred_element_type,
        tiling=tiling[:3],
        group_offset=group_offset,
        transpose_rhs=transpose_rhs,
        interpret=interpret,
        input_buffer_count=input_buffer_count[0],
    )
  else:
    out = backend.gmm(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling[:3],
        group_offset,
        existing_out,
        transpose_rhs=transpose_rhs,
        interpret=interpret,
    )
  return out, (lhs, rhs, group_sizes, group_offset, lhs_t)


def _is_blockwise_qarray(q: qpl.QArray) -> bool:
  """Check if a QArray has block-wise (not per-channel) scales."""
  return any(sd not in (1, qd) for sd, qd in zip(q.scale.shape, q.qvalue.shape))


def _gmm_bwd(
    lhs_dtype: jax.typing.DTypeLike,
    rhs_dtype: jax.typing.DTypeLike,
    preferred_element_type: jnp.dtype,
    tiling: tuple[int, int, int, int, int, int, int, int, int],
    input_buffer_count: tuple[int, int, int],
    combine_scopes: bool,
    transpose_rhs: bool,
    interpret: bool,
    quantization_rule: qwix.QtRule | None,
    use_tokamax_backend: bool,
    weight_gather_axes: List[Tuple[str, int]] | None,
    residual: tuple[
        jnp.ndarray | qpl.QArray,
        jnp.ndarray | qpl.QArray,
        jnp.ndarray,
        jnp.ndarray | None,
        jnp.ndarray | qpl.QArray | None,
    ],
    grad: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, None, None, jnp.ndarray]:
  """Backward function for throughput GMM VJP."""
  del preferred_element_type
  lhs, rhs, group_sizes, group_offset, lhs_t = residual
  num_actual_groups = rhs.shape[0]

  tile_size = getattr(quantization_rule, "tile_size", None) if quantization_rule else None
  if tile_size:
    # Cap kernel tiling to quantization block size (see _gmm_fwd).
    tiling = tuple(min(t, tile_size) for t in tiling)

  # Jargon used here:
  #  - lhs: input activation in forward pass, possibly quantized.
  #  - rhs: weight in forward pass, possibly quantized.
  #  - dout (or grad): the incoming gradient in the backward pass.
  #  - dlhs: gradient of the lhs in the backward pass, what we want to compute.
  #  - drhs: gradient of the rhs in the backward pass, what we want to compute.
  #  - dlhs_dout: the incoming gradient used to calculate dlhs.
  #  - drhs_dout: the incoming gradient used to calculate drhs.

  # dlhs_dout and drhs_dout can be different when quantization is enabled.
  dlhs_dout = grad
  drhs_dout = grad
  if isinstance(rhs, qpl.QArray):
    if _is_blockwise_qarray(rhs):
      # Block-wise: keep QArray as-is, kernel handles scale internally.
      pass
    else:
      # Per-channel: apply rhs.scale to dlhs_dout (dual quantization trick).
      # qvalue: [g, k, n] scale: [1, 1, n]
      dlhs_dout *= rhs.scale.astype(grad.dtype).reshape(1, -1)  # [1, n]
      rhs = rhs.qvalue
  if isinstance(lhs, qpl.QArray):
    if _is_blockwise_qarray(lhs):
      # Block-wise: keep QArray as-is, kernel handles scale internally.
      pass
    else:
      # Per-channel: apply lhs.scale to drhs_dout (dual quantization trick).
      # qvalue: [m, k] scale: [m, 1]
      drhs_dout *= lhs.scale.astype(grad.dtype)
      lhs = lhs.qvalue
  if quantization_rule and quantization_rule.bwd_qtype:
    # Enable backward pass quantization
    if lhs_t is not None:
      # TE-style dual quantization for dout [M, N]:
      #   dlhs_dout: tiled_axes={1: tile_size}, channelwise on M
      #     → scale [M, N//tile_size], 1×128 blocks (gmm, M is non-reduction)
      #   drhs_dout: tiled_axes={0: tile_size}, channelwise on N
      #     → scale [M//tile_size, N], 1×128 blocks on reduction axis
      dlhs_dout = qpl.quantize(
          dlhs_dout,
          quantization_rule.bwd_qtype,
          channelwise_axes=[] if quantization_rule.disable_channelwise_axes else [0],
          tiled_axes={1: tile_size},
          calibration_method=quantization_rule.bwd_calibration_method,
          scale_dtype=jnp.float32,
      )
      drhs_dout = qpl.quantize(
          drhs_dout,
          quantization_rule.bwd_qtype,
          channelwise_axes=[1],
          tiled_axes={0: tile_size},
          calibration_method=quantization_rule.bwd_calibration_method,
          scale_dtype=jnp.float32,
      )
    else:
      dlhs_dout = qpl.quantize(
          dlhs_dout,
          quantization_rule.bwd_qtype,
          channelwise_axes=[] if quantization_rule.disable_channelwise_axes else ([] if tile_size else [0]),
          tiled_axes={0: tile_size, 1: tile_size} if tile_size else None,
          calibration_method=quantization_rule.bwd_calibration_method,
          scale_dtype=jnp.float32,
      )
      drhs_dout = qpl.quantize(
          drhs_dout,
          quantization_rule.bwd_qtype,
          channelwise_axes=[] if quantization_rule.disable_channelwise_axes else ([] if tile_size else [1]),
          tiled_axes={0: tile_size, 1: tile_size} if tile_size else None,
          calibration_method=quantization_rule.bwd_calibration_method,
          scale_dtype=jnp.float32,
      )
  if use_tokamax_backend:
    dlhs = tokamax_backend.gmm(
        lhs=dlhs_dout,
        rhs=rhs,
        group_sizes=group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        out_dtype=lhs_dtype,
        tiling=tiling[3:6],
        group_offset=group_offset,
        transpose_rhs=not transpose_rhs,
        interpret=interpret,
        input_buffer_count=input_buffer_count[1],
    )
    drhs = tokamax_backend.tgmm(
        lhs=lhs_t if lhs_t is not None else lhs.swapaxes(0, 1),
        rhs=drhs_dout,
        group_sizes=group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        out_dtype=rhs_dtype,
        tiling=tiling[-3:],
        group_offset=group_offset,
        num_actual_groups=num_actual_groups,
        interpret=interpret,
        input_buffer_count=input_buffer_count[2],
        combine_scopes=combine_scopes,
    )
    if quantization_rule and quantization_rule.bwd_qtype and weight_gather_axes:
      # Scatter back in reverse order of gather
      for axis_name, axis_idx in reversed(weight_gather_axes):
        drhs = jax.lax.psum_scatter(drhs, axis_name, scatter_dimension=axis_idx, tiled=True)
  else:
    dlhs = backend.gmm(
        dlhs_dout,
        rhs,
        group_sizes,
        lhs_dtype,
        tiling[3:6],
        group_offset,
        transpose_rhs=not transpose_rhs,
        interpret=interpret,
    )
    drhs = backend.tgmm(
        lhs.swapaxes(0, 1),
        drhs_dout,
        group_sizes,
        rhs_dtype,
        tiling[-3:],
        group_offset,
        num_actual_groups,
        interpret=interpret,
    )

  # NOTE: If the rhs transposition is fused into the forward pass we need to
  # return the transpose of the rhs gradient that we calculated above.
  #
  # TODO(tgale, enriqueps, apaske): Fuse this transposition into the tgmm.
  drhs = drhs.swapaxes(1, 2) if transpose_rhs else drhs
  return dlhs, drhs, None, None, grad
