"""Wrapper around pallas-kernel's chunk GLA with custom_vjp.

Delegates to fused chunk GLA kernels (``tops.ops.gla.chunk_fused_kernels``)
for both forward and backward when conditions are met (no initial_state,
no output_final_state), registering a ``jax.custom_vjp`` so JAX can
differentiate through the Pallas kernels.

Falls back to ``tops.ops.simple_gla.chunk`` when initial_state or
output_final_state is used (the fused kernels do not support these).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

try:
  from tops.ops.simple_gla import simple_gla_bwd
  from tops.ops.simple_gla import simple_gla_fwd
  from tops.ops.simple_gla import SimpleGLAKernelMode
except ImportError as e:
  raise ImportError(
      "pallas-kernel (tops) is required for Pallas GLA kernels. " "Install it from the pallas-kernel submodule."
  ) from e

# Fused chunk GLA kernels require K and V to be multiples of this.
_KV_ALIGN = 128


# ---------------------------------------------------------------------------
# custom_vjp implementation
# ---------------------------------------------------------------------------


def _resolve_scale(scale: float | None, K: int) -> float:
  return scale if scale is not None else K**-0.5


def _to_1d_g_gamma(g: jnp.ndarray) -> jnp.ndarray:
  """Squeeze g_gamma to shape (H,) as required by chunk GLA kernels."""
  if g.ndim == 1:
    return g
  if g.ndim == 4:
    if not (g.shape[0] == 1 and g.shape[1] == 1 and g.shape[3] == 1):
      raise ValueError(f"4-D g_gamma must be broadcast-constant with shape (1, 1, H, 1), " f"got {g.shape}")
    return g[0, 0, :, 0]
  return g.reshape(-1)


def _pad_axis(x: jnp.ndarray, multiple: int, axis: int) -> jnp.ndarray:
  """Zero-pad tensor along *axis* so its size is a multiple of *multiple*."""
  rem = x.shape[axis] % multiple
  if rem == 0:
    return x
  pad_width = [(0, 0)] * x.ndim
  pad_width[axis] = (0, multiple - rem)
  return jnp.pad(x, pad_width)


def _pad_inputs_fused(q, k, v, chunk_size):
  """Pad T (axis 1) to chunk_size, K and V (axis -1) to _KV_ALIGN."""
  q = _pad_axis(_pad_axis(q, chunk_size, 1), _KV_ALIGN, -1)
  k = _pad_axis(_pad_axis(k, chunk_size, 1), _KV_ALIGN, -1)
  v = _pad_axis(_pad_axis(v, chunk_size, 1), _KV_ALIGN, -1)
  return q, k, v


def _pad_inputs(q, k, v, h0, chunk_size):
  """Pad T (axis 1) to chunk_size, K and V (axis -1) to _KV_ALIGN."""
  q = _pad_axis(_pad_axis(q, chunk_size, 1), _KV_ALIGN, -1)
  k = _pad_axis(_pad_axis(k, chunk_size, 1), _KV_ALIGN, -1)
  v = _pad_axis(_pad_axis(v, chunk_size, 1), _KV_ALIGN, -1)
  if h0 is not None:
    # h0: [B, H, K, V]
    h0 = _pad_axis(_pad_axis(h0, _KV_ALIGN, -2), _KV_ALIGN, -1)
  return q, k, v, h0


def _use_fused_kernels(initial_state, output_final_state) -> bool:
  """Whether to use the fused chunk GLA kernels.

  The fused kernels (chunk_fwd_fused_g_gamma / chunk_bwd_fused_g_gamma)
  merge h propagation + A recomputation + output into a single pallas_call,
  eliminating HBM round-trips.  They do not support initial_state or
  output_final_state.
  """
  return initial_state is None and not output_final_state


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
def pallas_chunk_gla(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g_gamma: jnp.ndarray,
    initial_state: jnp.ndarray | None = None,
    scale: float | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """Chunked GLA using pallas-kernel's Pallas TPU kernels.

  Uses fused chunk kernels when initial_state is None and
  output_final_state is False, falling back to the non-fused
  simple_gla path otherwise.

  Args:
      q: [B, T, H, K] queries
      k: [B, T, H, K] keys
      v: [B, T, H, V] values
      g_gamma: constant log-space gate, shape (H,).
          Typically (H,) for per-head ALiBi-style
          slopes; squeezed to (H,) internally.
      initial_state: optional [B, H, K, V] initial recurrent state
      scale: attention scale (default 1/sqrt(K))
      output_final_state: whether to return the final state
      chunk_size: chunk size BT

  Returns:
      (o, final_state) where o is [B, T, H, V] and final_state is
      [B, H, K, V] or None.
  """
  dtype = q.dtype
  T_orig, K_orig, V_orig = q.shape[1], q.shape[-1], v.shape[-1]
  g_gamma_f32 = _to_1d_g_gamma(g_gamma.astype(jnp.float32))
  s = _resolve_scale(scale, K_orig)

  if _use_fused_kernels(initial_state, output_final_state):
    # Fused path: single pallas_call for fwd, accepts original dtype.
    q_p, k_p, v_p = _pad_inputs_fused(q, k, v, chunk_size)
    o, _ = simple_gla_fwd(
        q_p, k_p, v_p, g_gamma=g_gamma_f32, scale=s, chunk_size=chunk_size, mode=SimpleGLAKernelMode.FUSED_CHUNK
    )
    o = o[:, :T_orig, :, :V_orig]
    return o.astype(dtype), None

  # Fallback: non-fused simple_gla path (supports initial_state / final_state).
  q_f32, k_f32, v_f32 = (x.astype(jnp.float32) for x in (q, k, v))
  h0_f32 = initial_state.astype(jnp.float32) if initial_state is not None else None
  q_f32, k_f32, v_f32, h0_f32 = _pad_inputs(q_f32, k_f32, v_f32, h0_f32, chunk_size)
  o, ht = simple_gla_fwd(
      q_f32,
      k_f32,
      v_f32,
      g_gamma=g_gamma_f32,
      scale=s,
      h0=h0_f32,
      use_ht=output_final_state,
      chunk_size=chunk_size,
      mode=SimpleGLAKernelMode.CHUNK,
  )
  o = o[:, :T_orig, :, :V_orig]
  if ht is not None:
    ht = ht[:, :, :K_orig, :V_orig]
  return o.astype(dtype), (ht if output_final_state else None)


def _pallas_chunk_gla_fwd(
    q,
    k,
    v,
    g_gamma,
    initial_state,
    scale,
    output_final_state,
    chunk_size,
):
  """Forward rule: run fused or non-fused forward and save residuals."""
  dtype = q.dtype
  T_orig, K_orig, V_orig = q.shape[1], q.shape[-1], v.shape[-1]
  g_gamma_f32 = _to_1d_g_gamma(g_gamma.astype(jnp.float32))
  s = _resolve_scale(scale, K_orig)

  if _use_fused_kernels(initial_state, output_final_state):
    # Fused path: the fused kernel accepts original dtype inputs and
    # keeps h in VMEM scratch (no HBM round-trip).
    q_p, k_p, v_p = _pad_inputs_fused(q, k, v, chunk_size)
    o, h_all = simple_gla_fwd(
        q_p, k_p, v_p, g_gamma=g_gamma_f32, scale=s, chunk_size=chunk_size, mode=SimpleGLAKernelMode.FUSED_CHUNK
    )
    o_out = o[:, :T_orig, :, :V_orig]
    primals_out = (o_out.astype(dtype), None)
    # Save padded inputs and h_all for fused backward.
    # Use a sentinel (True) to indicate fused path in residuals.
    residuals = (True, q_p, k_p, v_p, g_gamma_f32, h_all, T_orig, K_orig, V_orig)
    return primals_out, residuals

  # Fallback: non-fused simple_gla path.
  q_f32, k_f32, v_f32 = (x.astype(jnp.float32) for x in (q, k, v))
  h0_f32 = initial_state.astype(jnp.float32) if initial_state is not None else None
  q_f32, k_f32, v_f32, h0_f32 = _pad_inputs(q_f32, k_f32, v_f32, h0_f32, chunk_size)
  o, ht = simple_gla_fwd(
      q_f32,
      k_f32,
      v_f32,
      g_gamma=g_gamma_f32,
      scale=s,
      h0=h0_f32,
      use_ht=output_final_state,
      chunk_size=chunk_size,
      mode=SimpleGLAKernelMode.CHUNK,
  )
  o_out = o[:, :T_orig, :, :V_orig]
  ht_out = ht[:, :, :K_orig, :V_orig] if ht is not None else None
  primals_out = (o_out.astype(dtype), ht_out if output_final_state else None)
  # Use sentinel (False) to indicate non-fused path.
  residuals = (False, q_f32, k_f32, v_f32, g_gamma_f32, h0_f32, T_orig, K_orig, V_orig)
  return primals_out, residuals


def _pallas_chunk_gla_bwd(
    scale,
    output_final_state,
    chunk_size,
    residuals,
    grad_outputs,
):
  """Backward rule: call fused or non-fused backward to compute gradients."""
  is_fused = residuals[0]
  do, dht = grad_outputs

  if is_fused:
    _, q_p, k_p, v_p, g_gamma_f32, h_all, T_orig, K_orig, V_orig = residuals
    qkv_dtype = do.dtype
    s = _resolve_scale(scale, K_orig)

    # Pad do to match the padded T/V used in forward.
    do_p = _pad_axis(do, chunk_size, 1)
    do_p = _pad_axis(do_p, _KV_ALIGN, -1)

    dq, dk, dv, _ = simple_gla_bwd(
        q_p,
        k_p,
        v_p,
        do_p,
        h0=h_all,
        g_gamma=g_gamma_f32,
        scale=s,
        chunk_size=chunk_size,
        mode=SimpleGLAKernelMode.FUSED_CHUNK,
    )
    # Slice gradients back to original dimensions.
    dq = dq[:, :T_orig, :, :K_orig].astype(qkv_dtype)
    dk = dk[:, :T_orig, :, :K_orig].astype(qkv_dtype)
    dv = dv[:, :T_orig, :, :V_orig].astype(qkv_dtype)
    dg = jnp.zeros_like(g_gamma_f32)
    return dq, dk, dv, dg, None

  # Fallback: non-fused simple_gla backward.
  _, q_f32, k_f32, v_f32, g_gamma_f32, h0_f32, T_orig, K_orig, V_orig = residuals
  qkv_dtype = do.dtype
  s = _resolve_scale(scale, K_orig)

  do = do.astype(jnp.float32)
  do = _pad_axis(do, chunk_size, 1)
  do = _pad_axis(do, _KV_ALIGN, -1)
  if dht is not None:
    dht = dht.astype(jnp.float32)
    dht = _pad_axis(dht, _KV_ALIGN, -2)  # K
    dht = _pad_axis(dht, _KV_ALIGN, -1)  # V

  dq, dk, dv, dh0 = simple_gla_bwd(
      q_f32,
      k_f32,
      v_f32,
      do,
      g_gamma=g_gamma_f32,
      scale=s,
      h0=h0_f32,
      dht=dht,
      chunk_size=chunk_size,
      mode=SimpleGLAKernelMode.CHUNK,
  )
  dq = dq[:, :T_orig, :, :K_orig]
  dk = dk[:, :T_orig, :, :K_orig]
  dv = dv[:, :T_orig, :, :V_orig]
  dg = jnp.zeros_like(g_gamma_f32)
  if h0_f32 is None:
    dh0 = None
  else:
    dh0 = dh0[:, :, :K_orig, :V_orig]
  dq = dq.astype(qkv_dtype)
  dk = dk.astype(qkv_dtype)
  dv = dv.astype(qkv_dtype)
  if dh0 is not None:
    dh0 = dh0.astype(qkv_dtype)
  return dq, dk, dv, dg, dh0


pallas_chunk_gla.defvjp(_pallas_chunk_gla_fwd, _pallas_chunk_gla_bwd)
