"""Pallas kernel wrapper for KDA.

Delegates to ``tops.ops.kda.chunk_kda`` for the optimized Pallas
implementation. L2 normalization of Q/K is handled by the caller
(``layers/attention_kda.py``), not inside the kernel.

This module provides 1D single-sample interface (used via vmap from chunk_kda).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from tops.ops.kda import chunk_kda as tops_chunk_kda


def pallas_chunk_kda(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    scale: float | None = None,
    initial_state: jnp.ndarray | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    A_log: jnp.ndarray | None = None,
    dt_bias: jnp.ndarray | None = None,
    use_gate_in_kernel: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    segment_ids: jnp.ndarray | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    cp_context: Any | None = None,
    transpose_state_layout: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """1D single-sample (no batch dim) Pallas KDA kernel wrapper.

  This is called via jax.vmap from chunk_kda for row-level isolation.

  Args:
      q: [T, H, K] queries (single sample).
      k: [T, H, K] keys (single sample).
      v: [T, H, V] values (single sample).
      g: [T, H, K] gate values (single sample).
      beta: [T, H] beta gates (single sample).
      scale: attention scale (default 1/sqrt(K)).
      initial_state: [H, K, V] initial recurrent state (single sample).
      output_final_state: whether to return the final state.
      chunk_size: chunk size BT.
      A_log: [H] learnable decay in log space (no batch dim).
      dt_bias: [H*K] dt bias (no batch dim).
      use_gate_in_kernel: if True, apply A_log/dt_bias in kernel.
      use_qk_l2norm_in_kernel: apply L2 normalization to q/k in kernel.
      safe_gate: numerically safe gate mode.
      lower_bound: gate value lower bound.
      segment_ids: [T] segment IDs for varlen mode (1D, 1-based, 0=padding).
          When None, falls back to legacy behavior.

  Returns:
      (o, final_state) where o is [T, H, V] and final_state is [H, K, V] or None.
  """
  # tops kernel expects [B=1, T, H, *] for q/k/v/g/beta but a 1D [T]
  # segment_ids (it derives cu_seqlens internally).
  q = q[None]
  k = k[None]
  v = v[None]
  g = g[None]
  beta = beta[None]

  o, final_state = tops_chunk_kda(
      q=q,
      k=k,
      v=v,
      g=g,
      beta=beta,
      A_log=A_log,
      dt_bias=dt_bias,
      scale=scale,
      initial_state=initial_state,
      output_final_state=output_final_state,
      use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
      use_gate_in_kernel=use_gate_in_kernel,
      segment_ids=segment_ids,
      safe_gate=safe_gate,
      lower_bound=lower_bound,
      chunk_size=chunk_size,
      disable_recompute=disable_recompute,
      return_intermediate_states=return_intermediate_states,
      cp_context=cp_context,
      transpose_state_layout=transpose_state_layout,
  )

  o = o[0]
  if final_state is not None:
    final_state = final_state[0]
  return o, final_state
