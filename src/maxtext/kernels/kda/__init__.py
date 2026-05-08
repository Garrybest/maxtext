"""KDA (Kimi Delta Attention) kernels.

Entry point that delegates to the Pallas kernel wrapper, following the
same pattern as ``maxtext.kernels.gla``.

This module uses jax.vmap to provide row-level isolation for varlen mode.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from maxtext.kernels.kda.pallas import pallas_chunk_kda


def chunk_kda(
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
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """KDA entry point using Pallas TPU kernels.

  Row-level isolation is structurally guaranteed by vmap: each row's
  recurrent state ``S`` is computed in an independent kernel invocation,
  so segment ids in different rows can never collide.

  Args:
      q: [B, T, H, K] queries.
      k: [B, T, H, K] keys.
      v: [B, T, H, V] values.
      g: [B, T, H, K] gate values (raw or processed).
      beta: [B, T, H] delta rule mixing coefficient.
      scale: attention scale (default 1/sqrt(K)).
      initial_state: optional [B, H, K, V] initial recurrent state.
      output_final_state: whether to return the final state.
      chunk_size: chunk size BT.
      A_log: [H] learnable decay in log space.
      dt_bias: [H*K] dt bias.
      use_gate_in_kernel: if True, apply A_log/dt_bias inside kernel.
      use_qk_l2norm_in_kernel: apply L2 norm to q/k in kernel.
      safe_gate: numerically safe gate mode.
      lower_bound: gate value lower bound.
      segment_ids: [B, T] segment IDs for varlen mode (2D, 1-based, 0=padding).
          When None, falls back to legacy behavior (continuous state).

  Returns:
      (o, final_state) where o is [B, T, H, V] and final_state is
      [B, H, K, V] or None.
  """

  def _per_row(q_r, k_r, v_r, g_r, beta_r, seg_r, init_r):
    return pallas_chunk_kda(
        q=q_r,
        k=k_r,
        v=v_r,
        g=g_r,
        beta=beta_r,
        segment_ids=seg_r,
        initial_state=init_r,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        use_gate_in_kernel=use_gate_in_kernel,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )

  seg_in_axis = 0 if segment_ids is not None else None
  init_in_axis = 0 if initial_state is not None else None
  o, final_state = jax.vmap(
      _per_row,
      in_axes=(0, 0, 0, 0, 0, seg_in_axis, init_in_axis),
  )(q, k, v, g, beta, segment_ids, initial_state)
  return o, final_state
