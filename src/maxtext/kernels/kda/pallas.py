"""Pallas kernel wrapper for KDA.

Delegates to ``tops.ops.kda.chunk_kda`` for the optimized Pallas
implementation. L2 normalization of Q/K is handled by the caller
(``layers/attention_kda.py``), not inside the kernel.

Follows the same pattern as ``maxtext.kernels.gla.pallas``.
"""

from __future__ import annotations

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
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """Pallas KDA kernel wrapper.

  Args:
      q: [B, T, H, K] queries.
      k: [B, T, H, K] keys.
      v: [B, T, H, V] values.
      g: [B, T, H, K] gate values.
      beta: [B, T, H] beta gates.
      scale: attention scale (default 1/sqrt(K)).
      initial_state: [B, H, K, V] initial recurrent state.
      output_final_state: whether to return the final state.
      chunk_size: chunk size BT.
      A_log: [H] learnable decay in log space.
      dt_bias: [H*K] dt bias.
      use_gate_in_kernel: if True, apply A_log/dt_bias in kernel.
      use_qk_l2norm_in_kernel: apply L2 normalization to q/k in kernel.
      safe_gate: numerically safe gate mode.
      lower_bound: gate value lower bound.

  Returns:
      (o, final_state) where o is [B, T, H, V].
  """
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
      safe_gate=safe_gate,
      lower_bound=lower_bound,
      chunk_size=chunk_size,
  )
  return o, final_state
