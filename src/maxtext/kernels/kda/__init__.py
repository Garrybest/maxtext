"""KDA (Kimi Delta Attention) kernels.

Entry point that delegates to the Pallas kernel wrapper, following the
same pattern as ``maxtext.kernels.gla``.
"""

from __future__ import annotations

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
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """KDA entry point using Pallas TPU kernels.

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

  Returns:
      (o, final_state) where o is [B, T, H, V] and final_state is
      [B, H, K, V] or None.
  """
  return pallas_chunk_kda(
      q=q,
      k=k,
      v=v,
      g=g,
      beta=beta,
      scale=scale,
      initial_state=initial_state,
      output_final_state=output_final_state,
      chunk_size=chunk_size,
      A_log=A_log,
      dt_bias=dt_bias,
      use_gate_in_kernel=use_gate_in_kernel,
      use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
      safe_gate=safe_gate,
      lower_bound=lower_bound,
  )
