"""GLA (Gated Linear Attention) kernels."""

from __future__ import annotations

import math

import jax.numpy as jnp

from maxtext.kernels.gla.pallas import pallas_chunk_gla


def build_slope_tensor(n_attention_heads: int) -> jnp.ndarray:
  """Build Lightning Attention-2 slope tensor.

  This matches the implementation used in bailing_moe_v2.5/bailing_moe_linear_v2.py.
  """

  def get_slopes(n: int) -> list[float]:
    def get_slopes_power_of_2(n: int) -> list[float]:
      start = 2 ** (-(2 ** -(math.log2(n) - 3)))
      ratio = start
      return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
      return get_slopes_power_of_2(n)
    closest_power_of_2 = 2 ** math.floor(math.log2(n))
    return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]

  slopes = jnp.array(get_slopes(n_attention_heads), dtype=jnp.float32)
  return slopes


def chunk_gla(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g_gamma: jnp.ndarray,
    scale: float | None = None,
    initial_state: jnp.ndarray | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """GLA entry point using Pallas TPU kernels.

  Args:
      q: [B, T, H, K] queries.
      k: [B, T, H, K] keys.
      v: [B, T, H, V] values.
      g_gamma: per-head constant log-space gate, shape (H,).
      scale: attention scale (default 1/sqrt(K)).
      initial_state: optional [B, H, K, V] initial recurrent state.
      output_final_state: whether to return the final state.
      chunk_size: chunk size BT.

  Returns:
      (o, final_state) where o is [B, T, H, V] and final_state is
      [B, H, K, V] or None.
  """
  return pallas_chunk_gla(
      q,
      k,
      v,
      g_gamma,
      initial_state,
      scale,
      output_final_state,
      chunk_size,
  )
