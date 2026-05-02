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

"""Kimi Delta Attention (KDA) Layer Implementation.

This module implements the KDA (Kimi Delta Attention) layer as described in
RFC-0007. KDA is a linear attention mechanism with Delta Rule correction,
featuring:
  - Depthwise causal 1D convolution for local dependency modeling
  - Numerically safe gate mechanism
  - Q/K L2 normalization

The implementation wraps the optimized `tops.ops.kda.chunk_kda` kernel.
"""


import functools
import math

from flax import nnx
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
from maxtext.kernels.kda import chunk_kda
from tops.cpu.ops.common.l2norm import l2norm_fwd

from maxtext.common.common_types import Config, MODEL_MODE_AUTOREGRESSIVE
from maxtext.layers import linears
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils.sharding import logical_to_mesh_axes


class KimiDeltaAttention(nnx.Module):
  """Kimi Delta Attention (KDA) layer implementation.

    KDA is a linear attention mechanism that uses the Delta Rule for state
  correction:
      S' = S * exp(g_t)
      residual = v_t - k_t^T @ S'
      S = S' + beta_t * k_t (x) residual
      o_t = scale * q_t^T @ S

    Attributes:
      config: Model configuration containing KDA parameters.
      layer_idx: Index of this layer in the decoder stack.
      mesh: JAX device mesh for sharding.
  """

  def __init__(
      self,
      config: Config,
      layer_idx: int,
      mesh: Mesh,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.layer_idx = layer_idx
    self.mesh = mesh

    cfg = self.config

    # KDA head dimensions derived from global config (matching Megatron convention):
    #   key_head_dim = value_head_dim = config.head_dim (kv_channels)
    #   num_key_heads = num_value_heads = config.base_num_query_heads (num_attention_heads)
    self.key_head_dim = cfg.head_dim
    self.value_head_dim = cfg.head_dim
    self.num_key_heads = cfg.base_num_query_heads
    self.num_value_heads = cfg.base_num_query_heads
    self.num_query_heads = self.num_key_heads

    # Short convolution for local dependency modeling
    if cfg.linear_conv_kernel_dim > 0:
      # Q, K, V each have their own depthwise causal 1D conv layer.
      # nnx.Conv kernel shape: [K, in_features // feature_group_count, out_features].
      # With feature_group_count == features, this is depthwise (one channel per group),
      # matching Megatron's Conv1d(groups=features).
      q_features = self.num_query_heads * self.key_head_dim
      k_features = self.num_key_heads * self.key_head_dim
      v_features = self.num_value_heads * self.value_head_dim
      conv_kwargs = {
          "kernel_size": (cfg.linear_conv_kernel_dim,),
          "padding": "CAUSAL",
          "use_bias": False,
          "dtype": cfg.dtype,
          "param_dtype": cfg.weight_dtype,
          "rngs": rngs,
      }
      self.q_conv = nnx.Conv(
          in_features=q_features,
          out_features=q_features,
          feature_group_count=q_features,
          **conv_kwargs,
      )
      self.k_conv = nnx.Conv(
          in_features=k_features,
          out_features=k_features,
          feature_group_count=k_features,
          **conv_kwargs,
      )
      self.v_conv = nnx.Conv(
          in_features=v_features,
          out_features=v_features,
          feature_group_count=v_features,
          **conv_kwargs,
      )
    else:
      self.q_conv = None
      self.k_conv = None
      self.v_conv = None

    # QKV projections
    # Separate projections for Q, K, V (not fused) to allow independent conv
    self.q_proj = linears.DenseGeneral(
        in_features_shape=cfg.base_emb_dim,
        out_features_shape=(self.num_query_heads, self.key_head_dim),
        axis=-1,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "heads", "kv"),
        use_bias=cfg.attention_bias,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        input_activation_axes=("activation_batch", "activation_norm_length", None),
        rngs=rngs,
    )

    self.k_proj = linears.DenseGeneral(
        in_features_shape=cfg.base_emb_dim,
        out_features_shape=(self.num_key_heads, self.key_head_dim),
        axis=-1,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "heads", "kv"),
        use_bias=cfg.attention_bias,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        input_activation_axes=("activation_batch", "activation_norm_length", None),
        rngs=rngs,
    )

    self.v_proj = linears.DenseGeneral(
        in_features_shape=cfg.base_emb_dim,
        out_features_shape=(self.num_value_heads, self.value_head_dim),
        axis=-1,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "heads", "kv"),
        use_bias=cfg.attention_bias,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        input_activation_axes=("activation_batch", "activation_norm_length", None),
        rngs=rngs,
    )

    # Output projection
    self.o_proj = linears.DenseGeneral(
        in_features_shape=(self.num_value_heads, self.value_head_dim),
        out_features_shape=cfg.base_emb_dim,
        axis=(-2, -1),
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("heads", "kv", "embed"),
        use_bias=cfg.attention_bias,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        input_activation_axes=(
            "activation_batch",
            "activation_norm_length",
            "activation_heads",
            None,
        ),
        rngs=rngs,
    )

    # Gate projection for generating g (log-space gate)
    # g has shape [B, T, H, K] - per-head, per-dim gate
    self.g_proj = linears.DenseGeneral(
        in_features_shape=cfg.base_emb_dim,
        out_features_shape=(self.num_key_heads, self.key_head_dim),
        axis=-1,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "heads", "kv"),
        use_bias=False,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        input_activation_axes=("activation_batch", "activation_norm_length", None),
        rngs=rngs,
    )

    # Beta projection for generating beta (Delta rule mixing coefficient)
    # beta has shape [B, T, H] - per-head scalar
    self.b_proj = linears.DenseGeneral(
        in_features_shape=cfg.base_emb_dim,
        out_features_shape=(self.num_key_heads,),
        axis=-1,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "heads"),
        use_bias=False,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        input_activation_axes=("activation_batch", "activation_norm_length", None),
        rngs=rngs,
    )

    # Q/K L2 normalization is applied outside chunk_kda (matching Megatron)

    # Output gate projection: gate shape [B, T, H, V] (matching Megatron no_kda_lora path)
    self.gate_proj = linears.DenseGeneral(
        in_features_shape=cfg.base_emb_dim,
        out_features_shape=(self.num_value_heads, self.value_head_dim),
        axis=-1,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "heads", "kv"),
        use_bias=cfg.attention_bias,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        input_activation_axes=("activation_batch", "activation_norm_length", None),
        rngs=rngs,
    )

    # Output norm (per-head RMSNorm, applied before gating)
    self.out_norm = RMSNorm(
        num_features=self.value_head_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )

    # Gate parameters (matching Megatron kda.py:299-330)
    # A_log: [num_key_heads] — log of diagonal decay matrix
    A_init_range = (1.0, 16.0)
    A = jax.random.uniform(
        rngs.params(),
        shape=(self.num_key_heads,),
        minval=A_init_range[0],
        maxval=A_init_range[1],
    )
    self.A_log = nnx.Param(jnp.log(A))

    # dt_bias: [num_key_heads * key_head_dim] — gate bias
    # Initialize via inverse softplus of uniform(dt_min, dt_max)
    dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
    dt = jnp.exp(
        jax.random.uniform(
            rngs.params(),
            shape=(self.num_key_heads * self.key_head_dim,),
        )
        * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    )
    dt = jnp.clip(dt, min=dt_init_floor)
    # Inverse softplus: x = dt + log(-expm1(-dt))
    inv_dt = dt + jnp.log(-jnp.expm1(-dt))
    self.dt_bias = nnx.Param(inv_dt)

    # Axis names for shard_map (Pallas kernels cannot be auto-partitioned).
    self.qkv_axis_names = ("activation_batch", "activation_norm_length", "activation_heads", "activation_kv")
    self.beta_axis_names = ("activation_batch", "activation_norm_length", "activation_heads")

  def _logical_to_mesh_axes(self, logical_name):
    return logical_to_mesh_axes(logical_name, mesh=self.mesh, rules=self.config.logical_axis_rules)

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      decoder_positions: jnp.ndarray | None = None,
      deterministic: bool = True,
      model_mode: str = "train",
      *,
      layer_idx: int | None = None,
      decoder_segment_ids: jnp.ndarray | None = None,
  ) -> tuple[jnp.ndarray, None]:
    """Forward pass for KDA attention.

    Args:
      hidden_states: Input tensor of shape [B, T, emb_dim].
      decoder_positions: Position indices for RoPE (not used in KDA).
      deterministic: Whether to use deterministic mode.
      model_mode: Model mode (train/prefill/autoregressive).
      layer_idx: Optional layer index override.
      decoder_segment_ids: Optional segment IDs for packed sequences.

    Returns:
      Tuple of (output, None) where output has shape [B, T, emb_dim].
    """
    del decoder_positions  # KDA doesn't use RoPE
    del deterministic  # No dropout in KDA currently
    del layer_idx  # Not used

    cfg = self.config

    if decoder_segment_ids is not None:
      raise NotImplementedError("KDA does not yet support packed sequences.")

    if model_mode == MODEL_MODE_AUTOREGRESSIVE:
      raise NotImplementedError("KDA autoregressive mode not yet implemented.")

    B, T, _ = hidden_states.shape

    # tops chunk_kda kernel only supports chunk_size=64
    chunk_size = 64
    if T % chunk_size != 0:
      pad_len = chunk_size - (T % chunk_size)
      hidden_states = jnp.pad(hidden_states, ((0, 0), (0, pad_len), (0, 0)))
      T = hidden_states.shape[1]
      needs_unpad = True
    else:
      needs_unpad = False

    # QKV projections
    with jax.named_scope("qkv_proj"):
      q = self.q_proj(hidden_states)  # [B, T, H, K]
      k = self.k_proj(hidden_states)  # [B, T, H, K]
      v = self.v_proj(hidden_states)  # [B, T, H, V]

      q = checkpoint_name(q, "q_proj")
      k = checkpoint_name(k, "k_proj")
      v = checkpoint_name(v, "v_proj")

    # Apply short convolution if enabled (before activation, matching Megatron)
    if self.q_conv is not None:
      with jax.named_scope("short_conv"):
        # Reshape for conv: [B, T, H*D] -> conv -> [B, T, H*D] -> reshape back
        q_flat = q.reshape(B, T, -1)
        k_flat = k.reshape(B, T, -1)
        v_flat = v.reshape(B, T, -1)

        q_flat = self.q_conv(q_flat)
        k_flat = self.k_conv(k_flat)
        v_flat = self.v_conv(v_flat)

        q = q_flat.reshape(B, T, self.num_query_heads, self.key_head_dim)
        k = k_flat.reshape(B, T, self.num_key_heads, self.key_head_dim)
        v = v_flat.reshape(B, T, self.num_value_heads, self.value_head_dim)

    # Apply SiLU activation after conv on q, k, v (matching Megatron)
    q = jax.nn.silu(q)
    k = jax.nn.silu(k)
    v = jax.nn.silu(v)

    # Apply L2 normalization to Q/K outside the kernel (matching Megatron kda.py:824-828)
    if cfg.use_qk_norm:
      q, _ = l2norm_fwd(q)
      k, _ = l2norm_fwd(k)

    # Generate gate g (raw projection, gate transform done inside kernel)
    with jax.named_scope("gate_proj"):
      g = self.g_proj(hidden_states)  # [B, T, H, K]

    # Generate output gate (for gated norm after KDA kernel)
    with jax.named_scope("output_gate_proj"):
      output_gate = self.gate_proj(hidden_states)  # [B, T, H, V]

    # Generate beta (Delta rule mixing coefficient)
    with jax.named_scope("beta_proj"):
      beta = self.b_proj(hidden_states)  # [B, T, H]
      beta = beta.astype(jnp.float32)
      beta = jax.nn.sigmoid(beta)  # Ensure (0, 1) range, in fp32

    scale = self.key_head_dim**-0.5
    safe_gate = cfg.use_kda_safe_gate
    lower_bound = cfg.kda_lower_bound if safe_gate else None

    # Call KDA kernel via shard_map (Pallas/Mosaic kernels cannot be auto-partitioned).
    with jax.named_scope("kda_kernel"):
      qkv_pspec = self._logical_to_mesh_axes(self.qkv_axis_names)
      beta_pspec = self._logical_to_mesh_axes(self.beta_axis_names)
      a_log_pspec = self._logical_to_mesh_axes(("activation_heads",))
      dt_bias_2d_pspec = self._logical_to_mesh_axes(("activation_heads", "activation_kv"))

      # Reshape dt_bias from [H*K] to [H, K] for proper head-dim sharding.
      dt_bias_2d = self.dt_bias.value.reshape(self.num_key_heads, self.key_head_dim)

      @functools.partial(
          jax.shard_map,
          mesh=self.mesh,
          in_specs=(qkv_pspec, qkv_pspec, qkv_pspec, qkv_pspec, beta_pspec, a_log_pspec, dt_bias_2d_pspec),
          out_specs=qkv_pspec,
          check_vma=False,
      )
      def _shard_map_chunk_kda(q, k, v, g, beta, A_log, dt_bias_2d):
        dt_bias_flat = dt_bias_2d.reshape(-1)
        o, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias_flat,
            scale=scale,
            chunk_size=chunk_size,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=True,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
        )
        return o

      o = _shard_map_chunk_kda(q, k, v, g, beta, self.A_log.value, dt_bias_2d)

    # Output gated norm (matching Megatron _apply_gated_norm)
    # 1. Per-head RMSNorm on KDA output
    # 2. Sigmoid gate on normalized output
    with jax.named_scope("output_gated_norm"):
      # o: [B, T, H, V] → reshape to [..., V] for per-head norm
      o_shape = o.shape
      o_dtype = o.dtype
      o_flat = o.reshape(-1, self.value_head_dim)
      o_normed = self.out_norm(o_flat)
      # gate: [B, T, H, V] → reshape to [..., V]
      gate_flat = output_gate.reshape(-1, self.value_head_dim)
      o_gated = o_normed * jax.nn.sigmoid(gate_flat.astype(jnp.float32))
      o = o_gated.astype(o_dtype).reshape(o_shape)

    # Output projection
    with jax.named_scope("o_proj"):
      output = self.o_proj(o)
      output = checkpoint_name(output, "o_proj")

    # Unpad if needed
    if needs_unpad:
      output = output[:, : hidden_states.shape[1] - pad_len, :]

    return output, None
