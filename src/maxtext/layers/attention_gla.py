"""BailingMoeV2 Linear Attention (Lightning Attention-2 + chunk Simple GLA) in JAX/Flax."""

from __future__ import annotations

import functools

from flax import linen as nn
from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import Config, MODEL_MODE_AUTOREGRESSIVE
from maxtext.kernels.gla import chunk_gla, build_slope_tensor
from maxtext.layers import linears
from maxtext.layers.normalizations import RMSNorm, GroupRMSNorm
from maxtext.layers.embeddings import PartialRotaryEmbedding
from maxtext.utils.sharding import logical_to_mesh_axes


class BailingMoeV2LinearAttention(nnx.Module):
  """Lightning Attention-2 layer with chunk Simple GLA (NNX).

  Attributes:
    config: Model configuration.
    layer_idx: Index of this layer in the decoder stack (global by default).
  """

  def __init__(self, config: Config, layer_idx: int, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.layer_idx = layer_idx
    self.mesh = mesh
    cfg = self.config

    self.num_heads = cfg.base_num_query_heads
    self.num_key_value_heads = cfg.base_num_kv_heads
    self.head_dim = cfg.head_dim

    if self.num_key_value_heads != self.num_heads:
      raise NotImplementedError(
          f"GLA requires num_kv_heads == num_query_heads, got "
          f"{self.num_key_value_heads} vs {self.num_heads}. "
          f"GQA/MQA is not supported."
      )

    self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")
    self.qkv_axis_names = ("activation_batch", "activation_norm_length", "activation_heads", "activation_kv")
    self.g_axis_names = ("activation_batch", "activation_norm_length", "activation_heads")

    self.query_key_value = linears.DenseGeneral(
        in_features_shape=cfg.base_emb_dim,
        out_features_shape=(self.num_heads + 2 * self.num_key_value_heads, self.head_dim),
        axis=-1,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "heads", "kv"),
        use_bias=cfg.attention_bias,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )
    self.dense = linears.DenseGeneral(
        in_features_shape=(self.num_heads, self.head_dim),
        out_features_shape=cfg.base_emb_dim,
        axis=(-2, -1),
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("heads", "kv", "embed"),
        use_bias=cfg.attention_bias,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )
    self.g_proj = linears.DenseGeneral(
        in_features_shape=cfg.base_emb_dim,
        out_features_shape=(self.num_heads, self.head_dim),
        axis=-1,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "heads", "kv"),
        use_bias=False,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

    if cfg.use_qk_norm:
      self.query_layernorm = RMSNorm(
          num_features=self.head_dim,
          epsilon=cfg.normalization_layer_epsilon,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          rngs=rngs,
      )
      self.key_layernorm = RMSNorm(
          num_features=self.head_dim,
          epsilon=cfg.normalization_layer_epsilon,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          rngs=rngs,
      )
    else:
      self.query_layernorm = None
      self.key_layernorm = None

    self.rotary_emb = PartialRotaryEmbedding(
        min_timescale=cfg.rope_min_timescale,
        max_timescale=cfg.rope_max_timescale,
        mesh=self.mesh,
        embedding_dims=self.head_dim,
        partial_rotary_factor=cfg.partial_rotary_factor,
        cast_as_fprop_dtype=True,
        fprop_dtype=cfg.dtype,
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )

    self.g_norm = GroupRMSNorm(
        num_features=self.num_heads * self.head_dim,
        group_norm_size=cfg.group_norm_size,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )

    # NOTE: slope_base is NOT stored as a parameter. It is a deterministic constant
    # computed from num_heads and should not be trained. We compute it on-the-fly
    # in __call__ to avoid: (1) gradients flowing through it, (2) memory waste
    # when using nn.scan (which would stack params along scan axis).

  def _logical_to_mesh_axes(self, logical_name):
    return logical_to_mesh_axes(logical_name, mesh=self.mesh, rules=self.config.logical_axis_rules)

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      decoder_positions: jnp.ndarray | None,
      deterministic: bool,
      model_mode: str,
      *,
      layer_idx: jnp.ndarray | int | None = None,
      decoder_segment_ids: jnp.ndarray | None = None,
  ) -> tuple[jnp.ndarray, None]:
    """Forward pass (train and prefill only).

    Args:
      hidden_states: ``[B, T, emb_dim]``.
      decoder_positions: ``[B, T]`` position indices used to compute RoPE.
      deterministic: Unused (kept for signature compatibility).
      model_mode: Model mode string.
      layer_idx: Optional global layer index override for slope scaling.
      decoder_segment_ids: Not supported yet. Must be ``None``; passing a
        non-None value raises ``NotImplementedError``.

    Returns:
      ``(output, None)``.
    """
    del deterministic  # GLA has no dropout currently.
    cfg = self.config

    if decoder_segment_ids is not None:
      raise NotImplementedError(
          "GLA does not yet support packed sequences (decoder_segment_ids). "
          "Recurrent state reset at segment boundaries is not implemented."
      )

    if model_mode == MODEL_MODE_AUTOREGRESSIVE:
      raise NotImplementedError("GLA decode mode is not supported (autoregressive).")

    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)
    B, T, _ = hidden_states.shape

    # Mode selection: chunk only (raise for recurrent path).
    if T <= 128:
      raise NotImplementedError(
          "Recurrent mode (T <= 128) is not yet ported. " "Use naive_recurrent_simple_gla as a standalone function."
      )

    # QKV projection.
    with jax.named_scope("qkv_proj"):
      qkv = self.query_key_value(hidden_states)
      qkv = nn.with_logical_constraint(qkv, self.qkv_axis_names)
      qkv = checkpoint_name(qkv, "qkv_proj")
      if cfg.use_linear_silu:
        qkv = jax.nn.silu(qkv)
      query_states, key_states, value_states = jnp.split(
          qkv,
          [self.num_heads, self.num_heads + self.num_key_value_heads],
          axis=2,
      )
      query_states = nn.with_logical_constraint(query_states, self.qkv_axis_names)
      key_states = nn.with_logical_constraint(key_states, self.qkv_axis_names)
      value_states = nn.with_logical_constraint(value_states, self.qkv_axis_names)

      # Optional QK RMSNorm.
      if cfg.use_qk_norm:
        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

    # RoPE.
    with jax.named_scope("rope"):
      if decoder_positions is not None:
        query_states = self.rotary_emb(query_states, decoder_positions)
        key_states = self.rotary_emb(key_states, decoder_positions)

    query_states = checkpoint_name(query_states, "query_proj")
    key_states = checkpoint_name(key_states, "key_proj")
    value_states = checkpoint_name(value_states, "value_proj")

    # Slope as g_gamma: constant per-head log-space gate (H,).
    # Compute slope_base on-the-fly (deterministic, JAX will constant-fold).
    slope_base = build_slope_tensor(self.num_heads)
    layer_idx_val = self.layer_idx if layer_idx is None else layer_idx
    layer_idx_val = jnp.asarray(layer_idx_val, dtype=jnp.float32)
    denom = max(cfg.base_num_decoder_layers - 1, 1)
    slope_scale = 1.0 - layer_idx_val / denom + 1e-5
    g_gamma = -slope_base * slope_scale

    g_gamma = nn.with_logical_constraint(g_gamma, ("activation_heads",))

    # Chunk GLA via shard_map (Pallas/Mosaic kernels cannot be auto-partitioned).
    qkv_pspec = self._logical_to_mesh_axes(self.qkv_axis_names)
    g_gamma_pspec = self._logical_to_mesh_axes(("activation_heads",))

    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=(qkv_pspec, qkv_pspec, qkv_pspec, g_gamma_pspec),
        out_specs=qkv_pspec,
        check_vma=False,
    )
    def _shard_map_chunk_gla(q, k, v, g_gamma):
      o, _ = chunk_gla(
          q=q,
          k=k,
          v=v,
          g_gamma=g_gamma,
          scale=None,
          initial_state=None,
          output_final_state=False,
          chunk_size=64,
      )
      return o

    with jax.named_scope("gla_recurrence"):
      o = _shard_map_chunk_gla(query_states, key_states, value_states, g_gamma)
      o = checkpoint_name(o, "gla_context")

    # Reshape and output projection.
    o = o.reshape(B, T, -1)

    # Group RMSNorm.
    with jax.named_scope("group_norm"):
      o = self.g_norm(o)

    # Sigmoid gate.
    with jax.named_scope("gate"):
      g_proj = self.g_proj(hidden_states)
      g_proj = checkpoint_name(g_proj, "gate_proj")
      g_proj = g_proj.reshape(B, T, -1)
      o = o * jax.nn.sigmoid(g_proj)

    # Dense output.
    with jax.named_scope("out_proj"):
      o = o.reshape(B, T, self.num_heads, self.head_dim)
      o = self.dense(o)
      o = checkpoint_name(o, "out_proj")
      o = nn.with_logical_constraint(o, self.activation_axis_names)
    return o, None
