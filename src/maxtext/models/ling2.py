# Copyright 2023-2025 Google LLC
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

"""Ling2 decoder layers."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Any

from flax import nnx
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import Config, EP_AS_CONTEXT, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.inference import page_manager
from maxtext.layers import attention_gla
from maxtext.layers import attention_mla
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import max_utils
from maxtext.utils.sharding import create_sharding, maybe_shard_with_logical


class Ling2DecoderLayer(nnx.Module):
  """Ling2 decoder layer with MLA/GLA attention and dense/MoE MLP.

  Attributes:
    layer_idx: The layer index used at construction time to determine the layer's
      structure (MLA vs GLA attention, dense vs MoE MLP). When used inside a
      scannable block, this reflects only the first block's indices.

  Note on `global_layer_idx` in __call__:
    For computations that need the true global layer index (e.g., GLA slope
    scaling), pass `global_layer_idx` to __call__. If not provided, falls back
    to the construction-time `layer_idx`.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      layer_idx: int,
      quant: None | quantizations.AqtQuantization = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.quant = quant
    self.rngs = rngs
    cfg = self.config

    batch_size, sequence_length = max_utils.get_batch_seq_len_for_mode(self.config, self.model_mode)
    self.dummy_inputs_shape = (batch_size, sequence_length, cfg.emb_dim)

    self.out_sharding = create_sharding(self.mesh, self.logical_axis_names)
    self.mlp_intermediate_sharding = create_sharding(self.mesh, self.mlp_logical_axis_names)

    self.input_layernorm = RMSNorm(
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("norm",),
        epsilon=cfg.normalization_layer_epsilon,
        rngs=rngs,
    )
    self.post_attention_layernorm = RMSNorm(
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("norm",),
        epsilon=cfg.normalization_layer_epsilon,
        rngs=rngs,
    )

    # MTP layers (layer_idx >= num_decoder_layers) are always full-attention (MLA).
    is_full_attention_layer = (
        self.layer_idx + 1
    ) % cfg.inhomogeneous_layer_cycle_interval == 0 or self.layer_idx >= cfg.num_decoder_layers
    if is_full_attention_layer:
      self.attention = attention_mla.MLA(
          config=cfg,
          num_query_heads=cfg.num_query_heads,
          num_kv_heads=cfg.num_kv_heads,
          head_dim=cfg.head_dim,
          max_target_length=cfg.max_target_length,
          max_prefill_predict_length=cfg.max_prefill_predict_length,
          attention_kernel=cfg.attention,
          attention_type=cfg.attention_type,
          inputs_q_shape=self.dummy_inputs_shape,
          inputs_kv_shape=self.dummy_inputs_shape,
          mesh=mesh,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          dropout_rate=cfg.dropout_rate,
          name="self_attention",
          quant=quant,
          kv_quant=quantizations.configure_kv_quant(cfg),
          q_lora_rank=cfg.q_lora_rank,
          kv_lora_rank=cfg.kv_lora_rank,
          qk_nope_head_dim=cfg.qk_nope_head_dim,
          qk_rope_head_dim=cfg.qk_rope_head_dim,
          v_head_dim=cfg.v_head_dim,
          max_position_embeddings=cfg.max_position_embeddings,
          original_max_position_embeddings=cfg.original_max_position_embeddings,
          mscale=cfg.mscale,
          rope_factor=cfg.rope_factor,
          model_mode=model_mode,
          rngs=rngs,
          attn_logits_soft_cap=cfg.attn_logits_soft_cap,
      )
    else:
      self.attention = attention_gla.BailingMoeV2LinearAttention(
          config=cfg,
          layer_idx=self.layer_idx,
          mesh=mesh,
          rngs=rngs,
      )

    if self.layer_idx < cfg.first_num_dense_layers:
      self.mlp = linears.MlpBlock(
          config=cfg,
          mesh=mesh,
          in_features=cfg.emb_dim,
          intermediate_dim=cfg.mlp_dim,
          activations=cfg.mlp_activations,
          intermediate_dropout_rate=cfg.dropout_rate,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          quant=quant,
          model_mode=model_mode,
          rngs=rngs,
      )
    else:
      self.mlp = moe.RoutedAndSharedMoE(
          config=cfg,
          mesh=mesh,
          kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", None),
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          quant=quant,
          rngs=rngs,
      )

  def with_logical_constraint(self, x, logical_axes):
    return maybe_shard_with_logical(
        x,
        logical_axes=logical_axes,
        mesh=self.mesh,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
    )

  @property
  def logical_axis_names(self):
    """Return logical axis names for activation sharding."""
    if self.model_mode == MODEL_MODE_PREFILL:
      return (
          "activation_batch",
          "prefill_activation_norm_length",
          "activation_embed",
      )
    if self.model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      return (
          "activation_batch_no_exp",
          "activation_length",
          "activation_embed",
      )
    return (
        "activation_batch",
        "activation_norm_length",
        "activation_embed",
    )

  @property
  def mlp_logical_axis_names(self):
    """Return logical axis names for MLP activation sharding."""
    if self.model_mode == MODEL_MODE_PREFILL:
      return (
          "activation_batch",
          "prefill_activation_norm_length",
          "activation_mlp",
      )
    if self.model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      return (
          "activation_batch_no_exp",
          "activation_length",
          "activation_mlp",
      )
    return (
        "activation_batch",
        "activation_norm_length",
        "activation_mlp",
    )

  def post_process(
      self,
      layer_output,
      load_balance_loss,
      moe_z_loss,
      moe_expert_counts,
      router_stats=None,
      kv_cache=None,
  ):
    """Post-process layer output, recording losses and metrics."""
    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)

    if self.config.moe_z_loss_weight > 0.0 and moe_z_loss is not None:
      self.sow(nnx.Intermediate, "moe_z_loss", moe_z_loss)

    if self.config.routed_bias and self.config.routed_bias_update_rate > 0.0 and moe_expert_counts is not None:
      self.sow(nnx.Intermediate, "moe_expert_counts", moe_expert_counts)

    if router_stats is not None:
      for key, value in router_stats.items():
        self.sow(nnx.Intermediate, key, value)

    if self.config.record_internal_nn_metrics:
      self.sow(nnx.Intermediate, "activation_mean", jnp.mean(layer_output))
      self.sow(nnx.Intermediate, "activation_stdev", jnp.std(layer_output))
      self.sow(
          nnx.Intermediate,
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if self.config.scan_layers:
      return layer_output, None
    return layer_output, kv_cache

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
      kv_cache: None | jnp.ndarray = None,
      attention_metadata: None | dict[str, Any] = None,
      global_layer_idx: None | jnp.ndarray = None,
  ) -> tuple[jnp.ndarray, Any]:
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    residual = self.with_logical_constraint(inputs, self.logical_axis_names)

    hidden_states = self.input_layernorm(residual)
    hidden_states = self.with_logical_constraint(hidden_states, self.logical_axis_names)

    if isinstance(self.attention, attention_mla.MLA):
      attention_output, kv_cache = self.attention(
          hidden_states,
          hidden_states,
          decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=deterministic,
          model_mode=model_mode,
          out_sharding=self.out_sharding,
          previous_chunk=previous_chunk,
          page_state=page_state,
          slot=slot,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
      )
    else:
      # GLA does not support packed sequences; drop decoder_segment_ids
      attention_output, _ = self.attention(
          hidden_states,
          decoder_positions,
          deterministic,
          model_mode,
          layer_idx=global_layer_idx,
      )
      kv_cache = None

    hidden_states = residual + attention_output
    hidden_states = self.with_logical_constraint(hidden_states, self.logical_axis_names)
    residual = hidden_states

    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.with_logical_constraint(hidden_states, self.logical_axis_names)

    if isinstance(self.mlp, linears.MlpBlock):
      mlp_output = self.mlp(
          hidden_states,
          deterministic=deterministic,
          intermediate_sharding=self.mlp_intermediate_sharding,
          out_sharding=self.out_sharding,
      )
      load_balance_loss = None
      moe_z_loss = None
      moe_expert_counts = None
      router_stats = None
    else:
      mlp_output, load_balance_loss, moe_z_loss, moe_expert_counts, router_stats = self.mlp(
          hidden_states,
          intermediate_sharding=self.mlp_intermediate_sharding,
          out_sharding=self.out_sharding,
      )

    layer_output = residual + mlp_output
    layer_output = self.with_logical_constraint(layer_output, self.logical_axis_names)
    return self.post_process(layer_output, load_balance_loss, moe_z_loss, moe_expert_counts, router_stats, kv_cache)


Ling2DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Ling2DecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
