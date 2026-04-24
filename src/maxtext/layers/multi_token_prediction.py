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

"""JAX implementation of the Multi Token Prediction https://arxiv.org/pdf/2412.19437"""

from typing import Optional, Type, Union

from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import Config, DecoderBlockType, MODEL_MODE_TRAIN
from maxtext.utils.globals import EPS
from maxtext.layers.decoders import DecoderLayer
from maxtext.layers.nnx_decoders import NNXDecoderLayer
from maxtext.layers.initializers import variable_to_logically_partitioned
from maxtext.layers.linears import DenseGeneral
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding


# Custom Variable types for MTP intermediate outputs
# These will be automatically converted to Linen mutable collections by ToLinen wrapper
# The class names become collection names directly (no case conversion)
class mtp_losses(nnx.Variable):  # pylint: disable=invalid-name
  """Variable type for storing MTP loss components -> 'mtp_losses' collection."""


class mtp_acceptance(nnx.Variable):  # pylint: disable=invalid-name
  """Variable type for storing MTP acceptance predictions -> 'mtp_acceptance' collection."""


def roll_and_mask(x: jnp.ndarray, shift: int = -1) -> jnp.ndarray:
  """Performs a leftward roll on sequence axis and masks invalid positions.

  Args:
    x: Input array of shape [batch, seq_len, ...].
    shift: Number of positions to shift left.

  Returns:
    Rolled array with masked positions set to zero.
  """
  if shift == 0:
    return x
  return jnp.roll(x, shift, axis=1).at[:, shift:, ...].set(0)


def roll_and_mask_by_segment(x: jnp.ndarray, segment_ids: jnp.ndarray, shift: int = -1) -> jnp.ndarray:
  """Rolls sequence left within document boundaries defined by segment_ids.

  For each position, if the next position belongs to a different segment (or is
  the last position), the rolled value is zeroed out instead of wrapping around
  from the next document.

  When segment_ids is None or all positions have the same non-zero segment ID
  (reset_attention_mask=False mode), this behaves like roll_and_mask, only
  zeroing the last position.

  Args:
    x: Input array of shape [batch, seq_len, ...].
    segment_ids: Integer segment IDs of shape [batch, seq_len], or None.
      Same segment ID = same document. 0 = padding/EOD.
      If None, falls back to simple roll_and_mask behavior.
    shift: Number of positions to shift left (must be -1).

  Returns:
    Rolled array with cross-boundary and tail positions zeroed.
  """
  assert shift == -1, f"roll_and_mask_by_segment only supports shift=-1, got {shift}"

  # If segment_ids is None, fall back to simple roll_and_mask
  if segment_ids is None:
    return roll_and_mask(x, shift)

  # Standard left-roll by 1
  rolled = jnp.roll(x, shift, axis=1)
  # Zero out the absolute last position (same as original)
  rolled = rolled.at[:, shift:, ...].set(0)

  # Build boundary mask: True where position i and i+1 are in different segments,
  # or where position i is padding (segment_id == 0).
  seg_current = segment_ids  # [batch, seq_len]
  seg_next = jnp.roll(segment_ids, shift, axis=1)  # shifted segment_ids
  seg_next = seg_next.at[:, shift:].set(0)  # last pos -> 0

  # A position is a boundary if:
  #   1. current segment != next segment (document boundary), OR
  #   2. current segment == 0 (padding/EOD position)
  is_boundary = (seg_current != seg_next) | (seg_current == 0)  # [batch, seq_len]

  # Expand mask to match x's shape for broadcasting
  mask = is_boundary
  for _ in range(x.ndim - 2):
    mask = jnp.expand_dims(mask, axis=-1)

  return jnp.where(mask, 0, rolled)


class MultiTokenPredictionLayer(nnx.Module):
  """Multi-Token Prediction layer: normalize, concatenate, project, and transform.

  Implements: h_next = TransformerLayer(W_p(concat(RMSNorm(h_prev), RMSNorm(e_target))))
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      layer_number: int,
      transformer_layer_module: Type[Union[DecoderLayer, NNXDecoderLayer]],
      quant: Optional[quantizations.AqtQuantization] = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.layer_number = layer_number
    self.transformer_layer_module = transformer_layer_module
    self.rngs = rngs
    k = layer_number
    cfg = self.config

    self.embedding_norm = RMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.hidden_state_norm = RMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    if cfg.mtp_final_layernorm:
      self.final_layernorm = RMSNorm(
          num_features=cfg.emb_dim,
          epsilon=cfg.normalization_layer_epsilon,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          kernel_axes=("norm",),
          rngs=rngs,
      )
    self.projection_layer = DenseGeneral(
        in_features_shape=2 * cfg.emb_dim,
        out_features_shape=cfg.emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        use_bias=False,
        kernel_axes=("concat_embed", "embed"),
        rngs=rngs,
    )
    # Use MODEL_MODE_TRAIN for initialization; runtime model_mode is passed dynamically.
    # Some decoder types (e.g., Ling2/Ling3) require layer_idx to determine layer
    # structure (MLA vs GLA/KDA). MTP layers get unique indices starting from
    # num_decoder_layers so they don't collide with main model layers.
    layer_idx_kwargs = {}
    if cfg.decoder_block in (DecoderBlockType.LING2, DecoderBlockType.LING3):
      layer_idx_kwargs["layer_idx"] = cfg.num_decoder_layers + k - 1

    is_nnx_layer = issubclass(transformer_layer_module, nnx.Module)
    if is_nnx_layer:
      # Native NNX layer: instantiate directly (original upstream behavior).
      self.transformer_layer = transformer_layer_module(
          config=cfg,
          mesh=mesh,
          model_mode=MODEL_MODE_TRAIN,
          name=f"mtp_{k}_transformer_layer",
          rngs=rngs,
          **layer_idx_kwargs,
      )
    else:
      # Linen layer: wrap with ToNNX and lazy_init for parameter setup.
      mtp_transformer_layer = transformer_layer_module(
          config=cfg,
          mesh=mesh,
          model_mode=MODEL_MODE_TRAIN,
          name=f"mtp_{k}_transformer_layer",
          quant=quant,
          **layer_idx_kwargs,
      )
      self.transformer_layer = nnx_wrappers.ToNNX(mtp_transformer_layer, rngs=rngs)
      batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(cfg, MODEL_MODE_TRAIN)
      self.transformer_layer.lazy_init(
          inputs=jnp.zeros((batch_size, seq_len, cfg.emb_dim), dtype=cfg.dtype),
          decoder_segment_ids=jnp.ones((batch_size, seq_len), dtype=jnp.int32),
          decoder_positions=jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )

  @property
  def embedding_norm(self):
    return getattr(self, f"mtp_{self.layer_number}_embedding_norm")

  @embedding_norm.setter
  def embedding_norm(self, module):
    setattr(self, f"mtp_{self.layer_number}_embedding_norm", module)

  @property
  def hidden_state_norm(self):
    return getattr(self, f"mtp_{self.layer_number}_hidden_state_norm")

  @hidden_state_norm.setter
  def hidden_state_norm(self, module):
    setattr(self, f"mtp_{self.layer_number}_hidden_state_norm", module)

  @property
  def projection_layer(self):
    return getattr(self, f"mtp_{self.layer_number}_projection")

  @projection_layer.setter
  def projection_layer(self, module):
    setattr(self, f"mtp_{self.layer_number}_projection", module)

  @property
  def transformer_layer(self):
    return getattr(self, f"mtp_{self.layer_number}_transformer_layer")

  @transformer_layer.setter
  def transformer_layer(self, module):
    setattr(self, f"mtp_{self.layer_number}_transformer_layer", module)

  @property
  def final_layernorm(self):
    return getattr(self, f"mtp_{self.layer_number}_final_layernorm")

  @final_layernorm.setter
  def final_layernorm(self, module):
    setattr(self, f"mtp_{self.layer_number}_final_layernorm", module)

  def __call__(
      self,
      prev_hidden_state: jnp.ndarray,
      target_token_embedding: jnp.ndarray,
      *,
      position_ids: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str = MODEL_MODE_TRAIN,
  ) -> jnp.ndarray:
    """Applies MTP combination, projection, and transformer processing.

    Args:
        prev_hidden_state: Shape [batch, seq_len, hidden_size].
        target_token_embedding: Embedding for token t+k. Shape [batch, seq_len, embed_dim].
        position_ids: Shape [batch, seq_len].
        decoder_segment_ids: Shape [batch, seq_len] or None.
        deterministic: Whether to disable dropout.
        model_mode: Operational mode (train, eval, decode).

    Returns:
        Processed hidden state. Shape [batch, seq_len, hidden_size].
    """
    target_token_embedding = sharding.maybe_shard_with_logical(
        target_token_embedding,
        ("activation_batch", "activation_length", "activation_embed"),
        self.mesh,
        self.config.shard_mode,
        self.config.logical_axis_rules,
    )

    embedding_norm = self.embedding_norm(target_token_embedding)
    hidden_state_norm = self.hidden_state_norm(prev_hidden_state)
    concatenated_features = jnp.concatenate([embedding_norm, hidden_state_norm], axis=-1)
    projected_features = self.projection_layer(concatenated_features)

    output = self.transformer_layer(
        inputs=projected_features,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=position_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )

    output = output[0] if isinstance(output, tuple) else output
    if self.config.mtp_final_layernorm:
      output = self.final_layernorm(output)
    return output


class MultiTokenPredictionBlock(nnx.Module):
  """Orchestrates the MTP process by running a sequence of MTP layers."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      transformer_layer_module: Type[Union[DecoderLayer, NNXDecoderLayer]],
      decoder: nnx.Module,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
  ):
    self.config = config
    self.mesh = mesh
    self.transformer_layer_module = transformer_layer_module
    self.decoder = decoder
    self.rngs = rngs if rngs is not None else nnx.Rngs(0)

    # 1-indexed to match paper convention.
    for k in range(1, config.mtp_num_layers + 1):
      layer = MultiTokenPredictionLayer(
          config=config,
          mesh=mesh,
          layer_number=k,
          transformer_layer_module=transformer_layer_module,
          quant=quant,
          rngs=rngs.fork(),
      )
      setattr(self, f"mtp_layer_{k}", layer)

  def __call__(
      self,
      shared_embedding,
      main_hidden_state,
      input_ids,
      target_ids,
      target_mask,
      *,
      position_ids,
      decoder_segment_ids,
      model_mode,
      deterministic,
  ) -> dict:
    cfg = self.config
    mtp_hidden_state = main_hidden_state

    # Rolling variables move prediction window one token to the right per iteration.
    rolled_input_ids = input_ids
    rolled_target_ids = target_ids
    rolled_target_mask = target_mask
    rolled_position_id = position_ids

    mtp_losses_list = []
    mtp_weights_list = []
    mtp_preds_list = []
    mtp_masks_list = []
    mtp_segment_ids_list = []

    # Track segment boundaries for segment-aware rolling
    rolled_segment_ids = decoder_segment_ids

    for k in range(1, cfg.mtp_num_layers + 1):
      rolled_input_ids = roll_and_mask_by_segment(rolled_input_ids, rolled_segment_ids)
      rolled_target_ids = roll_and_mask_by_segment(rolled_target_ids, rolled_segment_ids)
      rolled_target_mask = roll_and_mask_by_segment(rolled_target_mask, rolled_segment_ids)
      rolled_position_id = roll_and_mask_by_segment(rolled_position_id, rolled_segment_ids)
      # Roll segment_ids itself for the next iteration (using plain roll)
      if rolled_segment_ids is not None:
        rolled_segment_ids = roll_and_mask(rolled_segment_ids)

      target_token_embedding = self.decoder._apply_embedding(
          shared_embedding,
          rolled_input_ids,
          rolled_position_id,
          deterministic,
          model_mode=model_mode,
      )

      mtp_layer = getattr(self, f"mtp_layer_{k}")
      mtp_hidden_state = mtp_layer(
          prev_hidden_state=mtp_hidden_state,
          target_token_embedding=target_token_embedding,
          position_ids=position_ids,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=deterministic,
          model_mode=model_mode,
      )

      if cfg.mtp_final_layernorm:
        # MTP layer has its own final_layernorm; skip shared decoder_norm
        # to avoid double normalization. Matches Megatron's architecture.
        mtp_logits = self.decoder.apply_output_projection(shared_embedding, mtp_hidden_state, deterministic, model_mode)
      else:
        # No MTP-specific final_layernorm; use apply_output_head which
        # applies the shared decoder_norm before projection.
        mtp_logits = self.decoder.apply_output_head(shared_embedding, mtp_hidden_state, deterministic, model_mode)

      mtp_xent, _ = max_utils.cross_entropy_with_logits(
          mtp_logits, jax.nn.one_hot(rolled_target_ids, cfg.vocab_size), 0.0
      )
      mtp_xent_masked = mtp_xent * rolled_target_mask

      if model_mode == MODEL_MODE_TRAIN:
        mtp_losses_list.append(jnp.sum(mtp_xent_masked))
        mtp_weights_list.append(jnp.sum(rolled_target_mask).astype(jnp.float32))

      if cfg.mtp_eval_target_module == k:
        # Float32 to avoid gradient errors; converted back to int32 in acceptance calculation.
        mtp_preds_list.append(jnp.argmax(mtp_logits, axis=-1).astype(jnp.float32))
        mtp_masks_list.append(rolled_target_mask)
        mtp_segment_ids_list.append(decoder_segment_ids)

    if mtp_losses_list:
      # Use self.sow to output results instead of storing them in the module state.
      # This ensures they are always materialized tracers and NOT subject to
      # checkpoint template issues.
      # Sow per-layer so the resulting tuple has mtp_num_layers entries.
      for loss, weight in zip(mtp_losses_list, mtp_weights_list):
        self.sow(mtp_losses, "losses", loss)
        self.sow(mtp_losses, "weights", weight)
    if mtp_preds_list:
      self.sow(mtp_acceptance, "mtp_preds", jnp.stack(mtp_preds_list))
      self.sow(mtp_acceptance, "mtp_mask", jnp.stack(mtp_masks_list))
      self.sow(mtp_acceptance, "segment_ids", jnp.stack(mtp_segment_ids_list))

    return {}


def calculate_mtp_loss(intermediate_outputs, config):
  """Calculates Multi-Token Prediction loss from intermediate outputs.

  Returns:
    A tuple of (scaled_mtp_loss, raw_mtp_loss) where raw_mtp_loss is the
    unscaled per-token average for logging purposes.
  """
  mtp_losses_data = maxtext_utils.get_nested_value(
      intermediate_outputs, ("mtp_losses", "mtp_block", "losses"), default=None
  )
  mtp_weights_data = maxtext_utils.get_nested_value(
      intermediate_outputs, ("mtp_losses", "mtp_block", "weights"), default=None
  )

  if mtp_losses_data is None:
    return 0.0, 0.0

  # Handle both tuple (Linen sow) and array (NNX Variable) formats.
  if isinstance(mtp_losses_data, (tuple, list)):
    if not mtp_losses_data:
      return 0.0, 0.0
    mtp_losses_array = jnp.array(mtp_losses_data)
    mtp_weights_array = jnp.array(mtp_weights_data)
  else:
    if mtp_losses_data.size == 0:
      return 0.0, 0.0
    mtp_losses_array = mtp_losses_data
    mtp_weights_array = mtp_weights_data

  if config.mtp_per_layer_loss_norm:
    # Per-layer independent normalization (Megatron-LM style): each layer's loss
    # is divided by its own valid token count, then averaged across layers.
    # This ensures each MTP layer contributes equally to the final loss regardless
    # of differing valid token counts across layers.
    per_layer_avg_losses = mtp_losses_array / (mtp_weights_array + EPS)
    avg_mtp_loss = jnp.mean(per_layer_avg_losses)
  else:
    # Global normalization (upstream default): sum all losses, divide by sum of all weights.
    avg_mtp_loss = jnp.sum(mtp_losses_array) / (jnp.sum(mtp_weights_array) + EPS)
  return avg_mtp_loss * config.mtp_loss_scaling_factor, avg_mtp_loss


def calculate_mtp_acceptance_rate(intermediate_outputs, config):
  """Calculates MTP acceptance rate from intermediate outputs."""
  sown_data = maxtext_utils.get_nested_value(intermediate_outputs, ("mtp_acceptance", "mtp_block"), {})

  # Handle both tuple (Linen sow) and array (NNX Variable) formats.
  mtp_preds_raw = maxtext_utils.get_nested_value(sown_data, ("mtp_preds",), None)
  valid_mask_raw = maxtext_utils.get_nested_value(sown_data, ("mtp_mask",), None)

  mtp_preds = mtp_preds_raw[0] if isinstance(mtp_preds_raw, (tuple, list)) and mtp_preds_raw else mtp_preds_raw
  valid_mask = valid_mask_raw[0] if isinstance(valid_mask_raw, (tuple, list)) and valid_mask_raw else valid_mask_raw

  segment_ids_raw = maxtext_utils.get_nested_value(sown_data, ("segment_ids",), None)
  segment_ids = segment_ids_raw[0] if isinstance(segment_ids_raw, (tuple, list)) and segment_ids_raw else segment_ids_raw

  # Only populated during eval for the target MTP module.
  if mtp_preds is None or valid_mask is None:
    return 0.0

  mtp_preds = mtp_preds.astype(jnp.int32)
  main_model_preds = jnp.argmax(intermediate_outputs["logits"], axis=-1)

  # Align main model predictions with MTP head target by rolling k steps.
  rolled_main_preds = main_model_preds
  if segment_ids is not None:
    rolled_seg = segment_ids
    for _ in range(config.mtp_eval_target_module):
      rolled_main_preds = roll_and_mask_by_segment(rolled_main_preds, rolled_seg)
      rolled_seg = roll_and_mask(rolled_seg)
  else:
    for _ in range(config.mtp_eval_target_module):
      rolled_main_preds = roll_and_mask(rolled_main_preds)

  correct_predictions = jnp.sum((mtp_preds == rolled_main_preds) * valid_mask)
  total_valid_tokens = jnp.sum(valid_mask)

  return (correct_predictions / (total_valid_tokens + EPS)) * 100


def multi_token_prediction_block_as_linen(
    *,
    config: Config,
    mesh: Mesh,
    transformer_layer_module: Type[Union[DecoderLayer, NNXDecoderLayer]],
    decoder: nnx.Module,
    rngs: nnx.Rngs,
    name: str | None = None,
    quant: Optional[quantizations.AqtQuantization] = None,
) -> nn.Module:
  """Initializes MultiTokenPredictionBlock as a Linen module.

  Args:
    config: Configuration object containing model hyperparameters.
    mesh: JAX Mesh for model parallelism.
    transformer_layer_module: The Transformer Decoder Layer class to use.
    decoder: The decoder module that provides embedding and output head.
    rngs: Random number generators for initialization.
    name: Optional name for the module.
    quant: Optional quantization configuration.

  Returns:
    An instance of MultiTokenPredictionBlock wrapped as a Linen module.
  """
  return nnx_wrappers.to_linen(
      MultiTokenPredictionBlock,
      config=config,
      mesh=mesh,
      transformer_layer_module=transformer_layer_module,
      decoder=decoder,
      rngs=rngs,
      metadata_fn=variable_to_logically_partitioned,
      name=name,
      quant=quant,
  )
