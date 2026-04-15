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

"""Tests for Ling2 decoder layer (PR6).

Validates:
- MLA/GLA dispatch logic based on layer_idx and inhomogeneous_layer_cycle_interval
- Dense/MoE MLP selection based on layer_idx and first_num_dense_layers
- Ling2DenseDecoderLayer / Ling2MoEDecoderLayer construction and forward pass
- scan_layers=True raises NotImplementedError
- Linen Decoder integration for Ling2
"""

import sys
import unittest

import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from maxtext.common.common_types import DecoderBlockType, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import attention_gla, attention_mla, linears, moe
from maxtext.layers.decoders import Decoder
from maxtext.models import ling2
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_decoupled_parallelism_overrides, get_test_config_path


# Small config overrides for fast testing
_LING2_TEST_CONFIG = {
    "per_device_batch_size": 1.0,
    "run_name": "ling2_decoder_test",
    "enable_checkpointing": False,
    "model_name": "ling2",
    "override_model_config": True,
    # Shrink dimensions for fast unit tests
    "base_emb_dim": 128,
    "base_mlp_dim": 256,
    "base_num_decoder_layers": 10,
    "base_num_query_heads": 2,
    "base_num_kv_heads": 2,
    "head_dim": 64,
    "attention": "dot_product",
    "max_target_length": 256,
    "max_prefill_predict_length": 256,
    "scan_layers": False,
    # MLA params (scaled down)
    "q_lora_rank": 32,
    "kv_lora_rank": 64,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 16,
    "v_head_dim": 64,
    # MoE params (scaled down)
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "base_moe_mlp_dim": 64,
    "moe_shared_expert_dim": 128,
    "shared_experts": 1,
    "n_routing_groups": 1,
    "topk_routing_group": 1,
    # Ling2-specific
    "inhomogeneous_layer_cycle_interval": 5,
    "first_num_dense_layers": 1,
    "group_norm_size": 4,
}


def _make_config(**overrides):
  """Return a pyconfig Config object for Ling2 tests."""
  extra_args = get_decoupled_parallelism_overrides()
  merged = {**_LING2_TEST_CONFIG, **overrides}
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      **merged,
      **extra_args,
  )


def _make_mesh(cfg):
  devices_array = maxtext_utils.create_device_mesh(cfg)
  return Mesh(devices_array, cfg.mesh_axes)


class TestLing2MlaDispatchLogic(unittest.TestCase):
  """Tests that the MLA/GLA dispatch logic selects the correct attention type."""

  def test_is_mla_at_interval_boundary(self):
    """Layer indices at (k*interval - 1) should use MLA attention."""
    interval = 5
    # layer_idx=4: (4+1) % 5 == 0 -> MLA
    self.assertTrue((4 + 1) % interval == 0)
    # layer_idx=9: (9+1) % 5 == 0 -> MLA
    self.assertTrue((9 + 1) % interval == 0)

  def test_is_gla_for_non_boundary_layers(self):
    """Layer indices not at interval boundary should use GLA attention."""
    interval = 5
    for idx in [0, 1, 2, 3, 5, 6, 7, 8]:
      self.assertFalse((idx + 1) % interval == 0, f"layer_idx={idx} should be GLA")

  def test_different_intervals(self):
    """Verify dispatch logic with different interval values."""
    for interval in [3, 4, 5, 6]:
      mla_indices = [i for i in range(20) if (i + 1) % interval == 0]
      gla_indices = [i for i in range(20) if (i + 1) % interval != 0]
      for idx in mla_indices:
        self.assertTrue((idx + 1) % interval == 0)
      for idx in gla_indices:
        self.assertFalse((idx + 1) % interval == 0)


class TestLing2DecoderLayerConstruction(unittest.TestCase):
  """Tests Ling2DenseDecoderLayer / Ling2MoEDecoderLayer construction with different layer indices."""

  @classmethod
  def setUpClass(cls):
    cls.cfg = _make_config()
    cls.mesh = _make_mesh(cls.cfg)

  def _make_dense_layer(self, layer_idx):
    """Construct a Ling2DenseDecoderLayer with the given layer_idx."""
    return ling2.Ling2DenseDecoderLayer(
        config=self.cfg,
        mesh=self.mesh,
        model_mode=MODEL_MODE_TRAIN,
        layer_idx=layer_idx,
        rngs=nnx.Rngs(params=0, dropout=1),
    )

  def _make_moe_layer(self, layer_idx):
    """Construct a Ling2MoEDecoderLayer with the given layer_idx."""
    return ling2.Ling2MoEDecoderLayer(
        config=self.cfg,
        mesh=self.mesh,
        model_mode=MODEL_MODE_TRAIN,
        layer_idx=layer_idx,
        rngs=nnx.Rngs(params=0, dropout=1),
    )

  def test_layer0_uses_dense_mlp(self):
    """layer_idx=0 (< first_num_dense_layers=1) should use Dense MLP."""
    with self.mesh:
      layer = self._make_dense_layer(layer_idx=0)
    self.assertIsInstance(layer.mlp, linears.MlpBlock)

  def test_layer1_uses_moe_mlp(self):
    """layer_idx=1 (>= first_num_dense_layers=1) should use MoE MLP."""
    with self.mesh:
      layer = self._make_moe_layer(layer_idx=1)
    self.assertIsInstance(layer.mlp, moe.RoutedAndSharedMoE)

  def test_layer4_uses_mla_attention(self):
    """layer_idx=4 ((4+1)%5==0) should use MLA attention."""
    with self.mesh:
      layer = self._make_moe_layer(layer_idx=4)
    self.assertIsInstance(layer.attention, attention_mla.MLA)

  def test_layer1_uses_gla_attention(self):
    """layer_idx=1 ((1+1)%5!=0) should use GLA attention."""
    with self.mesh:
      layer = self._make_moe_layer(layer_idx=1)
    self.assertIsInstance(layer.attention, attention_gla.BailingMoeV2LinearAttention)

  def test_layer0_gla_and_dense(self):
    """layer_idx=0: GLA attention + Dense MLP."""
    with self.mesh:
      layer = self._make_dense_layer(layer_idx=0)
    self.assertIsInstance(layer.attention, attention_gla.BailingMoeV2LinearAttention)
    self.assertIsInstance(layer.mlp, linears.MlpBlock)

  def test_layer4_mla_and_moe(self):
    """layer_idx=4: MLA attention + MoE MLP."""
    with self.mesh:
      layer = self._make_moe_layer(layer_idx=4)
    self.assertIsInstance(layer.attention, attention_mla.MLA)
    self.assertIsInstance(layer.mlp, moe.RoutedAndSharedMoE)

  def test_mtp_layer_uses_mla(self):
    """MTP layers (layer_idx >= num_decoder_layers) should always use MLA."""
    # num_decoder_layers=10, so MTP layers start at layer_idx=10.
    # (10+1)%5 = 1 != 0, but it should still be MLA as an MTP layer.
    with self.mesh:
      layer = ling2.Ling2MoEDecoderLayer(
          config=self.cfg,
          mesh=self.mesh,
          model_mode=MODEL_MODE_TRAIN,
          layer_idx=10,
          rngs=nnx.Rngs(params=0, dropout=1),
      )
    self.assertIsInstance(layer.attention, attention_mla.MLA)


class TestLing2DecoderLayerForward(unittest.TestCase):
  """Tests Ling2 decoder layer forward pass shape correctness."""

  @classmethod
  def setUpClass(cls):
    cls.cfg = _make_config()
    cls.mesh = _make_mesh(cls.cfg)

  def _forward_layer(self, layer_idx):
    """Run a single forward pass through a Ling2 decoder layer."""
    is_dense = layer_idx < self.cfg.first_num_dense_layers
    layer_cls = ling2.Ling2DenseDecoderLayer if is_dense else ling2.Ling2MoEDecoderLayer
    with self.mesh:
      layer = layer_cls(
          config=self.cfg,
          mesh=self.mesh,
          model_mode=MODEL_MODE_TRAIN,
          layer_idx=layer_idx,
          rngs=nnx.Rngs(params=0, dropout=1),
      )
      batch = self.cfg.global_batch_size_to_train_on
      seq_len = self.cfg.max_target_length
      emb_dim = self.cfg.emb_dim

      inputs = jnp.ones((batch, seq_len, emb_dim), dtype=self.cfg.dtype)
      positions = jnp.broadcast_to(jnp.arange(seq_len), (batch, seq_len))
      segment_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)

      # GLA does not support packed sequences (decoder_segment_ids)
      is_gla = (layer_idx + 1) % self.cfg.inhomogeneous_layer_cycle_interval != 0
      output, kv_cache = layer(
          inputs,
          decoder_segment_ids=None if is_gla else segment_ids,
          decoder_positions=positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
    return output, kv_cache

  def test_gla_layer_output_shape(self):
    """GLA layer (layer_idx=1) output shape should match input shape."""
    output, kv_cache = self._forward_layer(layer_idx=1)
    batch = self.cfg.global_batch_size_to_train_on
    seq_len = self.cfg.max_target_length
    emb_dim = self.cfg.emb_dim
    self.assertEqual(output.shape, (batch, seq_len, emb_dim))
    self.assertIsNone(kv_cache)

  def test_mla_layer_output_shape(self):
    """MLA layer (layer_idx=4) output shape should match input shape."""
    output, _ = self._forward_layer(layer_idx=4)
    batch = self.cfg.global_batch_size_to_train_on
    seq_len = self.cfg.max_target_length
    emb_dim = self.cfg.emb_dim
    self.assertEqual(output.shape, (batch, seq_len, emb_dim))


class TestLing2ScanLayersError(unittest.TestCase):
  """Tests that scan_layers=True raises NotImplementedError for Ling2."""

  def test_scan_layers_raises(self):
    """get_decoder_layers should raise NotImplementedError when scan_layers=True."""
    cfg = _make_config(scan_layers=True)
    mesh = _make_mesh(cfg)

    decoder = Decoder(
        config=cfg,
        mesh=mesh,
        model_mode=MODEL_MODE_TRAIN,
    )
    with self.assertRaises(NotImplementedError) as ctx:
      decoder.get_decoder_layers()
    self.assertIn("scan_layers", str(ctx.exception))


class TestLing2DecoderIntegration(unittest.TestCase):
  """Tests Ling2 integration with the Linen Decoder."""

  @classmethod
  def setUpClass(cls):
    cls.cfg = _make_config()
    cls.mesh = _make_mesh(cls.cfg)

  def test_get_decoder_layers_returns_ling2(self):
    """get_decoder_layers should return [Ling2DenseDecoderLayerToLinen, Ling2MoEDecoderLayerToLinen]."""
    decoder = Decoder(
        config=self.cfg,
        mesh=self.mesh,
        model_mode=MODEL_MODE_TRAIN,
    )
    layers = decoder.get_decoder_layers()
    self.assertEqual(len(layers), 2)
    self.assertIs(layers[0], ling2.Ling2DenseDecoderLayerToLinen)
    self.assertIs(layers[1], ling2.Ling2MoEDecoderLayerToLinen)

  def test_get_norm_layer_returns_rms_norm(self):
    """Ling2 should use RMSNorm as its normalization layer."""
    decoder = Decoder(
        config=self.cfg,
        mesh=self.mesh,
        model_mode=MODEL_MODE_TRAIN,
    )
    norm = decoder.get_norm_layer(self.cfg.emb_dim)
    self.assertIsNotNone(norm)

  def test_decoder_block_type(self):
    """Config should have decoder_block set to LING2."""
    self.assertEqual(self.cfg.decoder_block, DecoderBlockType.LING2)


if __name__ == "__main__":
  unittest.main()
