"""End-to-end verification for Ling2 decoder.

Runs on TPU with random weights to verify:
1. Ling2DenseDecoderLayer / Ling2MoEDecoderLayer construction for all layer types (GLA/MLA, Dense/MoE)
2. Single-layer forward pass for both GLA and MLA layers
3. Full Decoder init + forward pass with ling2 config
4. scan_layers=True raises NotImplementedError
5. Output shapes and finiteness
"""

import os
import sys
import unittest

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAXTEXT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SRC_DIR = os.path.join(MAXTEXT_ROOT, "src")
if SRC_DIR not in sys.path:
  sys.path.insert(0, SRC_DIR)

from maxtext.common.common_types import DecoderBlockType, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import attention_gla, attention_mla, linears, moe
from maxtext.layers.decoders import Decoder
from maxtext.models import ling2
from maxtext.utils import maxtext_utils

# Small config for fast testing
_LING2_OVERRIDES = [
    "run_name=ling2_e2e_verify",
    "model_name=ling2",
    "override_model_config=True",
    "enable_checkpointing=False",
    # Shrink dimensions
    "base_emb_dim=128",
    "base_mlp_dim=256",
    "base_num_decoder_layers=10",
    "base_num_query_heads=2",
    "base_num_kv_heads=2",
    "head_dim=64",
    "attention=dot_product",
    "max_target_length=256",
    "max_prefill_predict_length=256",
    "scan_layers=False",
    # MLA params (scaled down)
    "q_lora_rank=32",
    "kv_lora_rank=64",
    "qk_nope_head_dim=32",
    "qk_rope_head_dim=16",
    "v_head_dim=64",
    # MoE params (scaled down)
    "num_experts=4",
    "num_experts_per_tok=2",
    "base_moe_mlp_dim=64",
    "moe_shared_expert_dim=128",
    "shared_experts=1",
    "n_routing_groups=1",
    "topk_routing_group=1",
    # Ling2-specific
    "inhomogeneous_layer_cycle_interval=5",
    "first_num_dense_layers=1",
    "group_norm_size=4",
    # Disable MTP for simplicity
    "mtp_num_layers=0",
    # Training
    "per_device_batch_size=1",
    "ici_fsdp_parallelism=-1",
]


def _make_config(extra_overrides=None):
  config_path = os.path.join(SRC_DIR, "maxtext", "configs", "base.yml")
  overrides = list(_LING2_OVERRIDES)
  if extra_overrides:
    overrides.extend(extra_overrides)
  return pyconfig.initialize([sys.argv[0], config_path, *overrides])


def _make_mesh(cfg):
  devices_array = maxtext_utils.create_device_mesh(cfg)
  return Mesh(devices_array, cfg.mesh_axes)


class DummyEmbedding:
  """Minimal embedding for Decoder init."""

  def __init__(self, emb_dim: int):
    self.emb_dim = emb_dim

  def __call__(self, x, model_mode):
    return jnp.ones((x.shape[0], x.shape[1], self.emb_dim))


class TestLing2E2E(unittest.TestCase):
  """E2E tests for Ling2 decoder on TPU."""

  @classmethod
  def setUpClass(cls):
    cls.cfg = _make_config()
    cls.mesh = _make_mesh(cls.cfg)

  def test_config_loads(self):
    """Config loads with correct Ling2 values."""
    self.assertEqual(self.cfg.decoder_block, DecoderBlockType.LING2)
    self.assertEqual(self.cfg.inhomogeneous_layer_cycle_interval, 5)
    self.assertEqual(self.cfg.first_num_dense_layers, 1)
    self.assertFalse(self.cfg.scan_layers)

  def test_layer_construction_gla_dense(self):
    """layer_idx=0: GLA attention + Dense MLP."""
    with self.mesh:
      layer = ling2.Ling2DenseDecoderLayer(
          config=self.cfg,
          mesh=self.mesh,
          model_mode=MODEL_MODE_TRAIN,
          layer_idx=0,
          rngs=nnx.Rngs(params=0, dropout=1),
      )
    self.assertIsInstance(layer.attention, attention_gla.BailingMoeV2LinearAttention)
    self.assertIsInstance(layer.mlp, linears.MlpBlock)

  def test_layer_construction_gla_moe(self):
    """layer_idx=1: GLA attention + MoE MLP."""
    with self.mesh:
      layer = ling2.Ling2MoEDecoderLayer(
          config=self.cfg,
          mesh=self.mesh,
          model_mode=MODEL_MODE_TRAIN,
          layer_idx=1,
          rngs=nnx.Rngs(params=0, dropout=1),
      )
    self.assertIsInstance(layer.attention, attention_gla.BailingMoeV2LinearAttention)
    self.assertIsInstance(layer.mlp, moe.RoutedAndSharedMoE)

  def test_layer_construction_mla_moe(self):
    """layer_idx=4: MLA attention + MoE MLP ((4+1)%5==0)."""
    with self.mesh:
      layer = ling2.Ling2MoEDecoderLayer(
          config=self.cfg,
          mesh=self.mesh,
          model_mode=MODEL_MODE_TRAIN,
          layer_idx=4,
          rngs=nnx.Rngs(params=0, dropout=1),
      )
    self.assertIsInstance(layer.attention, attention_mla.MLA)
    self.assertIsInstance(layer.mlp, moe.RoutedAndSharedMoE)

  def test_layer_construction_mla_layer9(self):
    """layer_idx=9: MLA attention + MoE MLP ((9+1)%5==0)."""
    with self.mesh:
      layer = ling2.Ling2MoEDecoderLayer(
          config=self.cfg,
          mesh=self.mesh,
          model_mode=MODEL_MODE_TRAIN,
          layer_idx=9,
          rngs=nnx.Rngs(params=0, dropout=1),
      )
    self.assertIsInstance(layer.attention, attention_mla.MLA)
    self.assertIsInstance(layer.mlp, moe.RoutedAndSharedMoE)

  def test_gla_forward(self):
    """GLA layer forward pass: correct shape, None kv_cache, finite output."""
    cfg = self.cfg
    batch = cfg.global_batch_size_to_train_on
    seq_len = cfg.max_target_length
    emb_dim = cfg.emb_dim

    inputs = jnp.ones((batch, seq_len, emb_dim), dtype=cfg.dtype)
    positions = jnp.broadcast_to(jnp.arange(seq_len), (batch, seq_len))

    with self.mesh:
      layer = ling2.Ling2MoEDecoderLayer(
          config=cfg,
          mesh=self.mesh,
          model_mode=MODEL_MODE_TRAIN,
          layer_idx=1,
          rngs=nnx.Rngs(params=0, dropout=1),
      )
      output, kv_cache = layer(
          inputs,
          decoder_segment_ids=None,
          decoder_positions=positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )

    self.assertEqual(output.shape, (batch, seq_len, emb_dim))
    self.assertIsNone(kv_cache)
    self.assertTrue(jnp.isfinite(output).all())

  def test_mla_forward(self):
    """MLA layer forward pass: correct shape, finite output."""
    cfg = self.cfg
    batch = cfg.global_batch_size_to_train_on
    seq_len = cfg.max_target_length
    emb_dim = cfg.emb_dim

    inputs = jnp.ones((batch, seq_len, emb_dim), dtype=cfg.dtype)
    positions = jnp.broadcast_to(jnp.arange(seq_len), (batch, seq_len))
    segment_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)

    with self.mesh:
      layer = ling2.Ling2MoEDecoderLayer(
          config=cfg,
          mesh=self.mesh,
          model_mode=MODEL_MODE_TRAIN,
          layer_idx=4,
          rngs=nnx.Rngs(params=0, dropout=1),
      )
      output, _ = layer(
          inputs,
          decoder_segment_ids=segment_ids,
          decoder_positions=positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )

    self.assertEqual(output.shape, (batch, seq_len, emb_dim))
    self.assertTrue(jnp.isfinite(output).all())

  def test_scan_layers_raises(self):
    """scan_layers=True raises NotImplementedError."""
    scan_cfg = _make_config(["scan_layers=True"])
    mesh = _make_mesh(scan_cfg)

    decoder = Decoder(config=scan_cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
    with self.assertRaises(NotImplementedError) as ctx:
      decoder.get_decoder_layers()
    self.assertIn("scan_layers", str(ctx.exception))

  def test_decoder_get_layers(self):
    """Decoder.get_decoder_layers returns two Ling2 layer classes."""
    decoder = Decoder(config=self.cfg, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN)
    layers = decoder.get_decoder_layers()
    self.assertEqual(len(layers), 2)
    self.assertIs(layers[0], ling2.Ling2DenseDecoderLayerToLinen)
    self.assertIs(layers[1], ling2.Ling2MoEDecoderLayerToLinen)

  def test_full_decoder_forward(self):
    """Full Decoder init + forward pass with correct output shapes."""
    cfg = self.cfg
    batch = cfg.global_batch_size_to_train_on
    seq_len = cfg.max_target_length

    decoder = Decoder(config=cfg, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN)
    shared_embedding = DummyEmbedding(emb_dim=cfg.emb_dim)

    decoder_input_tokens = jnp.ones((batch, seq_len), dtype=jnp.int32)
    decoder_positions = jnp.broadcast_to(jnp.arange(seq_len), (batch, seq_len))
    decoder_segment_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)

    with self.mesh:
      variables = decoder.init(
          {
              "params": jax.random.PRNGKey(0),
              "dropout": jax.random.PRNGKey(1),
              "aqt": jax.random.PRNGKey(2),
          },
          shared_embedding=shared_embedding,
          decoder_input_tokens=decoder_input_tokens,
          decoder_positions=decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )

    # Verify all layer params exist (dense_layers_ / moe_layers_ naming)
    # Note: each group uses local indexing (dense_layers_0, moe_layers_0..N-1)
    params = variables.get("params", {})
    for lyr in range(cfg.first_num_dense_layers):
      self.assertIn(f"dense_layers_{lyr}", params)
    num_moe_layers = cfg.num_decoder_layers - cfg.first_num_dense_layers
    for lyr in range(num_moe_layers):
      self.assertIn(f"moe_layers_{lyr}", params)

    # Forward pass
    with self.mesh:
      result = decoder.apply(
          variables,
          shared_embedding=shared_embedding,
          decoder_input_tokens=decoder_input_tokens,
          decoder_positions=decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
          rngs={
              "dropout": jax.random.PRNGKey(1),
              "aqt": jax.random.PRNGKey(2),
          },
      )

    logits, hidden_state, _ = result
    self.assertEqual(logits.shape, (batch, seq_len, cfg.vocab_size))
    self.assertEqual(hidden_state.shape, (batch, seq_len, cfg.emb_dim))
    self.assertTrue(jnp.isfinite(logits).all())
    self.assertTrue(jnp.isfinite(hidden_state).all())


if __name__ == "__main__":
  unittest.main()
