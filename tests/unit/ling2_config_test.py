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

"""Tests for Ling2 config extension (PR1).

Validates that:
- ling2.yml loads correctly with expected values
- New config fields have backward-compatible defaults
- LING2 decoder block type is registered
"""

import inspect
import os
import unittest

from maxtext.common.common_types import DecoderBlockType
from maxtext.configs.pyconfig import initialize_pydantic
from maxtext.utils.globals import MAXTEXT_REPO_ROOT

_BASE_CONFIG_PATH = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")


class Ling2ConfigLoadingTest(unittest.TestCase):
  """Tests that ling2.yml loads correctly via initialize_pydantic."""

  @classmethod
  def setUpClass(cls):
    cls.cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "model_name=ling2"])

  def test_ling2_config_loads(self):
    """Tests that model_name=ling2 loads ling2.yml and sets expected values."""
    self.assertEqual(self.cfg.decoder_block, DecoderBlockType.LING2)
    self.assertEqual(self.cfg.base_emb_dim, 2048)
    self.assertEqual(self.cfg.base_num_decoder_layers, 20)
    self.assertEqual(self.cfg.vocab_size, 157184)

  def test_ling2_mla_fields(self):
    """Tests MLA-related fields from ling2.yml."""
    self.assertEqual(self.cfg.attention_type, "mla")
    self.assertEqual(self.cfg.q_lora_rank, 256)
    self.assertEqual(self.cfg.kv_lora_rank, 512)
    self.assertEqual(self.cfg.qk_nope_head_dim, 128)
    self.assertEqual(self.cfg.qk_rope_head_dim, 64)
    self.assertEqual(self.cfg.v_head_dim, 128)
    self.assertTrue(self.cfg.mla_interleaved_rope)

  def test_ling2_hybrid_attention_fields(self):
    """Tests hybrid attention grouping (MLA + GLA)."""
    self.assertEqual(self.cfg.inhomogeneous_layer_cycle_interval, 5)
    self.assertEqual(self.cfg.group_norm_size, 4)

  def test_ling2_moe_fields(self):
    """Tests MoE-related fields from ling2.yml."""
    self.assertEqual(self.cfg.num_experts, 256)
    self.assertEqual(self.cfg.num_experts_per_tok, 8)
    self.assertEqual(self.cfg.base_moe_mlp_dim, 512)
    self.assertEqual(self.cfg.moe_shared_expert_dim, 2048)
    self.assertEqual(self.cfg.first_num_dense_layers, 1)

  def test_ling2_routing_fields(self):
    """Tests routing-related fields from ling2.yml."""
    self.assertTrue(self.cfg.routed_bias)
    self.assertEqual(self.cfg.routed_bias_dtype, "float32")
    self.assertFalse(self.cfg.enable_routed_bias_grad)
    self.assertTrue(self.cfg.routed_bias_zero_mean_update)
    self.assertAlmostEqual(self.cfg.routed_bias_update_rate, 0.001)

  def test_ling2_training_fields(self):
    """Tests training-related fields from ling2.yml."""
    self.assertFalse(self.cfg.calculate_per_token_loss)
    self.assertFalse(self.cfg.use_linear_silu)


class Ling2DefaultsBackwardCompatTest(unittest.TestCase):
  """Tests that new fields have backward-compatible defaults for existing models."""

  @classmethod
  def setUpClass(cls):
    cls.cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "model_name=llama2-7b"])

  def test_existing_model_unaffected(self):
    """Tests that llama2-7b config is not affected by new fields."""
    cfg = self.cfg

    # New fields should all have safe defaults
    self.assertEqual(cfg.moe_z_loss_weight, 0.0)
    self.assertEqual(cfg.moe_shared_expert_dim, 0)
    self.assertEqual(cfg.routed_bias_dtype, "")
    self.assertTrue(cfg.enable_routed_bias_grad)
    self.assertFalse(cfg.routed_bias_zero_mean_update)
    self.assertTrue(cfg.mla_interleaved_rope)
    self.assertEqual(cfg.group_norm_size, 1)
    self.assertFalse(cfg.use_linear_silu)
    self.assertEqual(cfg.pad_id, 0)
    self.assertEqual(cfg.bos_id, 1)
    self.assertEqual(cfg.blend_cache_dir, "")
    self.assertEqual(cfg.blend_index_dir, "")
    self.assertTrue(cfg.reset_attention_mask)
    self.assertFalse(cfg.eod_mask_loss)
    self.assertFalse(cfg.mmap_split_sentences)
    self.assertTrue(cfg.calculate_per_token_loss)


class Ling2DecoderBlockTypeTest(unittest.TestCase):
  """Tests that LING2 is properly registered as a decoder block type."""

  @classmethod
  def setUpClass(cls):
    cls.cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "model_name=ling2"])

  def test_ling2_enum_exists(self):
    """Tests that DecoderBlockType.LING2 exists with correct value."""
    self.assertEqual(DecoderBlockType.LING2.value, "ling2")

  def test_ling2_decoder_block_from_config(self):
    """Tests that ling2.yml sets decoder_block to LING2."""
    self.assertEqual(self.cfg.decoder_block, DecoderBlockType.LING2)
    self.assertEqual(self.cfg.decoder_block.value, "ling2")

  def test_ling2_in_layer_map(self):
    """Tests that LING2 is registered in the Decoder layer_map (source-level check)."""
    nnx_decoders_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "layers", "nnx_decoders.py")
    with open(nnx_decoders_path, "r") as f:
      source = f.read()
    self.assertIn("DecoderBlockType.LING2", source)


if __name__ == "__main__":
  unittest.main()
