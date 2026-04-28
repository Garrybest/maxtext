# Copyright 2023-2026 Google LLC
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

"""Tests for Ling3 config extension (PR1).

Validates that:
- ling3-tiny.yml loads correctly with expected values
- New config fields (KDA + gated MLA) have backward-compatible defaults
- LING3 decoder block type is registered
- ling3-specific validator allowances do not regress other decoder blocks
"""

import os
import unittest

from maxtext.common.common_types import DecoderBlockType
from maxtext.configs.pyconfig import initialize_pydantic
from maxtext.utils.globals import MAXTEXT_REPO_ROOT

_BASE_CONFIG_PATH = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")


class Ling3ConfigLoadingTest(unittest.TestCase):
  """Tests that ling3-tiny.yml loads correctly via initialize_pydantic."""

  @classmethod
  def setUpClass(cls):
    cls.cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "model_name=ling3-tiny"])

  def test_ling3_config_loads(self):
    """Tests that model_name=ling3-tiny loads ling3-tiny.yml and sets expected values."""
    self.assertEqual(self.cfg.decoder_block, DecoderBlockType.LING3)
    self.assertEqual(self.cfg.base_emb_dim, 1536)
    self.assertEqual(self.cfg.base_num_decoder_layers, 24)
    self.assertEqual(self.cfg.vocab_size, 157184)
    self.assertEqual(self.cfg.max_target_length, 8192)

  def test_ling3_mla_fields(self):
    """Tests MLA-related fields from ling3-tiny.yml."""
    self.assertEqual(self.cfg.attention_type, "mla")
    self.assertEqual(self.cfg.q_lora_rank, 256)
    self.assertEqual(self.cfg.kv_lora_rank, 512)
    self.assertEqual(self.cfg.qk_nope_head_dim, 128)
    self.assertEqual(self.cfg.qk_rope_head_dim, 64)
    self.assertEqual(self.cfg.v_head_dim, 128)
    self.assertTrue(self.cfg.mla_interleaved_rope)

  def test_ling3_mla_gated_attention_type_field(self):
    """Tests Ling3-specific mla_gated_attention_type field (RFC §3.2)."""
    self.assertEqual(self.cfg.mla_gated_attention_type, "head_wise")

  def test_ling3_kda_fields(self):
    """Tests KDA-related fields from ling3-tiny.yml (RFC §3.1)."""
    self.assertEqual(self.cfg.linear_conv_kernel_dim, 4)
    self.assertFalse(self.cfg.use_kda_lora)
    self.assertTrue(self.cfg.use_kda_safe_gate)
    self.assertAlmostEqual(self.cfg.kda_lower_bound, -5.0)

  def test_ling3_hybrid_attention_fields(self):
    """Tests hybrid attention grouping (3 KDA + 1 MLA per group)."""
    self.assertEqual(self.cfg.inhomogeneous_layer_cycle_interval, 4)
    self.assertEqual(self.cfg.first_num_dense_layers, 1)

  def test_ling3_moe_fields(self):
    """Tests MoE-related fields from ling3-tiny.yml."""
    self.assertEqual(self.cfg.num_experts, 128)
    self.assertEqual(self.cfg.num_experts_per_tok, 8)
    self.assertEqual(self.cfg.base_moe_mlp_dim, 512)
    self.assertEqual(self.cfg.moe_shared_expert_dim, 512)

  def test_ling3_routing_fields(self):
    """Tests routing-related fields from ling3-tiny.yml."""
    self.assertTrue(self.cfg.routed_bias)
    self.assertEqual(self.cfg.routed_bias_dtype, "float32")
    self.assertFalse(self.cfg.enable_routed_bias_grad)
    self.assertTrue(self.cfg.routed_bias_zero_mean_update)
    self.assertAlmostEqual(self.cfg.routed_bias_update_rate, 0.001)

  def test_ling3_rope_fields(self):
    """Tests RoPE config — Ling3 uses original RoPE (no YaRN), partial rotary."""
    self.assertEqual(self.cfg.rope_factor, 1)
    self.assertAlmostEqual(self.cfg.partial_rotary_factor, 0.5)
    self.assertEqual(self.cfg.max_position_embeddings, 8192)

  def test_ling3_mtp_fields(self):
    """Tests MTP fields from ling3-tiny.yml."""
    self.assertTrue(self.cfg.mtp_final_layernorm)
    self.assertTrue(self.cfg.mtp_per_layer_loss_norm)
    self.assertAlmostEqual(self.cfg.mtp_loss_scaling_factor, 0.1)


class Ling3DefaultsBackwardCompatTest(unittest.TestCase):
  """Tests that newly added fields have backward-compatible defaults for existing models."""

  def test_llama2_unaffected(self):
    """Tests llama2-7b is not affected by Ling3-specific new fields."""
    cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "model_name=llama2-7b"])
    # Newly added fields must default to safe values
    self.assertEqual(cfg.mla_gated_attention_type, "disabled")
    self.assertEqual(cfg.linear_conv_kernel_dim, 4)
    self.assertFalse(cfg.use_kda_lora)
    self.assertFalse(cfg.use_kda_safe_gate)
    self.assertEqual(cfg.kda_lower_bound, 0.0)

  def test_ling2_unaffected(self):
    """Tests Ling2 still loads and the new MLA gate stays off (Ling2 has no gated attention)."""
    cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "model_name=ling2"])
    self.assertEqual(cfg.mla_gated_attention_type, "disabled")


class Ling3DecoderBlockTypeTest(unittest.TestCase):
  """Tests that LING3 is properly registered as a decoder block type."""

  @classmethod
  def setUpClass(cls):
    cls.cfg = initialize_pydantic(["", _BASE_CONFIG_PATH, "model_name=ling3-tiny"])

  def test_ling3_enum_exists(self):
    """Tests that DecoderBlockType.LING3 exists with correct value."""
    self.assertEqual(DecoderBlockType.LING3.value, "ling3")

  def test_ling3_decoder_block_from_config(self):
    """Tests that ling3-tiny.yml sets decoder_block to LING3."""
    self.assertEqual(self.cfg.decoder_block, DecoderBlockType.LING3)
    self.assertEqual(self.cfg.decoder_block.value, "ling3")

  def test_ling3_in_layer_map(self):
    """Tests that LING3 is registered in the Decoder layer_map (source-level check)."""
    nnx_decoders_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "layers", "nnx_decoders.py")
    with open(nnx_decoders_path, "r", encoding="utf-8") as f:
      source = f.read()
    self.assertIn("DecoderBlockType.LING3", source)


if __name__ == "__main__":
  unittest.main()
