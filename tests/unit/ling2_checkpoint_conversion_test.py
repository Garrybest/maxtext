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

"""Unit tests for Ling2 checkpoint conversion (param mapping + hook functions).

Validates mapping completeness, layer dispatch logic (dense/MoE, MLA/GLA),
expert stacking, hook coverage, reshape_kernel roundtrip, scan_layers guard,
and MTP toggle — all without JAX, TPU, or model weights.
"""

import json
import os
import unittest
from types import SimpleNamespace

import numpy as np

from maxtext.checkpoint_conversion.utils.hf_shape import HF_SHAPE
from maxtext.checkpoint_conversion.utils.param_mapping import (
    HOOK_FNS,
    LING2_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    LING2_MAXTEXT_TO_HF_PARAM_MAPPING,
    PARAM_MAPPING,
)
from maxtext.checkpoint_conversion.utils.utils import process_maxtext_param


def _make_configs(num_nextn_predict_layers=1):
  """Create mock HF config dict and MaxText config for Ling2 (matching ling2.yml)."""
  hf_config = {
      "num_hidden_layers": 20,
      "first_k_dense_replace": 1,
      "num_experts": 256,
      "num_nextn_predict_layers": num_nextn_predict_layers,
      "layer_group_size": 5,
      "q_lora_rank": 256,
      # Fields required by LING2_HF_WEIGHTS_TO_SHAPE (derived from ling2.yml):
      "hidden_size": 2048,
      "vocab_size": 157184,
      "kv_lora_rank": 512,
      "qk_nope_head_dim": 128,
      "qk_rope_head_dim": 64,
      "v_head_dim": 128,
      "num_attention_heads": 16,
      "head_dim": 128,
      "moe_intermediate_size": 512,
      "intermediate_size": 5120,
      "n_shared_experts": 1,
  }
  maxtext_config = SimpleNamespace(
      first_num_dense_layers=1,
      num_experts=256,
      inhomogeneous_layer_cycle_interval=5,
      q_lora_rank=256,
      mtp_num_layers=1 if num_nextn_predict_layers > 0 else 0,
  )
  return hf_config, maxtext_config


class Ling2CheckpointConversionTest(unittest.TestCase):

  def setUp(self):
    self.hf_config, self.maxtext_config = _make_configs()
    self.mapping = LING2_MAXTEXT_TO_HF_PARAM_MAPPING(self.hf_config, self.maxtext_config)
    self.hooks_to_hf = LING2_MAXTEXT_TO_HF_PARAM_HOOK_FN(self.hf_config, self.maxtext_config, saving_to_hf=True)
    self.hooks_to_mt = LING2_MAXTEXT_TO_HF_PARAM_HOOK_FN(self.hf_config, self.maxtext_config, saving_to_hf=False)

  def test_registered_in_global_dicts(self):
    """ling2 must be registered in PARAM_MAPPING and HOOK_FNS."""
    self.assertIn("ling2", PARAM_MAPPING)
    self.assertIn("ling2", HOOK_FNS)

  def test_mapping_key_count(self):
    """Verify total mapping entries match expected count for 20-layer Ling2 with MTP."""
    # 3 top-level + 20*2 norms + layer params + 21 MTP = 343
    # Layer 0 (GLA+Dense): 6 attn + 3 mlp = 9
    # 15 GLA+MoE layers: 15 * (6 attn + 8 moe) = 15 * 14 = 210
    # 4 MLA+MoE layers: 4 * (7 attn + 8 moe) = 4 * 15 = 60
    expected = 3 + 40 + 9 + 210 + 60 + 21
    self.assertEqual(len(self.mapping), expected)

  def test_layer_dispatch_dense_vs_moe(self):
    """Layer 0 should be dense MLP; layers 1+ should be MoE."""
    # Dense layer 0: has mlp-wi_0-kernel, no MoeBlock_0
    self.assertIn("params-decoder-dense_layers_0-mlp-wi_0-kernel", self.mapping)
    self.assertNotIn("params-decoder-dense_layers_0-mlp-MoeBlock_0-gate-kernel", self.mapping)

    # MoE layer 1 (global idx 1 -> moe_layers_0): has MoeBlock_0, no plain mlp kernels
    self.assertIn("params-decoder-moe_layers_0-mlp-MoeBlock_0-gate-kernel", self.mapping)
    self.assertIn("params-decoder-moe_layers_0-mlp-shared_experts-wi_0-kernel", self.mapping)
    self.assertNotIn("params-decoder-moe_layers_0-mlp-wi_0-kernel", self.mapping)

  def test_layer_dispatch_mla_vs_gla(self):
    """MLA layers at idx 4,9,14,19; GLA everywhere else."""
    # Layer 0 is GLA (cycle=5, idx 0 not last of group)
    self.assertIn("params-decoder-dense_layers_0-attention-query_key_value-kernel", self.mapping)
    self.assertNotIn("params-decoder-dense_layers_0-attention-wq_a-kernel", self.mapping)

    # Layer 4 is MLA (5th layer, last of first group) -> moe_layers_3
    self.assertIn("params-decoder-moe_layers_3-attention-wq_a-kernel", self.mapping)
    self.assertIn("params-decoder-moe_layers_3-attention-wkv_a-kernel", self.mapping)
    self.assertNotIn("params-decoder-moe_layers_3-attention-query_key_value-kernel", self.mapping)

    # Layer 5 is GLA -> moe_layers_4
    self.assertIn("params-decoder-moe_layers_4-attention-query_key_value-kernel", self.mapping)
    self.assertNotIn("params-decoder-moe_layers_4-attention-wq_a-kernel", self.mapping)

    # Layer 19 is MLA (last layer of model) -> moe_layers_18
    self.assertIn("params-decoder-moe_layers_18-attention-wq_a-kernel", self.mapping)

  def test_expert_stacking_list_values(self):
    """MoE expert weights should be lists of length num_experts=256."""
    expert_key = "params-decoder-moe_layers_0-mlp-MoeBlock_0-wi_0"
    self.assertIn(expert_key, self.mapping)
    val = self.mapping[expert_key]
    self.assertIsInstance(val, list)
    self.assertEqual(len(val), 256)
    self.assertEqual(val[0], "model.layers.1.mlp.experts.0.gate_proj.weight")
    self.assertEqual(val[255], "model.layers.1.mlp.experts.255.gate_proj.weight")

  def test_hook_coverage_matches_mapping_kernels(self):
    """Every kernel/weight key in mapping should have a hook; scale/bias keys should not."""
    # These suffixes denote weight tensors that need reshape/transpose hooks.
    # -bias (e.g., gate-bias = router expert_bias) is a 1D vector copied as-is, no hook needed.
    kernel_suffixes = ("-kernel", "-wi_0", "-wi_1", "-wo")
    no_hook_suffixes = ("-scale", "-bias", "-embedding")
    for key in self.mapping:
      if any(key.endswith(s) for s in kernel_suffixes):
        self.assertIn(key, self.hooks_to_hf, f"Missing hook for kernel key: {key}")
      elif any(key.endswith(s) for s in no_hook_suffixes):
        self.assertNotIn(key, self.hooks_to_hf, f"Unexpected hook for no-transform key: {key}")

  def test_reshape_kernel_roundtrip_from_shape_map(self):
    """reshape_kernel MT->HF->MT roundtrip, using HF_SHAPE as source of truth."""
    shape_map = HF_SHAPE["ling2"](self.hf_config)
    rng = np.random.RandomState(42)
    # Pick one representative of each kernel family. MT shape = flip(HF shape) for 2D kernels.
    sample_keys = [
        ("params-decoder-logits_dense-kernel", "lm_head.weight"),
        (
            "params-decoder-dense_layers_0-attention-query_key_value-kernel",
            "model.layers.0.attention.query_key_value.weight",
        ),
        ("params-decoder-moe_layers_3-attention-wq_a-kernel", "model.layers.4.attention.q_a_proj.weight"),
        (
            "params-decoder-moe_layers_0-mlp-shared_experts-wi_0-kernel",
            "model.layers.1.mlp.shared_experts.gate_proj.weight",
        ),
    ]
    for mt_key, hf_key in sample_keys:
      with self.subTest(mt_key=mt_key):
        hf_shape = tuple(shape_map[hf_key])
        mt_shape = hf_shape[::-1]  # 2D kernel: MT is transpose of HF
        mt_tensor = rng.randn(*mt_shape).astype(np.float32)
        hf_tensor = self.hooks_to_hf[mt_key](mt_tensor, hf_shape)
        self.assertEqual(tuple(hf_tensor.shape), hf_shape)
        restored = self.hooks_to_mt[mt_key](hf_tensor, mt_shape)
        np.testing.assert_array_almost_equal(mt_tensor, restored, decimal=6)

  def test_scan_layers_raises(self):
    """Both functions should raise NotImplementedError with scan_layers=True."""
    with self.assertRaises(NotImplementedError):
      LING2_MAXTEXT_TO_HF_PARAM_MAPPING(self.hf_config, self.maxtext_config, scan_layers=True)
    with self.assertRaises(NotImplementedError):
      LING2_MAXTEXT_TO_HF_PARAM_HOOK_FN(self.hf_config, self.maxtext_config, scan_layers=True)

  def test_no_mtp_when_disabled(self):
    """MTP keys should be absent when num_nextn_predict_layers=0."""
    hf_config, mt_config = _make_configs(num_nextn_predict_layers=0)
    mapping = LING2_MAXTEXT_TO_HF_PARAM_MAPPING(hf_config, mt_config)
    mtp_keys = [k for k in mapping if "mtp_block" in k]
    self.assertEqual(len(mtp_keys), 0, f"Found unexpected MTP keys: {mtp_keys}")

  def test_hf_shape_registered(self):
    """ling2 must be registered in HF_SHAPE."""
    self.assertIn("ling2", HF_SHAPE)

  def test_hf_shape_covers_all_mapping_targets(self):
    """Every HF target emitted by PARAM_MAPPING must have a shape in HF_SHAPE."""
    shape_map = HF_SHAPE["ling2"](self.hf_config)
    missing = []
    for hf_targets in self.mapping.values():
      if isinstance(hf_targets, list):
        # Expert-stacked: list of HF keys
        for hf_path in hf_targets:
          if hf_path not in shape_map:
            missing.append(hf_path)
      else:
        if hf_targets not in shape_map:
          missing.append(hf_targets)
    self.assertEqual(missing, [], f"Missing from HF_SHAPE: {missing[:5]}... ({len(missing)} total)")
    # Shapes must be list/tuple of ints
    for hf_path, shape in shape_map.items():
      self.assertIsInstance(shape, (list, tuple), f"{hf_path} shape not list/tuple")
      for d in shape:
        self.assertIsInstance(d, int, f"{hf_path} shape has non-int dim: {shape}")

  def test_param_map_covers_all_hf_shapes(self):
    """Every HF_SHAPE key must be emitted by PARAM_MAPPING (no dangling shapes)."""
    shape_map = HF_SHAPE["ling2"](self.hf_config)
    emitted = set()
    for hf_targets in self.mapping.values():
      if isinstance(hf_targets, list):
        emitted.update(hf_targets)
      else:
        emitted.add(hf_targets)
    dangling = [k for k in shape_map if k not in emitted]
    self.assertEqual(dangling, [], f"HF_SHAPE has dangling keys: {dangling[:5]}...")

  def test_hf_shape_with_mtp_toggle(self):
    """MTP shape keys disappear when num_nextn_predict_layers=0."""
    hf_config_no_mtp, _ = _make_configs(num_nextn_predict_layers=0)
    shape_map = HF_SHAPE["ling2"](hf_config_no_mtp)
    mtp_prefix = f"model.layers.{hf_config_no_mtp['num_hidden_layers']}."
    mtp_keys = [k for k in shape_map if k.startswith(mtp_prefix)]
    self.assertEqual(mtp_keys, [])

    # Sanity: with MTP enabled (default self.hf_config), keys exist
    shape_map_on = HF_SHAPE["ling2"](self.hf_config)
    mtp_prefix_on = f"model.layers.{self.hf_config['num_hidden_layers']}."
    mtp_keys_on = [k for k in shape_map_on if k.startswith(mtp_prefix_on)]
    self.assertGreater(len(mtp_keys_on), 0)

  def test_hf_shape_q_lora_disabled(self):
    """MLA layers use .query (not q_a_proj/q_b_proj) when q_lora_rank=0."""
    hf_config = dict(self.hf_config)
    hf_config["q_lora_rank"] = 0
    mt_config = SimpleNamespace(**vars(self.maxtext_config))
    mt_config.q_lora_rank = 0
    shape_map = HF_SHAPE["ling2"](hf_config)
    # Layer 4 is MLA in a 20-layer, group_size=5 model
    self.assertIn("model.layers.4.attention.query.weight", shape_map)
    self.assertNotIn("model.layers.4.attention.q_a_proj.weight", shape_map)
    self.assertNotIn("model.layers.4.attention.q_b_proj.weight", shape_map)
    self.assertNotIn("model.layers.4.attention.q_a_layernorm.weight", shape_map)
    # Cross-check: mapping also matches on the same config
    mapping_no_lora = LING2_MAXTEXT_TO_HF_PARAM_MAPPING(hf_config, mt_config)
    for key, value in mapping_no_lora.items():
      if "attention-query-kernel" in key:
        self.assertEqual(value, "model.layers.4.attention.query.weight")
        break
    else:
      self.fail("Expected attention-query-kernel not found in mapping")

  def _make_small_configs(self):
    """Small Ling2-shaped config for per-tensor end-to-end tests."""
    hf_config = {
        "num_hidden_layers": 6,
        "first_k_dense_replace": 1,
        "num_experts": 4,
        "num_nextn_predict_layers": 1,
        "layer_group_size": 3,
        "q_lora_rank": 16,
        "kv_lora_rank": 32,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 4,
        "v_head_dim": 8,
        "num_attention_heads": 4,
        "head_dim": 8,
        "hidden_size": 32,
        "moe_intermediate_size": 16,
        "intermediate_size": 48,
        "n_shared_experts": 1,
        "vocab_size": 100,
    }
    mt_config = SimpleNamespace(
        first_num_dense_layers=1,
        num_experts=4,
        inhomogeneous_layer_cycle_interval=3,
        q_lora_rank=16,
        mtp_num_layers=1,
        scan_layers=False,
        param_scan_axis=1,
    )
    return hf_config, mt_config

  def test_process_maxtext_param_end_to_end(self):
    """Drive random MaxText tensors through process_maxtext_param and check
    that output HF tensors match HF_SHAPE declarations exactly."""
    hf_config, mt_config = self._make_small_configs()
    mapping = LING2_MAXTEXT_TO_HF_PARAM_MAPPING(hf_config, mt_config)
    hooks = LING2_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_config, mt_config, saving_to_hf=True)
    shape_map = HF_SHAPE["ling2"](hf_config)

    # Representative keys: one per "form" we care about.
    # For each: inferred MT shape such that the hook produces the HF shape.
    # reshape_kernel(saving_to_hf=True): input.reshape(flip(hf_shape)).T
    # => MT shape must equal flip(HF shape) as a flat reshape source.
    # For 2D: MT_shape = (HF_in, HF_out), HF_shape = (HF_out, HF_in)
    hidden = hf_config["hidden_size"]
    num_heads = hf_config["num_attention_heads"]
    hd = hf_config["head_dim"]
    vocab = hf_config["vocab_size"]
    rope = hf_config["qk_rope_head_dim"]
    qlora = hf_config["q_lora_rank"]
    kvlora = hf_config["kv_lora_rank"]
    inter = hf_config["intermediate_size"]
    moe_inter = hf_config["moe_intermediate_size"]
    num_experts = hf_config["num_experts"]

    representative = [
        # (mt_key, mt_shape (np input to process_maxtext_param))
        ("params-token_embedder-embedding", (vocab, hidden)),
        ("params-decoder-decoder_norm-scale", (hidden,)),
        # Dense MLP (layer 0): wi_0 MT=(hidden, intermediate) -> HF=(inter, hidden)
        ("params-decoder-dense_layers_0-mlp-wi_0-kernel", (hidden, inter)),
        # MLA on layer 2 (last of group_size=3) -> moe_layers_1
        ("params-decoder-moe_layers_1-attention-wq_a-kernel", (hidden, qlora)),
        ("params-decoder-moe_layers_1-attention-wkv_a-kernel", (hidden, kvlora + rope)),
        # GLA on layer 1 -> moe_layers_0
        ("params-decoder-moe_layers_0-attention-query_key_value-kernel", (hidden, 3 * num_heads * hd)),
        # MoE expert stacking: MT shape (num_experts, hidden, moe_inter)
        ("params-decoder-moe_layers_0-mlp-MoeBlock_0-wi_0", (num_experts, hidden, moe_inter)),
        # MoE gate bias (1D, no hook)
        ("params-decoder-moe_layers_0-mlp-MoeBlock_0-gate-bias", (num_experts,)),
        # MTP projection (2*hidden -> hidden): MT=(2*hidden, hidden) -> HF=(hidden, 2*hidden)
        ("params-mtp_block-mtp_layer_1-mtp_1_projection-kernel", (2 * hidden, hidden)),
    ]

    rng = np.random.RandomState(0)
    for mt_key, mt_shape in representative:
      with self.subTest(mt_key=mt_key):
        self.assertIn(mt_key, mapping, f"{mt_key} missing from mapping")
        mt_weight = rng.randn(*mt_shape).astype(np.float32)
        out = process_maxtext_param(mt_key, mt_weight, mapping, hooks, shape_map, mt_config)
        hf_targets = mapping[mt_key]
        expected_count = len(hf_targets) if isinstance(hf_targets, list) else 1
        self.assertEqual(len(out), expected_count)
        for hf_path, hf_tensor in out:
          self.assertIn(hf_path, shape_map)
          self.assertEqual(
              tuple(hf_tensor.shape),
              tuple(shape_map[hf_path]),
              f"{hf_path} shape mismatch: got {hf_tensor.shape}, want {shape_map[hf_path]}",
          )

  def test_moe_expert_stacking_preserves_data(self):
    """MoE expert axis=0 slicing: HF experts.e.*.weight == reshape_kernel(MT[e])."""
    hf_config, mt_config = self._make_small_configs()
    mapping = LING2_MAXTEXT_TO_HF_PARAM_MAPPING(hf_config, mt_config)
    hooks = LING2_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_config, mt_config, saving_to_hf=True)
    shape_map = HF_SHAPE["ling2"](hf_config)

    num_experts = hf_config["num_experts"]
    hidden = hf_config["hidden_size"]
    moe_inter = hf_config["moe_intermediate_size"]

    # Build a MT MoE wi_0 tensor where slice e is filled with value (e+1).
    mt_key = "params-decoder-moe_layers_0-mlp-MoeBlock_0-wi_0"
    mt_weight = np.stack(
        [np.full((hidden, moe_inter), fill_value=(e + 1), dtype=np.float32) for e in range(num_experts)],
        axis=0,
    )

    out = process_maxtext_param(mt_key, mt_weight, mapping, hooks, shape_map, mt_config)
    self.assertEqual(len(out), num_experts)

    # Verify each HF tensor: path index matches expert_idx, and reshape-roundtrip
    # yields the original MT slice.
    hooks_back = LING2_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_config, mt_config, saving_to_hf=False)
    for expert_idx, (hf_path, hf_tensor) in enumerate(out):
      self.assertEqual(hf_path, f"model.layers.1.mlp.experts.{expert_idx}.gate_proj.weight")
      # Reverse the hook to recover MT slice, compare to original slice e
      mt_slice_shape = mt_weight.shape[1:]  # (hidden, moe_inter)
      restored = hooks_back[mt_key](hf_tensor, mt_slice_shape)
      np.testing.assert_array_almost_equal(restored, mt_weight[expert_idx], decimal=5)

  @unittest.skipUnless(
      os.getenv("LING2_HF_REF_PATH"),
      "Set LING2_HF_REF_PATH to a local Ling2 HF repo dir to enable this test.",
  )
  def test_shape_map_matches_reference_hf_index(self):
    """Compare HF_SHAPE against a real Ling2 HF repo's safetensors index."""
    ref_path = os.environ["LING2_HF_REF_PATH"]
    with open(os.path.join(ref_path, "config.json"), "r", encoding="utf-8") as f:
      real_config = json.load(f)
    index_path = os.path.join(ref_path, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
      real_index = json.load(f)
    # safetensors index has metadata.total_size + weight_map. We need shapes per tensor,
    # which requires opening each shard. Do a lightweight check: ensure every real
    # tensor path appears in HF_SHAPE["ling2"](real_config).
    shape_map = HF_SHAPE["ling2"](real_config)
    real_paths = set(real_index["weight_map"].keys())
    missing_in_our_map = real_paths - set(shape_map.keys())
    self.assertEqual(
        missing_in_our_map, set(), f"HF_SHAPE missing {len(missing_in_our_map)} keys: " f"{list(missing_in_our_map)[:5]}"
    )
    # And vice versa (shape_map should not have extras beyond MTP toggling etc.)
    extras = set(shape_map.keys()) - real_paths
    self.assertEqual(extras, set(), f"HF_SHAPE has extra keys not in real index: {list(extras)[:5]}")


if __name__ == "__main__":
  unittest.main()
