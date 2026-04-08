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

import unittest
from types import SimpleNamespace

import numpy as np

from maxtext.checkpoint_conversion.utils.param_mapping import (
    HOOK_FNS,
    LING2_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    LING2_MAXTEXT_TO_HF_PARAM_MAPPING,
    PARAM_MAPPING,
)


def _make_configs(num_nextn_predict_layers=1):
  """Create mock HF config dict and MaxText config for Ling2 (matching ling2.yml)."""
  hf_config = {
      "num_hidden_layers": 20,
      "first_k_dense_replace": 1,
      "num_experts": 256,
      "num_nextn_predict_layers": num_nextn_predict_layers,
      "layer_group_size": 5,
      "q_lora_rank": 256,
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

  def test_reshape_kernel_roundtrip(self):
    """reshape_kernel should be perfectly invertible (MT -> HF -> MT)."""
    rng = np.random.RandomState(42)
    test_cases = [
        ("params-decoder-logits_dense-kernel", (157184, 2048), (2048, 157184)),
        ("params-decoder-dense_layers_0-attention-query_key_value-kernel", (2048, 6144), (6144, 2048)),
        ("params-decoder-moe_layers_3-attention-wq_a-kernel", (2048, 256), (256, 2048)),
        ("params-decoder-moe_layers_0-mlp-shared_experts-wi_0-kernel", (2048, 2048), (2048, 2048)),
    ]
    for key, mt_shape, hf_shape in test_cases:
      with self.subTest(key=key):
        tensor = rng.randn(*mt_shape).astype(np.float32)
        hf_tensor = self.hooks_to_hf[key](tensor, hf_shape)
        restored = self.hooks_to_mt[key](hf_tensor, mt_shape)
        np.testing.assert_array_almost_equal(tensor, restored, decimal=6)

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


if __name__ == "__main__":
  unittest.main()
