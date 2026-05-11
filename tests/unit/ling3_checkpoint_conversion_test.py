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

"""Unit tests for Ling3 checkpoint conversion (param mapping + hook functions).

Validates mapping completeness, layer dispatch (dense/MoE, MLA/KDA),
KDA parameter-family coverage, gated-MLA conditional, depthwise conv hook
roundtrip, expert stacking, hook coverage, HF_SHAPE double coverage, MTP
toggle, and scan/unscan mode encoding — all without JAX, TPU, or model
weights. Mirrors the Ling2 test template (RFC-0003, RFC-0017 §测试方案).
"""

import json
import os
import re
import unittest
from types import SimpleNamespace

import numpy as np
import pytest

from maxtext.checkpoint_conversion.utils.hf_shape import HF_SHAPE
from maxtext.checkpoint_conversion.utils.param_mapping import (
    EXPERT_AXIS_KEYS,
    HOOK_FNS,
    LING3_EXPERT_AXIS_KEYS,
    LING3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    LING3_MAXTEXT_TO_HF_PARAM_MAPPING,
    PARAM_MAPPING,
    _ling3_unscan_prefix,
    get_expert_axis_keys,
)
from maxtext.checkpoint_conversion.utils.utils import process_maxtext_param
from maxtext.utils.globals import HF_IDS


def _make_configs(num_nextn_predict_layers=1, enable_gated_attention=True, mla_gated_attention_type=None):
  """Create mock HF config dict and MaxText config for Ling3-tiny.

  Values mirror `src/maxtext/configs/models/ling3-tiny.yml` plus the
  BailingMoeV3 HF config surface.

  Args:
    num_nextn_predict_layers: HF field controlling MTP.
    enable_gated_attention: v1 legacy bool. Maps to `head_wise` when True.
    mla_gated_attention_type: v2 canonical enum ("disabled"/"head_wise"/
      "element_wise"). When provided, takes precedence over the legacy bool
      and populates both the MaxText attr and the HF config's
      `gated_attention_proj_granularity_type`.
  """
  # Derive effective gating for both surfaces. `mla_gated_attention_type`
  # (when provided) is the authoritative v2 setting; otherwise fall back to
  # the v1 `enable_gated_attention` bool.
  if mla_gated_attention_type is None:
    mla_gated_attention_type = "head_wise" if enable_gated_attention else "disabled"
  hf_granularity = mla_gated_attention_type if mla_gated_attention_type != "disabled" else None

  hf_config = {
      # Core dims (from ling3-tiny.yml)
      "num_hidden_layers": 24,
      "first_k_dense_replace": 1,
      "num_experts": 128,
      "num_nextn_predict_layers": num_nextn_predict_layers,
      "layer_group_size": 4,
      "q_lora_rank": 256,
      "hidden_size": 1536,
      "vocab_size": 157184,
      "kv_lora_rank": 512,
      "qk_nope_head_dim": 128,
      "qk_rope_head_dim": 64,
      "v_head_dim": 128,
      "num_attention_heads": 16,
      "head_dim": 128,
      "moe_intermediate_size": 512,
      "intermediate_size": 4608,
      "num_shared_experts": 1,
      # Ling3-specific knobs (BailingMoeV3Config): expose both the v1 legacy
      # bool and the v2 granularity string so detection covers both paths.
      "enable_gated_attention": enable_gated_attention,
      "gated_attention_proj_granularity_type": hf_granularity,
      "short_conv_kernel_size": 4,
  }
  maxtext_config = SimpleNamespace(
      first_num_dense_layers=1,
      num_experts=128,
      inhomogeneous_layer_cycle_interval=4,
      q_lora_rank=256,
      mtp_num_layers=1 if num_nextn_predict_layers > 0 else 0,
      enable_gated_attention=enable_gated_attention,
      mla_gated_attention_type=mla_gated_attention_type,
      scan_layers=False,
      param_scan_axis=1,
  )
  return hf_config, maxtext_config


def _iter_hf_paths(mapping):
  """Flatten a PARAM_MAPPING dict's values into every leaf HF path string.

  Mapping values are one of: str (unscan single), list[str] (expert stacking or
  scan), list[list[str]] (scan + expert stacking, nested). Yield each leaf.
  """
  for hf_targets in mapping.values():
    if isinstance(hf_targets, str):
      yield hf_targets
      continue
    for hf_path in hf_targets:
      if isinstance(hf_path, list):
        yield from hf_path
      else:
        yield hf_path


class Ling3CheckpointConversionTest(unittest.TestCase):
  """Unscan-mode mapping, HF_SHAPE, and hook coverage tests."""

  def setUp(self):
    self.hf_config, self.maxtext_config = _make_configs()
    self.mapping = LING3_MAXTEXT_TO_HF_PARAM_MAPPING(self.hf_config, self.maxtext_config)
    self.hooks_to_hf = LING3_MAXTEXT_TO_HF_PARAM_HOOK_FN(self.hf_config, self.maxtext_config, saving_to_hf=True)
    self.hooks_to_mt = LING3_MAXTEXT_TO_HF_PARAM_HOOK_FN(self.hf_config, self.maxtext_config, saving_to_hf=False)

  def test_registered_in_global_dicts(self):
    """ling3-tiny must be registered in all four registries."""
    self.assertIn("ling3-tiny", PARAM_MAPPING)
    self.assertIn("ling3-tiny", HOOK_FNS)
    self.assertIn("ling3-tiny", HF_SHAPE)
    self.assertIn("ling3-tiny", HF_IDS)

  def test_layer_dispatch_dense_vs_moe(self):
    """Layer 0 is dense MLP; layers 1+ are MoE."""
    self.assertIn("params-decoder-dense_layers_0-mlp-wi_0-kernel", self.mapping)
    self.assertNotIn("params-decoder-dense_layers_0-mlp-MoeBlock_0-gate-kernel", self.mapping)
    self.assertIn("params-decoder-moe_layers_0-mlp-MoeBlock_0-gate-kernel", self.mapping)
    self.assertIn("params-decoder-moe_layers_0-mlp-shared_experts-wi_0-kernel", self.mapping)
    self.assertNotIn("params-decoder-moe_layers_0-mlp-wi_0-kernel", self.mapping)

  def test_layer_dispatch_mla_vs_kda(self):
    """With interval=4: MLA at HF idx 3, 7, 11, 15, 19, 23; KDA elsewhere."""
    # dense_layers_0 (HF idx 0) is KDA: (0+1) % 4 == 1 ≠ 0
    self.assertIn("params-decoder-dense_layers_0-attention-q_proj-kernel", self.mapping)
    self.assertNotIn("params-decoder-dense_layers_0-attention-wq_a-kernel", self.mapping)
    # moe_layers_0 (HF idx 1) is KDA
    self.assertIn("params-decoder-moe_layers_0-attention-q_proj-kernel", self.mapping)
    # moe_layers_2 (HF idx 3) is MLA (last of first group, interval=4)
    self.assertIn("params-decoder-moe_layers_2-attention-wq_a-kernel", self.mapping)
    self.assertIn("params-decoder-moe_layers_2-attention-wkv_a-kernel", self.mapping)
    self.assertNotIn("params-decoder-moe_layers_2-attention-q_proj-kernel", self.mapping)
    # moe_layers_3 (HF idx 4) is KDA
    self.assertIn("params-decoder-moe_layers_3-attention-q_proj-kernel", self.mapping)
    self.assertNotIn("params-decoder-moe_layers_3-attention-wq_a-kernel", self.mapping)
    # moe_layers_22 (HF idx 23, last of model) is MLA
    self.assertIn("params-decoder-moe_layers_22-attention-wq_a-kernel", self.mapping)

  def test_kda_param_family_complete(self):
    """All 13 KDA parameter families present in any non-MLA layer."""
    prefix = "params-decoder-moe_layers_0-attention"  # HF idx 1 is KDA
    expected_suffixes = [
        "q_proj-kernel",
        "k_proj-kernel",
        "v_proj-kernel",
        "q_conv-kernel",
        "k_conv-kernel",
        "v_conv-kernel",
        "A_log",
        "dt_bias",
        "g_proj-kernel",  # gate-decay (PR #76 naming, maps to HF f_a_proj)
        "b_proj-kernel",  # beta
        "gate_proj-kernel",  # output gate (PR #76 naming, maps to HF g_a_proj)
        "out_norm-scale",
        "o_proj-kernel",
    ]
    for suffix in expected_suffixes:
      key = f"{prefix}-{suffix}"
      self.assertIn(key, self.mapping, f"Missing KDA param: {key}")

  def test_kda_hf_side_names(self):
    """KDA MaxText attrs map to the correct bailing_moe_v3 HF names."""
    # gate-decay: MaxText g_proj ↔ HF state_dict key f_proj
    self.assertEqual(
        self.mapping["params-decoder-moe_layers_0-attention-g_proj-kernel"],
        "model.layers.1.attention.f_proj.weight",
    )
    # output gate: MaxText gate_proj ↔ HF state_dict key g_proj
    self.assertEqual(
        self.mapping["params-decoder-moe_layers_0-attention-gate_proj-kernel"],
        "model.layers.1.attention.g_proj.weight",
    )
    # depthwise conv: MaxText q_conv ↔ HF q_conv1d (attribute renamed in PR #76)
    self.assertEqual(
        self.mapping["params-decoder-moe_layers_0-attention-q_conv-kernel"],
        "model.layers.1.attention.q_conv1d.weight",
    )

  def test_mla_g_proj_conditional_on_gating_type(self):
    """MLA `g_proj` is emitted when `mla_gated_attention_type` is head_wise or element_wise."""
    # Default (head_wise): g_proj present at MLA layers, maps to HF g_proj.weight
    self.assertIn("params-decoder-moe_layers_2-attention-g_proj-kernel", self.mapping)
    self.assertEqual(
        self.mapping["params-decoder-moe_layers_2-attention-g_proj-kernel"],
        "model.layers.3.attention.g_proj.weight",
    )
    # Disabled (via v1 legacy bool): no MLA g_proj emission
    _, mt_off = _make_configs(enable_gated_attention=False)
    hf_off = {**self.hf_config, "enable_gated_attention": False, "gated_attention_proj_granularity_type": None}
    mapping_no_gate = LING3_MAXTEXT_TO_HF_PARAM_MAPPING(hf_off, mt_off)
    # MLA layer 2 (HF idx 3) has no g_proj when gating disabled
    self.assertNotIn("params-decoder-moe_layers_2-attention-g_proj-kernel", mapping_no_gate)
    # KDA layer still emits its own g_proj (KDA gate-decay), independent of MLA flag
    self.assertIn("params-decoder-moe_layers_0-attention-g_proj-kernel", mapping_no_gate)

  def test_mla_g_proj_v2_field_precedence(self):
    """`mla_gated_attention_type` (v2) takes precedence over v1 `enable_gated_attention`."""
    # element_wise overrides a False legacy bool
    hf_elem, mt_elem = _make_configs(enable_gated_attention=False, mla_gated_attention_type="element_wise")
    mapping_elem = LING3_MAXTEXT_TO_HF_PARAM_MAPPING(hf_elem, mt_elem)
    self.assertIn("params-decoder-moe_layers_2-attention-g_proj-kernel", mapping_elem)
    # HF_SHAPE reflects element_wise shape (num_heads * v_head_dim, hidden)
    shape_elem = HF_SHAPE["ling3-tiny"](hf_elem)
    self.assertEqual(
        list(shape_elem["model.layers.3.attention.g_proj.weight"]),
        [hf_elem["num_attention_heads"] * hf_elem["v_head_dim"], hf_elem["hidden_size"]],
    )
    # explicit "disabled" v2 value suppresses emission even if v1 bool is True
    hf_off, mt_off = _make_configs(enable_gated_attention=True, mla_gated_attention_type="disabled")
    mapping_off = LING3_MAXTEXT_TO_HF_PARAM_MAPPING(hf_off, mt_off)
    self.assertNotIn("params-decoder-moe_layers_2-attention-g_proj-kernel", mapping_off)

  def test_expert_stacking_list_values(self):
    """MoE expert weights are lists of length num_experts=128."""
    key = "params-decoder-moe_layers_0-mlp-MoeBlock_0-wi_0"
    self.assertIn(key, self.mapping)
    val = self.mapping[key]
    self.assertIsInstance(val, list)
    self.assertEqual(len(val), 128)
    self.assertEqual(val[0], "model.layers.1.mlp.experts.0.gate_proj.weight")
    self.assertEqual(val[127], "model.layers.1.mlp.experts.127.gate_proj.weight")

  def test_hook_coverage_matches_mapping_kernels(self):
    """Every kernel key in mapping has a hook; scale/bias/A_log/dt_bias keys do not."""
    kernel_suffixes = ("-kernel", "-wi_0", "-wi_1", "-wo")
    no_hook_endings = ("-scale", "-bias", "-embedding", "-A_log", "-dt_bias")
    for key in self.mapping:
      if any(key.endswith(s) for s in kernel_suffixes):
        self.assertIn(key, self.hooks_to_hf, f"Missing hook for kernel key: {key}")
      elif any(key.endswith(s) for s in no_hook_endings):
        self.assertNotIn(key, self.hooks_to_hf, f"Unexpected hook for no-transform key: {key}")

  def test_conv_hook_is_reshape_depthwise_conv_not_reshape_kernel(self):
    """q_conv / k_conv / v_conv must bind to reshape_depthwise_conv (different from reshape_kernel)."""
    conv_keys = [
        "params-decoder-moe_layers_0-attention-q_conv-kernel",
        "params-decoder-moe_layers_0-attention-k_conv-kernel",
        "params-decoder-moe_layers_0-attention-v_conv-kernel",
    ]
    other_kernel_key = "params-decoder-moe_layers_0-attention-q_proj-kernel"
    conv_hook = self.hooks_to_hf[conv_keys[0]]
    for k in conv_keys[1:]:
      self.assertIs(self.hooks_to_hf[k], conv_hook, f"{k} should share the conv hook")
    self.assertIsNot(self.hooks_to_hf[other_kernel_key], conv_hook)

  def test_reshape_depthwise_conv_roundtrip(self):
    """Depthwise conv weight MT [K, C] ↔ HF [C, 1, K] is a self-inverse roundtrip."""
    rng = np.random.RandomState(0)
    mt_key = "params-decoder-moe_layers_0-attention-q_conv-kernel"
    # MT shape [K_conv, H*head_dim] = [4, 16 * 128] = [4, 2048]
    # HF shape [H*head_dim, 1, K_conv] = [2048, 1, 4]
    mt_shape = (4, 2048)
    hf_shape = (2048, 1, 4)
    mt_tensor = rng.randn(*mt_shape).astype(np.float32)
    hf_tensor = self.hooks_to_hf[mt_key](mt_tensor)
    self.assertEqual(tuple(hf_tensor.shape), hf_shape)
    restored = self.hooks_to_mt[mt_key](hf_tensor)
    self.assertEqual(tuple(restored.shape), mt_shape)
    np.testing.assert_array_equal(mt_tensor, restored)

  def test_reshape_depthwise_conv_transposes_kernel(self):
    """Conv hook flips kernel axis + transposes between MT [K, C] and HF [C, 1, K].

    MaxText `ShortConvolution` uses convolution convention (kernel[k] is the
    coefficient at lag k); HF/PyTorch `nn.Conv1d` uses cross-correlation
    convention (weight[k] is the coefficient at lag K-1-k). Hence the flip.
    """
    mt_key = "params-decoder-moe_layers_0-attention-q_conv-kernel"
    K, C = 4, 8
    mt_tensor = np.arange(K * C, dtype=np.float32).reshape(K, C)
    hf_tensor = self.hooks_to_hf[mt_key](mt_tensor)
    self.assertEqual(tuple(hf_tensor.shape), (C, 1, K))
    # hf_tensor[c, 0, k] should equal mt_tensor[K-1-k, c] (flipped along K)
    for k in range(K):
      for c in range(C):
        self.assertEqual(hf_tensor[c, 0, k], mt_tensor[K - 1 - k, c])

  def test_reshape_kernel_roundtrip_from_shape_map(self):
    """reshape_kernel MT->HF->MT roundtrip, using HF_SHAPE as source of truth."""
    shape_map = HF_SHAPE["ling3-tiny"](self.hf_config)
    rng = np.random.RandomState(42)
    # 2D kernel: MT shape = flip(HF shape)
    sample_keys = [
        ("params-decoder-logits_dense-kernel", "lm_head.weight"),
        ("params-decoder-moe_layers_2-attention-wq_a-kernel", "model.layers.3.attention.q_a_proj.weight"),
        (
            "params-decoder-moe_layers_0-mlp-shared_experts-wi_0-kernel",
            "model.layers.1.mlp.shared_experts.gate_proj.weight",
        ),
    ]
    for mt_key, hf_key in sample_keys:
      with self.subTest(mt_key=mt_key):
        hf_shape = tuple(shape_map[hf_key])
        mt_shape = hf_shape[::-1]
        mt_tensor = rng.randn(*mt_shape).astype(np.float32)
        hf_tensor = self.hooks_to_hf[mt_key](mt_tensor, hf_shape)
        self.assertEqual(tuple(hf_tensor.shape), hf_shape)
        restored = self.hooks_to_mt[mt_key](hf_tensor, mt_shape)
        np.testing.assert_array_almost_equal(mt_tensor, restored, decimal=6)

  def test_no_mtp_when_disabled(self):
    """MTP keys absent when num_nextn_predict_layers=0."""
    hf_config, mt_config = _make_configs(num_nextn_predict_layers=0)
    mapping = LING3_MAXTEXT_TO_HF_PARAM_MAPPING(hf_config, mt_config)
    mtp_keys = [k for k in mapping if "mtp_block" in k]
    self.assertEqual(len(mtp_keys), 0, f"Found unexpected MTP keys: {mtp_keys}")

  def test_hf_shape_covers_all_mapping_targets(self):
    """Every HF target emitted by PARAM_MAPPING has a shape in HF_SHAPE."""
    shape_map = HF_SHAPE["ling3-tiny"](self.hf_config)
    missing = [p for p in _iter_hf_paths(self.mapping) if p not in shape_map]
    self.assertEqual(missing, [], f"Missing from HF_SHAPE: {missing[:5]}... ({len(missing)} total)")
    for hf_path, shape in shape_map.items():
      self.assertIsInstance(shape, (list, tuple), f"{hf_path} shape not list/tuple")
      for d in shape:
        self.assertIsInstance(d, int, f"{hf_path} shape has non-int dim: {shape}")

  def test_param_map_covers_all_hf_shapes(self):
    """Every HF_SHAPE key is emitted by PARAM_MAPPING (no dangling shapes)."""
    shape_map = HF_SHAPE["ling3-tiny"](self.hf_config)
    emitted = set(_iter_hf_paths(self.mapping))
    dangling = [k for k in shape_map if k not in emitted]
    self.assertEqual(dangling, [], f"HF_SHAPE has dangling keys: {dangling[:5]}...")

  def test_hf_shape_with_mtp_toggle(self):
    """MTP shape keys disappear when num_nextn_predict_layers=0."""
    hf_config_no_mtp, _ = _make_configs(num_nextn_predict_layers=0)
    shape_map = HF_SHAPE["ling3-tiny"](hf_config_no_mtp)
    mtp_prefix = f"model.layers.{hf_config_no_mtp['num_hidden_layers']}."
    mtp_keys = [k for k in shape_map if k.startswith(mtp_prefix)]
    self.assertEqual(mtp_keys, [])
    shape_map_on = HF_SHAPE["ling3-tiny"](self.hf_config)
    mtp_keys_on = [k for k in shape_map_on if k.startswith(f"model.layers.{self.hf_config['num_hidden_layers']}.")]
    self.assertGreater(len(mtp_keys_on), 0)

  def test_hf_shape_mla_g_proj_shape_is_head_wise(self):
    """Gated MLA output weight must be [num_heads, hidden_size] per RFC-0017 §3."""
    shape_map = HF_SHAPE["ling3-tiny"](self.hf_config)
    # HF idx 3 is the first MLA layer (interval=4, last-of-group)
    key = "model.layers.3.attention.g_proj.weight"
    self.assertIn(key, shape_map)
    self.assertEqual(
        list(shape_map[key]),
        [self.hf_config["num_attention_heads"], self.hf_config["hidden_size"]],
    )

  def test_mtp_with_q_lora_rank_zero_uses_query_path(self):
    """With q_lora_rank=0 + MTP enabled, MTP attention emits .query.weight (not q_a/q_b proj)."""
    hf_config, mt_config = _make_configs(num_nextn_predict_layers=1)
    hf_config["q_lora_rank"] = 0
    mt_config.q_lora_rank = 0
    mapping = LING3_MAXTEXT_TO_HF_PARAM_MAPPING(hf_config, mt_config)
    shape_map = HF_SHAPE["ling3-tiny"](hf_config)
    mtp_tf_prefix = "params-mtp_block-mtp_layer_1-mtp_1_transformer_layer-attention"
    # query path is emitted
    self.assertIn(f"{mtp_tf_prefix}-query-kernel", mapping)
    self.assertEqual(
        mapping[f"{mtp_tf_prefix}-query-kernel"],
        f"model.layers.{hf_config['num_hidden_layers']}.attention.query.weight",
    )
    # LoRA-decomposed paths are absent
    self.assertNotIn(f"{mtp_tf_prefix}-wq_a-kernel", mapping)
    self.assertNotIn(f"{mtp_tf_prefix}-wq_b-kernel", mapping)
    # HF_SHAPE matches: .query.weight present, q_a_proj / q_b_proj absent at MTP layer
    mtp_hf_prefix = f"model.layers.{hf_config['num_hidden_layers']}.attention"
    self.assertIn(f"{mtp_hf_prefix}.query.weight", shape_map)
    self.assertNotIn(f"{mtp_hf_prefix}.q_a_proj.weight", shape_map)
    # Hook for .query.weight also exists
    hooks = LING3_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_config, mt_config, saving_to_hf=True)
    self.assertIn(f"{mtp_tf_prefix}-query-kernel", hooks)

  def test_scan_layers_plus_mtp_no_longer_raises(self):
    """scan_layers=True + MTP is now supported via EXPERT_AXIS_KEYS registry."""
    hf_config, mt_config = _make_configs(num_nextn_predict_layers=1)
    # Both must NOT raise — the per-key registry replaces the prior NotImplementedError.
    LING3_MAXTEXT_TO_HF_PARAM_MAPPING(hf_config, mt_config, scan_layers=True)
    LING3_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_config, mt_config, scan_layers=True)
    # Sanity: scan_layers=True without MTP also works
    hf_config_no_mtp, mt_config_no_mtp = _make_configs(num_nextn_predict_layers=0)
    LING3_MAXTEXT_TO_HF_PARAM_MAPPING(hf_config_no_mtp, mt_config_no_mtp, scan_layers=True)
    LING3_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_config_no_mtp, mt_config_no_mtp, scan_layers=True)

  def test_hf_shape_mla_g_proj_absent_when_disabled(self):
    """With enable_gated_attention=False, MLA layers lose their g_proj entry.

    Note: KDA layers also have an `attention.g_proj.weight` (the KDA output
    gate), which is architectural and independent of the MLA gating flag.
    The test checks MLA-specific layer indices (3, 7, ...) where a non-MLA
    KDA g_proj cannot occur.
    """
    hf_config_on, _ = _make_configs(enable_gated_attention=True)
    hf_config_off, _ = _make_configs(enable_gated_attention=False)
    shape_on = HF_SHAPE["ling3-tiny"](hf_config_on)
    shape_off = HF_SHAPE["ling3-tiny"](hf_config_off)
    # Layer 3 is MLA (interval=4, last-of-group)
    self.assertIn("model.layers.3.attention.g_proj.weight", shape_on)
    self.assertNotIn("model.layers.3.attention.g_proj.weight", shape_off)
    # Layer 7 is MLA
    self.assertIn("model.layers.7.attention.g_proj.weight", shape_on)
    self.assertNotIn("model.layers.7.attention.g_proj.weight", shape_off)
    # Layer 0 is KDA: g_proj (KDA output gate) is always present, independent of flag
    self.assertIn("model.layers.0.attention.g_proj.weight", shape_on)
    self.assertIn("model.layers.0.attention.g_proj.weight", shape_off)


class Ling3CheckpointConversionScanTest(unittest.TestCase):
  """Scan-mode (scan_layers=True) mapping tests — RFC-0017 §7 / RFC-0012 §4.6.

  MTP is disabled here to keep these tests focused on scan-region mapping shape;
  scan_layers=True + MTP is now supported (see Ling3MixedScanMTPTest).
  """

  def setUp(self):
    self.hf_config, self.maxtext_config = _make_configs(num_nextn_predict_layers=0)
    self.mapping = LING3_MAXTEXT_TO_HF_PARAM_MAPPING(self.hf_config, self.maxtext_config, scan_layers=True)
    self.hooks = LING3_MAXTEXT_TO_HF_PARAM_HOOK_FN(
        self.hf_config, self.maxtext_config, scan_layers=True, saving_to_hf=True
    )
    # unscan_prefix = ceil(1/4)*4 = 4; scan_length = (24-4)/4 = 5
    self.unscan_prefix = 4
    self.scan_length = 5

  def test_scan_emits_moe_layers_layers_prefixes(self):
    """Scan mode should emit params-decoder-moe_layers-layers_{intra_idx} keys."""
    scan_keys = [k for k in self.mapping if "-moe_layers-layers_" in k]
    self.assertGreater(len(scan_keys), 0, "No scan-mode keys emitted")
    for intra in range(self.maxtext_config.inhomogeneous_layer_cycle_interval):
      prefix = f"params-decoder-moe_layers-layers_{intra}-"
      matching = [k for k in scan_keys if k.startswith(prefix)]
      self.assertGreater(len(matching), 0, f"No keys for intra_idx={intra}")

  def test_scan_value_is_list_of_scan_length(self):
    """Scan region single-tensor keys have list values of length scan_length."""
    key = "params-decoder-moe_layers-layers_0-attention-q_proj-kernel"
    self.assertIn(key, self.mapping)
    val = self.mapping[key]
    self.assertIsInstance(val, list)
    self.assertEqual(len(val), self.scan_length)
    # HF layer indices for intra_idx=0 are unscan_prefix + si * interval + 0
    #   = 4, 8, 12, 16, 20
    expected = [f"model.layers.{4 + si * 4}.attention.q_proj.weight" for si in range(self.scan_length)]
    self.assertEqual(val, expected)

  def test_scan_expert_stacking_is_nested_list(self):
    """Scan + expert stacking: outer=num_experts, inner=scan_length (DeepSeek convention)."""
    key = "params-decoder-moe_layers-layers_0-mlp-MoeBlock_0-wi_0"
    self.assertIn(key, self.mapping)
    val = self.mapping[key]
    self.assertIsInstance(val, list)
    self.assertEqual(len(val), 128)  # outer = num_experts
    self.assertIsInstance(val[0], list)
    self.assertEqual(len(val[0]), self.scan_length)  # inner = scan_length
    self.assertEqual(val[0][0], "model.layers.4.mlp.experts.0.gate_proj.weight")
    self.assertEqual(val[127][4], "model.layers.20.mlp.experts.127.gate_proj.weight")

  def test_scan_unscan_prefix_keys_are_single_strings(self):
    """The unscan-prefix layers still produce single-string mapping values in scan mode."""
    self.assertIsInstance(
        self.mapping["params-decoder-dense_layers_0-input_layernorm-scale"],
        str,
    )
    self.assertIsInstance(
        self.mapping["params-decoder-moe_layers_0-attention-q_proj-kernel"],
        str,
    )
    # HF idx 3 is MLA, still within unscan prefix (unscan_prefix=4)
    self.assertIsInstance(
        self.mapping["params-decoder-moe_layers_2-attention-wq_a-kernel"],
        str,
    )

  def test_scan_mla_position_in_cycle(self):
    """Scan region: intra_idx=interval-1 is MLA, others are KDA."""
    # In the scan region, intra_idx=3 is MLA (last of cycle).
    self.assertIn("params-decoder-moe_layers-layers_3-attention-wq_a-kernel", self.mapping)
    self.assertNotIn("params-decoder-moe_layers-layers_3-attention-q_proj-kernel", self.mapping)
    # intra_idx=0..2 are KDA
    for intra in range(3):
      self.assertIn(f"params-decoder-moe_layers-layers_{intra}-attention-q_proj-kernel", self.mapping)
      self.assertNotIn(f"params-decoder-moe_layers-layers_{intra}-attention-wq_a-kernel", self.mapping)

  def test_scan_hook_coverage(self):
    """Scan-region kernel keys must have hooks (reshape_kernel or reshape_depthwise_conv)."""
    kernel_suffixes = ("-kernel", "-wi_0", "-wi_1", "-wo")
    scan_keys = [k for k in self.mapping if "-moe_layers-layers_" in k]
    for key in scan_keys:
      if any(key.endswith(s) for s in kernel_suffixes):
        self.assertIn(key, self.hooks, f"Missing hook for scan-region kernel: {key}")


class Ling3ProcessMaxtextParamEndToEndTest(unittest.TestCase):
  """Drive random tensors through process_maxtext_param and verify output shapes
  match HF_SHAPE declarations exactly. Covers the integration of
  mapping + hook + shape_map for the three Ling3-specific new paths."""

  def _make_small_configs(self):
    """Return a small Ling3-tiny-shaped (config, maxtext_config) tuple for per-tensor E2E tests."""
    hf_config = {
        "num_hidden_layers": 8,
        "first_k_dense_replace": 1,
        "num_experts": 4,
        "num_nextn_predict_layers": 1,
        "layer_group_size": 4,
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
        "num_shared_experts": 1,
        "vocab_size": 100,
        "enable_gated_attention": True,
        "short_conv_kernel_size": 4,
    }
    mt_config = SimpleNamespace(
        first_num_dense_layers=1,
        num_experts=4,
        inhomogeneous_layer_cycle_interval=4,
        q_lora_rank=16,
        mtp_num_layers=1,
        scan_layers=False,
        param_scan_axis=1,
        enable_gated_attention=True,
    )
    return hf_config, mt_config

  def test_kda_and_mla_g_proj_through_process_maxtext_param(self):
    hf_config, mt_config = self._make_small_configs()
    mapping = LING3_MAXTEXT_TO_HF_PARAM_MAPPING(hf_config, mt_config)
    hooks = LING3_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_config, mt_config, saving_to_hf=True)
    shape_map = HF_SHAPE["ling3-tiny"](hf_config)

    hidden = hf_config["hidden_size"]
    num_heads = hf_config["num_attention_heads"]
    head_dim = hf_config["head_dim"]
    kda_proj = num_heads * head_dim
    k_conv = hf_config["short_conv_kernel_size"]

    # Representative MaxText input shapes (MT layout).
    # HF idx 1 is KDA (interval=4, (1+1)%4=2 ≠ 0), maps to moe_layers_0.
    # HF idx 3 is MLA (last of group), maps to moe_layers_2.
    representative = [
        ("params-token_embedder-embedding", (hf_config["vocab_size"], hidden)),
        ("params-decoder-decoder_norm-scale", (hidden,)),
        # KDA Q projection (2D kernel, MT=(hidden, H*K))
        ("params-decoder-moe_layers_0-attention-q_proj-kernel", (hidden, kda_proj)),
        # KDA depthwise conv (ShortConvolution MT [K_conv, H*K] ↔ HF [H*K, 1, K_conv]).
        ("params-decoder-moe_layers_0-attention-q_conv-kernel", (k_conv, kda_proj)),
        # KDA A_log (1D pass-through)
        ("params-decoder-moe_layers_0-attention-A_log", (num_heads,)),
        # KDA dt_bias (1D pass-through)
        ("params-decoder-moe_layers_0-attention-dt_bias", (kda_proj,)),
        # KDA output RMSNorm scale (1D pass-through)
        ("params-decoder-moe_layers_0-attention-out_norm-scale", (head_dim,)),
        # MLA gated output (2D kernel, MT=(hidden, num_heads))
        ("params-decoder-moe_layers_2-attention-g_proj-kernel", (hidden, num_heads)),
        # MoE gate bias (1D pass-through)
        ("params-decoder-moe_layers_0-mlp-MoeBlock_0-gate-bias", (hf_config["num_experts"],)),
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
          self.assertIn(hf_path, shape_map, f"{hf_path} missing from HF_SHAPE")
          self.assertEqual(
              tuple(hf_tensor.shape),
              tuple(shape_map[hf_path]),
              f"{hf_path} shape mismatch: got {hf_tensor.shape}, want {shape_map[hf_path]}",
          )


class Ling3RealCheckpointShapeCoverageTest(unittest.TestCase):
  """Validate HF_SHAPE + PARAM_MAPPING against a real bailing_moe_v3 checkpoint.

  Enabled when LING3_HF_REF_PATH points at a local HF checkpoint directory
  containing `config.json` + `model.safetensors.index.json`. Mirrors Ling2's
  `test_shape_map_matches_reference_hf_index`.
  """

  @unittest.skipUnless(
      os.getenv("LING3_HF_REF_PATH"),
      "Set LING3_HF_REF_PATH to a local Ling3 HF repo dir (containing config.json and "
      "model.safetensors.index.json) to enable this test.",
  )
  def test_shape_map_matches_reference_hf_index(self):
    ref_path = os.environ["LING3_HF_REF_PATH"]
    with open(os.path.join(ref_path, "config.json"), "r", encoding="utf-8") as f:
      real_config = json.load(f)
    with open(os.path.join(ref_path, "model.safetensors.index.json"), "r", encoding="utf-8") as f:
      real_index = json.load(f)

    declared = HF_SHAPE["ling3-tiny"](real_config)
    real_keys = set(real_index["weight_map"].keys())
    declared_keys = set(declared.keys())

    missing_in_ours = sorted(real_keys - declared_keys)
    extra_in_ours = sorted(declared_keys - real_keys)
    self.assertEqual(missing_in_ours, [], f"HF_SHAPE missing {len(missing_in_ours)} keys: {missing_in_ours[:5]}")
    self.assertEqual(extra_in_ours, [], f"HF_SHAPE has {len(extra_in_ours)} extras: {extra_in_ours[:5]}")

  @unittest.skipUnless(
      os.getenv("LING3_HF_REF_PATH"),
      "Set LING3_HF_REF_PATH to enable.",
  )
  def test_hf_tensor_shapes_match_declared(self):
    """Open each safetensors shard and verify the declared HF shape matches the file."""
    from safetensors import safe_open  # pylint: disable=import-outside-toplevel

    ref_path = os.environ["LING3_HF_REF_PATH"]
    with open(os.path.join(ref_path, "config.json"), "r", encoding="utf-8") as f:
      real_config = json.load(f)
    with open(os.path.join(ref_path, "model.safetensors.index.json"), "r", encoding="utf-8") as f:
      real_index = json.load(f)
    declared = HF_SHAPE["ling3-tiny"](real_config)

    # Group by shard to open each file only once
    by_shard = {}
    for hf_key, shard in real_index["weight_map"].items():
      by_shard.setdefault(shard, []).append(hf_key)

    mismatches = []
    for shard, hf_keys in by_shard.items():
      with safe_open(os.path.join(ref_path, shard), framework="np") as reader:
        for hf_key in hf_keys:
          real_shape = list(reader.get_slice(hf_key).get_shape())
          decl_shape = list(declared[hf_key])
          if real_shape != decl_shape:
            mismatches.append((hf_key, real_shape, decl_shape))
    self.assertEqual(
        mismatches,
        [],
        f"Shape mismatches ({len(mismatches)}): first 3: {mismatches[:3]}",
    )

  @unittest.skipUnless(
      os.getenv("LING3_HF_REF_PATH"),
      "Set LING3_HF_REF_PATH to enable.",
  )
  def test_param_mapping_covers_real_checkpoint(self):
    """Every HF key in the real ckpt is a target of LING3_MAXTEXT_TO_HF_PARAM_MAPPING."""
    ref_path = os.environ["LING3_HF_REF_PATH"]
    with open(os.path.join(ref_path, "config.json"), "r", encoding="utf-8") as f:
      real_config = json.load(f)
    with open(os.path.join(ref_path, "model.safetensors.index.json"), "r", encoding="utf-8") as f:
      real_index = json.load(f)
    # Build a maxtext_config mirror with fields the mapping function reads
    mt_config = SimpleNamespace(
        first_num_dense_layers=real_config.get("first_k_dense_replace", 1),
        num_experts=real_config.get("num_experts", 128),
        inhomogeneous_layer_cycle_interval=real_config.get("layer_group_size", 4),
        q_lora_rank=real_config.get("q_lora_rank", 256) or 0,
        mtp_num_layers=1 if real_config.get("num_nextn_predict_layers", 0) > 0 else 0,
        enable_gated_attention=True,
        scan_layers=False,
        param_scan_axis=1,
    )
    mapping = LING3_MAXTEXT_TO_HF_PARAM_MAPPING(real_config, mt_config)
    emitted = set(_iter_hf_paths(mapping))
    real_keys = set(real_index["weight_map"].keys())
    missing = sorted(real_keys - emitted)
    self.assertEqual(
        missing,
        [],
        f"{len(missing)} real HF keys not emitted by PARAM_MAPPING: {missing[:5]}",
    )


class Ling3MixedScanMTPTest(unittest.TestCase):
  """Tests for the EXPERT_AXIS_KEYS registry + scan_layers=True with MTP support.

  These tests cover the framework-side per-key axis-0 override that lets Ling3's
  mixed unscan-prefix + scan-region + MTP layout convert in scan mode without
  the prior NotImplementedError workaround.
  """

  # ---------- 5.1 不抛错 smoke ----------
  def test_scan_mtp_mapping_does_not_raise(self):
    hf_config, mt_config = _make_configs(num_nextn_predict_layers=1)
    mt_config.scan_layers = True
    LING3_MAXTEXT_TO_HF_PARAM_MAPPING(hf_config, mt_config, scan_layers=True)

  def test_scan_mtp_hook_does_not_raise(self):
    hf_config, mt_config = _make_configs(num_nextn_predict_layers=1)
    mt_config.scan_layers = True
    LING3_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_config, mt_config, scan_layers=True, saving_to_hf=True)
    LING3_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_config, mt_config, scan_layers=True, saving_to_hf=False)

  # ---------- 5.2 mtp_num_layers guard ----------
  def test_mtp_num_layers_2_raises_in_scan_mode(self):
    hf_config, mt_config = _make_configs(num_nextn_predict_layers=1)
    mt_config.mtp_num_layers = 2
    with self.assertRaises(NotImplementedError):
      LING3_MAXTEXT_TO_HF_PARAM_MAPPING(hf_config, mt_config, scan_layers=True)
    with self.assertRaises(NotImplementedError):
      LING3_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_config, mt_config, scan_layers=True)

  def test_mtp_num_layers_2_raises_in_unscan_mode(self):
    """Critical: guard must be independent of scan_layers (codex P1)."""
    hf_config, mt_config = _make_configs(num_nextn_predict_layers=1)
    mt_config.mtp_num_layers = 2
    with self.assertRaises(NotImplementedError):
      LING3_MAXTEXT_TO_HF_PARAM_MAPPING(hf_config, mt_config, scan_layers=False)
    with self.assertRaises(NotImplementedError):
      LING3_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_config, mt_config, scan_layers=False)

  def test_mtp_config_inconsistent_raises(self):
    """hf_config has MTP but maxtext_config says mtp_num_layers=0 (or vice versa) → ValueError."""
    hf_config, mt_config = _make_configs(num_nextn_predict_layers=1)
    mt_config.mtp_num_layers = 0
    with self.assertRaises(ValueError):
      LING3_MAXTEXT_TO_HF_PARAM_MAPPING(hf_config, mt_config, scan_layers=True)
    # Reverse: maxtext says MTP but HF doesn't.
    hf_config2, mt_config2 = _make_configs(num_nextn_predict_layers=0)
    mt_config2.mtp_num_layers = 1
    with self.assertRaises(ValueError):
      LING3_MAXTEXT_TO_HF_PARAM_MAPPING(hf_config2, mt_config2, scan_layers=True)

  # ---------- 5.3 Registry 注册 + 内容程序化 ----------
  def test_expert_axis_keys_registered(self):
    self.assertIn("ling3-tiny", EXPERT_AXIS_KEYS)
    # Other models fallback to empty set (codex regression guard against v2 heuristic).
    hf_config, mt_config = _make_configs(num_nextn_predict_layers=0)
    self.assertEqual(get_expert_axis_keys("mixtral-8x7b", hf_config, mt_config, True), set())
    self.assertEqual(get_expert_axis_keys("gpt-oss-20b", hf_config, mt_config, True), set())
    self.assertEqual(get_expert_axis_keys("deepseek3-671b", hf_config, mt_config, True), set())

  def test_expert_axis_keys_contents_default_ling3_tiny(self):
    hf_config, mt_config = _make_configs(num_nextn_predict_layers=1)
    self.assertEqual(LING3_EXPERT_AXIS_KEYS(hf_config, mt_config, scan_layers=False), set())

    hf_no_mtp, mt_no_mtp = _make_configs(num_nextn_predict_layers=0)
    expected_unscan_only = set()
    for moe_idx in (0, 1, 2):
      for suffix in ("wi_0", "wi_1", "wo"):
        expected_unscan_only.add(f"params-decoder-moe_layers_{moe_idx}-mlp-MoeBlock_0-{suffix}")
    self.assertEqual(LING3_EXPERT_AXIS_KEYS(hf_no_mtp, mt_no_mtp, scan_layers=True), expected_unscan_only)

    expected_with_mtp = set(expected_unscan_only)
    mtp_tf = "params-mtp_block-mtp_layer_1-mtp_1_transformer_layer"
    for suffix in ("wi_0", "wi_1", "wo"):
      expected_with_mtp.add(f"{mtp_tf}-mlp-MoeBlock_0-{suffix}")
    self.assertEqual(LING3_EXPERT_AXIS_KEYS(hf_config, mt_config, scan_layers=True), expected_with_mtp)

  def test_expert_axis_keys_scales_with_first_num_dense_layers(self):
    """Set scales programmatically with first_num_dense_layers, not hardcoded."""
    hf_config, mt_config = _make_configs(num_nextn_predict_layers=0)
    # first_num_dense_layers=0 → no unscan prefix → empty
    hf_config["first_k_dense_replace"] = 0
    mt_config.first_num_dense_layers = 0
    self.assertEqual(LING3_EXPERT_AXIS_KEYS(hf_config, mt_config, scan_layers=True), set())
    # first_num_dense_layers=5, interval=4 → unscan_prefix=8 → num_moe_prefix=3 → 9 keys
    hf_config["first_k_dense_replace"] = 5
    mt_config.first_num_dense_layers = 5
    keys = LING3_EXPERT_AXIS_KEYS(hf_config, mt_config, scan_layers=True)
    self.assertEqual(len(keys), 9)
    for moe_idx in (0, 1, 2):
      for suffix in ("wi_0", "wi_1", "wo"):
        self.assertIn(f"params-decoder-moe_layers_{moe_idx}-mlp-MoeBlock_0-{suffix}", keys)

  # ---------- 5.4 Registry vs Mapping 双向一致性(codex P2) ----------
  def test_expert_axis_keys_subset_of_mapping_keys_in_scan_mode(self):
    """Every registry entry must exist as a mapping key (防 dead key)."""
    hf_config, mt_config = _make_configs(num_nextn_predict_layers=1)
    mapping = LING3_MAXTEXT_TO_HF_PARAM_MAPPING(hf_config, mt_config, scan_layers=True)
    keys = LING3_EXPERT_AXIS_KEYS(hf_config, mt_config, scan_layers=True)
    self.assertTrue(keys, "registry should be non-empty for scan+MTP")
    for k in keys:
      self.assertIn(k, mapping, f"registry references dead key {k}")

  def test_expert_axis_keys_values_are_1d_string_lists(self):
    """Every registry entry's mapping value must be a flat list[str] (not nested 2D list)."""
    hf_config, mt_config = _make_configs(num_nextn_predict_layers=1)
    mapping = LING3_MAXTEXT_TO_HF_PARAM_MAPPING(hf_config, mt_config, scan_layers=True)
    for k in LING3_EXPERT_AXIS_KEYS(hf_config, mt_config, scan_layers=True):
      v = mapping[k]
      self.assertIsInstance(v, list, f"{k}: expected list, got {type(v)}")
      self.assertTrue(all(isinstance(x, str) for x in v), f"{k}: expected list[str], got nested list")

  def test_expert_axis_keys_equals_all_real_expert_lists(self):
    """REVERSE: every 1D list[str] in mapping with HF path matching .mlp.experts.{e}. must be in registry.

    Catches missing registry entries (e.g., if a future change adds a new expert family).
    """
    hf_config, mt_config = _make_configs(num_nextn_predict_layers=1)
    mapping = LING3_MAXTEXT_TO_HF_PARAM_MAPPING(hf_config, mt_config, scan_layers=True)
    expert_pat = re.compile(r"\.mlp\.experts\.\d+\.")
    real_expert_keys = set()
    for k, v in mapping.items():
      if not isinstance(v, list):
        continue
      if not v or not isinstance(v[0], str):
        continue
      if any(expert_pat.search(p) for p in v):
        real_expert_keys.add(k)
    registry = LING3_EXPERT_AXIS_KEYS(hf_config, mt_config, scan_layers=True)
    self.assertEqual(
        registry,
        real_expert_keys,
        f"Registry/mapping mismatch.\n  registry-only: {registry - real_expert_keys}\n"
        f"  mapping-only: {real_expert_keys - registry}",
    )

  # ---------- 5.5 底层 helper ----------
  def _fake_mt_config(self, scan_layers=True, param_scan_axis=1):
    return SimpleNamespace(scan_layers=scan_layers, param_scan_axis=param_scan_axis)

  def test_process_maxtext_param_unscan_prefix_expert_axis_0(self):
    mt_key = "params-decoder-moe_layers_0-mlp-MoeBlock_0-wi_0"
    paths = [f"model.layers.1.mlp.experts.{e}.gate_proj.weight" for e in range(4)]
    param_map = {mt_key: paths}
    hook_fn_map = {}
    hf_shape_map = {p: [8] for p in paths}
    weight = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
    out = process_maxtext_param(
        mt_key,
        weight,
        param_map,
        hook_fn_map,
        hf_shape_map,
        self._fake_mt_config(scan_layers=True),
        expert_axis_keys={mt_key},
    )
    self.assertEqual(len(out), 4)
    for i, (path, tensor) in enumerate(out):
      self.assertEqual(path, paths[i])
      self.assertEqual(tensor.shape, (8,))
      np.testing.assert_array_equal(tensor, weight[i])

  def test_process_maxtext_param_mtp_expert_axis_0(self):
    mt_key = "params-mtp_block-mtp_layer_1-mtp_1_transformer_layer-mlp-MoeBlock_0-wi_0"
    paths = [f"model.layers.24.mlp.experts.{e}.gate_proj.weight" for e in range(4)]
    param_map = {mt_key: paths}
    hf_shape_map = {p: [8] for p in paths}
    weight = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
    out = process_maxtext_param(
        mt_key,
        weight,
        param_map,
        {},
        hf_shape_map,
        self._fake_mt_config(scan_layers=True),
        expert_axis_keys={mt_key},
    )
    self.assertEqual(len(out), 4)
    for i, (_, tensor) in enumerate(out):
      self.assertEqual(tensor.shape, (8,))
      np.testing.assert_array_equal(tensor, weight[i])

  def test_process_maxtext_param_scan_region_axis_param_scan_axis(self):
    mt_key = "params-decoder-moe_layers-layers_0-attention-q_proj-kernel"
    paths = [f"model.layers.{4 + i * 4}.attention.q_proj.weight" for i in range(2)]
    param_map = {mt_key: paths}
    hf_shape_map = {p: [8, 8] for p in paths}
    # Shape (hidden=8, scan_length=2, hidden=8) along param_scan_axis=1.
    weight = np.arange(8 * 2 * 8, dtype=np.float32).reshape(8, 2, 8)
    out = process_maxtext_param(
        mt_key,
        weight,
        param_map,
        {},
        hf_shape_map,
        self._fake_mt_config(scan_layers=True, param_scan_axis=1),
        expert_axis_keys=set(),  # explicitly empty: not an expert key
    )
    self.assertEqual(len(out), 2)
    for i, (_, tensor) in enumerate(out):
      self.assertEqual(tensor.shape, (8, 8))
      np.testing.assert_array_equal(tensor, weight[:, i, :])

  @staticmethod
  def _fake_expert_getter(key):
    expert_idx = int(key.split(".experts.")[1].split(".")[0])
    return np.full((8,), expert_idx, dtype=np.float32)

  @staticmethod
  def _fake_scan_getter(_key):
    return np.zeros((8, 8), dtype=np.float32)

  def test_build_single_axis_stacked_tensor_expert_axis_0(self):
    # `to_maxtext` transitively imports torch via llama_or_mistral_ckpt;
    # skip the three to_maxtext tests when torch is unavailable.
    pytest.importorskip("torch")
    from maxtext.checkpoint_conversion.to_maxtext import (  # pylint: disable=import-outside-toplevel
        _build_single_axis_stacked_tensor,
    )

    paths = [f"model.layers.1.mlp.experts.{e}.gate_proj.weight" for e in range(4)]
    out = _build_single_axis_stacked_tensor(
        paths,
        self._fake_expert_getter,
        None,
        target_shape=(4, 8),
        config=self._fake_mt_config(scan_layers=True),
        is_expert_axis=True,
    )
    self.assertEqual(out.shape, (4, 8))
    for e in range(4):
      np.testing.assert_array_equal(out[e], np.full((8,), e))

  def test_build_single_axis_stacked_tensor_scan_axis(self):
    pytest.importorskip("torch")
    from maxtext.checkpoint_conversion.to_maxtext import (  # pylint: disable=import-outside-toplevel
        _build_single_axis_stacked_tensor,
    )

    paths = [f"model.layers.{4 + i * 4}.attention.q_proj.weight" for i in range(2)]
    out = _build_single_axis_stacked_tensor(
        paths,
        self._fake_scan_getter,
        None,
        target_shape=(8, 2, 8),
        config=self._fake_mt_config(scan_layers=True, param_scan_axis=1),
        is_expert_axis=False,
    )
    self.assertEqual(out.shape, (8, 2, 8))

  # ---------- 5.6 Wiring 中间层(codex P3) ----------
  def test_get_model_mappings_returns_expert_axis_keys(self):
    from maxtext.checkpoint_conversion.to_huggingface import (  # pylint: disable=import-outside-toplevel
        _get_model_mappings,
    )

    hf_config, mt_config = _make_configs(num_nextn_predict_layers=1)
    out = _get_model_mappings("ling3-tiny", True, hf_config, mt_config)
    self.assertIn("expert_axis_keys", out)
    self.assertIsInstance(out["expert_axis_keys"], set)
    self.assertEqual(len(out["expert_axis_keys"]), 12)

  def test_get_hf_loading_function_with_is_expert_axis_true(self):
    pytest.importorskip("torch")
    from maxtext.checkpoint_conversion.to_maxtext import (  # pylint: disable=import-outside-toplevel
        _get_hf_loading_function,
    )

    paths = [f"model.layers.1.mlp.experts.{e}.gate_proj.weight" for e in range(4)]
    load_fn = _get_hf_loading_function(
        paths,
        self._fake_expert_getter,
        None,
        (4, 8),
        self._fake_mt_config(scan_layers=True),
        is_expert_axis=True,
    )
    out = load_fn()
    self.assertEqual(out.shape, (4, 8))
    for e in range(4):
      np.testing.assert_array_equal(out[e], np.full((8,), e))

  def test_to_huggingface_main_loop_can_thread_expert_axis_keys(self):
    """Wiring sanity: the dict returned by _get_model_mappings has the key the main loop
    references, with the expected payload shape."""
    from maxtext.checkpoint_conversion.to_huggingface import (  # pylint: disable=import-outside-toplevel
        _get_model_mappings,
    )

    hf_config, mt_config = _make_configs(num_nextn_predict_layers=1)
    mappings = _get_model_mappings("ling3-tiny", True, hf_config, mt_config)
    # The to_huggingface main loop reads `mappings["expert_axis_keys"]` and forwards it
    # to process_maxtext_param. Verify the shape of that payload is what the call site expects.
    self.assertIsInstance(mappings["expert_axis_keys"], set)
    self.assertTrue(
        all(isinstance(k, str) for k in mappings["expert_axis_keys"]),
        "expert_axis_keys must be a set[str]",
    )
    self.assertIn("params-decoder-moe_layers_0-mlp-MoeBlock_0-wi_0", mappings["expert_axis_keys"])

  # ---------- 5.7 公式同步保障 ----------
  def test_unscan_prefix_matches_decoder_formula(self):
    """_ling3_unscan_prefix must match decoders.py:1259 formula for representative cases."""
    cases = [
        (0, 4, 0),
        (1, 4, 4),
        (3, 4, 4),
        (4, 4, 4),
        (5, 4, 8),
        (1, 1, 1),
        (2, 8, 8),
        (9, 8, 16),
    ]
    for first_dense, interval, expected in cases:
      with self.subTest(first_dense=first_dense, interval=interval):
        self.assertEqual(_ling3_unscan_prefix(first_dense, interval), expected)

  # ---------- 5.8 全 unscan 回归保险 ----------
  def test_existing_unscan_mode_unchanged_without_expert_axis_keys(self):
    """scan_layers=False expert MoE keys still slice axis 0 even without expert_axis_keys."""
    mt_key = "params-decoder-moe_layers_0-mlp-MoeBlock_0-wi_0"
    paths = [f"model.layers.1.mlp.experts.{e}.gate_proj.weight" for e in range(4)]
    param_map = {mt_key: paths}
    hf_shape_map = {p: [8] for p in paths}
    weight = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
    # Note: scan_layers=False, expert_axis_keys=None → falls through to else branch (axis=0).
    out = process_maxtext_param(
        mt_key,
        weight,
        param_map,
        {},
        hf_shape_map,
        self._fake_mt_config(scan_layers=False),
    )
    self.assertEqual(len(out), 4)
    for i, (_, tensor) in enumerate(out):
      self.assertEqual(tensor.shape, (8,))
      np.testing.assert_array_equal(tensor, weight[i])


if __name__ == "__main__":
  unittest.main()
