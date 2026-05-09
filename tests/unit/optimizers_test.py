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

"""Unit tests for all optimizers."""
import math
import re
import unittest
from unittest.mock import patch
import jax
import jax.numpy as jnp

import pytest
from absl.testing import parameterized
from third_party.optax_muon import MuonDimensionNumbers as mdn
from third_party.optax_muon import muon
from third_party.optax_muon._muon import _get_shape_products
from third_party.optax_muon._muon import _scale_update_for_consistent_rms

from maxtext.configs import pyconfig
from maxtext.optimizers import optimizers
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from maxtext.utils.muon_utils import get_model_mdn, transform_logic
from tests.utils.test_helpers import get_test_config_path
from typing import NamedTuple


# deepseek2, specific: q_lora_rank=0
# applicable: deepseek2-16, but not deepseek2-236b (q_lora_rank=1536)
_DEEPSEEK2_ATTENTION = {
    "self_attention": {
        "kv_norm": {"scale": None},
        "wkv_a": {"kernel": mdn((0,), (-1,))},
        "wkv_b": {"kernel": mdn((0,), (-1,), component_splits=(128, 128))},
        "out": {"kernel": mdn((0, -2), (-1,))},
        "query": {"kernel": mdn((0,), (-1,))},  # ds2
    },
    "post_self_attention_layer_norm": {"scale": None},
    "pre_self_attention_layer_norm": {"scale": None},
}

DEEPSEEK2_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "dense_layers": {
                "mlp": {
                    "wi_0": {"kernel": mdn((0,), (-1,))},
                    "wi_1": {"kernel": mdn((0,), (-1,))},
                    "wo": {"kernel": mdn((0,), (-1,))},
                },
                **_DEEPSEEK2_ATTENTION,
            },
            "moe_layers": {
                "DeepSeekMoeBlock_0": {
                    "MoeBlock_0": {
                        "wi_0": mdn((-2,), (-1,)),
                        "wi_1": mdn((-2,), (-1,)),
                        "wo": mdn((-2,), (-1,)),
                        "gate": {"kernel": mdn((0,), (-1,))},  # ds2
                    },
                    "shared_experts": {
                        "wi_0": {"kernel": mdn((0,), (-1,))},
                        "wi_1": {"kernel": mdn((0,), (-1,))},
                        "wo": {"kernel": mdn((0,), (-1,))},
                    },
                },
                **_DEEPSEEK2_ATTENTION,
            },
            "decoder_norm": {"scale": None},
            "logits_dense": {"kernel": None},
        },
        "token_embedder": {"embedding": None},
    }
}


# deepseek3
_DEEPSEEK3_ATTENTION = {
    "self_attention": {
        "kv_norm": {"scale": None},
        "wkv_a": {"kernel": mdn((0,), (-1,))},
        "wkv_b": {"kernel": mdn((0,), (-1,), component_splits=(128, 128))},
        "out": {"kernel": mdn((0, -2), (-1,))},
        "q_norm": {"scale": None},  # ds3
        "wq_a": {"kernel": mdn((0,), (-1,))},  # ds3
        "wq_b": {"kernel": mdn((0,), (-1,), component_splits=(128, 64))},  # ds3
    },
    "post_self_attention_layer_norm": {"scale": None},
    "pre_self_attention_layer_norm": {"scale": None},
}

DEEPSEEK3_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "dense_layers": {
                "mlp": {
                    "wi_0": {"kernel": mdn((0,), (-1,))},
                    "wi_1": {"kernel": mdn((0,), (-1,))},
                    "wo": {"kernel": mdn((0,), (-1,))},
                },
                **_DEEPSEEK3_ATTENTION,
            },
            "moe_layers": {
                "DeepSeekMoeBlock_0": {
                    "MoeBlock_0": {
                        "wi_0": mdn((-2,), (-1,)),
                        "wi_1": mdn((-2,), (-1,)),
                        "wo": mdn((-2,), (-1,)),
                        "gate": {"kernel": mdn((0,), (-1,)), "bias": None},  # ds3
                    },
                    "shared_experts": {
                        "wi_0": {"kernel": mdn((0,), (-1,))},
                        "wi_1": {"kernel": mdn((0,), (-1,))},
                        "wo": {"kernel": mdn((0,), (-1,))},
                    },
                },
                **_DEEPSEEK3_ATTENTION,
            },
            "decoder_norm": {"scale": None},
            "logits_dense": {"kernel": None},
        },
        "token_embedder": {"embedding": None},
    }
}

# gemma3
_GEMMA3_LAYER = {
    "mlp": {
        "wi_0": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "wi_1": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "wo": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
    },
    "post_ffw_norm": {"scale": None},
    "pre_ffw_norm": {"scale": None},
    "self_attention": {
        "query": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "key": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "value": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
        "out": {"kernel": mdn(reduction_axis=(0, -2), output_axis=(-1,))},
        "key_norm": {"scale": None},
        "query_norm": {"scale": None},
    },
    "post_self_attention_norm": {"scale": None},
    "pre_self_attention_norm": {"scale": None},
}

GEMMA3_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "decoder_norm": {"scale": None},
            "layers": {f"layers_{i}": _GEMMA3_LAYER for i in range(6)},
            "layers_remainder": {f"layers_{i}": _GEMMA3_LAYER for i in range(4)},
        },
        "token_embedder": {"embedding": None},
    }
}


# llama2 (also llama3)
LLAMA2_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "decoder_norm": {"scale": None},
            "layers": {
                "mlp": {
                    "wi_0": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                    "wi_1": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                    "wo": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                },
                "self_attention": {
                    "query": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                    "key": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                    "value": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                    "out": {"kernel": mdn(reduction_axis=(0, -2), output_axis=(-1,))},
                },
                "post_self_attention_layer_norm": {"scale": None},
                "pre_self_attention_layer_norm": {"scale": None},
            },
            "logits_dense": {"kernel": None},
        },
        "token_embedder": {"embedding": None},
    }
}


# qwen3, specific: logits_via_embedding=True
# applicable: qwen3-0.6b, qwen3-4b, but not: qwen3-8b, qwen3-14b (logits_via_embedding=False)
QWEN3_DIMENSION_NUMBER = {
    "params": {
        "decoder": {
            "decoder_norm": {"scale": None},
            "layers": {
                "mlp": {
                    "wi_0": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                    "wi_1": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                    "wo": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                },
                "self_attention": {
                    "query": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                    "key": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                    "value": {"kernel": mdn(reduction_axis=(0,), output_axis=(-1,))},
                    "out": {"kernel": mdn(reduction_axis=(0, -2), output_axis=(-1,))},
                    "key_norm": {"scale": None},
                    "query_norm": {"scale": None},
                },
                "post_self_attention_layer_norm": {"scale": None},
                "pre_self_attention_layer_norm": {"scale": None},
            },
        },
        "token_embedder": {"embedding": None},
    }
}


class MuonDimensionTest(parameterized.TestCase):
  """Unit tests for Muon dimension number generation.

  This suite verifies that the automatically generated Muon dimension numbers
  for various models match their hardcoded reference values.
  """

  @parameterized.named_parameters(
      ("deepseek2-16b", "deepseek2-16b", DEEPSEEK2_DIMENSION_NUMBER),
      ("deepseek3-671b", "deepseek3-671b", DEEPSEEK3_DIMENSION_NUMBER),
      ("kimi-k2-1t", "kimi-k2-1t", DEEPSEEK3_DIMENSION_NUMBER),
      ("llama2-7b", "llama2-7b", LLAMA2_DIMENSION_NUMBER),
      ("llama3-8b", "llama3-8b", LLAMA2_DIMENSION_NUMBER),
      ("llama3.1-8b", "llama3.1-8b", LLAMA2_DIMENSION_NUMBER),
      ("llama3.3-70b", "llama3.3-70b", LLAMA2_DIMENSION_NUMBER),
      ("gemma3-4b", "gemma3-4b", GEMMA3_DIMENSION_NUMBER),
      ("qwen3-0.6b", "qwen3-0.6b", QWEN3_DIMENSION_NUMBER),
  )
  @pytest.mark.tpu_only
  def test_model_integration(self, model_name, expected_output):
    """
    Initializes the specified MaxText model and asserts that the generated
    Muon dimension numbers match the hardcoded reference.
    """
    actual_output = get_model_mdn(model_name, scan_layers=True)
    self.assertEqual(actual_output, expected_output)


class MuonTransformLogicTest(parameterized.TestCase):
  """Tests Ling3-specific Muon dimension-number routing."""

  @parameterized.named_parameters(
      (
          "kda_q_proj",
          ("params", "decoder", "moe_layers", "layers_0", "attention", "q_proj", "kernel"),
          mdn((0,), (-1,)),
      ),
      (
          "kda_gate_proj",
          ("params", "decoder", "moe_layers", "layers_0", "attention", "gate_proj", "kernel"),
          mdn((0,), (-1,)),
      ),
      (
          "kda_o_proj",
          ("params", "decoder", "moe_layers", "layers_0", "attention", "o_proj", "kernel"),
          mdn((0, -2), (-1,)),
      ),
      (
          "kda_short_conv_kernel",
          ("params", "decoder", "moe_layers", "layers_0", "attention", "q_conv", "kernel"),
          None,
      ),
      (
          "kda_a_log",
          ("params", "decoder", "moe_layers", "layers_0", "attention", "A_log"),
          None,
      ),
      (
          "kda_dt_bias",
          ("params", "decoder", "moe_layers", "layers_0", "attention", "dt_bias"),
          None,
      ),
  )
  def test_transform_logic_for_ling3_kda(self, path, expected):
    self.assertEqual(transform_logic(path), expected)

  def test_ling3_muon_config_is_allowed(self):
    argv = ["", get_test_config_path(), "run_name=test", "model_name=ling3-tiny", "opt_type=muon"]
    config = pyconfig.initialize(argv)
    self.assertEqual(config.opt_type.value, "muon")


class MuonSplitHeadFanTest(parameterized.TestCase):
  """Verify fan_in/fan_out match Megatron split_head=True per-head semantics."""

  @staticmethod
  def _fan(shape, dim_nums):
    """Compute fan_in/fan_out from shape and MuonDimensionNumbers."""
    reduction_axes = tuple(ax % len(shape) for ax in dim_nums.reduction_axis)
    output_axes = tuple(ax % len(shape) for ax in dim_nums.output_axis)
    fan_in = math.prod(shape[ax] for ax in reduction_axes)
    fan_out = math.prod(shape[ax] for ax in output_axes)
    return fan_in, fan_out

  @parameterized.named_parameters(
      ("query_4096x32x128", (4096, 32, 128), "query", 4096, 128),
      ("key_4096x8x128", (4096, 8, 128), "key", 4096, 128),
      ("value_4096x8x128", (4096, 8, 128), "value", 4096, 128),
      ("out_32x128x4096", (32, 128, 4096), "out", 4096, 4096),
      ("wkv_b_512x8x128", (512, 8, 128), "wkv_b", 512, 128),
      # gate_proj: same split_head semantics as Q/K/V
      ("gate_proj_4096x32x128", (4096, 32, 128), "gate_proj", 4096, 128),
  )
  def test_split_head_fan_values(self, shape, param_name, expected_fan_in, expected_fan_out):
    """Per-head fan: fan_out=head_dim for Q/K/V, fan_in=head_dim for out."""
    path = ("params", "decoder", "self_attention", param_name, "kernel")
    dim_nums = transform_logic(path)
    fan_in, fan_out = self._fan(shape, dim_nums)
    self.assertEqual(fan_in, expected_fan_in, f"{param_name} fan_in mismatch: got {fan_in}, expected {expected_fan_in}")
    self.assertEqual(
        fan_out, expected_fan_out, f"{param_name} fan_out mismatch: got {fan_out}, expected {expected_fan_out}"
    )

  @parameterized.named_parameters(
      # Attention: heads are batch dim → fan excludes heads
      ("query_4096x32x128", (4096, 32, 128), "query", 4096, 128),
      ("key_4096x8x128", (4096, 8, 128), "key", 4096, 128),
      ("value_4096x8x128", (4096, 8, 128), "value", 4096, 128),
      ("out_32x128x4096", (32, 128, 4096), "out", 4096, 4096),
      # 2D standard linear: no batch dim
      ("mlp_wi_4096x11008", (4096, 11008), "wi", 4096, 11008),
  )
  def test_fork_get_shape_products(self, shape, param_name, expected_fan_in, expected_fan_out):
    """Verify the actual fork _get_shape_products matches Megatron expected fan values."""
    path = ("params", "decoder", "self_attention", param_name, "kernel")
    dim_nums = transform_logic(path)
    x = jnp.zeros(shape)
    fan_in, fan_out = _get_shape_products(x, dim_nums)
    self.assertEqual(fan_in, expected_fan_in, f"{param_name} fork fan_in: got {fan_in}, expected {expected_fan_in}")
    self.assertEqual(fan_out, expected_fan_out, f"{param_name} fork fan_out: got {fan_out}, expected {expected_fan_out}")

  @parameterized.named_parameters(
      # MoE wi_0: (num_experts, hidden, ffn_dim) → batch=experts, fan_in=hidden, fan_out=ffn_dim
      ("moe_wi0_8x4096x11008", (8, 4096, 11008), "wi_0", 4096, 11008),
      ("moe_wi1_8x4096x11008", (8, 4096, 11008), "wi_1", 4096, 11008),
      ("moe_wo_8x11008x4096", (8, 11008, 4096), "wo", 11008, 4096),
  )
  def test_moe_fan_values(self, shape, param_name, expected_fan_in, expected_fan_out):
    """MoE weights: expert dim is batch, fan computed from per-expert 2D shape."""
    path = ("params", "decoder", "MoeBlock_0", param_name, "kernel")
    dim_nums = transform_logic(path)
    x = jnp.zeros(shape)
    fan_in, fan_out = _get_shape_products(x, dim_nums)
    self.assertEqual(fan_in, expected_fan_in, f"MoE {param_name} fan_in: got {fan_in}, expected {expected_fan_in}")
    self.assertEqual(fan_out, expected_fan_out, f"MoE {param_name} fan_out: got {fan_out}, expected {expected_fan_out}")


class MuonConsistentRmsScalingTest(parameterized.TestCase):
  """Verify consistent_rms scaling factor matches Megatron adjust_lr_wd_for_muon.

  Megatron formula: scale = sqrt(max(fan_in, fan_out)) * consistent_rms
  where fan_in/fan_out are computed from per-head shape when split_head=True.
  """

  @parameterized.named_parameters(
      # Query (4096, 32, 128): per-head fan_in=4096, fan_out=128
      # scale = sqrt(max(4096, 128)) * 0.2 = sqrt(4096) * 0.2 = 64 * 0.2 = 12.8
      ("query_crms02", (4096, 32, 128), "query", 0.2, math.sqrt(4096) * 0.2),
      # Out (32, 128, 4096): mdn((0,-2),(-1,)) → fan_in=32*128=4096, fan_out=4096
      # scale = sqrt(max(4096, 4096)) * 0.2 = sqrt(4096) * 0.2 = 12.8
      ("out_crms02", (32, 128, 4096), "out", 0.2, math.sqrt(4096) * 0.2),
      # 2D MLP (4096, 11008): fan_in=4096, fan_out=11008
      # scale = sqrt(max(4096, 11008)) * 0.2 = sqrt(11008) * 0.2
      ("mlp_2d_crms02", (4096, 11008), "wi", 0.2, math.sqrt(11008) * 0.2),
      # Different consistent_rms value
      ("query_crms01", (4096, 32, 128), "query", 0.1, math.sqrt(4096) * 0.1),
  )
  def test_consistent_rms_scaling_factor(self, shape, param_name, crms, expected_scale):
    """Verify _scale_update_for_consistent_rms produces correct scaling factor."""
    path = ("params", "decoder", "self_attention", param_name, "kernel")
    dim_nums = transform_logic(path)
    # Use ones so the output directly shows the scale factor
    x = jnp.ones(shape)
    scaled = _scale_update_for_consistent_rms(x, dim_nums, crms)
    # Every element should be multiplied by expected_scale
    actual_scale = float(scaled.ravel()[0])
    self.assertAlmostEqual(
        actual_scale,
        expected_scale,
        places=5,
        msg=f"{param_name} scale: got {actual_scale}, expected {expected_scale}",
    )


class _DummyState:
  """Minimal stand-in for the sharding helper's state object."""

  def __init__(self, params, opt_state):
    self.params = params
    self.opt_state = opt_state

  def replace(self, **kwargs):
    return _DummyState(
        params=kwargs.get("params", self.params),
        opt_state=kwargs.get("opt_state", self.opt_state),
    )


class MuonOptimizerShardingTest(unittest.TestCase):
  """Tests optimizer-state sharding helpers with Muon."""

  def test_maybe_update_params_sharding_with_muon_partition_state(self):
    params = {"a": jax.numpy.ones((2, 3)), "b": jax.numpy.ones((3,))}
    tx = muon(
        learning_rate=0.1,
        muon_weight_dimension_numbers={"a": mdn((0,), (1,)), "b": None},
    )
    opt_state = tx.init(params)
    state_mesh_shardings = _DummyState(
        params={"params": {"a": "orig_a", "b": "orig_b"}, "other_collection": {"c": "keep_me"}},
        opt_state=opt_state,
    )
    config = type("Config", (), {"shard_optimizer_over_data": True})()

    prev_params_shardings, updated_state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(
        config, state_mesh_shardings
    )

    self.assertEqual(prev_params_shardings, state_mesh_shardings.params)
    self.assertEqual(updated_state_mesh_shardings.params["other_collection"], {"c": "keep_me"})
    self.assertTrue(jax.numpy.array_equal(updated_state_mesh_shardings.params["params"]["a"], params["a"] * 0))
    self.assertTrue(jax.numpy.array_equal(updated_state_mesh_shardings.params["params"]["b"], params["b"] * 0))


class AdamWMaskTest(parameterized.TestCase):
  """Tests for the AdamW mask functionality"""

  def test_get_adamw_mask_with_empty_mask(self):
    """Directly test the get_adamw_mask function with empty list"""
    # Case 1: No mask in config (empty list)
    argv = ["", get_test_config_path(), "run_name=test", "adamw_mask=[]"]
    config = pyconfig.initialize(argv)
    mask_fn = optimizers.get_adamw_mask(config)
    self.assertIsNone(mask_fn)

  def test_get_adamw_mask_with_valid_mask(self):
    """Directly test the get_adamw_mask function with valid mask"""
    # Case 2: Mask in config
    argv = ["", get_test_config_path(), "run_name=test", "adamw_mask=['bias', '.*norm', '.*ln.*']"]
    config = pyconfig.initialize(argv)
    mask_fn = optimizers.get_adamw_mask(config)
    self.assertTrue(callable(mask_fn))

    params = {
        "layer1": {"kernel": 1, "bias": 2},
        "layer2": {"layer_norm": {"scale": 3}},
        "layer3": {"ln": {"scale": 4}},
    }
    mask = mask_fn(params)
    self.assertTrue(mask["layer1"]["kernel"])
    self.assertFalse(mask["layer1"]["bias"])
    self.assertFalse(mask["layer2"]["layer_norm"]["scale"])
    self.assertFalse(mask["layer3"]["ln"]["scale"])

  def test_get_adamw_mask_with_invalid_mask(self):
    """Test that an invalid regex in the mask config raises an error when used"""
    # Create a config with an invalid regex (unbalanced bracket)
    argv = ["", get_test_config_path(), "run_name=test", "adamw_mask=['[']"]
    config = pyconfig.initialize(argv)

    # Applying the mask should raise re.error due to the invalid regex
    with self.assertRaises(re.error):
      optimizers.get_adamw_mask(config)

  def test_get_adamw_mask_with_getattrkey(self):
    """Test that get_adamw_mask correctly handles GetAttrKey (e.g. from NamedTuples)"""

    class MyParams(NamedTuple):
      kernel: jax.Array
      bias: jax.Array

    argv = ["", get_test_config_path(), "run_name=test", "adamw_mask=['bias']"]
    config = pyconfig.initialize(argv)
    mask_fn = optimizers.get_adamw_mask(config)

    params = MyParams(kernel=jax.numpy.ones((2, 2)), bias=jax.numpy.zeros((2,)))
    mask = mask_fn(params)

    self.assertTrue(mask.kernel)
    self.assertFalse(mask.bias)

  @parameterized.named_parameters(
      ("adamw", "adamw", "maxtext.optimizers.optimizers.optax.adamw"),
      ("adam_pax", "adam_pax", "maxtext.optimizers.optimizers.adam_pax"),
  )
  def test_optimizer_with_mask(self, opt_type, mock_path):
    """Test that optimizer receives the mask function from config and it works as expected"""
    # Create a config with a mask list including regex
    argv = [
        "",
        get_test_config_path(),
        "run_name=test",
        "adamw_mask=['bias', 'layer_norm', 'layer1/.*kernel']",
        f"opt_type={opt_type}",
    ]
    config = pyconfig.initialize(argv)
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)

    with patch(mock_path) as mock_opt:
      # Call get_optimizer
      optimizers.get_optimizer(config, learning_rate_schedule)

      # Check that optimizer was called with a mask function
      mock_opt.assert_called_once()
      _, kwargs = mock_opt.call_args
      mask_fn = kwargs["mask"]

      # Verify that mask_fn is not None
      self.assertIsNotNone(mask_fn)

      # Test the behavior of mask_fn
      params = {"layer1": {"kernel": 1, "bias": 2}, "layer2": {"layer_norm": {"scale": 3}}, "layer3": [4, 5]}

      mask = mask_fn(params)

      # kernel in layer1 should be False because of 'layer1/.*kernel'
      self.assertFalse(mask["layer1"]["kernel"])
      # bias in layer1 should be False because of 'bias'
      self.assertFalse(mask["layer1"]["bias"])
      # layer_norm should be False because of 'layer_norm'
      self.assertFalse(mask["layer2"]["layer_norm"]["scale"])
      # layer3 elements should be True
      self.assertTrue(mask["layer3"][0])
      self.assertTrue(mask["layer3"][1])

  @parameterized.named_parameters(
      ("adamw", "adamw", "maxtext.optimizers.optimizers.optax.adamw"),
      ("adam_pax", "adam_pax", "maxtext.optimizers.optimizers.adam_pax"),
  )
  def test_optimizer_without_mask(self, opt_type, mock_path):
    """Test that optimizer receives None for mask when config is empty"""
    argv = ["", get_test_config_path(), "run_name=test", f"opt_type={opt_type}"]
    config = pyconfig.initialize(argv)
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)

    with patch(mock_path) as mock_opt:
      # Call get_optimizer
      optimizers.get_optimizer(config, learning_rate_schedule)

      # Check that optimizer was called with mask=None
      mock_opt.assert_called_once()
      _, kwargs = mock_opt.call_args
      self.assertIsNone(kwargs["mask"])


class MuonNesterovStyleTest(unittest.TestCase):
  """Verify that optimizers.py wires nesterov_style='sgd' for Megatron parity."""

  def test_muon_optimizer_passes_sgd_nesterov_style(self):
    """get_optimizer(opt_type='muon') must pass nesterov_style='sgd' to muon()."""
    argv = ["", get_test_config_path(), "run_name=test", "opt_type=muon"]
    config = pyconfig.initialize(argv)
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)

    with (
        patch("maxtext.optimizers.optimizers.muon") as mock_muon,
        patch("maxtext.optimizers.optimizers.get_muon_weight_dimension_numbers") as mock_mdn,
    ):
      mock_mdn.return_value = mdn()
      mock_model = type("Model", (), {})()
      optimizers.get_optimizer(config, learning_rate_schedule, model=mock_model)

      mock_muon.assert_called_once()
      _, kwargs = mock_muon.call_args
      self.assertEqual(
          kwargs.get("nesterov_style"),
          "sgd",
          "muon() must be called with nesterov_style='sgd' for Megatron parity",
      )


class MuonAdamWeightDecayMaskTest(unittest.TestCase):
  """Tests for get_adam_wd_mask: per-param weight decay masking in Adam partition."""

  def _make_config(self, norm_params=False):
    return type("Config", (), {"muon_weight_decay_norm_params": norm_params})()

  def test_bias_always_excluded(self):
    """bias params must always be excluded from weight decay."""
    mask_fn = optimizers.get_adam_wd_mask(self._make_config(norm_params=False))
    params = {"layer": {"kernel": jax.numpy.ones((2, 2)), "bias": jax.numpy.ones((2,))}}
    mask = mask_fn(params)
    self.assertTrue(mask["layer"]["kernel"])
    self.assertFalse(mask["layer"]["bias"])

  def test_scale_excluded_when_norm_params_false(self):
    """scale (norm) params excluded when muon_weight_decay_norm_params=False."""
    mask_fn = optimizers.get_adam_wd_mask(self._make_config(norm_params=False))
    params = {"norm": {"scale": jax.numpy.ones((4,))}, "other": jax.numpy.ones((4,))}
    mask = mask_fn(params)
    self.assertFalse(mask["norm"]["scale"])
    self.assertTrue(mask["other"])

  def test_scale_included_when_norm_params_true(self):
    """scale (norm) params included when muon_weight_decay_norm_params=True."""
    mask_fn = optimizers.get_adam_wd_mask(self._make_config(norm_params=True))
    params = {"norm": {"scale": jax.numpy.ones((4,))}}
    mask = mask_fn(params)
    self.assertTrue(mask["norm"]["scale"])

  def test_embedding_and_logits_always_decayed(self):
    """embedding and logits_dense should always have weight decay (wd_mult=1.0)."""
    mask_fn = optimizers.get_adam_wd_mask(self._make_config(norm_params=False))
    params = {
        "token_embedder": {"embedding": jax.numpy.ones((100, 8))},
        "decoder": {"logits_dense": {"kernel": jax.numpy.ones((8, 100))}},
    }
    mask = mask_fn(params)
    self.assertTrue(mask["token_embedder"]["embedding"])
    self.assertTrue(mask["decoder"]["logits_dense"]["kernel"])

  def test_get_optimizer_passes_adam_weight_decay_mask(self):
    """get_optimizer(opt_type='muon') must pass adam_weight_decay_mask to muon()."""
    argv = [
        "",
        get_test_config_path(),
        "run_name=test",
        "opt_type=muon",
        "model_name=llama2-7b",
        "skip_jax_distributed_system=True",
    ]
    config = pyconfig.initialize(argv)
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)

    with (
        patch("maxtext.optimizers.optimizers.muon") as mock_muon,
        patch("maxtext.optimizers.optimizers.get_muon_weight_dimension_numbers") as mock_mdn,
    ):
      mock_mdn.return_value = mdn()
      mock_model = type("Model", (), {})()
      optimizers.get_optimizer(config, learning_rate_schedule, model=mock_model)

      mock_muon.assert_called_once()
      _, kwargs = mock_muon.call_args
      self.assertIn("adam_weight_decay_mask", kwargs, "muon() must be called with adam_weight_decay_mask")
      self.assertTrue(callable(kwargs["adam_weight_decay_mask"]), "adam_weight_decay_mask must be a callable")


class MuonWeightDecayNormParamsConfigTest(unittest.TestCase):
  """Verify muon_weight_decay_norm_params config field exists and defaults to False."""

  def test_muon_weight_decay_norm_params_defaults_false(self):
    """muon_weight_decay_norm_params should exist and default to False."""
    argv = ["", get_test_config_path(), "run_name=test", "opt_type=muon"]
    config = pyconfig.initialize(argv)
    self.assertFalse(config.muon_weight_decay_norm_params)

  def test_muon_weight_decay_norm_params_can_be_set_true(self):
    """muon_weight_decay_norm_params should be settable to True."""
    argv = ["", get_test_config_path(), "run_name=test", "opt_type=muon", "muon_weight_decay_norm_params=True"]
    config = pyconfig.initialize(argv)
    self.assertTrue(config.muon_weight_decay_norm_params)


class TrainableParametersMaskTest(parameterized.TestCase):
  """Tests for the trainable parameters mask functionality via get_optimizer"""

  def test_get_optimizer_with_trainable_mask(self):
    """Test get_optimizer with a valid trainable_parameters_mask."""
    argv = [
        "",
        get_test_config_path(),
        "run_name=test_with_trainable_mask",
        "trainable_parameters_mask=['.*indexer.*', 'layer_norm']",
    ]
    config = pyconfig.initialize(argv)

    # Use a constant learning rate > 0 to ensure non-zero updates
    def learning_rate_schedule(step):
      return 1.0

    opt = optimizers.get_optimizer(config, learning_rate_schedule)

    # We can test the optimizer by creating some dummy params and gradients
    # and checking if the updates are zeroed out for non-trainable parameters.
    params = {
        "layer1": {"kernel": jax.numpy.ones((2, 2)), "indexer": jax.numpy.ones((2, 2))},
        "layer2": {"layer_norm": {"scale": jax.numpy.ones((2, 2))}},
        "layer3": {"ln": {"scale": jax.numpy.ones((2, 2))}},
    }

    # Give some non-zero gradients
    grads = jax.tree_util.tree_map(lambda x: jax.numpy.ones_like(x) * 0.5, params)

    # Initialize optimizer state
    opt_state = opt.init(params)

    # Compute updates
    updates, _ = opt.update(grads, opt_state, params)

    # 'layer1/kernel' doesn't match the trainable mask, so it should be frozen (update == 0)
    self.assertTrue(jax.numpy.all(updates["layer1"]["kernel"] == 0))
    # 'layer3/ln/scale' doesn't match the trainable mask, so it should be frozen (update == 0)
    self.assertTrue(jax.numpy.all(updates["layer3"]["ln"]["scale"] == 0))
    # 'layer1/indexer' matches, so it should be trained (update != 0)
    self.assertFalse(jax.numpy.all(updates["layer1"]["indexer"] == 0))
    # 'layer2/layer_norm/scale' matches, so it should be trained (update != 0)
    self.assertFalse(jax.numpy.all(updates["layer2"]["layer_norm"]["scale"] == 0))

  def test_get_optimizer_without_trainable_mask(self):
    """Test get_optimizer when trainable_parameters_mask is empty."""
    argv = ["", get_test_config_path(), "run_name=test", "trainable_parameters_mask=[]"]
    config = pyconfig.initialize(argv)

    # Use a constant learning rate > 0 to ensure non-zero updates
    def learning_rate_schedule(step):
      return 1.0

    opt = optimizers.get_optimizer(config, learning_rate_schedule)

    params = {"layer1": {"kernel": jax.numpy.ones((2, 2))}}
    grads = {"layer1": {"kernel": jax.numpy.ones((2, 2)) * 0.5}}

    opt_state = opt.init(params)
    updates, _ = opt.update(grads, opt_state, params)

    # When no trainable mask is provided, nothing is frozen by this mechanism
    self.assertFalse(jax.numpy.all(updates["layer1"]["kernel"] == 0))


if __name__ == "__main__":
  unittest.main()
