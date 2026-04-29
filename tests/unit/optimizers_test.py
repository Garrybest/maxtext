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

""" Unit tests for all optimizers. """
import re
import unittest
from unittest.mock import patch
import jax

import pytest
from absl.testing import parameterized
from optax.contrib import MuonDimensionNumbers as mdn
from optax.contrib._muon import muon

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
        "wkv_b": {"kernel": mdn((0,), (-2, -1))},
        "out": {"kernel": mdn((0, -2), (-1,))},
        "query": {"kernel": mdn((0,), (-2, -1))},  # ds2
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
        "wkv_b": {"kernel": mdn((0,), (-2, -1))},
        "out": {"kernel": mdn((0, -2), (-1,))},
        "q_norm": {"scale": None},  # ds3
        "wq_a": {"kernel": mdn((0,), (-1,))},  # ds3
        "wq_b": {"kernel": mdn((0,), (-2, -1))},  # ds3
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
        "query": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
        "key": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
        "value": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
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
                    "query": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "key": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "value": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
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
                    "query": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "key": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
                    "value": {"kernel": mdn(reduction_axis=(0,), output_axis=(-2, -1))},
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
          mdn((0,), (-2, -1)),
      ),
      (
          "kda_gate_proj",
          ("params", "decoder", "moe_layers", "layers_0", "attention", "gate_proj", "kernel"),
          mdn((0,), (-2, -1)),
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

    params = {"layer1": {"kernel": 1, "bias": 2}, "layer2": {"layer_norm": {"scale": 3}}, "layer3": {"ln": {"scale": 4}}}
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
