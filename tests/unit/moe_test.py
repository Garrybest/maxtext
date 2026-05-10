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
"""Mixture of Experts (MoE) tests."""

import unittest

from flax import nnx
import flax.linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.common.common_types import Config, DType
from maxtext.layers import linears
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.layers.initializers import NdInitializer, nd_dense_init, variable_to_logically_partitioned
from maxtext.layers.quantizations import Fp8Quantization
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path, get_decoupled_parallelism_overrides
import pytest


class TokenDroppingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    extra_args = get_decoupled_parallelism_overrides()
    self.cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="token_dropping_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=False,
        sparse_matmul=False,
        max_target_length=80,
        per_device_batch_size=1,
        capacity_factor=2,
        **extra_args,
    )
    self.rngs = nnx.Rngs(params=0)
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.model = moe.RoutedMoE(
        config=self.cfg,
        num_experts=self.cfg.num_experts,
        num_experts_per_tok=self.cfg.num_experts_per_tok,
        mesh=Mesh(devices_array, self.cfg.mesh_axes),
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=self.cfg.dtype,
        rngs=self.rngs,
    )

  def test_generate_masks(self):
    # expert_capacity = (tokens_per_batch / num_experts) * capacity_factor
    # expert_capacity_in_batch = (4 * 2 / 8) * 2 = 2
    top_k_indices = jnp.array(
        [
            [[0, 5], [0, 4], [1, 0], [3, 5]],
            [[1, 2], [4, 1], [5, 0], [7, 1]],
            [[6, 2], [2, 3], [4, 2], [1, 2]],
            [[4, 1], [0, 7], [5, 0], [4, 7]],
        ]
    )
    softmax_probs = jnp.array(
        [
            [
                [0.20, 0, 0, 0, 0, 0.80, 0, 0],
                [0.68, 0, 0, 0, 0.32, 0, 0, 0],
                [0.22, 0.78, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0.32, 0, 0.68, 0, 0],
            ],
            [
                [0, 0.26, 0.74, 0, 0, 0, 0, 0],
                [0, 0.79, 0, 0, 0.21, 0, 0, 0],
                [0.89, 0, 0, 0, 0, 0.11, 0, 0],
                [0, 0.11, 0, 0, 0, 0, 0, 0.89],
            ],
            [
                [0, 0, 0.26, 0, 0, 0, 0.74, 0],
                [0, 0, 0.88, 0.12, 0, 0, 0, 0],
                [0, 0, 0.17, 0, 0.83, 0, 0, 0],
                [0, 0.35, 0.65, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0.47, 0, 0, 0.53, 0, 0, 0],
                [0.36, 0, 0, 0, 0, 0, 0, 0.64],
                [0.15, 0, 0, 0, 0, 0.85, 0, 0],
                [0, 0, 0, 0, 0.18, 0, 0, 0.82],
            ],
        ]
    )

    # As expert_capacity_in_batch=2, so updated softmax_probs become (4 tokens were dropped):
    # softmax_probs = jnp.array([[[0.20, 0, 0, 0, 0, 0.80, 0, 0],
    #                             [0.68, 0, 0, 0, 0.32, 0, 0, 0],
    #                             [0, 0.78, 0, 0, 0, 0, 0, 0],
    #                             [0, 0, 0, 0.32, 0, 0.68, 0, 0]],
    #                            [[0, 0.26, 0.74, 0, 0, 0, 0, 0],
    #                             [0, 0.79, 0, 0, 0.21, 0, 0, 0],
    #                             [0.89, 0, 0, 0, 0, 0.11, 0, 0],
    #                             [0, 0, 0, 0, 0, 0, 0, 0.89]],
    #                            [[0, 0, 0.26, 0, 0, 0, 0.74, 0],
    #                             [0, 0, 0.88, 0.12, 0, 0, 0, 0],
    #                             [0, 0, 0, 0, 0.83, 0, 0, 0],
    #                             [0, 0.35, 0, 0, 0, 0, 0, 0]],
    #                            [[0, 0.47, 0, 0, 0.53, 0, 0, 0],
    #                             [0.36, 0, 0, 0, 0, 0, 0, 0.64],
    #                             [0.15, 0, 0, 0, 0, 0.85, 0, 0],
    #                             [0, 0, 0, 0, 0.18, 0, 0, 0.82]]])

    # shape of dispatch_mask & combine_mask: (batch_size, seq_len, num_experts, expert_capacity_per_batch)
    expected_combine_mask = jnp.array(
        [
            [
                [[0.2, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0.8, 0], [0, 0], [0, 0]],
                [[0, 0.68], [0, 0], [0, 0], [0, 0], [0.32, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0.78, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0.32, 0], [0, 0], [0, 0.68], [0, 0], [0, 0]],
            ],
            [
                [[0, 0], [0.26, 0], [0.74, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0.79], [0, 0], [0, 0], [0.21, 0], [0, 0], [0, 0], [0, 0]],
                [[0.89, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0.11, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0.89, 0]],
            ],
            [
                [[0, 0], [0, 0], [0.26, 0], [0, 0], [0, 0], [0, 0], [0.74, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0.88], [0.12, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0, 0], [0.83, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0.35, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            ],
            [
                [[0, 0], [0.47, 0], [0, 0], [0, 0], [0.53, 0], [0, 0], [0, 0], [0, 0]],
                [[0.36, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0.64, 0]],
                [[0, 0.15], [0, 0], [0, 0], [0, 0], [0, 0], [0.85, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0.18], [0, 0], [0, 0], [0, 0.82]],
            ],
        ],
        dtype=jnp.float32,
    )
    expected_dispatch_mask = expected_combine_mask.astype(bool)
    actual_dispatch_mask, actual_combine_mask = self.model.generate_masks(top_k_indices, softmax_probs)

    self.assertTrue((expected_dispatch_mask == actual_dispatch_mask).all())
    self.assertTrue(jax.numpy.allclose(expected_combine_mask, actual_combine_mask, rtol=1e-02, atol=1e-02))


class MlpBlockTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.config = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="mlp_block_init_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=False,
        sparse_matmul=False,
        max_target_length=80,
        per_device_batch_size=1,
        capacity_factor=2,
    )
    self.rng = jax.random.PRNGKey(42)
    quant = Fp8Quantization()
    devices_array = maxtext_utils.create_device_mesh(self.config)
    self.model = linears.mlp_block(
        mesh=Mesh(devices_array, self.config.mesh_axes),
        config=self.config,
        in_features=2,
        intermediate_dim=2,
        activations=["silu", "linear"],
        intermediate_dropout_rate=0.0,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.bfloat16,
        name="mlp",
        quant=quant,
        use_bias=True,
    )

  @pytest.mark.external_serving
  def test_init(self):
    x = jnp.array([1.0, 2.0]).reshape((1, 1, 2))  # TODO(bug): need reshape due to error
    self.model.init({"params": self.rng, "dropout": self.rng}, x)


class DeepSeekRoutingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # Conditionally set ici_fsdp_parallelism to match device count in decoupled mode
    extra_args = get_decoupled_parallelism_overrides()
    self.cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="deepseek_routing_test",
        enable_checkpointing=False,
        decoder_block="deepseek",
        dtype="bfloat16",
        max_target_length=2,
        max_prefill_predict_length=1,
        per_device_batch_size=1,
        n_routing_groups=4,
        topk_routing_group=2,
        num_experts=16,
        num_experts_per_tok=4,
        sparse_matmul=True,
        **extra_args,
    )
    self.rngs = nnx.Rngs(params=0)
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.model = moe.RoutedMoE(
        config=self.cfg,
        num_experts=self.cfg.num_experts,
        num_experts_per_tok=self.cfg.num_experts_per_tok,
        mesh=Mesh(devices_array, self.cfg.mesh_axes),
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=self.cfg.dtype,
        rngs=self.rngs,
    )

  def test_deepseek_routing(self):
    # shape as [batch, sequence, num_experts] = [1,2,16]
    gate_logits = jnp.array(
        [
            [
                [0.20, 0.10, 0.05, 0.10, 0.10, 0.60, 0.30, 0.10, 0.80, 0.01, 0.01, 0.01, 0.05, 0.80, 0.20, 0.10],
                [0.68, 0.20, 0.06, 0.03, 0.32, 0.10, 0.05, 0.02, 0.65, 0.20, 0.04, 0.01, 0.32, 0.10, 0.05, 0.02],
            ]
        ]
    )
    pre_bias_logits = gate_logits - 0.5

    # 4 groups of 1st token:
    #  [0.20, 0.10, 0.05, 0.10] - sum top2 = 0.7
    #  [0.10, 0.60, 0.30, 0.10] - sum top2 = 0.9 (selected group) - index from 4 to 7
    #  [0.80, 0.01, 0.01, 0.01] - sum top2 = 0.81
    #  [0.05, 0.80, 0.20, 0.10] - sum top2 = 1.0 (selected group) - index from 12 to 15
    #
    # 4 groups of 2nd token
    #  [0.68, 0.20, 0.06, 0.03] - sum top2 = 0.88 (selected group) - index from 0 to 3
    #  [0.32, 0.10, 0.05, 0.02] - sum top2 = 0.42
    #  [0.65, 0.20, 0.04, 0.01] - sum top2 = 0.85 (selected group) - index from 8 to 11
    #  [0.32, 0.10, 0.05, 0.02] - sum top2 = 0.42
    #
    # From selected groups to choice top4 for each token
    expected_top_k_indices = jnp.array([[[13, 5, 6, 14], [0, 8, 1, 9]]])
    expected_top_k_weights = jnp.take_along_axis(pre_bias_logits, expected_top_k_indices, axis=-1)
    actual_top_k_weights, actual_top_k_indices = self.model.deepseek_routing(gate_logits, pre_bias_logits)
    self.assertTrue(
        jax.numpy.allclose(expected_top_k_indices, actual_top_k_indices, rtol=1e-05, atol=1e-05, equal_nan=False)
    )
    self.assertTrue(
        jax.numpy.allclose(expected_top_k_weights, actual_top_k_weights, rtol=1e-05, atol=1e-05, equal_nan=False)
    )

  def test_deepseek_bias_updates(self):
    num_experts = 4
    rate = 0.01
    # total tokens = 16, average tokens = 16 / 4 = 4
    # expert 0 assigned 5 tokens --> overload, update: -0.01
    # expert 1 assigned 4 tokens --> same, update: 0.0
    # expert 2 assigned 3 tokens --> underload, update: 0.01
    # expert 3 assigned 4 tokens --> same, update: 0.0
    # [batch, sequence, top_k] = [2, 4, 2]
    top_k_indices = jnp.array([[[0, 1], [3, 0], [3, 1], [1, 0]], [[0, 3], [2, 0], [1, 2], [2, 3]]])
    expected_counts = jnp.array([5, 4, 3, 4])
    actual_counts = moe.calculate_expert_counts(top_k_indices, num_experts)
    self.assertTrue(jax.numpy.allclose(expected_counts, actual_counts))

    expected_updates = jnp.array([-0.01, 0.0, 0.01, 0.0])
    actual_updates = moe.expert_counts_to_bias_update(actual_counts, num_experts, rate)
    self.assertTrue(jax.numpy.allclose(expected_updates, actual_updates, rtol=1e-05, atol=1e-05, equal_nan=False))


class Ling2RoutingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    extra_args = get_decoupled_parallelism_overrides()
    self.cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="ling2_routing_test",
        enable_checkpointing=False,
        decoder_block="ling2",
        dtype="bfloat16",
        max_target_length=2,
        max_prefill_predict_length=1,
        per_device_batch_size=1,
        n_routing_groups=4,
        topk_routing_group=2,
        num_experts=16,
        num_experts_per_tok=4,
        sparse_matmul=True,
        routed_score_func="sigmoid",
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        **extra_args,
    )
    self.rngs = nnx.Rngs(params=0)
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.model = moe.RoutedMoE(
        config=self.cfg,
        num_experts=self.cfg.num_experts,
        num_experts_per_tok=self.cfg.num_experts_per_tok,
        mesh=Mesh(devices_array, self.cfg.mesh_axes),
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=self.cfg.dtype,
        rngs=self.rngs,
    )

  def test_ling2_routing(self):
    """LING2 should use group routing (deepseek_routing) like DeepSeek."""
    # shape as [batch, sequence, num_experts] = [1,2,16]
    gate_logits = jnp.array(
        [
            [
                [0.20, 0.10, 0.05, 0.10, 0.10, 0.60, 0.30, 0.10, 0.80, 0.01, 0.01, 0.01, 0.05, 0.80, 0.20, 0.10],
                [0.68, 0.20, 0.06, 0.03, 0.32, 0.10, 0.05, 0.02, 0.65, 0.20, 0.04, 0.01, 0.32, 0.10, 0.05, 0.02],
            ]
        ]
    )
    pre_bias_logits = gate_logits - 0.5

    # Same group structure as DeepSeekRoutingTest:
    # 4 groups of 4 experts each, select top-2 groups, then top-4 experts.
    expected_top_k_indices = jnp.array([[[13, 5, 6, 14], [0, 8, 1, 9]]])
    expected_top_k_weights = jnp.take_along_axis(pre_bias_logits, expected_top_k_indices, axis=-1)
    actual_top_k_weights, actual_top_k_indices = self.model.deepseek_routing(gate_logits, pre_bias_logits)
    self.assertTrue(
        jax.numpy.allclose(expected_top_k_indices, actual_top_k_indices, rtol=1e-05, atol=1e-05, equal_nan=False)
    )
    self.assertTrue(
        jax.numpy.allclose(expected_top_k_weights, actual_top_k_weights, rtol=1e-05, atol=1e-05, equal_nan=False)
    )

  def test_ling2_weight_scaling(self):
    """LING2 weights should be normalized then scaled (no softmax, no double normalization)."""
    # Simulate pre-selected top-k weights (post-sigmoid, pre-bias scores)
    top_k_weights = jnp.array([[[0.3, 0.2, 0.4, 0.1]]])
    expected_sum = top_k_weights.sum(-1, keepdims=True)
    expected = (top_k_weights / expected_sum) * 2.5  # scaling_factor
    actual = self.model.deepseek_scale_weights(top_k_weights)
    self.assertTrue(jax.numpy.allclose(expected, actual, rtol=1e-05, atol=1e-05, equal_nan=False))
    # Verify weights sum to scaling_factor
    self.assertTrue(jax.numpy.allclose(actual.sum(-1), jnp.array([[2.5]]), rtol=1e-05, atol=1e-05, equal_nan=False))


class MoeLoopBlock(nnx.Module):
  """Reference implementation from https://github.com/mistralai/mistral-inference.
  This is not included anymore in our repo, due to a limitation of for-loop implementation in sharding.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      inputs_shape: tuple[int, ...],
      num_experts: int,
      num_experts_per_tok: int,
      kernel_init: NdInitializer,
      kernel_axes: tuple[str, ...],
      rngs: nnx.Rngs,
      weight_dtype: DType = jnp.float32,
      dtype: DType = jnp.bfloat16,
  ):
    self.config = config
    self.mesh = mesh
    self.inputs_shape = inputs_shape
    self.num_experts = num_experts
    self.num_experts_per_tok = num_experts_per_tok
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.gate = moe.GateLogit(
        in_features_shape=self.inputs_shape[-1],
        out_features_shape=self.num_experts,
        mesh=self.mesh,
        model_name=self.config.model_name,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        kernel_axes=self.kernel_axes,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )
    for k in range(self.num_experts):
      expert_module = linears.MlpBlock(
          config=self.config,
          mesh=self.mesh,
          in_features=self.inputs_shape[-1],
          intermediate_dim=self.config.mlp_dim,
          activations=["silu", "linear"],
          intermediate_dropout_rate=self.config.dropout_rate,
          dtype=dtype,
          weight_dtype=weight_dtype,
          rngs=rngs,
      )
      setattr(self, f"mlp_{k}", expert_module)

  def __call__(self, inputs, deterministic: bool = False):
    gate_logits = self.gate(inputs)[0]
    weights, selected_experts = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
    weights = jax.nn.softmax(weights.astype(jnp.float32), axis=-1).astype(self.weight_dtype)
    mlp_lnx = jnp.zeros_like(inputs)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_length_no_exp", "activation_embed"))

    for k in range(self.num_experts):
      weights_exp = jnp.sum(jnp.multiply(selected_experts == k, weights), axis=-1)
      getattr(self, f"mlp_{k}")
      mlp_lnx_exp = getattr(self, f"mlp_{k}")(inputs, deterministic=deterministic)
      mlp_lnx_exp = nn.with_logical_constraint(
          mlp_lnx_exp, ("activation_batch", "activation_length_no_exp", "activation_embed")
      )
      mlp_lnx_exp = weights_exp[:, :, None] * mlp_lnx_exp
      mlp_lnx += mlp_lnx_exp

    return mlp_lnx


def get_moe_loop(
    config: Config,
    mesh: Mesh,
    inputs_shape: tuple[int, ...],
    num_experts: int,
    num_experts_per_tok: int,
    kernel_init: NdInitializer,
    kernel_axes: tuple[str, ...],
    weight_dtype: DType = jnp.float32,
    dtype: DType = jnp.bfloat16,
):
  """Creates a MoeLoopBlock Linen module."""
  module = nnx_wrappers.to_linen(
      MoeLoopBlock,
      config=config,
      mesh=mesh,
      inputs_shape=inputs_shape,
      num_experts=num_experts,
      num_experts_per_tok=num_experts_per_tok,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes,
      weight_dtype=weight_dtype,
      dtype=dtype,
      metadata_fn=variable_to_logically_partitioned,
  )
  return module


class RoutedMoeTest(unittest.TestCase):
  """Routed Mixture of Experts test."""

  def get_expected_output(self, rng, hidden_states, cfg, mesh):
    """Retrieve expected output from Routed Mixture of Experts."""
    model = get_moe_loop(
        config=cfg,
        mesh=mesh,
        inputs_shape=hidden_states.shape,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
    )
    variables = model.init(
        {"params": rng, "dropout": rng},
        jax.random.normal(rng, (int(cfg.per_device_batch_size), cfg.max_target_length, cfg.base_emb_dim)),
    )

    output = jax.jit(model.apply)(variables, hidden_states)  # pylint: disable=not-callable
    return variables, output

  def get_moe_output(self, variables, hidden_states, cfg, mesh):
    """retrieve expected output from MoE"""
    model = moe.get_routed_moe(
        name="MoeBlock",
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.mlp_dim,
        dtype=cfg.dtype,
    )

    # convert format of parameters
    kernel = variables["params"]["gate"]["kernel"].value
    kernel = kernel.astype(cfg.weight_dtype)

    exp_wi_0 = []
    exp_wi_1 = []
    exp_wo = []

    for i in range(cfg.num_experts):
      tmp_wi_0 = variables["params"][f"mlp_{i}"]["wi_0"]["kernel"].value
      tmp_wi_0 = jnp.reshape(tmp_wi_0, (1, cfg.base_emb_dim, cfg.base_mlp_dim))
      tmp_wi_1 = variables["params"][f"mlp_{i}"]["wi_1"]["kernel"].value
      tmp_wi_1 = jnp.reshape(tmp_wi_1, (1, cfg.base_emb_dim, cfg.base_mlp_dim))
      tmp_wo = variables["params"][f"mlp_{i}"]["wo"]["kernel"].value
      tmp_wo = jnp.reshape(tmp_wo, (1, cfg.base_mlp_dim, cfg.base_emb_dim))

      exp_wi_0.append(tmp_wi_0)
      exp_wi_1.append(tmp_wi_1)
      exp_wo.append(tmp_wo)

    wi_0 = jnp.concatenate(exp_wi_0, axis=0, dtype=cfg.weight_dtype)
    wi_1 = jnp.concatenate(exp_wi_1, axis=0, dtype=cfg.weight_dtype)
    wo = jnp.concatenate(exp_wo, axis=0, dtype=cfg.weight_dtype)

    moe_variables = {"params": {"gate": {"kernel": kernel}, "wi_0": wi_0, "wi_1": wi_1, "wo": wo}}

    output = jax.jit(model.apply)(moe_variables, hidden_states)  # pylint: disable=not-callable
    return output

  @pytest.mark.tpu_only
  def test_megablox(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_megablox_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=1,
        max_target_length=128,
    )

    rng = jax.random.PRNGKey(1234)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg, mesh)
    actual_output, _, _, _, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
    self.assertTrue(jax.numpy.allclose(expected_output, actual_output, rtol=1e-02, atol=1e-02, equal_nan=False))

  @pytest.mark.tpu_only
  def test_ragged_dot(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_ragged_dot_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=False,
        sparse_matmul=True,
        per_device_batch_size=1,
        max_target_length=128,
    )

    rng = jax.random.PRNGKey(1234)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg, mesh)
    actual_output, _, _, _, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
    self.assertTrue(jax.numpy.allclose(expected_output, actual_output, rtol=1e-02, atol=1e-02, equal_nan=False))

  @pytest.mark.tpu_only
  def test_dense(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_dense_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="float32",
        megablox=False,
        sparse_matmul=False,
        per_device_batch_size=1,
        max_target_length=128,
    )

    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg, mesh)
    actual_output, _, _, _, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
    self.assertTrue(jax.numpy.allclose(expected_output, actual_output, rtol=1e-05, atol=1e-05, equal_nan=False))

  @pytest.mark.tpu_only
  def test_megablox_expert_parallelism(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_megablox_ep_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=4,  # TODO(b/450900273): sharding error if pdbs=1
        ici_expert_parallelism=4,
        max_target_length=128,
    )

    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg, mesh)
      actual_output, _, _, _, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
      self.assertTrue(jax.numpy.allclose(expected_output, actual_output, rtol=1e-02, atol=1e-02, equal_nan=False))

  @pytest.mark.tpu_only
  def test_ring_of_expert_and_tensor_parallelism(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_ring_ep_tp_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=4,  # TODO(b/450900273): sharding error if pdbs=1
        ici_expert_parallelism=2,
        use_ring_of_experts=True,
        ici_tensor_parallelism=2,
        max_target_length=128,
    )

    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg, mesh)
      actual_output, _, _, _, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
      self.assertTrue(jax.numpy.allclose(expected_output, actual_output, rtol=1e-02, atol=1e-02, equal_nan=False))

  @pytest.mark.tpu_only
  def test_moe_fsdp_two_stage_parallelism_tpu_only(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_megablox_ep_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=4,  # TODO(b/450900273): sharding error if pdbs=1
        ici_fsdp_parallelism=2,
        ici_fsdp_transpose_parallelism=2,
        moe_fsdp_use_two_stage_all_gather=True,
        max_target_length=128,
    )

    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg, mesh)
      actual_output, _, _, _, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
      self.assertTrue(jax.numpy.allclose(expected_output, actual_output, rtol=1e-02, atol=1e-02, equal_nan=False))

  @pytest.mark.tpu_only
  def test_megablox_tp_transpose_parallelism(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_megablox_tp_transpose_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=1,
        ici_tensor_transpose_parallelism=4,
        max_target_length=128,
    )

    cfg2 = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_megablox_tp_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=1,
        ici_tensor_parallelism=4,
        max_target_length=128,
    )

    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      variables, _ = self.get_expected_output(rng_model, hidden_states, cfg, mesh)
      tp_transpose_output, _, _, _, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
      tp_output, _, _, _, _ = self.get_moe_output(variables, hidden_states, cfg2, mesh)
      self.assertTrue(jax.numpy.allclose(tp_output, tp_transpose_output, rtol=1e-05, atol=1e-05, equal_nan=False))

  @pytest.mark.tpu_only
  def test_megablox_context_parallelism(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_megablox_cp_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=1,
        ici_context_parallelism=4,
        max_target_length=128,
    )

    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg, mesh)
      actual_output, _, _, _, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
      self.assertTrue(jax.numpy.allclose(expected_output, actual_output, rtol=1e-02, atol=1e-02, equal_nan=False))

  @pytest.mark.tpu_only
  def test_megablox_expert_context_parallelism(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_megablox_ep_cp_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=4,
        ici_context_parallelism=2,
        ici_expert_parallelism=2,
        packing=False,
        max_target_length=128,
    )

    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg, mesh)
      actual_output, _, _, _, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
      self.assertTrue(jax.numpy.allclose(expected_output, actual_output, rtol=1e-02, atol=1e-02, equal_nan=False))

  @pytest.mark.tpu_only
  def test_megablox_expert_tensor_parallelism(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_block_megablox_ep_tp_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=4,
        ici_tensor_parallelism=2,
        ici_expert_parallelism=2,
        max_target_length=128,
    )

    rng = jax.random.PRNGKey(2345)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      variables, expected_output = self.get_expected_output(rng_model, hidden_states, cfg, mesh)
      actual_output, _, _, _, _ = self.get_moe_output(variables, hidden_states, cfg, mesh)
      self.assertTrue(jax.numpy.allclose(expected_output, actual_output, rtol=1e-02, atol=1e-02, equal_nan=False))

  def test_random_routing(self):
    bs, seq_len, num_experts, num_experts_per_tok = 12, 1024, 8, 2
    rng = jax.random.PRNGKey(0)
    rng, logits_key = jax.random.split(rng)
    gate_logits = jax.random.normal(logits_key, (bs, seq_len, num_experts))

    rng, run_key = jax.random.split(rng)
    _, top_k_indices = moe.random_routing(run_key, gate_logits, num_experts_per_tok)

    flat_indices = top_k_indices.flatten()
    counts = jnp.bincount(flat_indices, length=num_experts)
    expected_count = bs * seq_len * num_experts_per_tok // num_experts
    tol = 0.05

    lower_bound = expected_count - expected_count * tol
    upper_bound = expected_count + expected_count * tol
    is_with_tolerance = (counts >= lower_bound) & (counts <= upper_bound)
    self.assertTrue(is_with_tolerance.all())

  def test_local_permute_no_offset(self):
    """Tests local_permute with is_offset=False across multiple shards."""
    num_experts = 8
    num_shards = 4
    experts_per_shard = num_experts // num_shards  # 2 experts per shard

    # Global group sizes for each of the 8 experts
    # Expert 0 gets 0 token, Expert 1 gets 1, ..., Expert 7 gets 7 tokens.
    global_group_sizes = jnp.arange(num_experts)
    total_assignments = jnp.sum(global_group_sizes)

    original_inputs = jnp.arange(total_assignments * 5, dtype=jnp.int32).reshape(total_assignments, 5)

    # Calculate the cumulative sum of global group sizes to determine shard input slices
    global_group_sizes_cumsum = jnp.cumsum(global_group_sizes)

    shard_start_indices = jnp.concatenate(
        [jnp.array([0]), global_group_sizes_cumsum[:-experts_per_shard:experts_per_shard]]
    )
    shard_end_indices = global_group_sizes_cumsum[experts_per_shard - 1 :: experts_per_shard]

    #               *****Expected outputs****
    # Shard 0: tokens for global experts 0, 1 (0+1=1 tokens)
    #  expected_local_group_size: [0, 1]
    #  expected_sorted_inputs: original_inputs[:0+1]
    #  expected_sorted_indices: [0]]
    #  expected_sorted_experts_ids: [1]
    # Shard 1: tokens for global experts 2, 3 (2+3=5 tokens)
    #  expected_local_group_size: [2, 3]
    #  expected_sorted_inputs: original_inputs[1:1+2+3]
    #  expected_sorted_indices: [0,1,2,3,4]
    #  expected_sorted_experts_ids: [0]*2 + [1]*3
    # Shard 2: tokens for global experts 4, 5 (4+5=9 tokens)
    #  expected_local_group_size: [4, 5]
    #  expected_sorted_inputs: original_inputs[6:6+4+5]
    #  expected_sorted_indices: [0,1,2,3,4,5,6,7,8]
    #  expected_sorted_experts_ids: [0]*4 + [1]*5
    # Shard 3: tokens for global experts 6, 7 (6+7=13 tokens)
    #  expected_local_group_size: [6, 7]
    #  expected_sorted_inputs: original_inputs[15:15+13]
    #  expected_sorted_indices: [0,1,2,3,4,5,6,7,8,9,10,11,12]
    #  expected_sorted_experts_ids: [0]*6 + [1]*7
    for shard_index in range(num_shards):
      # Determine the input slice for the current shard
      start_idx = shard_start_indices[shard_index]
      end_idx = shard_end_indices[shard_index]
      inputs_shard = original_inputs[start_idx:end_idx]
      shard_total_tokens = end_idx - start_idx

      # Get the global group sizes relevant to this shard's experts
      global_group_sizes_for_shard = global_group_sizes[
          shard_index * experts_per_shard : (shard_index + 1) * experts_per_shard
      ]

      # Get the actual local_permute outputs.
      sorted_inputs, sorted_indices, local_group_size, sorted_experts_ids = moe.RoutedMoE.local_permute(
          inputs_shard,
          global_group_sizes[None, :],
          experts_per_shard,
          shard_index,
          use_custom_sort_vjp=False,
          is_offset=False,
      )

      # Calculate expected outputs for the current shard
      expected_local_group_size = global_group_sizes_for_shard
      # With is_offset=False, input is assumed pre-sorted by expert, so sorted_inputs is the input itself.
      expected_sorted_inputs = inputs_shard
      # Indices are relative to inputs_shard, and since it's already sorted, they are just arange.
      expected_sorted_indices = jnp.arange(shard_total_tokens)
      # Local expert IDs: repeat local expert index (0, 1, ...) by its count
      expected_sorted_experts_ids = jnp.repeat(
          jnp.arange(experts_per_shard), expected_local_group_size, total_repeat_length=shard_total_tokens
      )

      self.assertTrue(
          jnp.array_equal(sorted_inputs, expected_sorted_inputs), f"Shard {shard_index}: sorted_inputs mismatch"
      )
      self.assertTrue(
          jnp.array_equal(sorted_indices, expected_sorted_indices), f"Shard {shard_index}: sorted_indices mismatch"
      )
      self.assertTrue(
          jnp.array_equal(local_group_size, expected_local_group_size), f"Shard {shard_index}: local_group_size mismatch"
      )
      self.assertTrue(
          jnp.array_equal(sorted_experts_ids, expected_sorted_experts_ids),
          f"Shard {shard_index}: sorted_experts_ids mismatch",
      )

  def test_local_permute_offset(self):
    experts_per_group = 2
    expert_groups = 4  # aka number of expert shards.
    num_experts = 8

    # Global group sizes for each of the 8 experts
    # Each entry i specifies the number of tokens assigned to expert i.
    simple_group_sizes = jnp.arange(8)
    manual_global_group_sizes = jnp.array([0, 0, 1, 1, 2, 0, 2, 2])
    for global_expert_counts in [simple_group_sizes, manual_global_group_sizes]:
      for shard_id in range(expert_groups):
        # Unpermuted data. shape: (sum(global_expert_counts), 5)
        x = jnp.tile(jnp.arange(1, jnp.sum(global_expert_counts) + 1).reshape(-1, 1), (1, 5))

        # The number of expert IDs assigned to each expert shard.
        local_group_sizes = jnp.sum(jnp.reshape(global_expert_counts, (expert_groups, experts_per_group)), axis=-1)

        # Expert assignments corresponding to each entry of x.
        # NOTE: It is assumed that x is sorted in order of expert ID (because it is previously
        # passed through permute()), so expert_assignments just repeats the expert ID using counts from
        # global_expert_counts.
        expert_assignments = jnp.repeat(jnp.arange(0, num_experts), repeats=global_expert_counts)

        # Offset for the start of each shard (aka expert group). Offset for shard i is the sum
        # of the number of tokens assigned to all shards (local_group_size) before i.
        input_offsets = jnp.concatenate((jnp.array([0]), jnp.cumsum(local_group_sizes)[:-1]))

        # Actual results of local_permute().
        permuted_x, local_sorted_indices, local_expert_counts, local_expert_assignments = moe.RoutedMoE.local_permute(
            x,
            global_expert_counts[None, :],
            experts_per_group,
            shard_index=shard_id,
            use_custom_sort_vjp=False,
            is_offset=True,
            global_sorted_experts=expert_assignments,
        )

        # permuted_x should be equivalent to slicing x at the input offset for that shard.
        assert jnp.all(
            permuted_x[: local_group_sizes[shard_id]]
            == x[input_offsets[shard_id] : input_offsets[shard_id] + local_group_sizes[shard_id]]
        ), f"Local permuted rows do not match their unpermuted original rows for shard_id={shard_id}"

        # local_sorted_indices should match the indices of the slice from x corresponding to this shard.
        # That can be computed by taking all of the indices between the input_offset for the current shard
        # until the last index belonging to the current shard (i.e. input_offset[shard_id] + local_group_sizes[shard_id]).
        assert jnp.all(
            local_sorted_indices[: local_group_sizes[shard_id]]
            == jnp.arange(input_offsets[shard_id], input_offsets[shard_id] + local_group_sizes[shard_id])
        ), (
            "Local permuted row indices do not match their respective unpermuted indices in the "
            f"original inputs for shard_id={shard_id}!"
        )

        # local_expert_counts should correspond to slicing experts_per_group values from global_expert_counts
        # for the shard_id.
        assert jnp.all(
            local_expert_counts == global_expert_counts[shard_id * experts_per_group : (shard_id + 1) * experts_per_group]
        ), "Local permuted group sizes do not match the respective unpermuted expert bincounts for shard_id={shard_id}."

        # local_expert_assignments should correspond to taking a slice out of expert_assignments.
        # The slice size is shard i's size (local_group_sizes[i]]) and the slice should start
        # at input_offsets[i].
        assert jnp.all(
            local_expert_assignments[: local_group_sizes[shard_id]]
            == jnp.mod(
                expert_assignments[input_offsets[shard_id] : input_offsets[shard_id] + local_group_sizes[shard_id]],
                experts_per_group,
            )
        ), (
            "Local permuted expert assignments to not match the expected unpermuted expert assignments "
            f"for shard_id={shard_id}."
        )

  def test_get_all_to_all_params_sharded_batch(self):
    num_expert_parallelism_sharded = 4

    # all_group_sizes[i, j] = num inputs batch_shard i sends to expert_shard j
    all_group_sizes_sharded = jnp.array([[1, 2, 0, 3], [4, 0, 1, 2], [0, 3, 2, 1], [2, 1, 4, 0]], dtype=jnp.int32)

    # The offset for the current batch shard (row) to send inputs to a particular expert
    # shard (column), will be the cumulative number of tokens sent to all previous experts.
    # Example: batch shard 1
    # all_group_sizes_sharded[1] = [4, 0, 1, 2]:
    # input_offsets = [0, 4, 4+0, 4+0+1] = [0, 4, 4, 5]
    expected_input_offsets_sharded = jnp.array([[0, 1, 3, 3], [0, 4, 4, 5], [0, 0, 3, 5], [0, 2, 3, 7]], dtype=jnp.int32)

    # The number of tokens that the current batch shard (row) sends to each expert shard (columns)
    # is recorded in all_group_sizes_sharded[row].
    expected_send_sizes_sharded = jnp.array([[1, 2, 0, 3], [4, 0, 1, 2], [0, 3, 2, 1], [2, 1, 4, 0]], dtype=jnp.int32)

    # The offset at which each expert shard (column) will receive the current batch shard's (row)
    # input is the cumulative number of tokens received by all previous batch shards (rows)
    # for that expert.
    # (batch shard 0) output_offsets = [0, 0, 0, 0]
    # (batch shard 1) output_offsets = [0+1, 0+2, 0+0, 0+3])
    # (batch shard 2) output_offsets = [0+1+4, 0+2+0, 0+0+1, 0+3+2])
    # ...
    expected_output_offsets_sharded = jnp.array([[0, 0, 0, 0], [1, 2, 0, 3], [5, 2, 1, 5], [5, 5, 3, 6]], dtype=jnp.int32)

    # The number of inputs a particular expert shard (col) receives from all of the batch_shards (rows).
    # Example: expert shard 1
    # Receives 2 from batch_shard 0, 0 from batch_shard 1, 3 from batch_shard 2, 1 from batch_shard 3
    # Which is the same as all_group_sizes_sharded[:, 1].
    expected_recv_sizes_sharded = jnp.array([[1, 4, 0, 2], [2, 0, 3, 1], [0, 1, 2, 4], [3, 2, 1, 0]], dtype=jnp.int32)

    for expert_shard_id in range(num_expert_parallelism_sharded):
      exp_in_off = expected_input_offsets_sharded[expert_shard_id]
      exp_send_sz = expected_send_sizes_sharded[expert_shard_id]
      exp_out_off = expected_output_offsets_sharded[expert_shard_id]
      exp_recv_sz = expected_recv_sizes_sharded[expert_shard_id]

      in_off, send_sz, out_off, recv_sz = moe.RoutedMoE.get_all_to_all_params(
          all_group_sizes_sharded, expert_shard_id, num_expert_parallelism_sharded, is_batch_sharded=True
      )
      self.assertTrue(
          jnp.array_equal(in_off, exp_in_off), f"Sharded Batch: Input offsets mismatch for shard {expert_shard_id}"
      )
      self.assertTrue(
          jnp.array_equal(send_sz, exp_send_sz), f"Sharded Batch: Send sizes mismatch for shard {expert_shard_id}"
      )
      self.assertTrue(
          jnp.array_equal(out_off, exp_out_off), f"Sharded Batch: Output offsets mismatch for shard {expert_shard_id}"
      )
      self.assertTrue(
          jnp.array_equal(recv_sz, exp_recv_sz), f"Sharded Batch: Receive sizes mismatch for shard {expert_shard_id}"
      )

  def test_get_all_to_all_params_unsharded_batch(self):
    """Tests get_all_to_all_params with a simple hard-coded example using 4 expert shards."""
    num_expert_parallelism_unsharded = 4

    # group_sizes_unsharded[i] = num inputs that each expert_shard i is responsible for.
    group_sizes_unsharded = jnp.array([6, 7, 6, 7], dtype=jnp.int32)

    # Each expert shard will send their data starting at index 0.
    expected_input_offsets_unsharded_template = jnp.array([0, 0, 0, 0], dtype=jnp.int32)

    # Each expert shard will send the amount of data they are responsible for
    # (indicated by group_sizes_unsharded).
    expected_send_sizes_unsharded_per_shard = jnp.array(
        [[6, 6, 6, 6], [7, 7, 7, 7], [6, 6, 6, 6], [7, 7, 7, 7]], dtype=jnp.int32
    )

    # When the batches are fully replicated (unsharded) then each batch will receive expert i's
    # data at the cumulative sum of the amount of input received from all previous experts.
    # (batch shard 0) output_offsets = [0, 0, 0, 0]
    # (batch shard 1) output_offsets = [0+6, 0+6, 0+6, 0+6])
    # (batch shard 2) output_offsets = [0+6+7, 0+6+7, 0+6+7, 0+6+7])
    # Which is just the cumulative sum of 0 and group_sizes_unsharded.
    expected_output_offsets_unsharded_per_shard = jnp.array(
        [[0, 0, 0, 0], [6, 6, 6, 6], [13, 13, 13, 13], [19, 19, 19, 19]], dtype=jnp.int32
    )

    # Each (replicated) batch shard will the amount of data from each expert specified by
    # group_sizes_unsharded.
    expected_recv_sizes_unsharded_template = jnp.array([6, 7, 6, 7], dtype=jnp.int32)

    for expert_shard_id in range(num_expert_parallelism_unsharded):
      exp_in_off = expected_input_offsets_unsharded_template
      exp_send_sz = expected_send_sizes_unsharded_per_shard[expert_shard_id]
      exp_out_off = expected_output_offsets_unsharded_per_shard[expert_shard_id]
      exp_recv_sz = expected_recv_sizes_unsharded_template

      in_off, send_sz, out_off, recv_sz = moe.RoutedMoE.get_all_to_all_params(
          group_sizes_unsharded, expert_shard_id, num_expert_parallelism_unsharded, is_batch_sharded=False
      )
      self.assertTrue(
          jnp.array_equal(in_off, exp_in_off), f"Unsharded Batch: Input offsets mismatch for shard {expert_shard_id}"
      )
      self.assertTrue(
          jnp.array_equal(send_sz, exp_send_sz), f"Unsharded Batch: Send sizes mismatch for shard {expert_shard_id}"
      )
      self.assertTrue(
          jnp.array_equal(out_off, exp_out_off), f"Unsharded Batch: Output offsets mismatch for shard {expert_shard_id}"
      )
      self.assertTrue(
          jnp.array_equal(recv_sz, exp_recv_sz), f"Unsharded Batch: Receive sizes mismatch for shard {expert_shard_id}"
      )


class ZeroMeanStateParamTest(unittest.TestCase):
  """Tests that zero_mean_state_param correctly zero-centers a target parameter."""

  def test_zero_mean_centering(self):
    """After applying zero_mean_state_param, the target parameter should have zero mean."""

    # Create a dummy state with nested params mimicking the MoE bias path
    params = {
        "params": {
            "decoder": {
                "moe_layers": {
                    "ALMoeBlock_0": {
                        "MoeBlock_0": {
                            "gate": {
                                "bias": jnp.array([0.5, -0.1, 0.3, -0.2, 0.1]),
                            }
                        }
                    }
                }
            }
        }
    }
    state = train_state.TrainState(step=0, apply_fn=None, params=params, tx=None, opt_state={})

    target_path = ("params", "decoder", "moe_layers", "ALMoeBlock_0", "MoeBlock_0", "gate", "bias")
    new_state = maxtext_utils.zero_mean_state_param(state, target_path)

    # Extract the updated bias
    new_bias = new_state.params["params"]["decoder"]["moe_layers"]["ALMoeBlock_0"]["MoeBlock_0"]["gate"]["bias"]

    # Verify mean is zero
    self.assertAlmostEqual(float(new_bias.mean()), 0.0, places=6)
    # Verify relative differences preserved
    original_bias = params["params"]["decoder"]["moe_layers"]["ALMoeBlock_0"]["MoeBlock_0"]["gate"]["bias"]
    expected = original_bias - original_bias.mean()
    self.assertTrue(jnp.allclose(new_bias, expected))

  def test_non_target_params_unchanged(self):
    """Non-target parameters should not be modified."""

    params = {
        "params": {
            "other_param": jnp.array([1.0, 2.0, 3.0]),
            "decoder": {
                "moe_layers": {
                    "ALMoeBlock_0": {
                        "MoeBlock_0": {
                            "gate": {
                                "bias": jnp.array([0.5, -0.1, 0.3]),
                            }
                        }
                    }
                }
            },
        }
    }
    state = train_state.TrainState(step=0, apply_fn=None, params=params, tx=None, opt_state={})

    target_path = ("params", "decoder", "moe_layers", "ALMoeBlock_0", "MoeBlock_0", "gate", "bias")
    new_state = maxtext_utils.zero_mean_state_param(state, target_path)

    # other_param should be unchanged
    self.assertTrue(
        jnp.array_equal(
            new_state.params["params"]["other_param"],
            params["params"]["other_param"],
        )
    )


class FuseGateUpProjTest(unittest.TestCase):
  """fuse_gate_up_proj=True must reduce wi GMM calls from 2 to 1."""

  def _build_cfg(self, fuse: bool):
    extra_args = get_decoupled_parallelism_overrides()
    return pyconfig.initialize(
        [None, get_test_config_path()],
        run_name=f"fuse_gate_up_proj_{fuse}_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=True,
        sparse_matmul=True,
        per_device_batch_size=1,
        max_target_length=128,
        fuse_gate_up_proj=fuse,
        **extra_args,
    )

  def _build_inputs_and_variables(self, cfg, mesh):
    """Initialize a RoutedMoE the same way RoutedMoeTest.get_moe_output does."""
    rng = jax.random.PRNGKey(1234)
    rng_model, rng_hidden_states = jax.random.split(rng)
    device_count = jax.device_count()
    hidden_states = jax.random.uniform(
        rng_hidden_states,
        (int(cfg.per_device_batch_size) * device_count, cfg.max_target_length, cfg.base_emb_dim),
        dtype=cfg.dtype,
    )

    # Use MoeLoopBlock to seed initial parameters, mirroring RoutedMoeTest.
    loop_model = get_moe_loop(
        config=cfg,
        mesh=mesh,
        inputs_shape=hidden_states.shape,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
    )
    variables = loop_model.init(
        {"params": rng_model, "dropout": rng_model},
        jax.random.normal(rng_model, (int(cfg.per_device_batch_size), cfg.max_target_length, cfg.base_emb_dim)),
    )

    routed_moe = moe.get_routed_moe(
        name="MoeBlock",
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=cfg.mlp_dim,
        dtype=cfg.dtype,
    )

    kernel = variables["params"]["gate"]["kernel"].value.astype(cfg.weight_dtype)
    exp_wi_0, exp_wi_1, exp_wo = [], [], []
    for i in range(cfg.num_experts):
      tmp_wi_0 = variables["params"][f"mlp_{i}"]["wi_0"]["kernel"].value
      tmp_wi_0 = jnp.reshape(tmp_wi_0, (1, cfg.base_emb_dim, cfg.base_mlp_dim))
      tmp_wi_1 = variables["params"][f"mlp_{i}"]["wi_1"]["kernel"].value
      tmp_wi_1 = jnp.reshape(tmp_wi_1, (1, cfg.base_emb_dim, cfg.base_mlp_dim))
      tmp_wo = variables["params"][f"mlp_{i}"]["wo"]["kernel"].value
      tmp_wo = jnp.reshape(tmp_wo, (1, cfg.base_mlp_dim, cfg.base_emb_dim))
      exp_wi_0.append(tmp_wi_0)
      exp_wi_1.append(tmp_wi_1)
      exp_wo.append(tmp_wo)
    wi_0 = jnp.concatenate(exp_wi_0, axis=0, dtype=cfg.weight_dtype)
    wi_1 = jnp.concatenate(exp_wi_1, axis=0, dtype=cfg.weight_dtype)
    wo = jnp.concatenate(exp_wo, axis=0, dtype=cfg.weight_dtype)
    moe_variables = {"params": {"gate": {"kernel": kernel}, "wi_0": wi_0, "wi_1": wi_1, "wo": wo}}
    return routed_moe, moe_variables, hidden_states

  @pytest.mark.tpu_only
  def test_fuse_reduces_gmm_call_count(self):
    """With fuse_gate_up_proj=True the wi gmm path should call mblx.gmm once instead of twice.

    Also runs an unfused-control branch to prove the patch actually intercepts:
    if the patch did not work, both branches would observe call_count == 0 (or
    the same nonzero number), not 2 vs. 3.
    """
    from unittest.mock import patch  # pylint: disable=import-outside-toplevel
    from maxtext.kernels import megablox as mblx_mod  # pylint: disable=import-outside-toplevel

    original_gmm = mblx_mod.gmm
    call_count = [0]

    def counting(*args, **kwargs):
      call_count[0] += 1
      return original_gmm(*args, **kwargs)

    # --- fused branch: wi=1 + wo=1 = 2 ---
    cfg_fused = self._build_cfg(fuse=True)
    devices_array = maxtext_utils.create_device_mesh(cfg_fused)
    mesh_fused = Mesh(devices_array, cfg_fused.mesh_axes)
    routed_moe_fused, moe_vars_fused, hs_fused = self._build_inputs_and_variables(cfg_fused, mesh_fused)

    call_count[0] = 0
    with patch("maxtext.kernels.megablox.gmm", side_effect=counting):
      _ = jax.jit(routed_moe_fused.apply)(moe_vars_fused, hs_fused)  # pylint: disable=not-callable
    fused_calls = call_count[0]

    # --- unfused control: wi=2 + wo=1 = 3 ---
    cfg_unfused = self._build_cfg(fuse=False)
    devices_array = maxtext_utils.create_device_mesh(cfg_unfused)
    mesh_unfused = Mesh(devices_array, cfg_unfused.mesh_axes)
    routed_moe_unfused, moe_vars_unfused, hs_unfused = self._build_inputs_and_variables(cfg_unfused, mesh_unfused)

    call_count[0] = 0
    with patch("maxtext.kernels.megablox.gmm", side_effect=counting):
      _ = jax.jit(routed_moe_unfused.apply)(moe_vars_unfused, hs_unfused)  # pylint: disable=not-callable
    unfused_calls = call_count[0]

    self.assertEqual(
        unfused_calls,
        3,
        f"Expected 3 gmm calls with fuse_gate_up_proj=False (control), got {unfused_calls}. "
        "If 0, the mock.patch target is not intercepting mblx.gmm.",
    )
    self.assertEqual(
        fused_calls,
        2,
        f"Expected 2 gmm calls with fuse_gate_up_proj=True, got {fused_calls}",
    )

  @pytest.mark.tpu_only
  def test_fuse_numerical_equivalence(self):
    """Fused (concat W_gate,W_up -> gmm -> split) must match unfused (two gmms) numerically.

    Catches regressions in n_half (split offset), slice direction, or concat axis.
    Uses identical PRNG seeding so weights/inputs are bitwise-identical between the two runs.
    """
    cfg_fused = self._build_cfg(fuse=True)
    devices_array = maxtext_utils.create_device_mesh(cfg_fused)
    mesh_fused = Mesh(devices_array, cfg_fused.mesh_axes)
    routed_moe_fused, moe_vars_fused, hs_fused = self._build_inputs_and_variables(cfg_fused, mesh_fused)
    out_fused = jax.jit(routed_moe_fused.apply)(moe_vars_fused, hs_fused)  # pylint: disable=not-callable

    cfg_unfused = self._build_cfg(fuse=False)
    devices_array = maxtext_utils.create_device_mesh(cfg_unfused)
    mesh_unfused = Mesh(devices_array, cfg_unfused.mesh_axes)
    routed_moe_unfused, moe_vars_unfused, hs_unfused = self._build_inputs_and_variables(cfg_unfused, mesh_unfused)
    out_unfused = jax.jit(routed_moe_unfused.apply)(moe_vars_unfused, hs_unfused)  # pylint: disable=not-callable

    # `_build_inputs_and_variables` reseeds with PRNGKey(1234) on every call, so the
    # MoE weights and hidden_states are identical across the two runs by construction.
    # bf16 numerics: match the tolerance RoutedMoeTest uses for end-to-end MoE outputs.
    def _arr(o):
      return o[0] if isinstance(o, tuple) else o

    self.assertTrue(
        jnp.allclose(_arr(out_fused), _arr(out_unfused), rtol=1e-02, atol=1e-02, equal_nan=False),
        "Fused gate+up output diverges from unfused output beyond bf16 tolerance; "
        "check n_half, split direction, and concat axis in moe.py fused branch.",
    )


class TokenPermutationOptTest(unittest.TestCase):
  """Math-only tests for the PR2 permute() optimizations.

  These run without TPU and validate the algebraic equivalences that the
  refactor relies on:

    1. ``searchsorted(diff)`` over an argsort-sorted experts vector produces
       the same per-expert counts as ``jnp.bincount``.
  """

  def test_searchsorted_diff_equals_bincount(self):
    rng = jax.random.PRNGKey(0)
    num_experts = 17  # deliberately not a power of 2
    n_tokens = 1024
    for seed in range(5):
      key = jax.random.fold_in(rng, seed)
      flatten_selected_experts = jax.random.randint(key, shape=(n_tokens,), minval=0, maxval=num_experts, dtype=jnp.int32)
      sorted_idx = jnp.argsort(flatten_selected_experts)
      sorted_experts = flatten_selected_experts[sorted_idx]

      # Old path.
      bincount_group_size = jnp.bincount(flatten_selected_experts, length=num_experts)

      # New path.
      boundaries = jnp.searchsorted(sorted_experts, jnp.arange(num_experts + 1))
      diff_group_size = jnp.diff(boundaries)

      self.assertEqual(bincount_group_size.shape, diff_group_size.shape)
      self.assertTrue(
          jnp.array_equal(bincount_group_size, diff_group_size),
          (
              f"searchsorted+diff disagrees with bincount on seed {seed}: "
              f"bincount={bincount_group_size}, diff={diff_group_size}"
          ),
      )

      # Sanity: total token count is preserved.
      self.assertEqual(int(jnp.sum(diff_group_size)), n_tokens)

    # Zero-bin edge case: ensure searchsorted+diff handles empty experts at
    # boundaries (expert 0 and expert num_experts-1 receive zero tokens).
    num_experts_zb = 8
    flatten_selected_experts = jnp.array([1, 1, 2, 3, 3, 3, 4, 5, 6, 6], dtype=jnp.int32)
    sorted_idx = jnp.argsort(flatten_selected_experts)
    sorted_experts = flatten_selected_experts[sorted_idx]
    boundaries = jnp.searchsorted(sorted_experts, jnp.arange(num_experts_zb + 1))
    group_size_new = jnp.diff(boundaries)
    group_size_ref = jnp.bincount(flatten_selected_experts, length=num_experts_zb)
    self.assertTrue(
        jnp.array_equal(group_size_new, group_size_ref),
        f"zero-bin case: searchsorted+diff={group_size_new} != bincount={group_size_ref}",
    )
    # Verify expert 0 and expert num_experts_zb-1 indeed got zero tokens.
    self.assertEqual(int(group_size_new[0]), 0)
    self.assertEqual(int(group_size_new[-1]), 0)


class TokenPermutationBackwardTest(unittest.TestCase):
  """TPU-only test that the backward pass through permute->op->unpermute
  performs only a single jnp.argsort (the one in permute's forward).

  Before PR2, unpermute called jnp.argsort(sorted_selected_experts) on every
  forward, and _sort_activations_custom's bwd called jnp.argsort again, so
  the trace through one permute+one unpermute had at least 3 argsort calls.
  After PR2 only the eager argsort in permute (and the inv_perm argsort
  staged alongside it) remain, so a forward+backward should observe
  exactly 2 argsort traces and zero additional ones in the bwd.
  """

  @pytest.mark.tpu_only
  def test_backward_no_extra_argsort(self):
    from unittest.mock import patch  # pylint: disable=import-outside-toplevel

    extra_args = get_decoupled_parallelism_overrides()
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="permute_bwd_no_argsort_test",
        enable_checkpointing=False,
        model_name="mixtral-8x7b",
        dtype="bfloat16",
        megablox=False,
        sparse_matmul=False,
        per_device_batch_size=1,
        max_target_length=64,
        **extra_args,
    )
    rngs = nnx.Rngs(params=0)
    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    model = moe.RoutedMoE(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=cfg.dtype,
        rngs=rngs,
    )

    bsz = int(cfg.per_device_batch_size) * jax.device_count()
    seq = cfg.max_target_length
    emb = cfg.base_emb_dim
    rng = jax.random.PRNGKey(0)
    rng_in, rng_logits = jax.random.split(rng)
    inputs = jax.random.normal(rng_in, (bsz, seq, emb), dtype=cfg.dtype)
    gate_logits = jax.random.normal(rng_logits, (bsz, seq, cfg.num_experts), dtype=jnp.float32)

    original_argsort = jnp.argsort
    call_count = [0]

    def counting_argsort(*args, **kwargs):
      call_count[0] += 1
      return original_argsort(*args, **kwargs)

    def loss_fn(x):
      sorted_inputs, sorted_idx, inv_perm, weights, *_ = model.permute(x, gate_logits, None, cfg.use_custom_sort_vjp)
      # Trivial op so backward must flow through both permute and unpermute.
      intermediate = sorted_inputs * 1.0
      out = model.unpermute(
          intermediate,
          sorted_idx,
          inv_perm,
          weights,
          batch_size=bsz,
          sequence_length=seq,
          use_custom_sort_vjp=cfg.use_custom_sort_vjp,
      )
      if isinstance(out, tuple):
        out = out[0]
      return jnp.sum(out.astype(jnp.float32))

    # Patch jnp.argsort *only inside the moe module* so we count just the
    # argsort calls that originate in permute/unpermute (not in the model
    # init or in the test scaffolding).
    with patch("maxtext.layers.moe.jnp.argsort", side_effect=counting_argsort):
      _, _ = jax.value_and_grad(loss_fn)(inputs)

    # PR2 invariant: exactly 2 argsort traces (sorted_idx + inv_perm in
    # permute fwd). The bwd of _gather_with_precomputed_inverse must NOT
    # add a third.
    self.assertEqual(
        call_count[0],
        2,
        f"Expected exactly 2 jnp.argsort calls in moe forward+backward, got {call_count[0]}. "
        "A third call indicates the backward pass is recomputing argsort "
        "instead of reusing the precomputed inverse permutation.",
    )


if __name__ == "__main__":
  unittest.main()
