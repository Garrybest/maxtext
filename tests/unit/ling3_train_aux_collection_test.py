# Copyright 2026 Google LLC
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

"""Regression tests for the LING3 scan-mode trainer-side aux collection.

Covers the bugs documented in docs/ling3-scan-loss-analysis.md:
  - `_collect_moe_intermediate_sum` must collect Phase 1b prefix + nested Phase 2
    layout under `decoder/moe_layers/layers_X/{key}` instead of the flat
    `decoder/moe_layers/{key}` path.
  - `_update_deepseek_bias` must update both Phase 1b prefix bias (1D) and
    Phase 2 sub-layer bias (2D `(num_experts, scan_length)`) with the correct
    transpose and zero-mean axis.
  - Both paths must raise (not silently fall back to 0 / None) when an
    expected-to-exist sown value is missing.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state

from maxtext.common.common_types import DecoderBlockType
from maxtext.trainers.pre_train import train as train_mod


# Mirror of ling3-tiny's pretrain config, just enough for the helpers under test.
NUM_EXPERTS = 128
INTERVAL = 4  # inhomogeneous_layer_cycle_interval
NUM_DENSE_PREFIX = 1  # first_num_dense_layers
NUM_MOE_PREFIX = INTERVAL - NUM_DENSE_PREFIX  # 3
SCAN_LENGTH = 5
NUM_DECODER_LAYERS = NUM_DENSE_PREFIX + NUM_MOE_PREFIX + SCAN_LENGTH * INTERVAL  # 24
NUM_BACKBONE_MOE = NUM_MOE_PREFIX + SCAN_LENGTH * INTERVAL  # 23


def make_ling3_config(scan_layers: bool, mtp_num_layers: int = 0) -> SimpleNamespace:
  return SimpleNamespace(
      decoder_block=DecoderBlockType.LING3,
      scan_layers=scan_layers,
      num_decoder_layers=NUM_DECODER_LAYERS,
      first_num_dense_layers=NUM_DENSE_PREFIX,
      inhomogeneous_layer_cycle_interval=INTERVAL,
      num_experts=NUM_EXPERTS,
      mtp_num_layers=mtp_num_layers,
      routed_bias=True,
      routed_bias_update_rate=0.001,
      routed_bias_zero_mean_update=True,
  )


def _wrap_sow(value):
  """Mimic Flax sow() accumulation: a single sow yields a 1-tuple."""
  return (value,)


def make_unscan_intermediates(per_layer_lb: float = 0.5, per_layer_z: float = 0.7):
  """23 separate moe_layers_{i}/{moe_lb_loss,moe_z_loss}, each scalar."""
  decoder = {}
  for i in range(NUM_BACKBONE_MOE):
    decoder[f"moe_layers_{i}"] = {
        "moe_lb_loss": _wrap_sow(jnp.float32(per_layer_lb)),
        "moe_z_loss": _wrap_sow(jnp.float32(per_layer_z)),
        "moe_expert_counts": _wrap_sow(jnp.full((NUM_EXPERTS,), float(i + 1), dtype=jnp.float32)),
    }
  return {"intermediates": {"decoder": decoder}}


def make_scan_intermediates(per_layer_lb: float = 0.5, per_layer_z: float = 0.7):
  """LING3 scan layout (the one the model actually produces, see dump_ling3_intermediates.py).

  Phase 1b prefix: 3 separate moe_layers_{0,1,2}/{key}, each scalar.
  Phase 2 scan:    moe_layers/layers_{0..3}/{key}, each shape (SCAN_LENGTH,) for scalars
                                                 or (SCAN_LENGTH, NUM_EXPERTS) for counts.

  Counts here are constructed so the equivalent sums match the unscan layout.
  """
  decoder = {}
  # Track per-layer-index (in unscan ordering) so the SUM matches across modes.
  unscan_idx = 0
  for i in range(NUM_MOE_PREFIX):
    decoder[f"moe_layers_{i}"] = {
        "moe_lb_loss": _wrap_sow(jnp.float32(per_layer_lb)),
        "moe_z_loss": _wrap_sow(jnp.float32(per_layer_z)),
        "moe_expert_counts": _wrap_sow(jnp.full((NUM_EXPERTS,), float(unscan_idx + 1), dtype=jnp.float32)),
    }
    unscan_idx += 1
  scan_subdict = {}
  for j in range(INTERVAL):
    # Each ScannableBlock sub-layer accumulates SCAN_LENGTH copies
    # (one per scan iteration), each contributing per_layer_lb / per_layer_z.
    scan_subdict[f"layers_{j}"] = {
        "moe_lb_loss": _wrap_sow(jnp.full((SCAN_LENGTH,), per_layer_lb, dtype=jnp.float32)),
        "moe_z_loss": _wrap_sow(jnp.full((SCAN_LENGTH,), per_layer_z, dtype=jnp.float32)),
        # counts are (scan_length, num_experts) per sub-layer.
        "moe_expert_counts": _wrap_sow(
            jnp.tile(jnp.arange(1, SCAN_LENGTH + 1, dtype=jnp.float32)[:, None], (1, NUM_EXPERTS))
            * 0  # placeholder — counts equivalence isn't asserted here
        ),
    }
  decoder["moe_layers"] = scan_subdict
  return {"intermediates": {"decoder": decoder}}


def _make_train_state(params):
  return train_state.TrainState(step=0, apply_fn=None, params=params, tx=None, opt_state={})


def make_ling3_scan_param_tree(num_experts: int = NUM_EXPERTS, scan_length: int = SCAN_LENGTH) -> dict:
  """Build the ling3 scan param subtree for routed bias.

  Mirrors the layout discovered by inspect_ling3_scan_ckpt.py. Note Flax wraps
  the actual weight tree under an outer "params" key, which is what the trainer's
  ``target_path = ("params", "decoder", ...)`` convention indexes into:
    params/decoder/moe_layers_{i}/mlp/MoeBlock_0/gate/bias       shape (num_experts,)
    params/decoder/moe_layers/layers_{j}/mlp/MoeBlock_0/gate/bias shape (num_experts, scan_length)
  """
  decoder = {}
  for i in range(NUM_MOE_PREFIX):
    decoder[f"moe_layers_{i}"] = {
        "mlp": {"MoeBlock_0": {"gate": {"bias": jnp.zeros((num_experts,), dtype=jnp.float32)}}}
    }
  decoder["moe_layers"] = {
      f"layers_{j}": {
          "mlp": {"MoeBlock_0": {"gate": {"bias": jnp.zeros((num_experts, scan_length), dtype=jnp.float32)}}}
      }
      for j in range(INTERVAL)
  }
  return {"params": {"decoder": decoder}}


# ---------------------------------------------------------------------------
# T1 — _collect_moe_intermediate_sum equivalence and fail-loud
# ---------------------------------------------------------------------------


class CollectMoeIntermediateSumTest(unittest.TestCase):
  """T1: scan vs unscan must produce identical aux totals when both layouts represent
  the same logical sown data."""

  def test_lb_loss_scan_unscan_equivalence(self):
    cfg_unscan = make_ling3_config(scan_layers=False)
    cfg_scan = make_ling3_config(scan_layers=True)

    # Unscan total: 23 layers * per_layer_lb.
    # Scan total:   3 prefix * per_layer_lb + 4 sub-layers * 5 scan iters * per_layer_lb
    #             = 3 * 0.5 + 20 * 0.5 = 11.5 ✓ matches 23 * 0.5
    unscan_sum = float(train_mod._collect_moe_intermediate_sum(  # pylint: disable=protected-access
        cfg_unscan, make_unscan_intermediates(), "moe_lb_loss"
    ))
    scan_sum = float(train_mod._collect_moe_intermediate_sum(  # pylint: disable=protected-access
        cfg_scan, make_scan_intermediates(), "moe_lb_loss"
    ))
    self.assertAlmostEqual(unscan_sum, NUM_BACKBONE_MOE * 0.5, places=5)
    self.assertAlmostEqual(scan_sum, NUM_BACKBONE_MOE * 0.5, places=5)
    self.assertAlmostEqual(unscan_sum, scan_sum, places=5)

  def test_z_loss_scan_unscan_equivalence(self):
    cfg_unscan = make_ling3_config(scan_layers=False)
    cfg_scan = make_ling3_config(scan_layers=True)
    unscan_sum = float(train_mod._collect_moe_intermediate_sum(  # pylint: disable=protected-access
        cfg_unscan, make_unscan_intermediates(), "moe_z_loss"
    ))
    scan_sum = float(train_mod._collect_moe_intermediate_sum(  # pylint: disable=protected-access
        cfg_scan, make_scan_intermediates(), "moe_z_loss"
    ))
    self.assertAlmostEqual(unscan_sum, scan_sum, places=5)
    self.assertAlmostEqual(scan_sum, NUM_BACKBONE_MOE * 0.7, places=5)

  def test_collect_raises_on_missing_phase2_path(self):
    """If a Phase 2 sub-layer is missing, the collector must raise — not silent 0."""
    cfg_scan = make_ling3_config(scan_layers=True)
    intermediates = make_scan_intermediates()
    # Drop one sub-layer to simulate layout drift.
    del intermediates["intermediates"]["decoder"]["moe_layers"]["layers_2"]
    with self.assertRaisesRegex(RuntimeError, r"layers_2"):
      train_mod._collect_moe_intermediate_sum(  # pylint: disable=protected-access
          cfg_scan, intermediates, "moe_lb_loss"
      )

  def test_collect_raises_on_missing_phase1b_path(self):
    cfg_scan = make_ling3_config(scan_layers=True)
    intermediates = make_scan_intermediates()
    del intermediates["intermediates"]["decoder"]["moe_layers_1"]
    with self.assertRaisesRegex(RuntimeError, r"moe_layers_1"):
      train_mod._collect_moe_intermediate_sum(  # pylint: disable=protected-access
          cfg_scan, intermediates, "moe_lb_loss"
      )


# ---------------------------------------------------------------------------
# T2 — _update_deepseek_bias shape / zero-mean / directionality
# ---------------------------------------------------------------------------


class UpdateLing3BiasScanTest(unittest.TestCase):
  """T2: scan-mode bias update must touch all 3 Phase 1b + 4 Phase 2 sub-layers,
  produce shape-correct deltas, zero-mean along the right axis, and push
  hot experts in the negative direction."""

  def setUp(self):
    self.cfg = make_ling3_config(scan_layers=True)
    self.params = make_ling3_scan_param_tree()
    self.state = _make_train_state(self.params)

  def _build_counts(self):
    """Phase 1b counts are (num_experts,); Phase 2 counts are (scan_length, num_experts).

    Construct counts with one hot expert (index 7) per layer so we can assert
    directionality.
    """
    hot = 7
    base = jnp.ones((NUM_EXPERTS,), dtype=jnp.float32)
    hot_1d = base.at[hot].set(NUM_EXPERTS * 10.0)  # massively overloaded
    prefix = [_wrap_sow(hot_1d) for _ in range(NUM_MOE_PREFIX)]
    base_2d = jnp.ones((SCAN_LENGTH, NUM_EXPERTS), dtype=jnp.float32)
    hot_2d = base_2d.at[:, hot].set(NUM_EXPERTS * 10.0)
    scan = [_wrap_sow(hot_2d) for _ in range(INTERVAL)]
    return {"prefix": prefix, "scan": scan}, hot

  def test_all_layers_updated_and_shapes_correct(self):
    counts, _ = self._build_counts()
    new_state = train_mod._update_ling3_scan_bias(  # pylint: disable=protected-access
        self.cfg, self.state, "mlp", counts
    )

    # Phase 1b: bias becomes non-zero
    for i in range(NUM_MOE_PREFIX):
      old = self.state.params["params"]["decoder"][f"moe_layers_{i}"]["mlp"]["MoeBlock_0"]["gate"]["bias"]  # pylint: disable=unsubscriptable-object
      new = new_state.params["params"]["decoder"][f"moe_layers_{i}"]["mlp"]["MoeBlock_0"]["gate"]["bias"]
      self.assertEqual(new.shape, (NUM_EXPERTS,))
      self.assertGreater(float(jnp.linalg.norm(new - old)), 0.0,
                         f"prefix moe_layers_{i} bias did not change")

    # Phase 2: bias shape (num_experts, scan_length); update applied per scan slice
    for j in range(INTERVAL):
      old = self.state.params["params"]["decoder"]["moe_layers"][f"layers_{j}"]["mlp"]["MoeBlock_0"]["gate"]["bias"]  # pylint: disable=unsubscriptable-object
      new = new_state.params["params"]["decoder"]["moe_layers"][f"layers_{j}"]["mlp"]["MoeBlock_0"]["gate"]["bias"]
      self.assertEqual(new.shape, (NUM_EXPERTS, SCAN_LENGTH))
      self.assertGreater(float(jnp.linalg.norm(new - old)), 0.0,
                         f"scan layers_{j} bias did not change")

  def test_phase2_zero_mean_axis_is_experts(self):
    """zero_mean_update must center along the expert axis (axis=0 for (num_experts, scan_length))."""
    counts, _ = self._build_counts()
    # Pre-perturb the Phase 2 bias so means are non-zero across both axes.
    rng = jax.random.PRNGKey(7)
    biased_init = jax.random.normal(rng, (NUM_EXPERTS, SCAN_LENGTH), dtype=jnp.float32) + 3.0
    perturbed_params = jax.tree_util.tree_map(lambda x: x, self.params)
    for j in range(INTERVAL):
      perturbed_params["params"]["decoder"]["moe_layers"][f"layers_{j}"]["mlp"]["MoeBlock_0"]["gate"]["bias"] = biased_init
    state = _make_train_state(perturbed_params)
    new_state = train_mod._update_ling3_scan_bias(  # pylint: disable=protected-access
        self.cfg, state, "mlp", counts
    )
    for j in range(INTERVAL):
      new_bias = new_state.params["params"]["decoder"]["moe_layers"][f"layers_{j}"]["mlp"]["MoeBlock_0"]["gate"]["bias"]
      mean_over_experts = jnp.mean(new_bias, axis=0)  # shape (scan_length,)
      mean_over_scan = jnp.mean(new_bias, axis=-1)    # shape (num_experts,)
      np.testing.assert_allclose(np.asarray(mean_over_experts), 0.0, atol=1e-5,
                                 err_msg=f"layers_{j} not zero-mean over expert axis")
      # If somebody flips zero_mean_axis back to -1, this expert-axis mean would still be ~0
      # only by coincidence; the scan-axis mean is what stays non-zero with the correct fix.
      self.assertGreater(float(jnp.max(jnp.abs(mean_over_scan))), 1e-3,
                         f"layers_{j} unexpectedly zero-mean over scan axis — "
                         "likely zero_mean_axis was flipped back to -1")

  def test_hot_expert_receives_negative_delta(self):
    """A hot expert should receive a negative bias update (loss-free balancing direction)."""
    counts, hot = self._build_counts()
    new_state = train_mod._update_ling3_scan_bias(  # pylint: disable=protected-access
        self.cfg, self.state, "mlp", counts
    )
    for i in range(NUM_MOE_PREFIX):
      bias = new_state.params["params"]["decoder"][f"moe_layers_{i}"]["mlp"]["MoeBlock_0"]["gate"]["bias"]
      # With zero-mean enabled, the *relative* delta of the hot expert is negative.
      self.assertLess(float(bias[hot]), float(jnp.mean(bias)),
                      f"prefix layer {i}: hot expert {hot} not below mean")
    for j in range(INTERVAL):
      bias = new_state.params["params"]["decoder"]["moe_layers"][f"layers_{j}"]["mlp"]["MoeBlock_0"]["gate"]["bias"]
      self.assertLess(float(jnp.mean(bias[hot, :])), float(jnp.mean(bias)),
                      f"scan layer {j}: hot expert {hot} not below mean")


# ---------------------------------------------------------------------------
# Fail-loud — _try_update_bias raises when params are missing
# ---------------------------------------------------------------------------


class TryUpdateBiasFailLoudTest(unittest.TestCase):

  def test_missing_required_path_raises(self):
    cfg = make_ling3_config(scan_layers=True)
    params = make_ling3_scan_param_tree()
    # Remove one sub-layer's bias to simulate layout drift.
    del params["params"]["decoder"]["moe_layers"]["layers_1"]
    state = _make_train_state(params)
    counts = _wrap_sow(jnp.ones((SCAN_LENGTH, NUM_EXPERTS), dtype=jnp.float32))
    target = ("params", "decoder", "moe_layers", "layers_1", "mlp", "MoeBlock_0", "gate", "bias")
    with self.assertRaisesRegex(RuntimeError, r"layers_1"):
      train_mod._try_update_bias(  # pylint: disable=protected-access
          cfg, state, target, counts, "ling3_scan_layers_1",
          zero_mean_axis=0, transpose_update=True,
      )

  def test_missing_optional_path_only_logs(self):
    cfg = make_ling3_config(scan_layers=True)
    params = make_ling3_scan_param_tree()
    state = _make_train_state(params)
    counts = _wrap_sow(jnp.ones((NUM_EXPERTS,), dtype=jnp.float32))
    target = ("params", "mtp_block", "mtp_layer_99", "mtp_99_transformer_layer", "mlp",
             "MoeBlock_0", "gate", "bias")
    # Should NOT raise.
    new_state = train_mod._try_update_bias(  # pylint: disable=protected-access
        cfg, state, target, counts, "mtp_layer_99", required=False,
    )
    # State unchanged.
    self.assertIs(new_state, state)


# ---------------------------------------------------------------------------
# update_state_param zero_mean_axis — direct test
# ---------------------------------------------------------------------------


class UpdateStateParamZeroMeanAxisTest(unittest.TestCase):
  """Standalone coverage for the new zero_mean_axis parameter."""

  def setUp(self):
    self.params = {"a": {"b": jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)}}
    self.state = _make_train_state(self.params)

  def test_default_axis_minus1_back_compat(self):
    from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel
    update = jnp.zeros_like(self.params["a"]["b"])
    new_state = maxtext_utils.update_state_param(
        self.state, ("a", "b"), update, zero_mean_update=True
    )
    new_b = new_state.params["a"]["b"]
    # axis=-1 means each row sums to 0
    np.testing.assert_allclose(np.asarray(jnp.mean(new_b, axis=-1)), 0.0, atol=1e-6)
    # axis=0 means generally NOT zero
    self.assertGreater(float(jnp.max(jnp.abs(jnp.mean(new_b, axis=0)))), 1e-3)

  def test_axis0_for_2d_bias(self):
    from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel
    update = jnp.zeros_like(self.params["a"]["b"])
    new_state = maxtext_utils.update_state_param(
        self.state, ("a", "b"), update, zero_mean_update=True, zero_mean_axis=0
    )
    new_b = new_state.params["a"]["b"]
    np.testing.assert_allclose(np.asarray(jnp.mean(new_b, axis=0)), 0.0, atol=1e-6)
    self.assertGreater(float(jnp.max(jnp.abs(jnp.mean(new_b, axis=-1)))), 1e-3)


if __name__ == "__main__":
  unittest.main()
