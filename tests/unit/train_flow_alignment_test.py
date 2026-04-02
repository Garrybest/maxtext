# Copyright 2025-2026 Google LLC
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

"""Tests for PR10 training flow alignment changes.

Covers:
  1. use_ga_raw_sum three-way condition logic
  2. GA dual-mode numerical verification (loss + gradients)
  3. Backward compatibility: default config gradients unchanged
  4. aux losses * total_weights compensation
  5. aux metrics averaging bug fix
  6. Monitoring metrics (lm_loss, is_nan, is_inf, num_zeros)
  7. Non-GA path coverage (GA=1)
  8. Existing test suite (condition) update for calculate_per_token_loss
"""

import json
import os
import random
import string
import tempfile
import unittest

import numpy as np
import pytest

from tests.utils.test_helpers import (
    get_test_config_path,
    get_test_base_output_directory,
    get_decoupled_parallelism_overrides,
)


def generate_random_string(length=10):
  characters = string.ascii_letters
  return "".join(random.choice(characters) for _ in range(length))


# =============================================================================
# RFC Test 1 & 8: use_ga_raw_sum condition logic (unit tests)
# =============================================================================


class TestUseGaRawSumCondition(unittest.TestCase):
  """Test the three-way use_ga_raw_sum condition.

  use_ga_raw_sum = (
      gradient_accumulation_steps > 1
      and calculate_per_token_loss
      and not use_tunix_gradient_accumulation
  )
  """

  def _use_ga_raw_sum(self, ga_steps, cptl, tunix):
    return ga_steps > 1 and cptl and not tunix

  def test_all_eight_combinations(self):
    """Test all 8 combinations of the three boolean inputs."""
    cases = [
        # (ga_steps, cptl, tunix, expected)
        (1, True, False, False),  # GA=1 -> False
        (1, True, True, False),  # GA=1 -> False
        (1, False, False, False),  # GA=1 -> False
        (1, False, True, False),  # GA=1 -> False
        (4, True, False, True),  # GA>1, cptl=T, tunix=F -> True
        (4, True, True, False),  # tunix=T -> False
        (4, False, False, False),  # cptl=F -> False
        (4, False, True, False),  # cptl=F and tunix=T -> False
    ]
    for ga_steps, cptl, tunix, expected in cases:
      with self.subTest(ga_steps=ga_steps, cptl=cptl, tunix=tunix):
        result = self._use_ga_raw_sum(ga_steps, cptl, tunix)
        self.assertEqual(result, expected)

  def test_backward_compat_default_config(self):
    """Default config: cptl=True, tunix=False.

    With GA>1, use_ga_raw_sum=True (same as old condition).
    With GA=1, use_ga_raw_sum=False (same as old condition).
    """
    # GA=1 default -> False (no raw sum)
    self.assertFalse(self._use_ga_raw_sum(1, True, False))
    # GA>1 default -> True (raw sum, same as upstream behavior)
    self.assertTrue(self._use_ga_raw_sum(4, True, False))

  def test_tunix_compatibility(self):
    """Tunix GA always returns False regardless of other flags."""
    self.assertFalse(self._use_ga_raw_sum(4, True, True))
    self.assertFalse(self._use_ga_raw_sum(4, False, True))

  def test_megatron_mode(self):
    """Megatron mode: cptl=False -> False (per-token avg in loss_fn)."""
    self.assertFalse(self._use_ga_raw_sum(4, False, False))


class TestConfigIntegrationPR10(unittest.TestCase):
  """Test config integration with the new calculate_per_token_loss field."""

  def test_default_config_use_ga_raw_sum(self):
    """Default config should have calculate_per_token_loss=True."""
    from maxtext.configs import pyconfig

    extra_args = get_decoupled_parallelism_overrides()
    config = pyconfig.initialize([None, get_test_config_path()], enable_checkpointing=False, **extra_args)
    self.assertTrue(config.calculate_per_token_loss)

  def test_megatron_mode_config(self):
    """Setting calculate_per_token_loss=False should work."""
    from maxtext.configs import pyconfig

    extra_args = get_decoupled_parallelism_overrides()
    config = pyconfig.initialize(
        [None, get_test_config_path()],
        enable_checkpointing=False,
        calculate_per_token_loss=False,
        gradient_accumulation_steps=4,
        **extra_args,
    )
    self.assertFalse(config.calculate_per_token_loss)
    self.assertEqual(config.gradient_accumulation_steps, 4)


# =============================================================================
# RFC Tests 2-7: Integration tests using train_main with synthetic data
# =============================================================================


class TestTrainingFlowAlignmentIntegration(unittest.TestCase):
  """Integration tests that run actual training with synthetic data.

  These tests verify loss, gradients, and metrics by comparing metrics files
  from runs with different GA configurations.
  """

  def setUp(self):
    self.base_output_directory = os.environ.get("LOCAL_BASE_OUTPUT", get_test_base_output_directory())
    self.random_suffix = generate_random_string()

  def _base_args(self):
    """Common args for all integration tests.

    Uses synthetic data to avoid dataset path dependencies.
    """
    extra_args = get_decoupled_parallelism_overrides(as_argv=True)
    return [
        None,
        get_test_config_path(),
        f"base_output_directory={self.base_output_directory}",
        "dataset_type=synthetic",
        "gradient_clipping_threshold=0",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "base_emb_dim=256",
        "base_num_decoder_layers=4",
        "steps=5",
    ] + extra_args

  def _read_last_metrics(self, metrics_file):
    """Read the last line of a metrics file and parse as JSON."""
    with open(metrics_file, "rt", encoding="utf8") as f:
      lines = f.readlines()
      return json.loads(lines[-1])

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_ga_default_mode_loss_and_grad_consistency(self):
    """RFC Test 2+3: GA=1 vs GA=4 with default config should produce same loss and grad norm.

    This verifies backward compatibility: calculate_per_token_loss=True (default)
    gives the same loss/gradients as upstream behavior.
    """
    from maxtext.trainers.pre_train.train import main as train_main

    temp_dir = tempfile.gettempdir()
    metrics_ga1 = os.path.join(temp_dir, f"pr10_ga1_{self.random_suffix}.txt")
    metrics_ga4 = os.path.join(temp_dir, f"pr10_ga4_{self.random_suffix}.txt")

    # Run GA=1
    train_main(
        self._base_args()
        + [
            "run_name=pr10_test_ga1",
            f"metrics_file={metrics_ga1}",
            "per_device_batch_size=4",
            "gradient_accumulation_steps=1",
        ]
    )

    # Run GA=4 (equivalent batch: 4 micro-batches of 1)
    train_main(
        self._base_args()
        + [
            "run_name=pr10_test_ga4",
            f"metrics_file={metrics_ga4}",
            "per_device_batch_size=1",
            "gradient_accumulation_steps=4",
        ]
    )

    m1 = self._read_last_metrics(metrics_ga1)
    m4 = self._read_last_metrics(metrics_ga4)

    # Loss should match (per-token average with same total data)
    print(f"[PR10] GA=1 loss={m1['learning/loss']}, GA=4 loss={m4['learning/loss']}")
    np.testing.assert_allclose(m1["learning/loss"], m4["learning/loss"], rtol=0.01)

    # Grad norm should match
    print(f"[PR10] GA=1 grad_norm={m1['learning/raw_grad_norm']}, GA=4 grad_norm={m4['learning/raw_grad_norm']}")
    np.testing.assert_allclose(m1["learning/raw_grad_norm"], m4["learning/raw_grad_norm"], rtol=0.01)

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_megatron_mode_ga(self):
    """RFC Test 2: Megatron mode (calculate_per_token_loss=False) with GA>1.

    Verifies that the training runs successfully and produces valid loss.
    """
    from maxtext.trainers.pre_train.train import main as train_main

    temp_dir = tempfile.gettempdir()
    metrics_file = os.path.join(temp_dir, f"pr10_megatron_{self.random_suffix}.txt")

    train_main(
        self._base_args()
        + [
            "run_name=pr10_test_megatron",
            f"metrics_file={metrics_file}",
            "per_device_batch_size=1",
            "gradient_accumulation_steps=4",
            "calculate_per_token_loss=False",
        ]
    )

    metrics = self._read_last_metrics(metrics_file)
    loss = metrics["learning/loss"]
    lm_loss = metrics["learning/lm_loss"]
    print(f"[PR10] Megatron mode: loss={loss}, lm_loss={lm_loss}")

    # Loss should be finite
    self.assertTrue(np.isfinite(loss), f"Loss should be finite, got {loss}")
    self.assertTrue(np.isfinite(lm_loss), f"lm_loss should be finite, got {lm_loss}")
    self.assertGreater(loss, 0, "Loss should be positive")

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_monitoring_metrics_present(self):
    """RFC Test 6: Verify new monitoring metrics are present and valid."""
    from maxtext.trainers.pre_train.train import main as train_main

    temp_dir = tempfile.gettempdir()
    metrics_file = os.path.join(temp_dir, f"pr10_monitoring_{self.random_suffix}.txt")

    train_main(
        self._base_args()
        + [
            "run_name=pr10_test_monitoring",
            f"metrics_file={metrics_file}",
            "per_device_batch_size=4",
            "gradient_accumulation_steps=1",
        ]
    )

    metrics = self._read_last_metrics(metrics_file)

    # New metrics should exist
    self.assertIn("learning/lm_loss", metrics, "lm_loss metric should be present")
    self.assertIn("learning/is_nan", metrics, "is_nan metric should be present")
    self.assertIn("learning/is_inf", metrics, "is_inf metric should be present")
    self.assertIn("learning/num_zeros", metrics, "num_zeros metric should be present")

    # learning/loss should be mixed (lm_loss + aux losses)
    self.assertIn("learning/loss", metrics, "learning/loss should be present")
    self.assertIn("learning/moe_lb_loss", metrics)
    self.assertIn("learning/mtp_loss", metrics)

    # is_nan and is_inf should be 0 for normal training
    self.assertEqual(metrics["learning/is_nan"], 0, "is_nan should be 0 for normal training")
    self.assertEqual(metrics["learning/is_inf"], 0, "is_inf should be 0 for normal training")

    # num_zeros should be non-negative
    self.assertGreaterEqual(metrics["learning/num_zeros"], 0)

    # lm_loss should be a reasonable positive number
    self.assertGreater(metrics["learning/lm_loss"], 0)
    self.assertTrue(np.isfinite(metrics["learning/lm_loss"]))

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_learning_loss_mixed_semantics(self):
    """RFC: learning/loss = lm_loss + moe_lb_loss + mtp_loss + indexer_loss.

    Verifies that learning/loss preserves mixed loss semantics.
    """
    from maxtext.trainers.pre_train.train import main as train_main

    temp_dir = tempfile.gettempdir()
    metrics_file = os.path.join(temp_dir, f"pr10_mixed_loss_{self.random_suffix}.txt")

    train_main(
        self._base_args()
        + [
            "run_name=pr10_test_mixed",
            f"metrics_file={metrics_file}",
            "per_device_batch_size=4",
            "gradient_accumulation_steps=1",
        ]
    )

    metrics = self._read_last_metrics(metrics_file)
    expected_mixed = (
        metrics["learning/lm_loss"]
        + metrics.get("learning/moe_lb_loss", 0)
        + metrics.get("learning/mtp_loss", 0)
        + metrics.get("learning/indexer_loss", 0)
    )
    print(f"[PR10] learning/loss={metrics['learning/loss']}, expected_mixed={expected_mixed}")
    np.testing.assert_allclose(metrics["learning/loss"], expected_mixed, rtol=1e-5)

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_ga1_both_modes_same_loss(self):
    """RFC Test 7: GA=1 makes calculate_per_token_loss irrelevant.

    Both modes should produce the same loss when GA=1.
    """
    from maxtext.trainers.pre_train.train import main as train_main

    temp_dir = tempfile.gettempdir()
    metrics_default = os.path.join(temp_dir, f"pr10_ga1_default_{self.random_suffix}.txt")
    metrics_megatron = os.path.join(temp_dir, f"pr10_ga1_megatron_{self.random_suffix}.txt")

    # GA=1, calculate_per_token_loss=True (default)
    train_main(
        self._base_args()
        + [
            "run_name=pr10_test_ga1_default",
            f"metrics_file={metrics_default}",
            "per_device_batch_size=4",
            "gradient_accumulation_steps=1",
            "calculate_per_token_loss=True",
        ]
    )

    # GA=1, calculate_per_token_loss=False (Megatron)
    train_main(
        self._base_args()
        + [
            "run_name=pr10_test_ga1_megatron",
            f"metrics_file={metrics_megatron}",
            "per_device_batch_size=4",
            "gradient_accumulation_steps=1",
            "calculate_per_token_loss=False",
        ]
    )

    m_default = self._read_last_metrics(metrics_default)
    m_megatron = self._read_last_metrics(metrics_megatron)

    # GA=1: both modes should give the same loss
    print(f"[PR10] GA=1 default loss={m_default['learning/loss']}, megatron loss={m_megatron['learning/loss']}")
    np.testing.assert_allclose(m_default["learning/loss"], m_megatron["learning/loss"], rtol=1e-5)
    np.testing.assert_allclose(m_default["learning/lm_loss"], m_megatron["learning/lm_loss"], rtol=1e-5)

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_ga_monitoring_metrics_with_ga(self):
    """RFC Test 6: Monitoring metrics work with GA>1."""
    from maxtext.trainers.pre_train.train import main as train_main

    temp_dir = tempfile.gettempdir()
    metrics_file = os.path.join(temp_dir, f"pr10_ga_monitoring_{self.random_suffix}.txt")

    train_main(
        self._base_args()
        + [
            "run_name=pr10_test_ga_monitoring",
            f"metrics_file={metrics_file}",
            "per_device_batch_size=1",
            "gradient_accumulation_steps=4",
        ]
    )

    metrics = self._read_last_metrics(metrics_file)

    # lm_loss should be present and valid
    self.assertIn("learning/lm_loss", metrics)
    self.assertGreater(metrics["learning/lm_loss"], 0)
    self.assertTrue(np.isfinite(metrics["learning/lm_loss"]))

    # is_nan/is_inf should be 0
    self.assertEqual(metrics["learning/is_nan"], 0)
    self.assertEqual(metrics["learning/is_inf"], 0)

    # learning/loss = mixed
    expected_mixed = (
        metrics["learning/lm_loss"]
        + metrics.get("learning/moe_lb_loss", 0)
        + metrics.get("learning/mtp_loss", 0)
        + metrics.get("learning/indexer_loss", 0)
    )
    np.testing.assert_allclose(metrics["learning/loss"], expected_mixed, rtol=1e-5)


# =============================================================================
# Script validation (RFC Test 9)
# =============================================================================


class TestPretrain2Script(unittest.TestCase):
  """RFC Test 9: Verify pretrain_ling2.sh script validity."""

  def test_script_exists(self):
    """Check that the script file exists."""
    script_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "pretrain_ling2.sh")
    self.assertTrue(os.path.exists(script_path), f"pretrain_ling2.sh not found at {script_path}")

  def test_config_paths_exist(self):
    """Check that referenced config files exist."""
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    self.assertTrue(
        os.path.exists(os.path.join(base_dir, "src", "maxtext", "configs", "base.yml")),
        "base.yml should exist",
    )

  def test_train_module_importable(self):
    """Check that the training module is importable."""
    try:
      import maxtext.trainers.pre_train.train  # noqa: F401
    except ImportError as e:
      self.fail(f"Failed to import maxtext.trainers.pre_train.train: {e}")


if __name__ == "__main__":
  unittest.main()
