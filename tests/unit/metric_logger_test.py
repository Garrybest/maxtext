"""Tests for MFU setup and emission in MetricLogger."""

import unittest
from types import SimpleNamespace
from unittest import mock

from maxtext.common.metric_logger import MetadataKey, MetricLogger


def _make_config(**overrides):
  """Build a minimal config SimpleNamespace for MetricLogger construction."""
  defaults = {
      "tensorboard_dir": "/tmp/test_tb",
      "run_name": "test_run",
      "gcs_metrics": False,
      "managed_mldiagnostics": False,
      "report_heartbeat_metric_for_gcp_monitoring": False,
      "report_performance_metric_for_gcp_monitoring": False,
      "rampup_end_step": 0,
      "mtp_num_layers": 0,
      "hide_profiler_step_metric": False,
      "profiler": [],
      "skip_first_n_steps_for_profiler": 0,
      "profiler_steps": 0,
      "enable_tensorboard": False,
      "log_period": 10,
      "steps": 100,
      "metrics_file": None,
      "global_batch_size_to_train_on": 1024,
  }
  defaults.update(overrides)
  return SimpleNamespace(**defaults)


def _constant_lr_schedule(step):
  """Test-only LR schedule that ignores step and returns a constant."""
  del step
  return 0.001


def _make_logger(config):
  """Construct a MetricLogger with a mocked TB writer.

  The mock.patch context manager is scoped to the constructor call; once
  MetricLogger is built, ``self.writer`` already holds the MagicMock, so the
  patch can safely unwind on exit.
  """
  with mock.patch(
      "maxtext.common.metric_logger.max_utils.initialize_summary_writer",
      return_value=mock.MagicMock(),
  ):
    return MetricLogger(config, _constant_lr_schedule)


class MFUSetupTest(unittest.TestCase):
  """Test that write_setup_info_to_tensorboard populates the peak metadata."""

  @mock.patch("maxtext.common.metric_logger.get_peak_tflops_per_device", return_value=197.0)
  @mock.patch(
      "maxtext.common.metric_logger.maxtext_utils.calculate_tokens_training_per_device",
      return_value=500.0,
  )
  @mock.patch(
      "maxtext.common.metric_logger.maxtext_utils.calculate_tflops_training_per_device",
      return_value=(100.0, None, None),
  )
  @mock.patch("maxtext.common.metric_logger.maxtext_utils.add_config_to_summary_writer")
  @mock.patch("maxtext.common.metric_logger.max_utils.add_text_to_summary_writer")
  @mock.patch(
      "maxtext.common.metric_logger.max_utils.calculate_num_params_from_pytree",
      return_value=1_000_000_000,
  )
  def test_setup_populates_peak_metadata(
      self,
      _num_params,
      _add_text,
      _add_config,
      _tflops,
      _tokens,
      mock_get_peak,
  ):
    config = _make_config()
    logger = _make_logger(config)

    logger.write_setup_info_to_tensorboard(params={"fake": "pytree"})

    self.assertEqual(logger.metadata[MetadataKey.PEAK_TFLOPS_PER_DEVICE], 197.0)
    mock_get_peak.assert_called_once_with(config)


class MFUEmissionTest(unittest.TestCase):
  """Tests for perf/mfu emission in record_train_metrics."""

  def _logger_with_metadata(self, rampup_end_step, peak):
    config = _make_config(rampup_end_step=rampup_end_step)
    logger = _make_logger(config)
    logger.metadata = {
        MetadataKey.PER_DEVICE_TFLOPS: 100.0,
        MetadataKey.PER_DEVICE_TOKENS: 500.0,
        MetadataKey.PEAK_TFLOPS_PER_DEVICE: peak,
    }
    return logger

  def test_mfu_emitted_when_peak_positive(self):
    logger = self._logger_with_metadata(rampup_end_step=0, peak=197.0)
    metrics = {"scalar": {}}
    logger.record_train_metrics(metrics, step=10, step_time=2.0)

    self.assertIn("perf/mfu", metrics["scalar"])
    # tflops_per_sec = 100.0 / 2.0 = 50.0; mfu = 50.0 / 197.0
    expected_mfu = 50.0 / 197.0
    self.assertAlmostEqual(metrics["scalar"]["perf/mfu"], expected_mfu, places=6)

  def test_mfu_absent_when_peak_zero(self):
    logger = self._logger_with_metadata(rampup_end_step=0, peak=0.0)
    metrics = {"scalar": {}}
    logger.record_train_metrics(metrics, step=10, step_time=2.0)

    self.assertNotIn("perf/mfu", metrics["scalar"])
    # Other perf metrics should still be present.
    self.assertIn("perf/per_device_tflops_per_sec", metrics["scalar"])

  def test_mfu_absent_during_rampup(self):
    # step < rampup_end_step → entire rampup-gated block is skipped.
    logger = self._logger_with_metadata(rampup_end_step=20, peak=197.0)
    metrics = {"scalar": {}}
    logger.record_train_metrics(metrics, step=10, step_time=2.0)

    self.assertNotIn("perf/mfu", metrics["scalar"])
    self.assertNotIn("perf/per_device_tflops_per_sec", metrics["scalar"])
    self.assertNotIn("perf/per_device_tokens_per_sec", metrics["scalar"])


if __name__ == "__main__":
  unittest.main()
