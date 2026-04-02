#!/usr/bin/env bash
# CI Smoke Test: Minimal MaxText training with synthetic data.
# Validates that the training pipeline can run end-to-end.
#
# Environment variables (optional overrides):
#   STEPS       - Number of training steps (default: 10)
#   RUN_NAME    - Run name for output directory (default: ci-smoke-test)

set -euo pipefail

STEPS="${STEPS:-10}"
RUN_NAME="${RUN_NAME:-ci-smoke-test}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ramdisk/ci_output}"

echo "========================================================"
echo "MaxText CI Smoke Test"
echo "  steps:      ${STEPS}"
echo "  run_name:   ${RUN_NAME}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "========================================================"

python3 -m maxtext.trainers.pre_train.train \
  src/maxtext/configs/base.yml \
  run_name="${RUN_NAME}" \
  steps="${STEPS}" \
  per_device_batch_size=1 \
  max_target_length=128 \
  dataset_type=synthetic \
  base_output_directory="${OUTPUT_DIR}"

echo "========================================================"
echo "Smoke test completed successfully!"
echo "========================================================"
