#!/bin/bash
# job_log_sync.sh — Persistent log capture for CI training jobs.
#
# Usage: source scripts/job_log_sync.sh <log_dir>
#   - Sets up LOG_FILE at /tmp/train.log
#   - Copies LOG_FILE to <log_dir>/train.log on EXIT (success or failure)
#
# Example:
#   source scripts/job_log_sync.sh /models/pretrain/ling2/my-job
#   bash train.sh 2>&1 | tee -a "$LOG_FILE"

LOG_DIR="${1:?Usage: source job_log_sync.sh <log_dir>}"
LOG_FILE="/tmp/train.log"

mkdir -p "$LOG_DIR"

trap 'cp "$LOG_FILE" "$LOG_DIR/train-$(date +%Y%m%d-%H%M%S).log" 2>/dev/null || true' EXIT
