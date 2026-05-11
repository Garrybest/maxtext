#!/bin/bash
# Submit Ling3 Tiny Pretrain job to GKE (64 chips / 128 devices, 4x4x4)
#
# Usage:
#   scripts/submit_ling3_64chips_job.sh                    # use current branch
#   scripts/submit_ling3_64chips_job.sh feat/my-branch     # specify branch
#   STEPS=1000 scripts/submit_ling3_64chips_job.sh         # override defaults
#
# Prerequisites:
#   - kubectl configured with GKE cluster credentials
#   - K8s secret 'perf-64chips-token' with a valid GitHub PAT (ghp_...)
#
# NOTE on envsubst: The job YAML mixes deploy-time variables (K8s metadata
# and env value: fields) with runtime variables used in the inline bash
# script (GITHUB_TOKEN, JOB_COMPLETION_INDEX, LOG_FILE, RUN_NAME).
# We MUST use an explicit variable whitelist so envsubst only substitutes
# deploy-time variables and preserves runtime ones.

set -euo pipefail

# ============================================================================
# 1. Branch & Job Name
# ============================================================================
BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
if [ -z "${JOB_NAME:-}" ]; then
  JOB_NAME="ling3-64-$(echo "$BRANCH" | sed 's/[^a-zA-Z0-9]/-/g' | head -c 27)-$(date +%m%d%H%M)"
  JOB_NAME=$(echo "$JOB_NAME" | tr '[:upper:]' '[:lower:]' | head -c 63)
fi
BRANCH_LABEL=$(echo "$BRANCH" | sed 's/[^a-zA-Z0-9._-]/-/g' | head -c 63 | sed 's/^[^a-zA-Z0-9]//;s/[^a-zA-Z0-9]$//')
USER="${USER:-$(whoami)}"

# ============================================================================
# 2. Training Parameters (override via environment)
# ============================================================================
export STEPS="${STEPS:-1000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
export OPT_TYPE="${OPT_TYPE:-muon}"

# Dataset selection (default: opensource via grain)
#   Opensource: DATASET_TYPE=grain  DATASETS_YAML=""
#   Ant data:   DATASET_TYPE=lazy   DATASETS_YAML=scripts/datasets/ant_datasets_dev.yml
export DATASET_TYPE="${DATASET_TYPE:-grain}"
export DATASETS_YAML="${DATASETS_YAML:-}"

# Parallelism (16 hosts × 8 devices = 128 devices total)
# Default: EP=1, FSDP=32, DP=4 — 1 × 32 × 4 = 128 ✓
export ICI_EXPERT_PARALLELISM="${ICI_EXPERT_PARALLELISM:-1}"
export ICI_DATA_PARALLELISM="${ICI_DATA_PARALLELISM:-4}"
export ICI_FSDP_PARALLELISM="${ICI_FSDP_PARALLELISM:-32}"
export ICI_TENSOR_PARALLELISM="${ICI_TENSOR_PARALLELISM:-1}"
export ICI_CONTEXT_PARALLELISM="${ICI_CONTEXT_PARALLELISM:-1}"
export SHARD_EXP_ON_FSDP="${SHARD_EXP_ON_FSDP:-true}"

# Performance
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export REMAT_POLICY="${REMAT_POLICY:-save_out_proj}"

# Lazy dataloader scatter group sharding.
# -1 = single-host debug: every process reads ALL shards.
# -4 = production: 4 scatter groups, each process reads 1/4 of shards.
#       Requires num_hosts to be a multiple of 4. (16 hosts ✓)
# WARNING: -1 and -4 produce DIFFERENT data consumption order — results
# from single-host debug (-1) are NOT comparable with production (-4).
# Default: -4 (production for the 16-host 64-chip job).
export LAZY_LOADER_SCATTER="${LAZY_LOADER_SCATTER:--4}"

# Profiler (empty = disabled)
export PROFILER="${PROFILER:-}"
export SKIP_FIRST_N_STEPS_FOR_PROFILER="${SKIP_FIRST_N_STEPS_FOR_PROFILER:-}"
export PROFILER_STEPS="${PROFILER_STEPS:-}"

# Extra XLA flags (empty = use script defaults only)
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-}"

export JOB_NAME BRANCH BRANCH_LABEL USER

# ============================================================================
# 3. Submit
# ============================================================================
TEMPLATE=".github/ci/tpu-64chips-ling3-job.yaml"
if [ ! -f "$TEMPLATE" ]; then
  echo "ERROR: Template not found: $TEMPLATE" >&2
  echo "Run this script from the repo root." >&2
  exit 1
fi

# Explicit whitelist: only substitute deploy-time variables.
# Runtime variables (GITHUB_TOKEN, JOB_COMPLETION_INDEX, LOG_FILE, RUN_NAME)
# are preserved for the pod's bash script to resolve at execution time.
SUBST_VARS='$JOB_NAME $BRANCH $BRANCH_LABEL $USER'
SUBST_VARS+=' $STEPS $EVAL_INTERVAL $OPT_TYPE'
SUBST_VARS+=' $DATASET_TYPE $DATASETS_YAML'
SUBST_VARS+=' $ICI_EXPERT_PARALLELISM $ICI_DATA_PARALLELISM $ICI_FSDP_PARALLELISM'
SUBST_VARS+=' $ICI_TENSOR_PARALLELISM $ICI_CONTEXT_PARALLELISM $SHARD_EXP_ON_FSDP'
SUBST_VARS+=' $PER_DEVICE_BATCH_SIZE $GRADIENT_ACCUMULATION_STEPS $REMAT_POLICY'
SUBST_VARS+=' $LAZY_LOADER_SCATTER'
SUBST_VARS+=' $PROFILER $SKIP_FIRST_N_STEPS_FOR_PROFILER $PROFILER_STEPS'
SUBST_VARS+=' $LIBTPU_INIT_ARGS'

echo "=== Submitting Ling3 64-chips Job ==="
echo "  JobSet:     $JOB_NAME"
echo "  Branch:     $BRANCH"
echo "  Topology:   4x4x4 (64 chips / 128 devices, 16 hosts)"
echo "  Optimizer:  $OPT_TYPE"
echo "  Dataset:    $DATASET_TYPE${DATASETS_YAML:+ ($DATASETS_YAML)}"
echo "  Steps:      $STEPS"
echo "  Eval:       every $EVAL_INTERVAL steps"
echo "  Batch:      $PER_DEVICE_BATCH_SIZE per device × $GRADIENT_ACCUMULATION_STEPS accum"
echo "  Parallelism: EP=$ICI_EXPERT_PARALLELISM DP=$ICI_DATA_PARALLELISM FSDP=$ICI_FSDP_PARALLELISM TP=$ICI_TENSOR_PARALLELISM CP=$ICI_CONTEXT_PARALLELISM"
echo "  ShardExpFSDP: $SHARD_EXP_ON_FSDP"
echo "  Remat:      $REMAT_POLICY"
echo "  Scatter:    $LAZY_LOADER_SCATTER"
echo "====================================="

envsubst "$SUBST_VARS" < "$TEMPLATE" | kubectl apply -f -

echo ""
echo "Monitor with:"
echo "  kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$JOB_NAME -w"
echo "  kubectl logs -f job/$JOB_NAME-worker-0 -c jax-tpu"
echo ""
echo "Cleanup:"
echo "  kubectl delete jobset $JOB_NAME"
