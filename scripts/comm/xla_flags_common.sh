#!/bin/bash
# Common XLA Flags for TPU Training Scripts
# This file contains shared LIBTPU XLA flags to avoid duplication across training scripts.
# Source this file in your training script: source "$(dirname "${BASH_SOURCE[0]}")/comm/xla_flags_common.sh"
#
# Usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/comm/xla_flags_common.sh"
#   (This will automatically export LIBTPU_INIT_ARGS)
#   echo "   LIBTPU_INIT_ARGS: $LIBTPU_INIT_ARGS"

# ============================================================================
# LIBTPU XLA flags organized by functional groups for clarity and maintenance
# ============================================================================

# --- Async Collective Fusion ---
# NOTE: Async collective fusion conflicts with Sparse Core (SC) offload.
# When Sparse Core is enabled (xla_tpu_enable_sparse_core_collective_offload_*=true),
# these flags must all be set to 'false' to avoid conflicts.
# SC offload and async fusion cannot be used simultaneously.
LIBTPU_ASYNC_FUSION="\
--xla_tpu_enable_async_collective_fusion=false \
--xla_tpu_enable_async_collective_fusion_multiple_steps=false \
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false \
--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false \
--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false"

# --- Async Collective Operations ---
LIBTPU_ASYNC_OPS="\
--xla_enable_async_all_gather=true \
--xla_enable_async_collective_permute=true"

# --- Scheduler & Compute-Collective Overlap ---
LIBTPU_SCHEDULER="\
--xla_tpu_enable_all_experimental_scheduler_features=true \
--xla_tpu_overlap_compute_collective_tc=false"

# --- Sparse Core (SC) Configuration ---
LIBTPU_SPARSE_CORE="\
--xla_tpu_use_tc_device_shape_on_sc=true \
--xla_sc_enable_instruction_fusion=false \
--xla_sc_disjoint_spmem=false \
--xla_sc_disable_megacore_partitioning=true \
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true"

# --- Memory & Power Management ---
LIBTPU_MEMORY_POWER="\
--xla_tpu_scoped_vmem_limit_kib=65536 \
--xla_tpu_dvfs_p_state=7"

# --- Data Parallel Optimization ---
LIBTPU_DATA_PARALLEL="\
--xla_tpu_data_parallel_opt_different_sized_ops=true \
--xla_tpu_enable_data_parallel_all_reduce_opt=true"

# --- Combine all LIBTPU args into a single default variable ---
LIBTPU_INIT_ARGS_DEFAULT="${LIBTPU_ASYNC_FUSION} ${LIBTPU_ASYNC_OPS} ${LIBTPU_SCHEDULER} ${LIBTPU_SPARSE_CORE} ${LIBTPU_MEMORY_POWER} ${LIBTPU_DATA_PARALLEL}"

# --- Export LIBTPU_INIT_ARGS combining defaults with any user-provided values ---
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS_DEFAULT} ${LIBTPU_INIT_ARGS:-}"
