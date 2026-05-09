#!/bin/bash
# Ling3 Pretraining Script
# Architecture: Ling3 (Hybrid MLA/KDA + MoE)
# Dataset: Megatron MMap indexed datasets (.bin/.idx)

set -e

# ============================================================================
# 1. Tokenizer Config
# ============================================================================
# Vocabulary size must match the tokenizer used to create the mmap data
# (vocab_size, mmap_eod_id, bos_id are set in configs/models/ling3-tiny.yml)

# ============================================================================
# 2. Basic Environment Config (GCS Bucket & Run Name)
# ============================================================================
BASE_OUTPUT_DIR=${GCS_BUCKET:-"gs://ant-pretrain/pretrain/dev"}
RUN_NAME=${RUN_NAME:-"ling3-pretrain-$(date +%Y%m%d-%H%M)"}
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"

# JAX compilation cache: GCS path for multi-host sharing (one host compiles, all reuse).
JAX_CACHE_DIR=${JAX_CACHE_DIR:-"${BASE_OUTPUT_DIR}/jax_cache"}

# ============================================================================
# 3. JAX Multi-node Config (Auto-detect)
# ============================================================================
if [[ -n "$TPU_PROCESS_ADDRESSES" ]]; then
    JAX_COORDINATOR_ADDRESS=$(echo "$TPU_PROCESS_ADDRESSES" | cut -d',' -f1)
    export JAX_COORDINATOR_ADDRESS
    echo "Multi-node detection: Coordinator -> $JAX_COORDINATOR_ADDRESS"
    echo "   Worker ID: $TPU_WORKER_ID"
    echo "   TPU Topology: $TPU_TOPOLOGY"
else
    echo "TPU_PROCESS_ADDRESSES not detected, assuming single machine."
fi

# ============================================================================
# 4. Dataset Config (Megatron MMap Indexed)
# ============================================================================
# Path prefixes for .idx/.bin files (without extension), or directories containing them
# Path format for mmap_npy:
#   single dataset: "npy_dir|bin_prefix_or_dir"
#   blended dataset: "npy_dir|bin_prefix_or_dir,weight;npy_dir|bin_prefix_or_dir,weight"
# The .npy index files encode document ordering, sampling, and shuffle.
# bin_dirs provide the raw token data (.bin/.idx files).
DATASET_TYPE="grain"
GRAIN_FILE_TYPE="mmap_npy"

NEMO_HQ_T2E_WEIGHT="0.597200"
NEMO_MHQ_T2E_WEIGHT="0.402800"
NEMO_HQ_T2E_BIN_PREFIX="/models/datasets/nemotron-cc-v2.1_megatron_indexed/High-Quality-Translated-To-English_text_document"
NEMO_MHQ_T2E_BIN_PREFIX="/models/datasets/nemotron-cc-v2.1_megatron_indexed/Medium-High-Quality-Translated-To-English_text_document"
# Replace these with the actual per-component npy index directories.
NEMO_HQ_T2E_NPY_DIR="/models/datasets/hqt-npy-next-ci"
NEMO_MHQ_T2E_NPY_DIR="/models/datasets/mqt-npy-next-ci"

GRAIN_TRAIN_FILES="${NEMO_HQ_T2E_NPY_DIR}|${NEMO_HQ_T2E_BIN_PREFIX}"
GRAIN_EVAL_FILES=${GRAIN_EVAL_FILES:-$GRAIN_TRAIN_FILES}
# Pre-generated Megatron blended indices directory.
# This directory should contain:
#   - dataset_index.npy
#   - dataset_sample_index.npy
BLEND_INDEX_DIR="/models/datasets/ling2.5-blend/"
# Conservative Grain dataloader parallelism for multi-host stability.
GRAIN_WORKER_COUNT=${GRAIN_WORKER_COUNT:-8}
GRAIN_PER_WORKER_BUFFER_SIZE=${GRAIN_PER_WORKER_BUFFER_SIZE:-32}
GRAIN_NUM_THREADS=${GRAIN_NUM_THREADS:-16}
GRAIN_PREFETCH_BUFFER_SIZE=${GRAIN_PREFETCH_BUFFER_SIZE:-500}
MMAP_SPLIT_SENTENCES="true"  # Data was generated with --split-sentences
MMAP_NPY_SPLIT=${MMAP_NPY_SPLIT:-"999,1,0"}  # Megatron-style split: 99.9% train, 0.1% eval, 0% test
# MTP Plan C: Allow cross-document attention with packing for efficiency
PACKING="true"  # Enable sequence packing for better GPU/TPU utilization
RESET_ATTENTION_MASK="true"  # Reset attention at document boundaries (enables KDA varlen mode)
EOD_MASK_LOSS="true"  # Exclude EOD tokens from loss (matching Megatron --eod-mask-loss)

# ============================================================================
# 5. Training Hyperparameters
# ============================================================================
MODEL_NAME="ling3-tiny"
CONFIG_FILE="src/maxtext/configs/base.yml"

# Training Steps and Batch Size
STEPS=${STEPS:-100000}
if [ "$STEPS" -le 0 ] 2>/dev/null; then
    echo "Error: STEPS must be > 0 (got: $STEPS)" >&2
    exit 1
fi
MAX_SEQ_LEN=8192

PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}

EVAL_INTERVAL=${EVAL_INTERVAL:-1490}
EVAL_STEPS=1
OPT_TYPE=${OPT_TYPE:-"muon"}
# AdamW parameters (also used as AdamW fallback within Muon optimizer)
ADAM_B1=0.9
ADAM_B2=0.95
ADAM_WEIGHT_DECAY=0.1
MU_DTYPE="float32"  # Megatron uses fp32 master weights; must match to avoid multi-step divergence
# Muon optimizer parameters (matching Megatron run_v3.sh --optimizer muon)
# muon_consistent_rms=0.2 maps to Megatron --muon-matched-adamw-rms 0.2
# muon_split_head / muon_split_linear_fc1 are configurable (auto-detected in muon_utils.py)
MUON_BETA=${MUON_BETA:-0.95}
MUON_WEIGHT_DECAY=${MUON_WEIGHT_DECAY:-0.1}
MUON_CONSISTENT_RMS=${MUON_CONSISTENT_RMS:-0.2}
# Megatron: --weight-decay-norm-params (applies weight decay to norm parameters in Adam partition)
MUON_WEIGHT_DECAY_NORM_PARAMS=${MUON_WEIGHT_DECAY_NORM_PARAMS:-true}
# muon_split_head / muon_split_linear_fc1: auto-detected by muon_utils.py transform_logic;
# these flags are accepted for Megatron parity but the actual splitting is always on.
MUON_SPLIT_HEAD=${MUON_SPLIT_HEAD:-true}
MUON_SPLIT_LINEAR_FC1=${MUON_SPLIT_LINEAR_FC1:-true}
# Megatron: --muon-batch-update --muon-batch-update-size 16
MUON_BATCH_UPDATE=${MUON_BATCH_UPDATE:-true}
MUON_BATCH_UPDATE_SIZE=${MUON_BATCH_UPDATE_SIZE:-16}
GRADIENT_CLIPPING_THRESHOLD=1.0
LEARNING_RATE=0.000339
MIN_LEARNING_RATE=0.000339  # Constant LR: min_lr = lr
WARMUP_ITERS=${WARMUP_ITERS:-250}
# LEARNING_RATE_SCHEDULE_STEPS defaults to STEPS (so warmup_fraction = warmup/STEPS) but
# can be overridden — Megatron-reference loss-validation runs set this to the
# reference's --train-iters so the LR curve matches the reference exactly.
# For short CI runs where STEPS < WARMUP_ITERS (e.g. profiling with STEPS=10),
# the default would produce warmup_fraction > 1 which Pydantic rejects;
# auto-scale LEARNING_RATE_SCHEDULE_STEPS to keep fraction valid (LR shape
# is irrelevant when STEPS never reaches the warmup tail anyway).
if [ -z "$LEARNING_RATE_SCHEDULE_STEPS" ] && [ "$STEPS" -le "$WARMUP_ITERS" ]; then
  LEARNING_RATE_SCHEDULE_STEPS=$((WARMUP_ITERS * 4))
  echo "   [auto-scale] STEPS=$STEPS <= WARMUP_ITERS=$WARMUP_ITERS, raising LEARNING_RATE_SCHEDULE_STEPS to $LEARNING_RATE_SCHEDULE_STEPS to keep warmup_steps_fraction in [0, 1]"
fi
LEARNING_RATE_SCHEDULE_STEPS=${LEARNING_RATE_SCHEDULE_STEPS:-$STEPS}
WARMUP_STEPS_FRACTION=$(python3 -c "print(${WARMUP_ITERS} / ${LEARNING_RATE_SCHEDULE_STEPS})")
# Constant learning rate schedule (matching Megatron --lr-decay-style constant)
LEARNING_RATE_FINAL_FRACTION=1.0  # Keep at 1.0 for constant LR
DATA_SHUFFLE_SEED=42
INIT_WEIGHTS_SEED=42
REMAT_POLICY=${REMAT_POLICY:-"save_out_proj"}

# Unscan mode: checkpoint was saved without scan, so disable scan_layers
SCAN_LAYERS="false"

# MFU override (optional). Auto-detected by default from jax device_kind.
# Only set for fp8/int8 training, or if your chip isn't in the bf16 peak table.
# See src/maxtext/utils/peak_tflops_map.py for the supported chip table.
PEAK_TFLOPS_PER_DEVICE=${PEAK_TFLOPS_PER_DEVICE:-0.0}

CHECKPOINT_PERIOD=${CHECKPOINT_PERIOD:-100}
ENABLE_CHECKPOINTING=${ENABLE_CHECKPOINTING:-false}

# ============================================================================
# 5b. Vertex AI TensorBoard (optional, off by default)
# ============================================================================
# Set USE_VERTEX_TENSORBOARD=true to enable uploading metrics to Vertex AI TensorBoard.
# Requires: VERTEX_TB_PROJECT and VERTEX_TB_REGION.
USE_VERTEX_TENSORBOARD=${USE_VERTEX_TENSORBOARD:-false}
VERTEX_TB_PROJECT=${VERTEX_TB_PROJECT:-""}
VERTEX_TB_REGION=${VERTEX_TB_REGION:-""}

# ============================================================================
# 5c. Goodput Monitoring (on by default for Ling3 pretraining)
# ============================================================================
# Goodput recording/monitoring uploads job health metrics to Cloud Logging and
# Tensorboard on the lead host. The CI stub (DECOUPLE_GCLOUD=TRUE) no-ops this.
# Set ENABLE_GOODPUT=false to disable entirely (e.g. local debugging).
ENABLE_GOODPUT=${ENABLE_GOODPUT:-false}
MONITOR_STEP_TIME_DEVIATION=${MONITOR_STEP_TIME_DEVIATION:-true}

# ============================================================================
# 6. Start Training Command
# ============================================================================
echo "========================================================"
echo "Starting Ling3 Training"
echo "   Model Arch : Ling3 (MLA + KDA + MoE)"
echo "   Output Dir : $OUTPUT_DIR"
echo "   Dataset    : $GRAIN_TRAIN_FILES"
echo "   Eval Data  : $GRAIN_EVAL_FILES"
echo "   BlendIndex : $BLEND_INDEX_DIR"
echo "   Per-Device Batch: $PER_DEVICE_BATCH_SIZE"
echo "   Grad Accum : $GRADIENT_ACCUMULATION_STEPS"
echo "   Scan Layers: $SCAN_LAYERS"
echo "   LR Schedule: Constant (warmup=${WARMUP_ITERS}, lr=${LEARNING_RATE})"
echo "   Optimizer  : $OPT_TYPE (muon_consistent_rms=$MUON_CONSISTENT_RMS, batch_update=$MUON_BATCH_UPDATE, batch_size=$MUON_BATCH_UPDATE_SIZE)"
echo "========================================================"

# ============================================================================
# 7. LIBTPU Configuration
# ============================================================================
# Load common XLA flags from shared library
source "$(dirname "${BASH_SOURCE[0]}")/comm/xla_flags_common.sh"

echo "   LIBTPU_INIT_ARGS: $LIBTPU_INIT_ARGS"

python3 -m maxtext.trainers.pre_train.train "$CONFIG_FILE" \
    model_name=$MODEL_NAME \
    override_model_config=true \
    run_name=$RUN_NAME \
    base_output_directory=$BASE_OUTPUT_DIR \
    \
    `# --- Dataset Loading (Megatron MMap) ---` \
    dataset_type=$DATASET_TYPE \
    grain_file_type=$GRAIN_FILE_TYPE \
    grain_train_files=$GRAIN_TRAIN_FILES \
    grain_eval_files=$GRAIN_EVAL_FILES \
    grain_worker_count=$GRAIN_WORKER_COUNT \
    grain_per_worker_buffer_size=$GRAIN_PER_WORKER_BUFFER_SIZE \
    grain_num_threads=$GRAIN_NUM_THREADS \
    grain_prefetch_buffer_size=$GRAIN_PREFETCH_BUFFER_SIZE \
    grain_worker_count_eval=$GRAIN_WORKER_COUNT \
    grain_per_worker_buffer_size_eval=$GRAIN_PER_WORKER_BUFFER_SIZE \
    grain_num_threads_eval=$GRAIN_NUM_THREADS \
    grain_prefetch_buffer_size_eval=$GRAIN_PREFETCH_BUFFER_SIZE \
    mmap_split_sentences=$MMAP_SPLIT_SENTENCES \
    mmap_npy_split=$MMAP_NPY_SPLIT \
    blend_index_dir=$BLEND_INDEX_DIR \
    packing=$PACKING \
    reset_attention_mask=$RESET_ATTENTION_MASK \
    eod_mask_loss=$EOD_MASK_LOSS \
    \
    `# --- Training Parameters ---` \
    steps=$STEPS \
    eval_interval=$EVAL_INTERVAL \
    eval_steps=$EVAL_STEPS \
    max_target_length=$MAX_SEQ_LEN \
    per_device_batch_size=$PER_DEVICE_BATCH_SIZE \
    gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    data_shuffle_seed=$DATA_SHUFFLE_SEED \
    init_weights_seed=$INIT_WEIGHTS_SEED \
    \
    `# --- Optimizer (muon with AdamW fallback) ---` \
    opt_type=$OPT_TYPE        \
    adam_b1=$ADAM_B1               \
    adam_b2=$ADAM_B2               \
    adam_weight_decay=$ADAM_WEIGHT_DECAY      \
    mu_dtype=$MU_DTYPE                        \
    muon_beta=$MUON_BETA \
    muon_weight_decay=$MUON_WEIGHT_DECAY \
    muon_consistent_rms=$MUON_CONSISTENT_RMS \
    muon_weight_decay_norm_params=$MUON_WEIGHT_DECAY_NORM_PARAMS \
    muon_batch_update=$MUON_BATCH_UPDATE \
    muon_batch_update_size=$MUON_BATCH_UPDATE_SIZE \
    `# --- learning rate ---` \
    gradient_clipping_threshold=$GRADIENT_CLIPPING_THRESHOLD \
    learning_rate=$LEARNING_RATE \
    warmup_steps_fraction=$WARMUP_STEPS_FRACTION \
    learning_rate_final_fraction=$LEARNING_RATE_FINAL_FRACTION \
    learning_rate_schedule_steps=$LEARNING_RATE_SCHEDULE_STEPS \
    \
    `# --- Parallelism Strategy (configurable via ENV) ---` \
    ici_data_parallelism=${ICI_DATA_PARALLELISM:-1} \
    ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM:-1} \
    ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM:-1} \
    ici_context_parallelism=${ICI_CONTEXT_PARALLELISM:-1} \
    ici_expert_parallelism=${ICI_EXPERT_PARALLELISM:-8} \
    shard_exp_on_fsdp=${SHARD_EXP_ON_FSDP:-false} \
    \
    `# --- Scan / Unscan ---` \
    scan_layers=$SCAN_LAYERS \
    \
    `# --- Performance Optimization ---` \
    remat_policy=$REMAT_POLICY \
    `# --- System Config ---` \
    enable_checkpointing=$ENABLE_CHECKPOINTING \
    save_checkpoint_on_completion=$ENABLE_CHECKPOINTING \
    enable_emergency_checkpoint=false \
    enable_multi_tier_checkpointing=false \
    checkpoint_period=$CHECKPOINT_PERIOD \
    async_checkpointing=false \
    gcs_metrics=false \
    peak_tflops_per_device=$PEAK_TFLOPS_PER_DEVICE \
    save_config_to_gcs=false \
    jax_cache_dir=$JAX_CACHE_DIR \
    load_parameters_path=/models/pretrain/ling3/maxtext_ckpt/ling3-tiny/0/items/ \
    log_period=10 \
    \
    `# --- Vertex AI TensorBoard (optional, controlled by ENV) ---` \
    use_vertex_tensorboard=$USE_VERTEX_TENSORBOARD \
    ${VERTEX_TB_PROJECT:+vertex_tensorboard_project=$VERTEX_TB_PROJECT} \
    ${VERTEX_TB_REGION:+vertex_tensorboard_region=$VERTEX_TB_REGION} \
    \
    `# --- Goodput Monitoring (controlled by ENV, defaults on) ---` \
    enable_goodput_recording=$ENABLE_GOODPUT \
    monitor_goodput=$ENABLE_GOODPUT \
    monitor_step_time_deviation=$MONITOR_STEP_TIME_DEVIATION \
    \
    `# --- Profiler (optional, controlled by ENV) ---` \
    ${PROFILER:+profiler=$PROFILER} \
    ${PROFILER:+skip_first_n_steps_for_profiler=${SKIP_FIRST_N_STEPS_FOR_PROFILER:-1}} \
    ${PROFILER:+profiler_steps=${PROFILER_STEPS:-5}} \
    \
    "$@"


echo "Training finished (or submitted). Check GCS for logs."
