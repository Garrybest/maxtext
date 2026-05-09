#!/bin/bash
# Ling2 Pretraining Script
# Architecture: Ling2 (Hybrid MLA/GLA + MoE)
# Dataset: Megatron MMap indexed datasets (.bin/.idx)

set -e

# ============================================================================
# 1. Tokenizer Config
# ============================================================================
# Vocabulary size must match the tokenizer used to create the mmap data
# (vocab_size, mmap_eod_id, bos_id are set in configs/models/ling2.yml)

# ============================================================================
# 2. Basic Environment Config (GCS Bucket & Run Name)
# ============================================================================
BASE_OUTPUT_DIR=${GCS_BUCKET:-"gs://ant-pretrain/pretrain/dev"}
RUN_NAME=${RUN_NAME:-"ling2-pretrain-$(date +%Y%m%d-%H%M)"}
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

# NEMO_HQ_T2E_WEIGHT="0.597200"
# NEMO_MHQ_T2E_WEIGHT="0.402800"
# NEMO_HQ_T2E_BIN_PREFIX="/models/datasets/nemotron-cc-v2.1_megatron_indexed/High-Quality-Translated-To-English_text_document"
# NEMO_MHQ_T2E_BIN_PREFIX="/models/datasets/nemotron-cc-v2.1_megatron_indexed/Medium-High-Quality-Translated-To-English_text_document"
# Replace these with the actual per-component npy index directories.
# NEMO_HQ_T2E_NPY_DIR="/models/datasets/hqt-npy-ci"
# NEMO_MHQ_T2E_NPY_DIR="/models/datasets/mqt-npy"
NPY_DIR_CACHE="/models/datasets/maxtext_npy_cache"

# Blended dataset: 112 sub-datasets (fineweb-edu + nemotron-cc-v2.1) with sampling weights
# Format: npy_dir|bin_prefix,weight;npy_dir|bin_prefix,weight;...
GRAIN_TRAIN_FILES="\
${NPY_DIR_CACHE}|/models/datasets/nemotron-cc-v2.1_megatron_indexed/Medium-High-Quality-Translated-To-English_text_document,0.016689;\
${NPY_DIR_CACHE}|/models/datasets/nemotron-cc-v2.1_megatron_indexed/High-Quality-Translated-To-English_text_document,0.024743;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2015-06_text_document,0.006726;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2023-50_text_document,0.014558;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2014-35_text_document,0.007214;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2017-04_text_document,0.009099;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2018-30_text_document,0.008332;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2021-25_text_document,0.007986;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2015-35_text_document,0.006768;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2019-30_text_document,0.007623;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2016-07_text_document,0.006583;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2016-30_text_document,0.006724;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2024-22_text_document,0.010917;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2017-34_text_document,0.007856;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2024-51_text_document,0.007927;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2020-50_text_document,0.007856;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2015-32_text_document,0.006677;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2022-27_text_document,0.011674;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2025-21_text_document,0.010188;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2024-26_text_document,0.010846;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2018-09_text_document,0.008201;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2019-47_text_document,0.007463;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2025-26_text_document,0.009214;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2017-47_text_document,0.007854;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2018-39_text_document,0.007613;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2021-21_text_document,0.008864;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2019-04_text_document,0.007659;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2018-05_text_document,0.008518;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2020-29_text_document,0.009503;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2017-30_text_document,0.007921;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2017-17_text_document,0.010796;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2016-18_text_document,0.005586;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2024-33_text_document,0.008564;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2021-39_text_document,0.010234;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2025-08_text_document,0.010233;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2014-52_text_document,0.007261;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2018-13_text_document,0.007785;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2022-40_text_document,0.012226;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2020-10_text_document,0.007055;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2016-22_text_document,0.005857;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2015-22_text_document,0.007197;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2016-44_text_document,0.008730;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2019-43_text_document,0.008008;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2020-45_text_document,0.008088;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2019-22_text_document,0.008008;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2025-13_text_document,0.010758;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2019-35_text_document,0.008317;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2017-51_text_document,0.006932;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2024-38_text_document,0.011102;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2017-39_text_document,0.007656;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2015-14_text_document,0.006481;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2016-36_text_document,0.006579;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2018-34_text_document,0.007446;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2016-50_text_document,0.008641;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2022-21_text_document,0.012910;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2015-48_text_document,0.006694;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2020-24_text_document,0.007390;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2023-23_text_document,0.013643;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2019-09_text_document,0.008545;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2018-47_text_document,0.007985;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2022-49_text_document,0.012898;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2025-18_text_document,0.011162;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2019-26_text_document,0.007633;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2021-49_text_document,0.007979;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2024-18_text_document,0.011474;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2018-26_text_document,0.008357;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2017-26_text_document,0.008347;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2019-51_text_document,0.006891;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2014-10_text_document,0.007095;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2024-46_text_document,0.010370;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2015-27_text_document,0.006418;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2018-51_text_document,0.008751;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2021-17_text_document,0.010779;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2021-04_text_document,0.010555;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2020-05_text_document,0.009364;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2014-15_text_document,0.006765;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2021-31_text_document,0.011487;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2018-22_text_document,0.006620;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2014-41_text_document,0.007429;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2015-40_text_document,0.005452;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2020-16_text_document,0.008817;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2024-10_text_document,0.012754;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2017-13_text_document,0.010771;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2022-33_text_document,0.008458;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2024-30_text_document,0.009745;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2020-40_text_document,0.010126;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2025-05_text_document,0.012315;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2018-43_text_document,0.008711;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2014-49_text_document,0.006172;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2023-06_text_document,0.012860;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2014-23_text_document,0.007606;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2017-43_text_document,0.008702;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2013-48_text_document,0.006890;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2013-20_text_document,0.007006;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2019-13_text_document,0.007943;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2021-10_text_document,0.008685;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2023-14_text_document,0.012681;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2019-18_text_document,0.008067;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2014-42_text_document,0.006992;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2017-09_text_document,0.009295;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2022-05_text_document,0.010249;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2024-42_text_document,0.009408;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2023-40_text_document,0.015087;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2016-26_text_document,0.004921;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2016-40_text_document,0.007286;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2017-22_text_document,0.007870;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2021-43_text_document,0.011802;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2018-17_text_document,0.007415;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2019-39_text_document,0.007378;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2020-34_text_document,0.007527;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2015-18_text_document,0.007269;\
${NPY_DIR_CACHE}|/models/datasets/fineweb-edu_megatron_indexed/CC-MAIN-2015-11_text_document,0.006865"

MMAP_NPY_SPLIT=${MMAP_NPY_SPLIT:-"999,1,0"}  # Megatron-style split: 99.9% train, 0.1% eval, 0% test
GRAIN_EVAL_FILES=${GRAIN_EVAL_FILES:-$GRAIN_TRAIN_FILES}
# Auto-generate and cache Megatron blended indices
BLEND_CACHE_DIR="/models/datasets/blend_cache"
# Conservative Grain dataloader parallelism for multi-host stability.
GRAIN_WORKER_COUNT=${GRAIN_WORKER_COUNT:-8}
GRAIN_PER_WORKER_BUFFER_SIZE=${GRAIN_PER_WORKER_BUFFER_SIZE:-32}
GRAIN_NUM_THREADS=${GRAIN_NUM_THREADS:-16}
GRAIN_PREFETCH_BUFFER_SIZE=${GRAIN_PREFETCH_BUFFER_SIZE:-500}
MMAP_SPLIT_SENTENCES="true"  # Data was generated with --split-sentences
# MTP Plan C: Allow cross-document attention with packing for efficiency
PACKING="true"  # Enable sequence packing for better GPU/TPU utilization
RESET_ATTENTION_MASK="false"  # Allow cross-document attention (Megatron default mode)
EOD_MASK_LOSS="true"  # Exclude EOD tokens from loss (matching Megatron --eod-mask-loss)

# ============================================================================
# 5. Training Hyperparameters
# ============================================================================
MODEL_NAME="ling2"
CONFIG_FILE="src/maxtext/configs/base.yml"

# Training Steps and Batch Size
STEPS=${STEPS:-100000}
if [ "$STEPS" -le 0 ] 2>/dev/null; then
    echo "Error: STEPS must be > 0 (got: $STEPS)" >&2
    exit 1
fi
MAX_SEQ_LEN=4096

PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}

EVAL_INTERVAL=${EVAL_INTERVAL:-1490}
EVAL_STEPS=1
OPT_TYPE="adamw"
ADAM_B1=0.9
ADAM_B2=0.95
ADAM_WEIGHT_DECAY=0.1
MU_DTYPE="float32"  # Megatron uses fp32 master weights; must match to avoid multi-step divergence
GRADIENT_CLIPPING_THRESHOLD=1.0
LEARNING_RATE=0.000336
MIN_LEARNING_RATE=0.000336  # Constant LR: min_lr = lr
WARMUP_ITERS=${WARMUP_ITERS:-2000}
WARMUP_STEPS_FRACTION=$(python3 -c "print(${WARMUP_ITERS} / ${STEPS})")
# Constant learning rate schedule (matching Megatron --lr-decay-style constant)
LEARNING_RATE_FINAL_FRACTION=1.0  # Keep at 1.0 for constant LR
LEARNING_RATE_SCHEDULE_STEPS=$STEPS
DATA_SHUFFLE_SEED=42
INIT_WEIGHTS_SEED=42
REMAT_POLICY=${REMAT_POLICY:-"save_out_proj"}

CHECKPOINT_PERIOD=${CHECKPOINT_PERIOD:-1192}

# ============================================================================
# 5b. Vertex AI TensorBoard (optional, off by default)
# ============================================================================
# Set USE_VERTEX_TENSORBOARD=true to enable uploading metrics to Vertex AI TensorBoard.
# Requires: VERTEX_TB_PROJECT and VERTEX_TB_REGION.
USE_VERTEX_TENSORBOARD=${USE_VERTEX_TENSORBOARD:-false}
VERTEX_TB_PROJECT=${VERTEX_TB_PROJECT:-""}
VERTEX_TB_REGION=${VERTEX_TB_REGION:-""}

# ============================================================================
# 5c. Goodput Monitoring (on by default for Ling2 pretraining)
# ============================================================================
# Goodput recording/monitoring uploads job health metrics to Cloud Logging and
# Tensorboard on the lead host. The CI stub (DECOUPLE_GCLOUD=TRUE) no-ops this.
# Set ENABLE_GOODPUT=false to disable entirely (e.g. local debugging).
ENABLE_GOODPUT=${ENABLE_GOODPUT:-true}
MONITOR_STEP_TIME_DEVIATION=${MONITOR_STEP_TIME_DEVIATION:-true}

# ============================================================================
# 6. Start Training Command
# ============================================================================
echo "========================================================"
echo "Starting Ling2 Training"
echo "   Model Arch : Ling2 (MLA + GLA + MoE)"
echo "   Output Dir : $OUTPUT_DIR"
echo "   Dataset    : $GRAIN_TRAIN_FILES"
echo "   Eval Data  : $GRAIN_EVAL_FILES"
echo "   BlendCache : $BLEND_CACHE_DIR"
echo "   Per-Device Batch: $PER_DEVICE_BATCH_SIZE"
echo "   Grad Accum : $GRADIENT_ACCUMULATION_STEPS"
echo "   LR Schedule: Constant (warmup=${WARMUP_ITERS}, lr=${LEARNING_RATE})"
echo "========================================================"

# ============================================================================
# 6. LIBTPU Configuration
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
    blend_cache_dir=$BLEND_CACHE_DIR \
    packing=$PACKING \
    reset_attention_mask=$RESET_ATTENTION_MASK \
    eod_mask_loss=$EOD_MASK_LOSS \
    \
    `# --- Training Parameters ---` \
    steps=$STEPS \
    num_epoch=1 \
    eval_interval=$EVAL_INTERVAL \
    eval_steps=$EVAL_STEPS \
    max_target_length=$MAX_SEQ_LEN \
    per_device_batch_size=$PER_DEVICE_BATCH_SIZE \
    gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    data_shuffle_seed=$DATA_SHUFFLE_SEED \
    init_weights_seed=$INIT_WEIGHTS_SEED \
    \
    `# --- Optimizer (adam) ---` \
    opt_type=$OPT_TYPE        \
    adam_b1=$ADAM_B1               \
    adam_b2=$ADAM_B2               \
    adam_weight_decay=$ADAM_WEIGHT_DECAY      \
    mu_dtype=$MU_DTYPE                        \
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
    ici_expert_parallelism=${ICI_EXPERT_PARALLELISM:-1} \
    shard_exp_on_fsdp=${SHARD_EXP_ON_FSDP:-false} \
    \
    `# TBD: async_checkpointing 和 load_parameters_path 可能有冲突` \
    `# --- Performance Optimization ---` \
    remat_policy=$REMAT_POLICY \
    `# --- System Config ---` \
    enable_checkpointing=true \
    save_checkpoint_on_completion=false \
    enable_emergency_checkpoint=false \
    enable_multi_tier_checkpointing=false \
    checkpoint_period=$CHECKPOINT_PERIOD \
    async_checkpointing=true \
    gcs_metrics=false \
    save_config_to_gcs=false \
    jax_cache_dir=$JAX_CACHE_DIR \
    load_parameters_path=/models/gpu-ckpt-ling2.5/ling2.5-maxtext/0/items/ \
    log_period=1 \
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
