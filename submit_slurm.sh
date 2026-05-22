#!/usr/bin/env bash
#
# SLURM submission script — GR00T-N1.7 fine-tuning via Apptainer
#
# Usage:
#   sbatch submit_slurm.sh
#
# Override defaults at submission time, e.g.:
#   sbatch --gres=gpu:4 --time=72:00:00 submit_slurm.sh

########################
# SLURM CONFIGURATION  #
########################

#SBATCH --job-name=gr00t-n1d7-finetune
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/gr00t-%j.out
#SBATCH --error=logs/gr00t-%j.err

set -euo pipefail

########################
# PATHS                #
########################

# Root of this repo on the host — defaults to the directory where sbatch was invoked.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "${BASH_SOURCE[0]}")")}}"

# Apptainer SIF image (devel variant with .venv-gr00t already baked in)
SIF_PATH="${SIF_PATH:-${PROJECT_DIR}/industrial_humanoids_psi0-train.gr00t_devel.sif}"

# .env file with secrets (HF_TOKEN, WANDB_API_KEY, …)
ENV_FILE="${ENV_FILE:-${PROJECT_DIR}/.env}"

########################
# SANITY CHECKS        #
########################

echo "=================================================="
echo "GR00T-N1.7 Fine-tuning — Apptainer + SLURM"
echo "=================================================="
echo "Job ID:       ${SLURM_JOB_ID:-<local>}"
echo "Node:         ${SLURM_NODELIST:-<local>}"
echo "Start time:   $(date)"
echo "PROJECT_DIR:  ${PROJECT_DIR}"
echo "SIF_PATH:     ${SIF_PATH}"
echo "=================================================="

if [ ! -f "${SIF_PATH}" ]; then
    echo "ERROR: Apptainer image not found: ${SIF_PATH}"
    exit 1
fi

########################
# PREPARE DIRECTORIES  #
########################

mkdir -p \
    "${PROJECT_DIR}/logs" \
    "${PROJECT_DIR}/checkpoints/gr00t_n1d7_finetune_output" \
    "${PROJECT_DIR}/cache/huggingface" \
    "${PROJECT_DIR}/cache/torch" \
    "${PROJECT_DIR}/cache/wandb"

########################
# LOAD MODULES         #
########################

module load apptainer 2>/dev/null || true

########################
# LOAD SECRETS         #
########################

if [ -f "${ENV_FILE}" ]; then
    # Export only non-comment, non-empty lines
    set -a
    # shellcheck disable=SC1090
    source <(grep -v '^\s*#' "${ENV_FILE}" | grep -v '^\s*$')
    set +a
    echo "[INFO] Loaded environment from ${ENV_FILE}"
fi

########################
# APPTAINER EXEC       #
########################

apptainer exec \
    --nv \
    --bind "${PROJECT_DIR}/data:/workspace/data" \
    --bind "${PROJECT_DIR}/checkpoints:/workspace/checkpoints" \
    --bind "${PROJECT_DIR}/src:/workspace/src" \
    --bind "${PROJECT_DIR}/baselines:/workspace/baselines" \
    --bind "${PROJECT_DIR}/scripts:/workspace/scripts" \
    --bind "${PROJECT_DIR}/cache/huggingface:/workspace/cache/huggingface" \
    --bind "${PROJECT_DIR}/cache/torch:/workspace/cache/torch" \
    --bind "${PROJECT_DIR}/cache/wandb:/workspace/cache/wandb" \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env WANDB_ENTITY="${WANDB_ENTITY:-industrial_humanoids}" \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    --env HF_HOME="/workspace/cache/huggingface" \
    --env TORCH_HOME="/workspace/cache/torch" \
    --env OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
    --env TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}" \
    --env CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-true}" \
    --env TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}" \
    --env NCCL_DEBUG="WARN" \
    "${SIF_PATH}" \
    bash -c '
        set -euo pipefail
        cd /workspace

        echo ""
        echo "--- Container environment ---"
        python_bin="/workspace/src/gr00t/.venv-gr00t/bin/python"
        echo "Python:  $($python_bin --version)"
        echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-all}"
        echo ""

        export DATASET_PATH="/workspace/data/simple/simple/G1WholebodyPickAndPlaceAndHugContainerTeleop-v0"
        export CUDA_VISIBLE_DEVICES="0,1"

        $python_bin -m torch.distributed.run \
            --nproc_per_node 2 \
            --master_port 29502 \
            /workspace/baselines/gr00t-n1.7/launch_finetune_n1d7_inner.py \
            --base-model-path /workspace/checkpoints/GR00T-N1.7-3B \
            --dataset-path "$DATASET_PATH" \
            --embodiment-tag G1_LOCO_DOWNSTREAM \
            --save-steps 10000 \
            --save-total-limit 4 \
            --max-steps 50000 \
            --warmup-ratio 0.05 \
            --weight-decay 1e-05 \
            --learning-rate 0.0001 \
            --global-batch-size 16 \
            --gradient-accumulation-steps 2 \
            --dataloader-num-workers 16 \
            --output-dir /workspace/checkpoints/gr00t_n1d7_finetune_output \
            --eval-strategy no \
            --num-gpus 2 \
            --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
            --modality-config-path src/gr00t/gr00t/configs/modality/g1_locomanip_n1d7.py \
            --gradient-checkpointing \
            --use-wandb
    '

echo ""
echo "=================================================="
echo "Job finished: $(date)"
echo "=================================================="
