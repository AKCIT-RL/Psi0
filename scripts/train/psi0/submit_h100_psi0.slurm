#!/usr/bin/env bash
#SBATCH --job-name=psi0-h100
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/psi0-h100-%j.out
#SBATCH --error=logs/psi0-h100-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIF_PATH="${SIF_PATH:-$PROJECT_DIR/containers/industrial_humanoids_psi0-train.gr00t_devel.sif}"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/secrets/psi0.env}"
DATASET_NAME="${DATASET_NAME:?Set DATASET_NAME to a validated LeRobot dataset under $PROJECT_DIR/data}"
DATASET_REPO_ID="${DATASET_REPO_ID:?Set DATASET_REPO_ID to the Hugging Face dataset repo_id}"
DATASET_REVISION="${DATASET_REVISION:?Set DATASET_REVISION to an immutable Hugging Face commit}"
EXP_NAME="${EXP_NAME:-psi0-${DATASET_NAME}-h100-${SLURM_JOB_ID:-manual}}"
SMOKE_MODE="${SMOKE_MODE:-0}"
TARGET_EPOCHS="${TARGET_EPOCHS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_EPISODE_FRACTION="${VAL_EPISODE_FRACTION:-0.1}"
VALIDATION_STEPS="${VALIDATION_STEPS:-2500}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

DATA_ROOT="$PROJECT_DIR/data"
RUNS_ROOT="$PROJECT_DIR/.runs"
PSI_HOME_HOST="$PROJECT_DIR/psi_home"
HF_CACHE_HOST="$PROJECT_DIR/cache/huggingface"
WANDB_CACHE_HOST="$PROJECT_DIR/cache/wandb"
GIT_COMMIT="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
MODEL_REPO_ID="USC-PSI-Lab/psi-model"
MODEL_REVISION_FILE="$DATA_ROOT/.provenance/psi-model.revision"

for required in \
    "$SIF_PATH" \
    "$ENV_FILE" \
    "$DATA_ROOT/$DATASET_NAME/meta/info.json" \
    "$DATA_ROOT/$DATASET_NAME/meta/stats_psi0.json" \
    "$MODEL_REVISION_FILE"; do
    [[ -e "$required" ]] || { echo "ERROR: required path not found: $required" >&2; exit 1; }
done
VLM_DIR="$PSI_HOME_HOST/cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"
ACTION_DIR="$PSI_HOME_HOST/cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k"
compgen -G "$VLM_DIR/model*.safetensors" >/dev/null \
    || compgen -G "$VLM_DIR/pytorch_model*.bin" >/dev/null \
    || { echo "ERROR: VLM weights are incomplete: $VLM_DIR" >&2; exit 1; }
[[ -s "$ACTION_DIR/action_header.safetensors" ]] \
    || { echo "ERROR: action header is incomplete: $ACTION_DIR" >&2; exit 1; }
MODEL_REVISION="$(<"$MODEL_REVISION_FILE")"
CONTAINER_SHA256="$(sha256sum "$SIF_PATH" | awk '{print $1}')"

mkdir -p "$PROJECT_DIR/logs" "$RUNS_ROOT" "$HF_CACHE_HOST" "$WANDB_CACHE_HOST"
module load apptainer 2>/dev/null || true

if [[ "$SMOKE_MODE" == 1 ]]; then
    TARGET_EPOCHS=1
    TRAIN_BATCH_SIZE=1
    MAX_TRAINING_STEPS=2
    CHECKPOINTING_STEPS=1
    VALIDATION_STEPS=1
    VAL_NUM_BATCHES=1
    WARMUP_STEPS=0
else
    MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-}"
    CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-}"
    VAL_NUM_BATCHES="${VAL_NUM_BATCHES:-20}"
    WARMUP_STEPS="${WARMUP_STEPS:-500}"
fi

echo "Job:       ${SLURM_JOB_ID:-unknown}"
echo "Node:      ${SLURM_NODELIST:-unknown}"
echo "Dataset:   $DATA_ROOT/$DATASET_NAME"
echo "HF source: $DATASET_REPO_ID@$DATASET_REVISION"
echo "Weights:   $MODEL_REPO_ID@$MODEL_REVISION"
echo "Git:       $GIT_COMMIT"
echo "SIF SHA:   $CONTAINER_SHA256"
echo "Run:       $EXP_NAME"
echo "Smoke:     $SMOKE_MODE"
echo "Container: $SIF_PATH"

apptainer exec --nv --cleanenv \
    --bind "$PROJECT_DIR/src:/workspace/src" \
    --bind "$PROJECT_DIR/scripts:/workspace/scripts" \
    --bind "$PROJECT_DIR/docker:/workspace/docker:ro" \
    --bind "$PROJECT_DIR/pyproject.toml:/workspace/pyproject.toml:ro" \
    --bind "$ENV_FILE:/workspace/.env:ro" \
    --bind "$DATA_ROOT:/workspace/data:ro" \
    --bind "$RUNS_ROOT:/workspace/.runs" \
    --bind "$PSI_HOME_HOST:/workspace/psi_home" \
    --bind "$HF_CACHE_HOST:/workspace/hf_cache" \
    --bind "$WANDB_CACHE_HOST:/workspace/wandb_cache" \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env CUDA_LAUNCH_BLOCKING=0 \
    --env PYTHONPATH=/workspace/src \
    --env PSI_HOME=/workspace/psi_home \
    --env DATA_HOME=/workspace/data \
    --env HF_HOME=/workspace/hf_cache \
    --env WANDB_CACHE_DIR=/workspace/wandb_cache \
    --env GIT_COMMIT="$GIT_COMMIT" \
    --env DATASET_REPO_ID="$DATASET_REPO_ID" \
    --env DATASET_REVISION="$DATASET_REVISION" \
    --env MODEL_REPO_ID="$MODEL_REPO_ID" \
    --env MODEL_REVISION="$MODEL_REVISION" \
    --env CONTAINER_SHA256="$CONTAINER_SHA256" \
    --env SOURCE_SLURM_JOB_ID="${SLURM_JOB_ID:-unknown}" \
    --env TARGET_EPOCHS="$TARGET_EPOCHS" \
    --env TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    --env VAL_EPISODE_FRACTION="$VAL_EPISODE_FRACTION" \
    --env VALIDATION_STEPS="$VALIDATION_STEPS" \
    --env MAX_TRAINING_STEPS="$MAX_TRAINING_STEPS" \
    --env CHECKPOINTING_STEPS="$CHECKPOINTING_STEPS" \
    --env VAL_NUM_BATCHES="$VAL_NUM_BATCHES" \
    --env WARMUP_STEPS="$WARMUP_STEPS" \
    --env RESUME_FROM_CHECKPOINT="$RESUME_FROM_CHECKPOINT" \
    "$SIF_PATH" \
    bash -lc '
        set -euo pipefail
        cd /workspace
        source /workspace/.env
        [[ -n "${WANDB_API_KEY:-}" ]] || { echo "ERROR: WANDB_API_KEY is not configured" >&2; exit 1; }
        [[ -n "${WANDB_ENTITY:-}" ]] || { echo "ERROR: WANDB_ENTITY is not configured" >&2; exit 1; }
        [[ -n "${WANDB_PROJECT:-}" ]] || { echo "ERROR: WANDB_PROJECT is not configured" >&2; exit 1; }
        if /workspace/.venv-psi/bin/python -c "import torch" 2>/dev/null; then
            source /workspace/.venv-psi/bin/activate
        else
            source /workspace/.venv/bin/activate
        fi
        python -c "import flash_attn, torch; assert torch.cuda.get_device_capability() == (9, 0)"
        exec bash scripts/train/psi0/finetune-lerobot-psi0.sh "/workspace/data/'"$DATASET_NAME"'" "'"$EXP_NAME"'"
    '