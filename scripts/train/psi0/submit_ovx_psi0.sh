#!/usr/bin/env bash
#SBATCH --job-name=psi0-wmo-totes
#SBATCH --nodes=1
#SBATCH --partition=h100n3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/psi0-%j.out
#SBATCH --error=logs/psi0-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RAID_ROOT="${RAID_ROOT:-/raid/${USER}/psi0}"
SIF_PATH="${SIF_PATH:-${RAID_ROOT}/containers/industrial_humanoids_psi0-train.psi_devel.sif}"
ENV_FILE="${ENV_FILE:-${RAID_ROOT}/secrets/psi0.env}"
DATASET_NAME="${DATASET_NAME:-G1WholebodyLocomotionPickTotesShelfToTableTeleop-psi0}"
EXP_NAME="${EXP_NAME:-psi0-wmo-totes-original-ovx}"
TARGET_EPOCHS="${TARGET_EPOCHS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

DATA_ROOT="${RAID_ROOT}/data"
RUNS_ROOT="${RAID_ROOT}/runs"
PSI_HOME_HOST="${RAID_ROOT}/psi_home"
HF_CACHE_HOST="${RAID_ROOT}/cache/huggingface"

for required in "$SIF_PATH" "$ENV_FILE" "$DATA_ROOT/$DATASET_NAME/meta/info.json"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: required path not found: $required" >&2
        exit 1
    fi
done

mkdir -p "$RUNS_ROOT" "$PSI_HOME_HOST" "$HF_CACHE_HOST"
module load apptainer 2>/dev/null || true

echo "Job:       ${SLURM_JOB_ID:-unknown}"
echo "Node:      ${SLURM_NODELIST:-unknown}"
echo "Dataset:   $DATA_ROOT/$DATASET_NAME"
echo "Runs:      $RUNS_ROOT"
echo "Container: $SIF_PATH"
echo "Started:   $(date --iso-8601=seconds)"

apptainer exec --nv --cleanenv \
    --bind "$PROJECT_DIR/src:/workspace/src" \
    --bind "$PROJECT_DIR/scripts:/workspace/scripts" \
    --bind "$PROJECT_DIR/docker:/workspace/docker" \
    --bind "$PROJECT_DIR/pyproject.toml:/workspace/pyproject.toml:ro" \
    --bind "$PROJECT_DIR/README.md:/workspace/README.md:ro" \
    --bind "$ENV_FILE:/workspace/.env:ro" \
    --bind "$DATA_ROOT:/workspace/data" \
    --bind "$RUNS_ROOT:/workspace/.runs" \
    --bind "$PSI_HOME_HOST:/workspace/psi_home" \
    --bind "$HF_CACHE_HOST:/workspace/hf_cache" \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env TARGET_EPOCHS="$TARGET_EPOCHS" \
    --env TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    --env RESUME_FROM_CHECKPOINT="$RESUME_FROM_CHECKPOINT" \
    "$SIF_PATH" \
    bash -lc '
        set -euo pipefail
        cd /workspace
        test -x .venv-psi/bin/python || {
            echo "ERROR: the SIF does not contain /workspace/.venv-psi" >&2
            exit 1
        }
        source .venv-psi/bin/activate
        python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
        exec bash scripts/train/psi0/finetune-lerobot-psi0.sh \
            "/workspace/data/'"$DATASET_NAME"'" \
            "'"$EXP_NAME"'"
    '

echo "Finished: $(date --iso-8601=seconds)"