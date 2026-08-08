#!/usr/bin/env bash
#SBATCH --job-name=dp-wmo-totes
#SBATCH --nodes=1
#SBATCH --partition=ovx01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/dp-%j.out
#SBATCH --error=logs/dp-%j.err

# Baseline Diffusion Policy on the OVX cluster. Hyperparameters mirror
# baselines/dp/train_dp_g1_real.sh; only paths/venv/val-split differ.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RAID_ROOT="${RAID_ROOT:-/raid/${USER}/psi0}"
SIF_PATH="${SIF_PATH:-${RAID_ROOT}/containers/industrial_humanoids_psi0-train.gr00t_devel.sif}"
ENV_FILE="${ENV_FILE:-${RAID_ROOT}/secrets/psi0.env}"
DATASET_NAME="${DATASET_NAME:-G1WholebodyLocomotionPickTotesShelfToTableTeleop-psi0}"
EXP_NAME="${EXP_NAME:-dp-wmo-totes-ovx}"
VAL_EPISODE_FRACTION="${VAL_EPISODE_FRACTION:-0.1}"
STATE_NOISE_STD="${STATE_NOISE_STD:-0.0}"
STATE_NOISE_STD_WAIST="${STATE_NOISE_STD_WAIST:-0.0}"

DATA_ROOT="${RAID_ROOT}/data"
RUNS_ROOT="${RAID_ROOT}/runs"
PSI_HOME_HOST="${RAID_ROOT}/psi_home"
HF_CACHE_HOST="${RAID_ROOT}/cache/huggingface"

for required in "$SIF_PATH" "$ENV_FILE" "$DATA_ROOT/$DATASET_NAME/meta/stats_psi0.json"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: required path not found: $required" >&2
        exit 1
    fi
done

mkdir -p "$RUNS_ROOT" "$PSI_HOME_HOST" "$HF_CACHE_HOST"

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
    --env DATASET_NAME="$DATASET_NAME" \
    --env EXP_NAME="$EXP_NAME" \
    --env VAL_EPISODE_FRACTION="$VAL_EPISODE_FRACTION" \
    --env STATE_NOISE_STD="$STATE_NOISE_STD" \
    --env STATE_NOISE_STD_WAIST="$STATE_NOISE_STD_WAIST" \
    "$SIF_PATH" \
    bash -lc '
        set -euo pipefail
        cd /workspace
        if .venv-psi/bin/python -c "import torch" 2>/dev/null; then
            source .venv-psi/bin/activate
        elif .venv/bin/python -c "import torch" 2>/dev/null; then
            source .venv/bin/activate
        else
            echo "ERROR: no venv with torch found in the SIF" >&2
            exit 1
        fi
        export PYTHONPATH="/workspace/src${PYTHONPATH:+:$PYTHONPATH}"
        set -a; source /workspace/.env; set +a
        python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
        export OMP_NUM_THREADS=32
        ulimit -n 65535 2>/dev/null || true

        exec python scripts/train.py \
            real_dp_config \
            --seed=2026 \
            --exp="$EXP_NAME" \
            --train.name=diffusion-policy-g1 \
            --log.report-to=wandb \
            --train.data_parallel=ddp \
            --train.mixed_precision=bf16 \
            --train.train-batch-size=16 \
            --train.validation_steps=500 \
            --train.val_num_batches=20 \
            --train.max-training-steps=42000 \
            --train.learning-rate=1e-4 \
            --train.max-grad-norm=1.0 \
            --train.lr_scheduler_kwargs.weight_decay=1e-6 \
            --train.lr_scheduler_kwargs.betas 0.95 0.999 \
            --train.lr_scheduler_type=cosine \
            --train.warmup-steps=1000 \
            --train.warmup-ratio=None \
            --train.checkpointing-steps=5000 \
            --data.root_dir=/workspace/data \
            --data.train-repo-ids="$DATASET_NAME" \
            --data.val_episode_fraction="$VAL_EPISODE_FRACTION" \
            --data.transform.field.state-noise-std="$STATE_NOISE_STD" \
            --data.transform.field.state-noise-std-waist="$STATE_NOISE_STD_WAIST" \
            --data.transform.repack.action-chunk-size=16 \
            --data.transform.repack.pad-action-dim=36 \
            --data.transform.repack.pad-state-dim=32 \
            --data.transform.field.stat-path=meta/stats_psi0.json \
            --data.transform.field.stat-action-key=action \
            --data.transform.field.stat-state-key=states \
            --data.transform.field.normalize-state \
            --data.transform.field.action-norm-type=bounds \
            --data.transform.model.img-aug \
            --model.action-chunk-size=16 \
            --model.action-dim=36 \
            --model.obs-dim=32
    '

echo "Finished: $(date --iso-8601=seconds)"
