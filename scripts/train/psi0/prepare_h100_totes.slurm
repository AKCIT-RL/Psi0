#!/usr/bin/env bash
#SBATCH --job-name=psi0-totes-prepare
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/psi0-totes-prepare-%j.out
#SBATCH --error=logs/psi0-totes-prepare-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIF_PATH="${SIF_PATH:-$PROJECT_DIR/containers/industrial_humanoids_psi0-train.gr00t_devel.sif}"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/secrets/psi0.env}"
DATASET_REPO_ID="lGabrielJJ/G1WholebodyLocomotionPickTotesShelfToTableTeleop"
DATASET_REVISION="219d98f9228f9d2db41d723a56744f3b9093353a"
DATASET_REMOTE_DIR="G1WholebodyLocomotionPickTotesShelfToTableTeleopPsi0"
DATASET_NAME="G1WholebodyLocomotionPickTotesShelfToTableTeleop-psi0"
MODEL_REPO_ID="USC-PSI-Lab/psi-model"
VLM_REMOTE_DIR="psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"
ACTION_REMOTE_DIR="psi0/postpre.1by1.pad36.2601131206.ckpt.he30k"

DATA_ROOT="$PROJECT_DIR/data"
DATASET_DIR="$DATA_ROOT/$DATASET_NAME"
STAGE_DIR="$DATA_ROOT/.staging-$DATASET_NAME"
PSI_HOME_HOST="$PROJECT_DIR/psi_home"
CHECKPOINT_ROOT="$PSI_HOME_HOST/cache/checkpoints"
HF_CACHE_HOST="$PROJECT_DIR/cache/huggingface"
METADATA_DIR="$PROJECT_DIR/data/.provenance"

for required in "$SIF_PATH" "$ENV_FILE"; do
    [[ -s "$required" ]] || { echo "ERROR: required file not found: $required" >&2; exit 1; }
done

mkdir -p "$PROJECT_DIR/logs" "$DATA_ROOT" "$CHECKPOINT_ROOT" "$HF_CACHE_HOST" "$METADATA_DIR"
module load apptainer 2>/dev/null || true

apptainer exec --cleanenv \
    --bind "$PROJECT_DIR/scripts:/workspace/scripts:ro" \
    --bind "$ENV_FILE:/workspace/.env:ro" \
    --bind "$DATA_ROOT:/workspace/data" \
    --bind "$PSI_HOME_HOST:/workspace/psi_home" \
    --bind "$HF_CACHE_HOST:/workspace/hf_cache" \
    --env HF_HOME=/workspace/hf_cache \
    --env DATASET_REPO_ID="$DATASET_REPO_ID" \
    --env DATASET_REVISION="$DATASET_REVISION" \
    --env DATASET_REMOTE_DIR="$DATASET_REMOTE_DIR" \
    --env DATASET_NAME="$DATASET_NAME" \
    --env MODEL_REPO_ID="$MODEL_REPO_ID" \
    --env VLM_REMOTE_DIR="$VLM_REMOTE_DIR" \
    --env ACTION_REMOTE_DIR="$ACTION_REMOTE_DIR" \
    --env STAGE_DIR="/workspace/data/$(basename "$STAGE_DIR")" \
    "$SIF_PATH" \
    bash -lc '
        set -euo pipefail
        source /workspace/.env
        source /workspace/.venv/bin/activate
        export HF_HOME=/workspace/hf_cache
        dataset_dir="/workspace/data/$DATASET_NAME"
        checkpoint_root=/workspace/psi_home/cache/checkpoints
        provenance=/workspace/data/.provenance

        if [[ ! -f "$dataset_dir/meta/info.json" ]]; then
            mkdir -p "$STAGE_DIR"
            hf download "$DATASET_REPO_ID" \
                --repo-type dataset \
                --revision "$DATASET_REVISION" \
                --include "$DATASET_REMOTE_DIR/**" \
                --local-dir "$STAGE_DIR"
            mv "$STAGE_DIR/$DATASET_REMOTE_DIR" "$dataset_dir"
            rm -rf "$STAGE_DIR"
        fi

        model_revision=$(python -c "import os; from huggingface_hub import HfApi; print(HfApi().model_info(\"$MODEL_REPO_ID\", token=os.environ.get(\"HF_TOKEN\")).sha)")
        printf "%s\n" "$model_revision" > "$provenance/psi-model.revision"
        printf "%s\n" "$DATASET_REVISION" > "$provenance/$DATASET_NAME.revision"

        for remote_dir in "$VLM_REMOTE_DIR" "$ACTION_REMOTE_DIR"; do
            hf download "$MODEL_REPO_ID" \
                --repo-type model \
                --revision "$model_revision" \
                --include "$remote_dir/**" \
                --local-dir "$checkpoint_root"
        done

        python /workspace/scripts/validate_lerobot_modality.py \
            --expect-psi0 "$dataset_dir"
        compgen -G "$checkpoint_root/$VLM_REMOTE_DIR/model*.safetensors" >/dev/null \
            || compgen -G "$checkpoint_root/$VLM_REMOTE_DIR/pytorch_model*.bin" >/dev/null
        test -s "$checkpoint_root/$ACTION_REMOTE_DIR/action_header.safetensors"
        echo "dataset=$DATASET_REPO_ID@$DATASET_REVISION"
        echo "model=$MODEL_REPO_ID@$model_revision"
        echo "dataset_path=$dataset_dir"
        echo "checkpoint_root=$checkpoint_root"
    '