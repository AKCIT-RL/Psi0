#!/usr/bin/env bash
#SBATCH --job-name=psi0-totes-desk-prepare
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/psi0-totes-desk-prepare-%j.out
#SBATCH --error=logs/psi0-totes-desk-prepare-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIF_PATH="${SIF_PATH:-$PROJECT_DIR/containers/industrial_humanoids_psi0-train.gr00t_devel.sif}"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/secrets/psi0.env}"
DATASET_REPO_ID="${DATASET_REPO_ID:-AKCITWMOPOC/G1PickUpToteFromShelfToDeskPsi0}"
DATASET_REVISION="${DATASET_REVISION:-40ae4f97d1ba9422769a6a220547df7ae70e6761}"
DATASET_NAME="${DATASET_NAME:-G1PickUpToteFromShelfToDesk-psi0}"

DATA_ROOT="$PROJECT_DIR/data"
STAGE_DIR="$DATA_ROOT/.staging-$DATASET_NAME"
HF_CACHE_HOST="$PROJECT_DIR/cache/huggingface"
METADATA_DIR="$DATA_ROOT/.provenance"

for required in "$SIF_PATH" "$ENV_FILE"; do
    [[ -s "$required" ]] || { echo "ERROR: required file not found: $required" >&2; exit 1; }
done

mkdir -p "$PROJECT_DIR/logs" "$DATA_ROOT" "$HF_CACHE_HOST" "$METADATA_DIR"
module load apptainer 2>/dev/null || true

apptainer exec --cleanenv \
    --bind "$PROJECT_DIR/scripts:/workspace/scripts:ro" \
    --bind "$ENV_FILE:/workspace/.env:ro" \
    --bind "$DATA_ROOT:/workspace/data" \
    --bind "$HF_CACHE_HOST:/workspace/hf_cache" \
    --env HF_HOME=/workspace/hf_cache \
    --env DATASET_REPO_ID="$DATASET_REPO_ID" \
    --env DATASET_REVISION="$DATASET_REVISION" \
    --env DATASET_NAME="$DATASET_NAME" \
    --env STAGE_DIR="/workspace/data/$(basename "$STAGE_DIR")" \
    "$SIF_PATH" \
    bash -lc '
        set -euo pipefail
        # set -a: export .env vars so hf download authenticates (avoids IP rate limit)
        set -a; source /workspace/.env; set +a
        source /workspace/.venv/bin/activate
        export HF_HOME=/workspace/hf_cache
        dataset_dir="/workspace/data/$DATASET_NAME"
        provenance=/workspace/data/.provenance

        # completeness guard: info.json alone is not enough (a failed run can
        # leave partial meta); require one video per episode before skipping
        expected=$(python -c "import json;print(json.load(open(\"$dataset_dir/meta/info.json\"))[\"total_episodes\"])" 2>/dev/null || echo -1)
        # (find || true): fresh download has no videos/ yet and pipefail would abort
        actual=$( (find "$dataset_dir/videos" -name "*.mp4" 2>/dev/null || true) | wc -l)
        if [[ "$actual" != "$expected" ]]; then
            # repo stores data/, meta/, videos/ at root (no remote subdir);
            # single --include: repeated flags override each other in hf cli
            hf download "$DATASET_REPO_ID" \
                --repo-type dataset \
                --revision "$DATASET_REVISION" \
                --include "data/**" "meta/**" "videos/**" \
                --local-dir "$dataset_dir"
            rm -rf "$dataset_dir/.cache"
        fi

        printf "%s\n" "$DATASET_REVISION" > "$provenance/$DATASET_NAME.revision"

        python /workspace/scripts/validate_lerobot_modality.py \
            --expect-psi0 "$dataset_dir"
        echo "dataset=$DATASET_REPO_ID@$DATASET_REVISION"
        echo "dataset_path=$dataset_dir"
    '
