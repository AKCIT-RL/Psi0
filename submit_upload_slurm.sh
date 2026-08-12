#!/usr/bin/env bash
#
# SLURM submission script — publish a finished run to Hugging Face, then reclaim disk.
#
# CPU only, on purpose: uploading ~21 GB takes long enough that holding a GPU for it
# would waste more machine time than the training itself.
#
# Usage:
#   sbatch --export=ALL,RUN_NAME=gr00t_n1d7_finetune_output_close_door submit_upload_slurm.sh
#
# Chained after a training job (the normal path — run_pipeline.sh does this for you):
#   sbatch --dependency=afterok:<train_job_id> --export=ALL,RUN_NAME=... submit_upload_slurm.sh
#
# Extra flags for scripts/upload_checkpoints_hf.py go in UPLOAD_ARGS, e.g.
#   --export=ALL,RUN_NAME=...,UPLOAD_ARGS="--no-delete"

########################
# SLURM CONFIGURATION  #
########################

#SBATCH --job-name=hf-upload
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/hfupload-%j.out
#SBATCH --error=logs/hfupload-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "${BASH_SOURCE[0]}")")}}"
ENV_FILE="${ENV_FILE:-${PROJECT_DIR}/.env}"
PY="${PY:-${PROJECT_DIR}/src/gr00t/.venv-gr00t/bin/python}"

HF_BRANCH="${HF_BRANCH:-simple-converted}"
HF_REPO_ID="${HF_REPO_ID:-lucasolives/gr00t_1.7_Psi}"
UPLOAD_WHAT="${UPLOAD_WHAT:-final+last}"

echo "=================================================="
echo "Hugging Face upload"
echo "=================================================="
echo "Job ID:      ${SLURM_JOB_ID:-<local>}"
echo "Start time:  $(date)"
echo "RUN_NAME:    ${RUN_NAME:-<all>}"
echo "Repo:        ${HF_REPO_ID} @ ${HF_BRANCH}"
echo "Uploading:   ${UPLOAD_WHAT}"
echo "=================================================="

if [ ! -x "${PY}" ]; then
    echo "ERROR: interpreter not found: ${PY}"
    exit 1
fi

# The upload runs natively — it only needs huggingface_hub, which is already in the
# gr00t venv on the host. No container, no GPU, no bind mounts to get wrong.
if [ -f "${ENV_FILE}" ]; then
    set -a
    # shellcheck disable=SC1090
    source <(grep -v '^\s*#' "${ENV_FILE}" | grep -v '^\s*$')
    set +a
    echo "[INFO] Loaded environment from ${ENV_FILE}"
fi

TARGET_ARGS=()
if [ -n "${RUN_NAME:-}" ]; then
    TARGET_ARGS=(--run "${RUN_NAME}")
else
    TARGET_ARGS=(--all)
fi

# shellcheck disable=SC2086
"${PY}" "${PROJECT_DIR}/scripts/upload_checkpoints_hf.py" \
    "${TARGET_ARGS[@]}" \
    --checkpoints-dir "${PROJECT_DIR}/checkpoints" \
    --repo-id "${HF_REPO_ID}" \
    --branch "${HF_BRANCH}" \
    --what "${UPLOAD_WHAT}" \
    ${UPLOAD_ARGS:-}

echo ""
echo "=================================================="
echo "Upload job finished: $(date)"
echo "=================================================="
