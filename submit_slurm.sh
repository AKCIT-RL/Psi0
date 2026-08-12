#!/usr/bin/env bash
#
# SLURM submission script — GR00T-N1.7 fine-tuning via Apptainer
#
# Usage:
#   sbatch submit_slurm.sh
#
# Pick the dataset (a directory name under data/simple/simple-converted/):
#   sbatch --export=ALL,DATASET_NAME=G1WholebodyCloseDoorTeleop-v0 submit_slurm.sh
#
# The run directory defaults to checkpoints/gr00t_n1d7_finetune_output_<slug>, where
# <slug> comes from scripts/prepare_simple_datasets.py; override with RUN_NAME.
#
# Override resources at submission time, e.g.:
#   sbatch --gres=gpu:4 --time=72:00:00 submit_slurm.sh

########################
# SLURM CONFIGURATION  #
########################

#SBATCH --job-name=gr00t-n1d7-finetune
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
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
# WHAT TO TRAIN        #
########################

# Dataset directory name under data/simple/simple-converted/ — produced and validated by
# scripts/prepare_simple_datasets.py. Kept as a name, not a path, so the run name can be
# derived from it and the container path stays fixed.
DATASET_NAME="${DATASET_NAME:-G1WholebodyOpenOvenTeleop-v0}"
DATASET_HOST_PATH="${PROJECT_DIR}/data/simple/simple-converted/${DATASET_NAME}"

# Run slug: taken from the dataset's PROVENANCE.json when present (single source of truth
# with the prepare script), otherwise from RUN_NAME, otherwise lower-cased dataset name.
if [ -z "${RUN_NAME:-}" ]; then
    RUN_SLUG=""
    if [ -f "${DATASET_HOST_PATH}/PROVENANCE.json" ]; then
        RUN_SLUG=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('run_slug',''))" \
                   "${DATASET_HOST_PATH}/PROVENANCE.json" 2>/dev/null || true)
    fi
    if [ -z "${RUN_SLUG}" ]; then
        # Datasets prepared before this pipeline existed have no PROVENANCE.json. Ask the
        # prepare script for the slug rather than re-deriving it here: a second rule would
        # eventually disagree, and disagreeing means training into a fresh directory
        # instead of resuming the run that is already there.
        RUN_SLUG=$("${PROJECT_DIR}/src/gr00t/.venv-gr00t/bin/python" \
                   "${PROJECT_DIR}/scripts/prepare_simple_datasets.py" \
                   --print-run-slug "${DATASET_NAME}" 2>/dev/null || true)
    fi
    if [ -z "${RUN_SLUG}" ]; then
        RUN_SLUG=$(echo "${DATASET_NAME}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' \
                   | sed 's/__*/_/g; s/^_//; s/_$//')
    fi
    RUN_NAME="gr00t_n1d7_finetune_output_${RUN_SLUG}"
fi

########################
# QUOTA PREFLIGHT      #
########################

# Free space on /raid is not the constraint — the per-user quota is. Running out of it
# mid-run does not fail fast and loudly: it truncates the checkpoint being written seven
# hours in, and then breaks every other file creation, including this pipeline's own
# bookkeeping. Refuse before burning the GPU time instead.
#
# A run needs ~21 GB (a 12 GB checkpoint plus the 9 GB merged model at the end), so the
# default leaves room for two lanes plus headroom.
REQUIRED_GB="${REQUIRED_GB:-50}"
QUOTA_FREE_GB=$(quota 2>/dev/null | python3 -c "
import sys, re
for line in sys.stdin:
    v = [int(x) for x in re.findall(r'\b\d+\b', line)]
    if len(v) >= 6:                      # blocks, soft, hard, files, soft, hard
        used, soft, hard = v[0], v[1], v[2]
        limit = soft or hard             # 0 means 'no soft limit'; fall back to hard
        print(int((limit - used) / 1024 / 1024))
        break
else:
    print(-1)
" 2>/dev/null || echo -1)

if [ "${QUOTA_FREE_GB}" = "-1" ]; then
    echo "[WARN] Could not read the disk quota; continuing without the check."
elif [ "${QUOTA_FREE_GB}" -lt "${REQUIRED_GB}" ]; then
    echo "ERROR: only ${QUOTA_FREE_GB} GB left under your disk quota, ${REQUIRED_GB} GB needed."
    echo "       A run that hits the quota corrupts its checkpoint hours in — refusing to start."
    echo "       Free space (published runs can be reclaimed) and resubmit:"
    echo "         ./run_pipeline.sh --status"
    echo "         quota -s"
    exit 1
else
    echo "[INFO] Quota headroom: ${QUOTA_FREE_GB} GB (need ${REQUIRED_GB} GB)"
fi

########################
# RENDEZVOUS PORT      #
########################

# torchrun opens a TCP store on the node. A fixed port only works while a single job runs
# at a time: with several fine-tuning jobs sharing this node they all race for it and
# every loser dies with EADDRINUSE within seconds of starting. Derive a distinct port
# from the job id, then scan upward for one that is actually free.
if [ -z "${MASTER_PORT:-}" ]; then
    MASTER_PORT=$(python3 - "$((29500 + ${SLURM_JOB_ID:-$$} % 10000))" <<'PYEOF'
import socket, sys
start = int(sys.argv[1])
for port in range(start, start + 200):
    with socket.socket() as sock:
        try:
            sock.bind(("", port))
        except OSError:
            continue
    print(port)
    break
else:
    raise SystemExit("no free rendezvous port found")
PYEOF
)
fi

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
echo "DATASET:      ${DATASET_NAME}"
echo "RUN_NAME:     ${RUN_NAME}"
echo "MASTER_PORT:  ${MASTER_PORT}"
echo "=================================================="

if [ ! -f "${SIF_PATH}" ]; then
    echo "ERROR: Apptainer image not found: ${SIF_PATH}"
    exit 1
fi

# Fail before allocating hours of GPU rather than after: a missing modality.json means
# the dataset was never validated, and training on it produces a model that looks fine
# and is unusable (see docs/runbook_modality.md).
if [ ! -d "${DATASET_HOST_PATH}" ]; then
    echo "ERROR: dataset not found: ${DATASET_HOST_PATH}"
    echo "       run: scripts/prepare_simple_datasets.py --only ${DATASET_NAME}"
    exit 1
fi
if [ ! -f "${DATASET_HOST_PATH}/meta/modality.json" ]; then
    echo "ERROR: ${DATASET_HOST_PATH}/meta/modality.json is missing — dataset not prepared"
    exit 1
fi
if [ ! -f "${DATASET_HOST_PATH}/PROVENANCE.json" ]; then
    echo "WARNING: no PROVENANCE.json — this dataset was not produced by"
    echo "         scripts/prepare_simple_datasets.py, so it may never have been validated."
fi

########################
# PREPARE DIRECTORIES  #
########################

mkdir -p \
    "${PROJECT_DIR}/logs" \
    "${PROJECT_DIR}/checkpoints/${RUN_NAME}" \
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

# The exit status is needed after the run: it decides whether the upload is submitted,
# so `set -e` must not abort here. A failed training still hands the lane on to the next
# dataset; it just publishes nothing.
TRAIN_STATUS=0
set +e

apptainer exec \
    --nv \
    --bind "${PROJECT_DIR}/data:/workspace/data" \
    --bind "${PROJECT_DIR}/checkpoints:/workspace/checkpoints" \
    --bind "${PROJECT_DIR}/src:/workspace/src" \
    --bind "${PROJECT_DIR}/baselines:/workspace/baselines" \
    --bind "${PROJECT_DIR}/scripts:/workspace/scripts" \
    --bind "${PROJECT_DIR}/third_party:/workspace/third_party" \
    --bind "${PROJECT_DIR}/cache/huggingface:/workspace/cache/huggingface" \
    --bind "${PROJECT_DIR}/cache/torch:/workspace/cache/torch" \
    --bind "${PROJECT_DIR}/cache/wandb:/workspace/cache/wandb" \
    --env DATASET_NAME="${DATASET_NAME}" \
    --env RUN_NAME="${RUN_NAME}" \
    --env MASTER_PORT="${MASTER_PORT}" \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env WANDB_ENTITY="${WANDB_ENTITY:-industrial_humanoids}" \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    --env HF_HOME="/workspace/cache/huggingface" \
    --env TORCH_HOME="/workspace/cache/torch" \
    --env OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
    --env TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}" \
    --env CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-false}" \
    --env TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}" \
    --env NCCL_DEBUG="WARN" \
    --env NCCL_ASYNC_ERROR_HANDLING="1" \
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

        export DATASET_PATH="/workspace/data/simple/simple-converted/${DATASET_NAME}"
        export CUDA_VISIBLE_DEVICES="0"
        echo "Dataset: $DATASET_PATH"
        echo "Output:  /workspace/checkpoints/${RUN_NAME}"
        echo ""

        $python_bin -m torch.distributed.run \
            --nproc_per_node 1 \
            --master_port "${MASTER_PORT}" \
            /workspace/baselines/gr00t-n1.7/launch_finetune_n1d7_inner.py \
            --base-model-path /workspace/checkpoints/GR00T-N1.7-3B \
            --dataset-path "$DATASET_PATH" \
            --embodiment-tag G1_LOCO_DOWNSTREAM \
            --save-steps 10000 \
            --save-total-limit 1 \
            --max-steps 50000 \
            --warmup-ratio 0.05 \
            --weight-decay 1e-05 \
            --learning-rate 0.0001 \
            --global-batch-size 16 \
            --gradient-accumulation-steps 2 \
            --dataloader-num-workers 16 \
            --output-dir "/workspace/checkpoints/${RUN_NAME}" \
            --eval-strategy no \
            --num-gpus 1 \
            --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
            --modality-config-path src/gr00t/gr00t/configs/modality/g1_locomanip_n1d7.py \
            --gradient-checkpointing \
            --use-wandb
    '
TRAIN_STATUS=$?
set -e

echo ""
echo "=================================================="
echo "Training finished: $(date)  (exit ${TRAIN_STATUS})"
echo "=================================================="

########################
# HAND OFF THE LANE    #
########################

# Set by run_pipeline.sh. The follow-up work is submitted from *inside* this job rather
# than queued up front, so the cluster only ever holds the jobs that are actually
# running. On a shared node, priority here is age-based with no fair-share weighting, so
# a wall of pending jobs really would sit in front of everyone else's work.
# PIPELINE_UPLOAD=1  publish this run when it succeeds.
# PIPELINE_ADVANCE=1 also pull the next dataset in (implies PIPELINE_UPLOAD).
# An isolated run wants the first without the second: publish and free the disk, but do
# not drag the rest of the queue along.
if [ "${PIPELINE_ADVANCE:-0}" = "1" ] || [ "${PIPELINE_UPLOAD:-0}" = "1" ]; then
    if [ "${TRAIN_STATUS}" -eq 0 ]; then
        upload_id=$(sbatch --parsable \
            --job-name "hfup-${RUN_SLUG:-${RUN_NAME#gr00t_n1d7_finetune_output_}}" \
            --export "ALL,RUN_NAME=${RUN_NAME},HF_REPO_ID=${HF_REPO_ID:-},HF_BRANCH=${HF_BRANCH:-},UPLOAD_WHAT=${UPLOAD_WHAT:-}" \
            "${PROJECT_DIR}/submit_upload_slurm.sh" 2>&1) \
            && echo "[INFO] Upload job submitted: ${upload_id}" \
            || echo "[WARN] Could not submit the upload job: ${upload_id}"
    else
        echo "[INFO] Training failed — no upload submitted, nothing deleted."
    fi

    if [ "${PIPELINE_ADVANCE:-0}" = "1" ]; then
        # Pull in the next dataset regardless of this one's outcome: a dataset that
        # cannot train must not stall the queue behind it.
        echo "[INFO] Advancing the pipeline..."
        "${PROJECT_DIR}/run_pipeline.sh" --advance || \
            echo "[WARN] --advance failed; run ./run_pipeline.sh manually to resume"
    else
        echo "[INFO] Isolated run — not pulling in another dataset."
    fi
fi

echo ""
echo "=================================================="
echo "Job finished: $(date)"
echo "=================================================="

exit "${TRAIN_STATUS}"
