#!/usr/bin/env bash
#
# SLURM submission script — GR00T-N1.7 fine-tuning via Apptainer
#
# Usage:
#   sbatch submit_slurm.sh
#
# NOTE: run sbatch from the repo root, and make sure logs/ exists there first
# (`mkdir -p logs`). The --output path below is relative to the submitting directory, and
# SLURM creates that file before this script runs, so a missing logs/ kills the job with
# no log to explain it. logs/ is gitignored, so a fresh clone does not have it.
#
# Pick the dataset (a directory name under data/simple/simple-converted/):
#   sbatch --export=ALL,DATASET_NAME=G1WholebodyCloseDoorTeleop-v0 submit_slurm.sh
#
# One specialist per dataset, one after another, from a single submission:
#   sbatch --array=0-2%1 --export=ALL,DATASETS="carry_totes screwdrivers screws" submit_slurm.sh
#
# Start from other weights than the base model (a directory name under checkpoints/):
#   sbatch --export=ALL,BASE_MODEL=gr00t_n1d7_finetune_output_totes_shelf_to_table_render/final,DATASET_NAME=carry_totes submit_slurm.sh
#
# Shorten training for a small dataset:
#   sbatch --export=ALL,DATASET_NAME=carry_totes,MAX_STEPS=8000 submit_slurm.sh
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

# Under `sbatch --array`, the task id picks the dataset: one specialist per subtask from a
# single submission, serialised with %1. This must run BEFORE the RUN_NAME block below,
# which derives the run name from DATASET_NAME. Outside an array the block is inert.
#   sbatch --array=0-2%1 --export=ALL,DATASETS="carry_totes screwdrivers screws" submit_slurm.sh
DATASETS="${DATASETS:-}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${DATASETS}" ]; then
    read -ra _DATASET_LIST <<< "${DATASETS}"
    if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#_DATASET_LIST[@]}" ]; then
        echo "ERROR: --array index ${SLURM_ARRAY_TASK_ID} is out of range."
        echo "       DATASETS holds ${#_DATASET_LIST[@]} entries: ${DATASETS}"
        echo "       Use --array=0-$(( ${#_DATASET_LIST[@]} - 1 ))%1"
        exit 1
    fi
    DATASET_NAME="${_DATASET_LIST[${SLURM_ARRAY_TASK_ID}]}"
    echo "[INFO] Array task ${SLURM_ARRAY_TASK_ID}/$(( ${#_DATASET_LIST[@]} - 1 )) -> ${DATASET_NAME}"
fi

# Dataset directory name under data/simple/simple-converted/ — produced and validated by
# scripts/prepare_simple_datasets.py. Kept as a name, not a path, so the run name can be
# derived from it and the container path stays fixed.
DATASET_NAME="${DATASET_NAME:-G1WholebodyOpenOvenTeleop-v0}"
DATASET_HOST_PATH="${PROJECT_DIR}/data/simple/simple-converted/${DATASET_NAME}"

# Starting weights: a directory name under checkpoints/. Kept as a name for the same reason
# as the dataset — the host path is checkable before submitting, the container path is fixed.
#
# GR00T-N1.7-3B is the pristine base model and is not public. A checkpoint already
# fine-tuned on the same embodiment works as a starting point too, e.g.
#   BASE_MODEL=gr00t_n1d7_finetune_output_totes_shelf_to_table_render/final
# Doing so means you are no longer fine-tuning from base: record which one a run used.
BASE_MODEL="${BASE_MODEL:-GR00T-N1.7-3B}"
BASE_MODEL_HOST_PATH="${PROJECT_DIR}/checkpoints/${BASE_MODEL}"

# Training length. Two ways to say it:
#   MAX_STEPS=16000       exact number of steps (the historical knob)
#   TARGET_EPOCHS=20      passes over THIS dataset; steps are derived from its frame count
#
# TARGET_EPOCHS is what you want across an --array: a fixed MAX_STEPS means very different
# amounts of training for datasets of different sizes, so the small one memorises while the
# large one barely converges.
MAX_STEPS="${MAX_STEPS:-50000}"
TARGET_EPOCHS="${TARGET_EPOCHS:-}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"

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

# Without this the run allocates a GPU, loads the dataset, and only then dies inside
# launch_finetune_n1d7_inner.py's load_checkpoint_model_config().
if [ ! -d "${BASE_MODEL_HOST_PATH}" ]; then
    echo "ERROR: base model not found: ${BASE_MODEL_HOST_PATH}"
    echo "       Set BASE_MODEL to a directory name under checkpoints/, e.g."
    echo "         BASE_MODEL=gr00t_n1d7_finetune_output_totes_shelf_to_table_render/final"
    exit 1
fi
if [ ! -f "${BASE_MODEL_HOST_PATH}/config.json" ]; then
    echo "ERROR: ${BASE_MODEL_HOST_PATH}/config.json is missing."
    echo "       --base-model-path needs a model directory, not the run directory above it."
    exit 1
fi

########################
# DERIVE THE SCHEDULE  #
########################

# Done on the host so the numbers land in the log before a GPU is allocated.
TOTAL_FRAMES=""
TOTAL_EPISODES=""
if [ -f "${DATASET_HOST_PATH}/meta/info.json" ]; then
    read -r TOTAL_FRAMES TOTAL_EPISODES <<<"$(python3 -c "
import json; d = json.load(open('${DATASET_HOST_PATH}/meta/info.json'))
print(d.get('total_frames', 0), d.get('total_episodes', 0))
" 2>/dev/null || echo "0 0")"
fi

if [ -n "${TARGET_EPOCHS}" ]; then
    if [ "${TOTAL_FRAMES:-0}" -le 0 ]; then
        echo "ERROR: TARGET_EPOCHS needs total_frames from ${DATASET_HOST_PATH}/meta/info.json"
        exit 1
    fi
    MAX_STEPS=$(python3 -c "print(max(500, int(${TOTAL_FRAMES} * ${TARGET_EPOCHS} / ${GLOBAL_BATCH_SIZE})))")
fi

# --save-steps was a fixed 10000. That silently produces ZERO intermediate checkpoints on
# any run shorter than that: the job writes nothing until final/, and a crash at 9k steps
# loses everything. Keep 10000 for long runs, scale down for short ones.
SAVE_STEPS="${SAVE_STEPS:-$(python3 -c "print(min(10000, max(200, ${MAX_STEPS} // 5)))")}"

echo ""
echo "--- Schedule ---"
echo "Episodes:         ${TOTAL_EPISODES:-?}"
echo "Frames:           ${TOTAL_FRAMES:-?}"
echo "Global batch:     ${GLOBAL_BATCH_SIZE}"
[ -n "${TARGET_EPOCHS}" ] && echo "Target epochs:    ${TARGET_EPOCHS}"
echo "Max steps:        ${MAX_STEPS}"
echo "Checkpoint every: ${SAVE_STEPS}"
if [ "${TOTAL_FRAMES:-0}" -gt 0 ]; then
    EPOCHS_EQ=$(python3 -c "print(round(${MAX_STEPS} * ${GLOBAL_BATCH_SIZE} / ${TOTAL_FRAMES}, 1))")
    echo "Epochs equivalent: ${EPOCHS_EQ}"
fi
echo ""

########################
# VENV INTERPRETER     #
########################

# The training Python is the HOST venv, reached through the src/ bind. Its bin/python is a
# symlink to an absolute path, and that path must also exist inside the container:
#
#   /usr/bin/python3.10          the image has it — nothing to do
#   /raid/.../uv/python/...      a uv-managed interpreter — must be bind-mounted at the
#                                SAME path, or bin/python is a dangling symlink and
#                                apptainer dies with "stat ...: no such file or directory"
#
# Failing here costs a queue slot on a busy cluster, so resolve it before submitting work.
# The user's scratch area, mounted at its OWN path inside the container. This is what makes
# host conventions survive the container boundary:
#
#   ~/.cache -> /raid/$USER/cache      the symlink resolves instead of dangling
#   TRITON_CACHE_DIR=/raid/...         the variable points somewhere writable
#   a uv-managed interpreter on /raid   the venv's bin/python symlink resolves
#
# Apptainer mounts $HOME, the working directory and /tmp automatically, and nothing else.
# Everything under /raid other than this repo is a read-only shell created just to hold the
# working-directory mountpoint — writing there fails with EROFS, and the traceback names a
# library's cache directory rather than anything about containers.
#
# Path-identical is the point: a translated bind (/raid/x -> /workspace/y) fixes neither
# symlinks nor environment variables, because both store the original path.
SCRATCH_DIR="${SCRATCH:-/raid/${USER}}"
SCRATCH_BIND=()
if [ -d "${SCRATCH_DIR}" ]; then
    SCRATCH_BIND=(--bind "${SCRATCH_DIR}:${SCRATCH_DIR}")
    echo "[INFO] Binding ${SCRATCH_DIR} at the same path inside the container."
else
    echo "[WARN] ${SCRATCH_DIR} not found — not bound. Cache paths pointing there will fail."
fi

VENV_PY="${PROJECT_DIR}/src/gr00t/.venv-gr00t/bin/python"
PY_BINDS=()

# -e follows symlinks, so a dangling one reads as absent; -L separates the two cases,
# which need different fixes.
if [ ! -e "${VENV_PY}" ] && [ ! -L "${VENV_PY}" ]; then
    echo "ERROR: training interpreter not found: ${VENV_PY}"
    echo "       Create it with (must be 3.10 + cuda12, to match the image):"
    echo "         cd src/gr00t && uv venv .venv-gr00t --python 3.10 && uv sync --active --extra cuda12"
    exit 1
fi

VENV_PY_REAL=$(readlink -f "${VENV_PY}" 2>/dev/null || true)
if [ -z "${VENV_PY_REAL}" ] || [ ! -e "${VENV_PY_REAL}" ]; then
    echo "ERROR: ${VENV_PY} points at ${VENV_PY_REAL:-<unresolved>}, which does not exist."
    echo "       The venv's base interpreter is gone. Recreate it with 3.10 + cuda12:"
    echo "         cd src/gr00t && rm -rf .venv-gr00t"
    echo "         uv venv .venv-gr00t --python 3.10 && uv sync --active --extra cuda12"
    exit 1
fi

case "${VENV_PY_REAL}" in
    /usr/*)
        # A system interpreter. It resolves inside the container only if the image ships
        # the same version — the image is Python 3.10, so a 3.12 venv fails here.
        echo "[INFO] Training interpreter: ${VENV_PY_REAL} (expected to exist in the image)"
        ;;
    *)
        PY_ROOT="${VENV_PY_REAL%/bin/*}"
        PY_BINDS+=(--bind "${PY_ROOT}:${PY_ROOT}")
        echo "[INFO] Training interpreter is managed: ${VENV_PY_REAL}"
        echo "[INFO] Binding ${PY_ROOT} at the same path so the symlink resolves."
        ;;
esac

########################
# PREPARE DIRECTORIES  #
########################

mkdir -p \
    "${PROJECT_DIR}/logs" \
    "${PROJECT_DIR}/checkpoints/${RUN_NAME}" \
    "${PROJECT_DIR}/cache/huggingface" \
    "${PROJECT_DIR}/cache/torch" \
    "${PROJECT_DIR}/cache/wandb" \
    "${PROJECT_DIR}/cache/triton"

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
# Apptainer injects the host's driver libraries with --nv, from a fixed list that includes
# the OpenGL stack. When the host distro is newer than the image, those libraries need a
# glibc the image does not have, and anything that resolves OpenGL fails to load — torchcodec
# reaches it through ffmpeg, so video decoding dies in the dataloader with a GLIBC error that
# names none of this.
#
#   NV_FLAGS="--nv --nvccli"   delegate injection to nvidia-container-cli, which brings what
#                              CUDA needs and leaves the graphics stack out
#   NV_FLAGS="--nv"            the default, and what a matching host/image pair wants
read -ra NV_FLAGS <<< "${NV_FLAGS:---nv}"
echo "[INFO] Apptainer GPU flags: ${NV_FLAGS[*]}"

# --nvccli exposes GPUs through nvidia-container-cli, which reads NVIDIA_VISIBLE_DEVICES.
# Apptainer defaults it to "all" to emulate --nv, and "all" on a shared node means every
# GPU on the machine, including the ones SLURM allocated to other people. The device cgroup
# usually blocks the access anyway, but relying on that is one mistake away from stepping on
# a colleague's run. Expose exactly what SLURM granted.
NV_ENV=()
for f in "${NV_FLAGS[@]}"; do
    if [ "$f" = "--nvccli" ]; then
        _gpus="${SLURM_JOB_GPUS:-${GPU_DEVICE_ORDINAL:-${CUDA_VISIBLE_DEVICES:-}}}"
        if [ -n "${_gpus}" ]; then
            NV_ENV=(--env NVIDIA_VISIBLE_DEVICES="${_gpus}")
            echo "[INFO] NVIDIA_VISIBLE_DEVICES=${_gpus} (from SLURM, not 'all')"
        else
            echo "[WARN] --nvccli without a SLURM GPU allocation: the container may see"
            echo "       every GPU on the node. Do not run training this way."
        fi
        break
    fi
done

TRAIN_STATUS=0
set +e

apptainer exec \
    "${NV_FLAGS[@]}" \
    "${NV_ENV[@]}" \
    --bind "${PROJECT_DIR}/data:/workspace/data" \
    --bind "${PROJECT_DIR}/checkpoints:/workspace/checkpoints" \
    --bind "${PROJECT_DIR}/src:/workspace/src" \
    --bind "${PROJECT_DIR}/baselines:/workspace/baselines" \
    --bind "${PROJECT_DIR}/scripts:/workspace/scripts" \
    --bind "${PROJECT_DIR}/third_party:/workspace/third_party" \
    `# Bind the cache ROOT, not individual subdirectories: everything the container`  \
    `# writes under /workspace/cache then lands on the host, including directories`    \
    `# created at runtime. Binding only the known subdirs leaves /workspace/cache`     \
    `# itself read-only, so a library inventing a new cache path still dies.`          \
    --bind "${PROJECT_DIR}/cache:/workspace/cache" \
    "${SCRATCH_BIND[@]}" \
    "${PY_BINDS[@]}" \
    --env DATASET_NAME="${DATASET_NAME}" \
    --env RUN_NAME="${RUN_NAME}" \
    --env BASE_MODEL="${BASE_MODEL}" \
    --env MAX_STEPS="${MAX_STEPS}" \
    --env SAVE_STEPS="${SAVE_STEPS}" \
    --env GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
    --env MASTER_PORT="${MASTER_PORT}" \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env WANDB_ENTITY="${WANDB_ENTITY:-industrial_humanoids}" \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    --env HF_HOME="/workspace/cache/huggingface" \
    --env TORCH_HOME="/workspace/cache/torch" \
    `# Everything else cache-related is inherited from the host and resolves through the`  \
    `# scratch bind above, so it is deliberately not overridden here.`                     \
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
            --base-model-path "/workspace/checkpoints/${BASE_MODEL}" \
            --dataset-path "$DATASET_PATH" \
            --embodiment-tag G1_LOCO_DOWNSTREAM \
            --save-steps "${SAVE_STEPS}" \
            --save-total-limit 1 \
            --max-steps "${MAX_STEPS}" \
            --warmup-ratio 0.05 \
            --weight-decay 1e-05 \
            --learning-rate 0.0001 \
            --global-batch-size "${GLOBAL_BATCH_SIZE}" \
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
