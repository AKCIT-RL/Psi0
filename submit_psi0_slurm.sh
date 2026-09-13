#!/usr/bin/env bash
#
# SLURM submission script — Ψ₀ fine-tuning via Apptainer
#
# Counterpart of submit_slurm.sh (GR00T-N1.7) for the native Ψ₀ stack. Same skeleton — quota
# preflight, rendezvous port, sanity gates — and the SAME Apptainer image. What differs is the
# interpreter (the image's own /workspace/.venv rather than the host venv), the binds, and the
# inner command.
#
# Usage — one dataset:
#   sbatch --export=ALL,DATASET_NAME=carry_totes submit_psi0_slurm.sh
#
# Usage — one specialist per dataset, trained one after another, from a single submission:
#   sbatch --array=0-2%1 submit_psi0_slurm.sh
#
# The %1 is what serialises them: without it all three start at once and fight for GPUs.
# Each array task is a separate job with its own time limit and its own exit status, so a
# failure in the second does not stop the third, and re-running just that one is
# `--array=1`. A `for` loop inside a single job would share one 48h budget across all
# three and lose the whole allocation when it runs out.
#
# The dataset list comes from DATASETS (space-separated), indexed by SLURM_ARRAY_TASK_ID:
#   sbatch --array=0-1%1 --export=ALL,DATASETS="screws screwdrivers" submit_psi0_slurm.sh
#
# Run names are derived per dataset, so the three write to separate directories and their
# checkpoints never collide.
#
# Common overrides:
#   TARGET_EPOCHS=30 TRAIN_BATCH_SIZE=8 sbatch --export=ALL,DATASET_NAME=screws,TARGET_EPOCHS,TRAIN_BATCH_SIZE submit_psi0_slurm.sh
#   sbatch --gres=gpu:2 --partition=gpu submit_psi0_slurm.sh
#
# Not wired to run_pipeline.sh: that pipeline's bookkeeping (PROVENANCE.json, run_slug,
# the gr00t_n1d7_finetune_output_* naming that submit_upload_slurm.sh keys off) is
# GR00T-specific. Publish a Ψ₀ run by hand until that is generalised.

########################
# SLURM CONFIGURATION  #
########################

#SBATCH --job-name=psi0-finetune
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/psi0-%j.out
#SBATCH --error=logs/psi0-%j.err
# TODO(cluster): add `#SBATCH --partition=<name>` once Phase 0 tells you the name, and
# adjust --gres to the exact GRES string this cluster expects (e.g. gpu:h100:1).

set -euo pipefail

########################
# PATHS                #
########################

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "${BASH_SOURCE[0]}")")}}"

# The SAME image submit_slurm.sh uses — one .sif serves both stacks. Built from
# docker/Dockerfile and named industrial_humanoids_psi0-train.gr00t_devel.sif by
# docs/pipeline_finetune.md. The difference is which interpreter each stack runs: GR00T uses
# the host venv bind-mounted at /workspace/src/gr00t/.venv-gr00t, while Ψ₀ uses the image's
# own /workspace/.venv, where docker/Dockerfile installed the psi package editable.
SIF_PATH="${SIF_PATH:-${PROJECT_DIR}/industrial_humanoids_psi0-train.gr00t_devel.sif}"

ENV_FILE="${ENV_FILE:-${PROJECT_DIR}/.env}"

# Pre-trained Ψ₀ checkpoints (VLM backbone + action header). Downloaded on the LOGIN node —
# compute nodes commonly have no route to huggingface.co, and discovering that inside the
# job wastes the whole allocation. See docs/plano_cluster.md, Phase 3.
PSI_CKPT_DIR="${PSI_CKPT_DIR:-${SCRATCH:-/raid/${USER}}/cache/checkpoints/psi0}"
VLM_CKPT_NAME="${VLM_CKPT_NAME:-pre.fast.1by1.2601091803.ckpt.ego200k.he30k}"
ACTION_CKPT_NAME="${ACTION_CKPT_NAME:-postpre.1by1.pad36.2601131206.ckpt.he30k}"

# Where runs are written. Kept out of the repo tree by default so a full disk never
# corrupts the working copy.
RUNS_DIR="${RUNS_DIR:-${SCRATCH:-/raid/${USER}}/psi0-runs}"

########################
# WHAT TO TRAIN        #
########################

# Under `sbatch --array`, the task id picks the dataset: one specialist per subtask from a
# single submission. Outside an array this whole block is inert and DATASET_NAME rules.
DATASETS="${DATASETS:-carry_totes screwdrivers screws}"

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
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

DATASET_NAME="${DATASET_NAME:-carry_totes}"
DATASET_ROOT_HOST="${DATASET_ROOT_HOST:-${PROJECT_DIR}/data/simple/simple-converted}"
DATASET_HOST_PATH="${DATASET_ROOT_HOST}/${DATASET_NAME}"

# Inside the container the data root is fixed, so the inner command never varies.
DATASET_ROOT_CTR="/workspace/data"

EXP="${EXP:-$(echo "${DATASET_NAME}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-' \
      | sed 's/--*/-/g; s/^-//; s/-$//')}"
RUN_NAME="${RUN_NAME:-psi0_finetune_${EXP//-/_}}"

TARGET_EPOCHS="${TARGET_EPOCHS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"

########################
# DISK PREFLIGHT       #
########################

# submit_slurm.sh parses `quota`, which reports the HOME filesystem. With the repo and the
# runs on /raid that check measures the wrong thing entirely, so measure the filesystem
# the run actually writes to instead.
#
# Caveat worth knowing: df reports filesystem-wide free space, not your per-user quota. If
# this cluster enforces a quota on /raid, df can read as roomy while you are already at
# your limit. Check `quota -s` by hand once, at setup time.
REQUIRED_GB="${REQUIRED_GB:-60}"
mkdir -p "${RUNS_DIR}"
AVAIL_GB=$(df -BG --output=avail "${RUNS_DIR}" 2>/dev/null | tail -1 | tr -dc '0-9' || echo "")

if [ -z "${AVAIL_GB}" ]; then
    echo "[WARN] Could not read free space on ${RUNS_DIR}; continuing without the check."
elif [ "${AVAIL_GB}" -lt "${REQUIRED_GB}" ]; then
    echo "ERROR: only ${AVAIL_GB} GB free on ${RUNS_DIR}, ${REQUIRED_GB} GB needed."
    echo "       Running out mid-run truncates the checkpoint hours in — refusing to start."
    exit 1
else
    echo "[INFO] Free space on ${RUNS_DIR}: ${AVAIL_GB} GB (need ${REQUIRED_GB} GB)"
fi

########################
# RENDEZVOUS PORT      #
########################

# Identical rationale to submit_slurm.sh: a fixed port only works while one job runs at a
# time. Derive from the job id, then scan upward for one actually free.
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
echo "Ψ₀ Fine-tuning — Apptainer + SLURM"
echo "=================================================="
echo "Job ID:       ${SLURM_JOB_ID:-<local>}"
echo "Node:         ${SLURM_NODELIST:-<local>}"
echo "Start time:   $(date)"
echo "PROJECT_DIR:  ${PROJECT_DIR}"
echo "SIF_PATH:     ${SIF_PATH}"
echo "DATASET:      ${DATASET_NAME}"
echo "EXP:          ${EXP}"
echo "RUN_NAME:     ${RUN_NAME}"
echo "RUNS_DIR:     ${RUNS_DIR}"
echo "MASTER_PORT:  ${MASTER_PORT}"
echo "=================================================="

[ -f "${SIF_PATH}" ] || { echo "ERROR: Apptainer image not found: ${SIF_PATH}"; exit 1; }
[ -d "${DATASET_HOST_PATH}" ] || { echo "ERROR: dataset not found: ${DATASET_HOST_PATH}"; exit 1; }

# The Ψ₀ gate is stats_psi0.json, not modality.json: normalisation statistics are read from
# it (--data.transform.field.stat-path). Missing means the dataset was never prepared for
# this stack, and training would either crash late or normalise against the wrong ranges.
#
# Note: scripts/data/calc_modality_stats.py writes meta/stats.json, not meta/stats_psi0.json.
# If you generated it with that script, rename the file.
for rel in meta/info.json meta/stats_psi0.json meta/tasks.jsonl meta/episodes.jsonl data videos; do
    if [ ! -e "${DATASET_HOST_PATH}/${rel}" ]; then
        echo "ERROR: missing ${DATASET_HOST_PATH}/${rel}"
        echo "       stats_psi0.json can be produced with:"
        echo "         python scripts/data/calc_modality_stats.py --task-dir ${DATASET_HOST_PATH}"
        echo "         mv ${DATASET_HOST_PATH}/meta/stats.json ${DATASET_HOST_PATH}/meta/stats_psi0.json"
        exit 1
    fi
done

for ck in "${PSI_CKPT_DIR}/${VLM_CKPT_NAME}" "${PSI_CKPT_DIR}/${ACTION_CKPT_NAME}"; do
    if [ ! -e "${ck}" ]; then
        echo "ERROR: pre-trained checkpoint not found: ${ck}"
        echo "       Download it on the LOGIN node (compute nodes usually have no internet):"
        echo "         hf download USC-PSI-Lab/psi-model --repo-type=model \\"
        echo "           --include \"psi0/${VLM_CKPT_NAME}/**\" \\"
        echo "           --include \"psi0/${ACTION_CKPT_NAME}/**\" \\"
        echo "           --local-dir $(dirname "${PSI_CKPT_DIR}")"
        exit 1
    fi
done

########################
# DERIVE THE SCHEDULE   #
########################

# Computed on the host so the numbers appear in the log before a GPU is allocated: with 20
# episodes the default 50 epochs is deep into memorisation territory, and you want to see
# that in the header rather than in a deployment failure.
read -r TOTAL_FRAMES TOTAL_EPISODES <<<"$(python3 -c "
import json; d = json.load(open('${DATASET_HOST_PATH}/meta/info.json'))
print(d['total_frames'], d['total_episodes'])
")"

EFFECTIVE_BATCH=$(( TRAIN_BATCH_SIZE * NUM_GPUS ))
read -r MAX_STEPS CKPT_STEPS VAL_STEPS <<<"$(python3 -c "
steps = max(1000, int(${TOTAL_FRAMES} / ${EFFECTIVE_BATCH} * ${TARGET_EPOCHS}))
print(steps, max(200, steps // 10), max(100, steps // 20))
")"

echo ""
echo "--- Schedule ---"
echo "Episodes:         ${TOTAL_EPISODES}"
echo "Frames:           ${TOTAL_FRAMES}"
echo "GPUs:             ${NUM_GPUS}"
echo "Effective batch:  ${EFFECTIVE_BATCH} (${TRAIN_BATCH_SIZE} x ${NUM_GPUS})"
echo "Target epochs:    ${TARGET_EPOCHS}"
echo "Max steps:        ${MAX_STEPS}"
echo "Checkpoint every: ${CKPT_STEPS}"
echo "Validate every:   ${VAL_STEPS}"
if [ "${TOTAL_EPISODES}" -lt 30 ] && [ "${TARGET_EPOCHS}" -gt 30 ]; then
    echo "[WARN] ${TOTAL_EPISODES} episodes x ${TARGET_EPOCHS} epochs — high overfitting risk."
    echo "       Consider TARGET_EPOCHS=20..30 for a dataset this small."
fi
echo ""

########################
# PREPARE DIRECTORIES  #
########################

mkdir -p \
    "${PROJECT_DIR}/logs" \
    "${RUNS_DIR}/${RUN_NAME}" \
    "${SCRATCH:-/raid/${USER}}/cache/huggingface" \
    "${SCRATCH:-/raid/${USER}}/cache/torch" \
    "${SCRATCH:-/raid/${USER}}/cache/wandb"

module load apptainer 2>/dev/null || true

########################
# LOAD SECRETS         #
########################

if [ -f "${ENV_FILE}" ]; then
    set -a
    # shellcheck disable=SC1090
    source <(grep -v '^\s*#' "${ENV_FILE}" | grep -v '^\s*$')
    set +a
    echo "[INFO] Loaded environment from ${ENV_FILE}"
fi

# scripts/train.py line 3 is `assert load_dotenv()` — it aborts before printing anything
# useful when no .env is reachable from the working directory. The host .env cannot be
# reused as-is: its paths (PSI_HOME, DATA_HOME, HF_HOME) refer to host locations that mean
# something different inside the container. So synthesise a container-local one.
#
# load_dotenv() does not override variables already set, so the --env flags below still win.
CONTAINER_ENV="${RUNS_DIR}/${RUN_NAME}/.env.container"
cat > "${CONTAINER_ENV}" <<EOF
PSI_HOME=/workspace/psi_ckpts
DATA_HOME=/workspace/data
HF_HOME=/workspace/cache/huggingface
TORCH_HOME=/workspace/cache/torch
OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
TOKENIZERS_PARALLELISM=false
EOF

########################
# APPTAINER EXEC       #
########################

# Bind notes:
#   src/psi   — the image installs psi with `uv pip install -e .`, so the editable install
#               resolves through /workspace/src/psi. Mounting the host copy there means the
#               code you edited is the code that runs. Set MOUNT_SRC=0 to train against the
#               code baked into the image instead (reproducible, but needs a rebuild to change).
#   NOT src/  — mounting the whole src/ tree would drag in the other baselines and their
#               host venvs, which are built for a different Python than this image's.
#   runs      — outside the repo, so a full disk cannot corrupt the working copy.

BINDS=(
    --bind "${DATASET_ROOT_HOST}:${DATASET_ROOT_CTR}"
    --bind "${RUNS_DIR}:/workspace/runs"
    --bind "${PSI_CKPT_DIR}:/workspace/psi_ckpts"
    --bind "${CONTAINER_ENV}:/workspace/.env"
    --bind "${SCRATCH:-/raid/${USER}}/cache/huggingface:/workspace/cache/huggingface"
    --bind "${SCRATCH:-/raid/${USER}}/cache/torch:/workspace/cache/torch"
    --bind "${SCRATCH:-/raid/${USER}}/cache/wandb:/workspace/cache/wandb"
)
if [ "${MOUNT_SRC:-1}" = "1" ]; then
    BINDS+=(--bind "${PROJECT_DIR}/src/psi:/workspace/src/psi")
    BINDS+=(--bind "${PROJECT_DIR}/scripts:/workspace/scripts")
fi

LOG_BACKEND="wandb"
if [ -z "${WANDB_API_KEY:-}" ] || [ "${WANDB_DISABLED:-0}" = "1" ]; then
    LOG_BACKEND="tensorboard"
    echo "[INFO] No WANDB_API_KEY — logging to tensorboard."
fi

TRAIN_STATUS=0
set +e

apptainer exec \
    --nv \
    "${BINDS[@]}" \
    --env DATASET_NAME="${DATASET_NAME}" \
    --env EXP="${EXP}" \
    --env RUN_NAME="${RUN_NAME}" \
    --env MASTER_PORT="${MASTER_PORT}" \
    --env NUM_GPUS="${NUM_GPUS}" \
    --env TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
    --env MAX_STEPS="${MAX_STEPS}" \
    --env CKPT_STEPS="${CKPT_STEPS}" \
    --env VAL_STEPS="${VAL_STEPS}" \
    --env VLM_CKPT="/workspace/psi_ckpts/${VLM_CKPT_NAME}" \
    --env ACTION_CKPT="/workspace/psi_ckpts/${ACTION_CKPT_NAME}" \
    --env LOG_BACKEND="${LOG_BACKEND}" \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env WANDB_ENTITY="${WANDB_ENTITY:-industrial_humanoids}" \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    --env HF_HOME="/workspace/cache/huggingface" \
    --env TORCH_HOME="/workspace/cache/torch" \
    --env OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
    --env TOKENIZERS_PARALLELISM="false" \
    --env TF_CPP_MIN_LOG_LEVEL="3" \
    --env DS_BUILD_OPS="0" \
    --env DS_SKIP_CUDA_CHECK="1" \
    --env NCCL_DEBUG="WARN" \
    --env NCCL_ASYNC_ERROR_HANDLING="1" \
    "${SIF_PATH}" \
    bash -c '
        set -euo pipefail
        cd /workspace

        echo "--- Container environment ---"
        python_bin="/workspace/.venv/bin/python"
        echo "Python:  $($python_bin --version)"
        echo "Torch:   $($python_bin -c "import torch; print(torch.__version__, torch.cuda.is_available())")"
        echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-all}"
        echo ""

        # Many small parquet shards plus video decoding open a lot of descriptors at once.
        ulimit -n 65535 2>/dev/null || true

        # NOTE: --train.output_dir combines with --train.name into <output_dir>/<name>/<exp>.<timestamp>.
        # NOTE: auto_tag_run defaults to False (src/psi/config/config.py). Leave it off — it
        #       runs `git add . && git commit` inside the job.
        $python_bin -m torch.distributed.run \
            --nproc_per_node "${NUM_GPUS}" \
            --master_port "${MASTER_PORT}" \
            /workspace/scripts/train.py \
            finetune_real_psi0_config \
            --seed=292285 \
            --exp="${EXP}" \
            --train.name=finetune \
            --train.output_dir="/workspace/runs/${RUN_NAME}" \
            --train.data_parallel=ddp \
            --train.mixed_precision=bf16 \
            --train.train_batch_size="${TRAIN_BATCH_SIZE}" \
            --train.max_checkpoints_to_keep=3 \
            --train.gradient_accumulation_steps=1 \
            --train.learning_rate=1e-4 \
            --train.max_training_steps="${MAX_STEPS}" \
            --train.warmup_ratio=None \
            --train.warmup_steps=500 \
            --train.checkpointing_steps="${CKPT_STEPS}" \
            --train.validation_steps="${VAL_STEPS}" \
            --train.val_num_batches=20 \
            --train.max_grad_norm=1.0 \
            --train.lr_scheduler_type=cosine \
            --train.lr_scheduler_kwargs.weight_decay=1e-6 \
            --train.lr_scheduler_kwargs.betas 0.95 0.999 \
            --log.report_to="${LOG_BACKEND}" \
            --data.root_dir="/workspace/data" \
            --data.train_repo_ids="${DATASET_NAME}" \
            --data.transform.repack.pad-action-dim=36 \
            --data.transform.repack.pad-state-dim=36 \
            --data.transform.field.stat-path=meta/stats_psi0.json \
            --data.transform.field.stat-action-key=action \
            --data.transform.field.stat-state-key=states \
            --data.transform.field.action_norm_type=bounds \
            --data.transform.field.no-use-norm-mask \
            --data.transform.field.normalize-state \
            --data.transform.field.pad-action-dim=36 \
            --data.transform.field.pad-state-dim=36 \
            --data.transform.model.img-aug \
            --data.transform.model.resize.size 240 320 \
            --data.transform.model.center_crop.size 240 320 \
            --model.model_name_or_path="${VLM_CKPT}" \
            --model.pretrained-action-header-path="${ACTION_CKPT}" \
            --model.noise-scheduler=flow \
            --model.train-diffusion-steps=1000 \
            --model.n_conditions=0 \
            --model.action-chunk-size=30 \
            --model.action-dim=36 \
            --model.action-exec-horizon=30 \
            --model.observation-horizon=1 \
            --model.odim=36 \
            --model.view_feature_dim=2048 \
            --model.no-tune-vlm \
            --model.no-use_film \
            --model.no-combined_temb \
            --model.rtc \
            --model.max-delay=8
    '
TRAIN_STATUS=$?
set -e

echo ""
echo "=================================================="
echo "Training finished: $(date)  (exit ${TRAIN_STATUS})"
echo "Run directory: ${RUNS_DIR}/${RUN_NAME}/finetune"
echo "=================================================="

# Resuming is not automatic. src/psi/config/config.py only auto-resumes when
# --train.resume_from_checkpoint=latest AND an explicit --timestamp matches an existing run
# directory (the resume-from-newest branch is commented out). To resume, read the timestamp
# off the run directory and pass both flags.

exit "${TRAIN_STATUS}"
