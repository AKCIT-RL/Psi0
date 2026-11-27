#!/usr/bin/env bash
# A training and then the eval of the model it produced, in ONE SLURM job: the GPU goes straight
# from one to the other, so the eval does not wait in the queue again. Everything is set in a config
# file (see train_eval.env.example). Run it on the login shell; it submits itself:
#
#   ovx/slurm/train_eval.sh [config]        # default ovx/slurm/train_eval.env
#
# The config is frozen at submission into logs/train-eval-<run>.env, and the job reads that copy:
# editing the config afterwards does not change a queued job. The CODE is read from the clone when
# the job starts, so edits made while it waits go into it (point OVX_SRC at a snapshot to freeze it).
#
# The eval runs only if the training succeeded. To evaluate again, or another model, use eval.sbatch.
# Both stages run through train.sbatch and eval.sbatch, so the quota check and the logs are theirs.

#SBATCH --job-name=ovx-train-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=54:00:00
#SBATCH --output=logs/ovx-train-eval-%j.out

set -euo pipefail

# load <config>: defaults, then the file. set -a exports the OVX_* variables to sbatch and to the
# container launcher.
load() {
    RUN_NAME=""; EVAL_DATA=""; TRAIN_ARGS=(); EVAL_ARGS=(); SBATCH_ARGS=()
    set -a
    # shellcheck disable=SC1090
    source "$1"
    set +a
}

if [ "${1:-}" != --in-job ]; then
    # ---- on the login shell: check, freeze the config, submit ----
    cd "$(dirname "${BASH_SOURCE[0]}")/../.."
    CONFIG="${1:-ovx/slurm/train_eval.env}"
    [ -f "${CONFIG}" ] || { echo "config not found: ${CONFIG} -- copy ovx/slurm/train_eval.env.example"; exit 1; }
    load "${CONFIG}"
    for v in OVX_SIF DATASET BASE_MODEL ENV_ID; do
        [ -n "${!v:-}" ] || { echo "${v} is not set in ${CONFIG}"; exit 1; }
    done
    [ -f "${OVX_SIF}" ] || [ -d "${OVX_SIF}" ] || { echo "OVX_SIF not found: ${OVX_SIF}"; exit 1; }

    RUN_NAME="${RUN_NAME:-${DATASET}_$(date +%Y%m%d-%H%M)}"
    mkdir -p logs
    FROZEN="logs/train-eval-${RUN_NAME}.env"
    { cat "${CONFIG}"; printf '\n# frozen by train_eval.sh at submission\nRUN_NAME=%s\n' "${RUN_NAME}"; } > "${FROZEN}"

    JOB=$(sbatch --parsable "${SBATCH_ARGS[@]}" ovx/slurm/train_eval.sh --in-job "${FROZEN}")
    echo "run:    ${RUN_NAME}   (config: ${FROZEN})"
    echo "job:    ${JOB}   log: logs/ovx-train-eval-${JOB}.out"
    echo "model:  ${OVX_CKPT:-checkpoints}/${RUN_NAME}"
    echo "eval:   ${OVX_EVALS:-evals}/${RUN_NAME}/"
    exit 0
fi

# ---- inside the job ----
cd "${SLURM_SUBMIT_DIR:?}"
load "$2"
EVAL_DATA="${EVAL_DATA:-${DATASET}}"

echo "=== train: start $(date +%T)"
bash ovx/slurm/train.sbatch --dataset "${DATASET}" --base-model "${BASE_MODEL}" \
    --run-name "${RUN_NAME}" "${TRAIN_ARGS[@]}"
echo "=== eval: start $(date +%T)"
exec bash ovx/slurm/eval.sbatch --model "${RUN_NAME}" --env-id "${ENV_ID}" \
    --data-dir "${EVAL_DATA}" "${EVAL_ARGS[@]}"
