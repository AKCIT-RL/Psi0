#!/usr/bin/env bash
# First run of the whole pipeline on a cluster, in one allocation: preflight, smoke, a short
# training and a one-episode eval of the model it trained. Every stage runs and logs to its own
# file; only the eval is skipped when training produced no model. The last line of the job log is
# the summary, and the job fails unless every stage passed.
#
#   mkdir -p logs
#   export OVX_SIF=/raid/$USER/images/ovx-gr00t_eval_<version>.sif
#   sbatch -p <l40 partition> ovx/slurm/validate.sbatch --base-model <model dir>
#
# Options:
#   --base-model DIR   required; relative to OVX_CKPT, as for train.sbatch
#   --dataset NAME     default screws (a directory under OVX_DATA)
#   --env-id ID        default simple/G1IndustrialScrewToToteTeleop-v0 (must match --dataset)
#   --max-steps N      default 200
#
# Logs: logs/ovx-validate-<jobid>.out (everything) and logs/ovx-validate-<jobid>/<stage>.log.
# The model goes to OVX_CKPT/ovx_validate_<jobid>, the eval to OVX_EVALS/ovx_validate_<jobid>/.
# L40 partition: the smoke and the eval boot Isaac Sim, and the H100s have no RT cores.

#SBATCH --job-name=ovx-validate
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ovx-validate-%j.out

set -uo pipefail   # no -e: a failed stage is recorded, not fatal

cd "${SLURM_SUBMIT_DIR:?submit this with sbatch}"
[ -n "${OVX_SIF:-}" ] || { echo "OVX_SIF is not set -- export it before sbatch"; exit 1; }
export OVX_ENGINE=apptainer   # never fall back to docker on the cluster

BASE_MODEL=""; DATASET=screws; ENV_ID=simple/G1IndustrialScrewToToteTeleop-v0; MAX_STEPS=200
while [ $# -gt 0 ]; do
    case "$1" in
        --base-model) BASE_MODEL="$2"; shift 2 ;;
        --dataset)    DATASET="$2"; shift 2 ;;
        --env-id)     ENV_ID="$2"; shift 2 ;;
        --max-steps)  MAX_STEPS="$2"; shift 2 ;;
        *) echo "unknown option: $1"; exit 1 ;;
    esac
done
[ -n "${BASE_MODEL}" ] || { echo "--base-model is required"; exit 1; }

RUN="ovx_validate_${SLURM_JOB_ID}"
LOGDIR="logs/ovx-validate-${SLURM_JOB_ID}"
mkdir -p "${LOGDIR}"
RESULTS=()

# stage <name> <command...>: runs it, logs to the job log and to LOGDIR/<name>.log, records the exit.
stage() {
    local name="$1"; shift
    echo "=== ${name}: start $(date +%T)"
    "$@" 2>&1 | tee "${LOGDIR}/${name}.log"
    local status="${PIPESTATUS[0]}"
    echo "=== ${name}: exit ${status} at $(date +%T)"
    RESULTS+=("${name}=${status}")
    return "${status}"
}

# What the smoke does not cover and training needs: the node's driver and internet access, and the
# paths as the container sees them.
preflight() {
    local fails=0 model="${BASE_MODEL}"
    case "${model}" in /*) ;; *) model="/home/ovx/checkpoints/${model}" ;; esac
    echo "node: $(hostname)"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || fails=$((fails + 1))
    for url in https://huggingface.co https://pypi.org; do
        curl -s -o /dev/null -m 10 -w "%{http_code}  ${url}\n" "${url}" \
            || { echo "NO ACCESS  ${url}"; fails=$((fails + 1)); }
    done
    for key in WANDB_API_KEY WANDB_ENTITY HF_TOKEN; do
        grep -q "^${key}=" .env 2> /dev/null && echo "${key}: set in .env" \
            || { echo "${key}: MISSING from .env"; fails=$((fails + 1)); }
    done
    quota -s 2> /dev/null || true
    ovx/run.sh shell -lc "cat /opt/ovx/build-info.json && echo &&
        ls /home/ovx/data/${DATASET}/meta/modality.json &&
        ls ${model}/config.json &&
        touch /home/ovx/cache/.w && rm /home/ovx/cache/.w && echo 'cache writable'" \
        || fails=$((fails + 1))
    return "${fails}"
}

echo "job ${SLURM_JOB_ID} on ${SLURMD_NODENAME:-?}, $(date)"
echo "image: ${OVX_SIF}"
echo "base model: ${BASE_MODEL}  dataset: ${DATASET}  env: ${ENV_ID}  steps: ${MAX_STEPS}"

stage preflight preflight
stage smoke ovx/run.sh smoke --isaac
if stage train ovx/run.sh train --dataset "${DATASET}" --base-model "${BASE_MODEL}" \
        --max-steps "${MAX_STEPS}" --save-steps "$((MAX_STEPS / 2))" --run-name "${RUN}"; then
    stage eval ovx/run.sh eval --model "${RUN}" --env-id "${ENV_ID}" \
        --data-dir "${DATASET}" --num-episodes 1
else
    RESULTS+=("eval=skipped")
fi

echo "=== SUMMARY: ${RESULTS[*]}   (0 = ok; logs in ${LOGDIR}/)"
for r in "${RESULTS[@]}"; do [ "${r#*=}" = 0 ] || exit 1; done
