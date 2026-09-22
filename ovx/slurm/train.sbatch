#!/usr/bin/env bash
# GR00T N1.7 fine-tuning on the OVX, through the image (ovx/run.sh train).
#
#   mkdir -p logs                          # SLURM opens the log file before this script runs
#   export OVX_SIF=/raid/$USER/images/ovx-gr00t_train_<version>.sif
#   sbatch ovx/slurm/train.sbatch --dataset carry_totes --base-model <model dir> --epochs 30
#
# One specialist per dataset, one after another, from a single submission:
#
#   sbatch --array=0-2%1 --export=ALL,DATASETS="carry_totes screwdrivers screws" \
#       ovx/slurm/train.sbatch --base-model <model dir> --epochs 30
#
# Everything after the script name goes to ovx/bin/train.sh (`ovx/run.sh train --help`).
# Resources can be overridden at submission, e.g. `sbatch --time=72:00:00 ...`.

#SBATCH --job-name=ovx-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ovx-train-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this with sbatch}"
[ -n "${OVX_SIF:-}" ] || { echo "OVX_SIF is not set -- export it before sbatch"; exit 1; }

args=("$@")
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${DATASETS:-}" ]; then
    read -ra list <<< "${DATASETS}"
    if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#list[@]}" ]; then
        echo "array index ${SLURM_ARRAY_TASK_ID} out of range: DATASETS has ${#list[@]} entries"
        exit 1
    fi
    args=(--dataset "${list[${SLURM_ARRAY_TASK_ID}]}" "${args[@]}")
fi

# The per-user quota, not free space, is the limit on /raid. Hitting it mid-run truncates the
# checkpoint being written hours in; refuse up front instead. A run needs ~21 GB.
REQUIRED_GB="${REQUIRED_GB:-50}"
free_gb=$(quota 2> /dev/null | python3 -c '
import re, sys
for line in sys.stdin:
    v = [int(x) for x in re.findall(r"\b\d+\b", line)]
    if len(v) >= 6:
        used, soft, hard = v[0], v[1], v[2]
        print(int(((soft or hard) - used) / 1024 / 1024))
        break
else:
    print(-1)' 2> /dev/null || echo -1)
if [ "${free_gb}" = -1 ]; then
    echo "[WARN] could not read the disk quota; continuing without the check"
elif [ "${free_gb}" -lt "${REQUIRED_GB}" ]; then
    echo "ERROR: ${free_gb} GB left under the quota, ${REQUIRED_GB} GB needed -- refusing to start"
    exit 1
fi

echo "job ${SLURM_JOB_ID} on ${SLURMD_NODENAME:-?}, $(date)"
echo "image: ${OVX_SIF}"
echo "args:  ${args[*]}"
exec ovx/run.sh train "${args[@]}"
