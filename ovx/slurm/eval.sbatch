#!/usr/bin/env bash
# SIMPLE closed-loop evaluation on the OVX, through the eval image (ovx/run.sh eval): policy
# server and simulator in one job, on one GPU.
#
# Isaac Sim renders with RT cores. With --sim-mode mujoco_isaac (the default) submit to the L40
# partition -- the H100s have no RT cores. --sim-mode mujoco needs no Isaac at all.
#
#   mkdir -p logs
#   export OVX_SIF=/raid/$USER/images/ovx-gr00t_eval_<version>.sif
#   sbatch -p <l40 partition> ovx/slurm/eval.sbatch \
#       --model gr00t_n1d7_finetune_output_screws \
#       --env-id simple/G1IndustrialScrewToToteTeleop-v0 --data-dir screws
#
# Evaluate each specialist as soon as it finishes training -- array task i waits for training
# task i, and takes its arguments from line i of EVAL_SPEC (see evals.example.tsv):
#
#   TRAIN=$(sbatch --parsable --array=0-2%1 --export=ALL,DATASETS="carry_totes screwdrivers screws" \
#               ovx/slurm/train.sbatch --base-model <model dir> --epochs 30)
#   sbatch -p <l40 partition> --array=0-2 --dependency=aftercorr:${TRAIN} \
#       --export=ALL,EVAL_SPEC=ovx/slurm/evals.example.tsv,OVX_SIF=<eval .sif> \
#       ovx/slurm/eval.sbatch --num-episodes 20
#
# Arguments after the script name are appended to every task (`ovx/run.sh eval --help`).

#SBATCH --job-name=ovx-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ovx-eval-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this with sbatch}"
[ -n "${OVX_SIF:-}" ] || { echo "OVX_SIF is not set -- export it before sbatch"; exit 1; }

args=("$@")
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${EVAL_SPEC:-}" ]; then
    [ -f "${EVAL_SPEC}" ] || { echo "EVAL_SPEC not found: ${EVAL_SPEC}"; exit 1; }
    line="$(grep -vE '^[[:space:]]*(#|$)' "${EVAL_SPEC}" | sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p")"
    [ -n "${line}" ] || { echo "EVAL_SPEC has no line for array task ${SLURM_ARRAY_TASK_ID}"; exit 1; }
    read -ra spec <<< "${line}"
    args=("${spec[@]}" "${args[@]}")
fi

echo "job ${SLURM_JOB_ID} on ${SLURMD_NODENAME:-?}, $(date)"
echo "image: ${OVX_SIF}"
echo "args:  ${args[*]}"
exec ovx/run.sh eval "${args[@]}"
