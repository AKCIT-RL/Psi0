#!/usr/bin/env bash
#SBATCH --job-name=simple-wmo-smoke
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=eval/results/simple-wmo-smoke-%j.out
#SBATCH --error=eval/results/simple-wmo-smoke-%j.err

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUN_ROOT="${PSI0_RUN_DIR:-$ROOT/.runs/finetune/psi0-wmo-totes-h100-smoke-v2.real.flow1000.cosine.lr1.0e-04.b1.gpus1.2608111150}"

export ROOT
export PSI0_RUN_DIR="$RUN_ROOT"
export CHECKPOINT="${CHECKPOINT:-2}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export DATASET_RELATIVE_PATH="${DATASET_RELATIVE_PATH:-G1WholebodyLocomotionPickTotesShelfToTableTeleop-psi0}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-$ROOT/cache/huggingface}"
export PSI_HOME_DIR="${PSI_HOME_DIR:-$ROOT/psi_home}"
export SIF_PATH="${SIF_PATH:-$ROOT/containers/industrial_humanoids_psi0-train.gr00t_devel.sif}"
export ENV_FILE="${ENV_FILE:-$ROOT/secrets/psi0.env}"
export SIMPLE_COMMIT="${SIMPLE_COMMIT:-599d6c0e3364b31095f1a59fdbfa4ea507ae90c9}"
export DATASET_COMMIT="${DATASET_COMMIT:-219d98f9228f9d2db41d723a56744f3b9093353a}"
export IMAGE="${IMAGE:-simple-teleoperation:251025-wmo-eval}"
export EPISODES=1
export SMOKE_ONLY=1
export RUN_ID="${RUN_ID:-slurm-${SLURM_JOB_ID}}"

module load apptainer 2>/dev/null || true
command -v docker >/dev/null
command -v apptainer >/dev/null

bash "$ROOT/eval/run_wmo_totes_20260807.sh"