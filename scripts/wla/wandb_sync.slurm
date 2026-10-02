#!/usr/bin/env bash
#SBATCH --job-name=wandb-sync
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
# CPU-only. Uso: sbatch --export=ALL,ARGS="train <dir>" scripts/wla/wandb_sync.slurm
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/wla/env_raid.sh
unset WANDB_BASE_URL
# shellcheck disable=SC2086
third_party/unifolm-wla/.venv/bin/python scripts/wla/wandb_sync.py $ARGS
