#!/bin/bash
#SBATCH --job-name=f2a_golden
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
#
# F2a — golden test (pipeline independente vs dataset oficial) + round-trip WBT. CPU-only.
set -euo pipefail

PSI0=/raid/user_marcospaulo/Psi0
source "$PSI0/scripts/wla/env_raid.sh"
cd "$PSI0"
source "$WLA_REPO/.venv/bin/activate"
export PYTHONPATH="$PSI0/src:$WLA_REPO"  # editable install do WLA tem MAPPING vazio
mkdir -p "$WLA_EXP/f2_validation"

python scripts/wla/validate_official_wla.py --out "$WLA_EXP/f2_validation"
