#!/bin/bash
#SBATCH --job-name=f2b_roundtrip
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
# Round-trip em TRAIN (desenvolvimento). Não toca em TEST.
set -euo pipefail
PSI0=/raid/user_marcospaulo/Psi0
source "$PSI0/scripts/wla/env_raid.sh"
cd "$PSI0"
source "$WLA_REPO/.venv/bin/activate"
export PYTHONPATH="$PSI0/src:$WLA_REPO"
python scripts/wla/f2b_roundtrip.py --split train --limit 8 \
  --out "$WLA_EXP/f2_validation/roundtrip_train8.json"
