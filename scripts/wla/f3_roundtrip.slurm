#!/bin/bash
#SBATCH --job-name=f3_roundtrip
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
# F3: TEST congelado. Só reporta. Não escreve stats nem altera o conversor.
set -euo pipefail
PSI0=/raid/user_marcospaulo/Psi0
source "$PSI0/scripts/wla/env_raid.sh"
cd "$PSI0"
source "$WLA_REPO/.venv/bin/activate"
export PYTHONPATH="$PSI0/src:$WLA_REPO"
mkdir -p "$WLA_EXP/f3_test"
git -C "$PSI0" rev-parse HEAD > "$WLA_EXP/f3_test/git_commit.txt"
python scripts/wla/f2b_roundtrip.py --split test --allow-test \
  --out "$WLA_EXP/f3_test/roundtrip_report.json"
