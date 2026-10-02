#!/usr/bin/env bash
#SBATCH --job-name=s0-eval-data
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
# CPU-only: raízes LeRobot por split (symlinks) + ações oráculo @30 FPS para o eval no SIMPLE.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/wla/env_raid.sh
source "$WLA_REPO/.venv/bin/activate"
export PYTHONPATH="$PWD/src:$WLA_REPO"
OUT=$WLA_EXP/s0_eval_data
for s in train val test; do python scripts/wla/s0_make_split_roots.py --split $s --out $OUT/roots/$s; done
python scripts/wla/s3_prepare_oracle.py --split val --name val --out $OUT/oracle/val
python scripts/wla/s3_prepare_oracle.py --split train --name train --limit 8 --out $OUT/oracle/train
git rev-parse HEAD > $OUT/git_commit.txt
echo S0_DONE
