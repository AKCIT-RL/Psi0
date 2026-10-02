#!/usr/bin/env bash
#SBATCH --job-name=s9-ablation
#SBATCH --partition=b200n1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/wla/env_raid.sh
module load cuda12.6/toolkit/12.6.3 2>/dev/null || true
export CUDA_HOME=/cm/shared/apps/cuda12.6/toolkit/12.6.3
cd "$WLA_REPO"; source .venv/bin/activate
export PYTHONPATH="$OLDPWD/src:$WLA_REPO"
python "$OLDPWD/scripts/wla/s9_input_ablation.py" $ARGS 2>&1 | grep -v "av1\|Sequence\|warn"
