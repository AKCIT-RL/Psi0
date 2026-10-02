#!/usr/bin/env bash
#SBATCH --job-name=s8-offline-eval
#SBATCH --partition=b200n1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
# Eval offline (malha aberta): saída do modelo x dado convertido, por checkpoint, em episódios de VAL (EM AMOSTRA: VAL foi visto no treino).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/wla/env_raid.sh
module load cuda12.6/toolkit/12.6.3 2>/dev/null || true
export CUDA_HOME=/cm/shared/apps/cuda12.6/toolkit/12.6.3
OUT=${OUT:-$WLA_EXP/s8_offline_val3}; mkdir -p "$OUT"
{ echo "psi0: $(git rev-parse HEAD)"; echo "unifolm-wla: $(git -C $WLA_REPO rev-parse HEAD)"; } > "$OUT/git_commit.txt"
cp "$0" "$OUT/s8_offline_eval.slurm"
cd "$WLA_REPO"; source .venv/bin/activate
export PYTHONPATH="$OLDPWD/src:$WLA_REPO"
python "$OLDPWD/scripts/wla/s8_offline_eval.py" ${EXTRA:---base} --val_idx "${VAL_IDX:-12,24,27}" --out "$OUT" 2>&1 | tee "$OUT/eval.log"
