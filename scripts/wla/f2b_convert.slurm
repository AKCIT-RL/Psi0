#!/bin/bash
#SBATCH --job-name=f2b_convert
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
# Converte TRAIN+VAL (nunca TEST) para LeRobot v3 @30FPS. Reentrante: --overwrite só se F2B_OVERWRITE=1.
set -euo pipefail
PSI0=/raid/user_marcospaulo/Psi0
source "$PSI0/scripts/wla/env_raid.sh"
cd "$PSI0"
source "$WLA_REPO/.venv/bin/activate"
export PYTHONPATH="$PSI0/src:$WLA_REPO"
OUT="$WLA_DATA_ROOT/Psi0_Tote_Dataset/G1ToteMix_psi0_trainval"
EXTRA=()
if [[ "${F2B_OVERWRITE:-1}" == 1 ]]; then EXTRA+=(--overwrite); fi
python -m wla_adapter.write_dataset --split train,val --out "$OUT" \
  --stats-out "$WLA_DATA_ROOT/stats_psi0_tote_train" "${EXTRA[@]}"
echo "convert_done $(git -C "$PSI0" rev-parse HEAD)" | tee "$WLA_EXP/f2_validation/convert_version.txt"
