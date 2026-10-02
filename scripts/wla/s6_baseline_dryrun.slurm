#!/usr/bin/env bash
#SBATCH --job-name=s6-baseline-dryrun
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=00:50:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
# CPU-only (sem GPU na fila): dry-run do servidor com o WLA-Base num chunk real de VAL. Só valida o caminho.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/wla/env_raid.sh
source "$WLA_REPO/.venv/bin/activate"
export PYTHONPATH="$PWD/src:$WLA_REPO" CUDA_VISIBLE_DEVICES=
B=$WLA_MODELS/UnifoLM-WLA-1.0-Base
python scripts/wla/s6_baseline_dryrun.py --ckpt_path "$B/checkpoints/model.safetensors" --base_vlm "$B/tokenizer" \
   --source_name UnifoLM_WBT --profile wbt --device cpu --out "$WLA_EXP/s6_baseline_dryrun"
echo DRYRUN_DONE
