#!/usr/bin/env bash
#SBATCH --job-name=wla-setup
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
# CPU-only: venv + downloads. Idempotent (hf download skips completed files).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/wla/env_raid.sh

cd "$WLA_REPO"
uv sync
source .venv/bin/activate
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
if ! python -c "import flash_attn" 2>/dev/null; then
    uv pip install flash-attn --no-build-isolation || echo "WARN: flash-attn install failed (needed only for training/eval on GPU)"
fi

hf download unitreerobotics/UnifoLM-WLA-1.0-Base --local-dir "$WLA_MODELS/UnifoLM-WLA-1.0-Base"
hf download unitreerobotics/G1_WBT_Brainco_Supermarket_Shelf_Organizing --repo-type dataset \
    --local-dir "$WLA_DATA_ROOT/UnifoLM_WBT_Dataset/G1_WBT_Brainco_Supermarket_Shelf_Organizing"
hf download agentereal/G1ToteMix-psi0 --repo-type dataset --local-dir "$PSI0_DATA/G1ToteMix-psi0"

du -sh "$WLA_MODELS"/* "$WLA_DATA_ROOT"/*/* "$PSI0_DATA"/*
echo SETUP_DONE
