#!/bin/bash
#SBATCH --job-name=pi05-weights
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Downloads pi05_droid pytorch weights into cache/checkpoints/openpi/pi05_droid.
set -euo pipefail
ROOT=/raid/user_marcospaulo/Psi0
cd "$ROOT"

apptainer exec --cleanenv \
    --bind "$ROOT/cache/huggingface:/workspace/hf_cache" \
    --bind "$ROOT/cache/checkpoints:/workspace/checkpoints" \
    --bind "$ROOT/secrets/psi0.env:/workspace/.env:ro" \
    --env HF_HOME=/workspace/hf_cache \
    containers/industrial_humanoids_psi0-train.gr00t_devel.sif bash -lc '
        set -euo pipefail
        set -a; source /workspace/.env; set +a
        source /workspace/.venv/bin/activate
        hf download USC-PSI-Lab/psi-model \
            --local-dir=/workspace/checkpoints \
            --include "openpi/pi05_droid/*" \
            --repo-type=model
        ls -la /workspace/checkpoints/openpi/pi05_droid/
    '
echo "weights OK"
