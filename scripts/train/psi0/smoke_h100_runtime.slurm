#!/usr/bin/env bash
#SBATCH --job-name=psi0-h100-runtime
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/psi0-h100-runtime-%j.out
#SBATCH --error=logs/psi0-h100-runtime-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIF_PATH="${SIF_PATH:-$PROJECT_DIR/containers/industrial_humanoids_psi0-train.gr00t_devel.sif}"

[[ -s "$SIF_PATH" ]] || { echo "ERROR: SIF not found: $SIF_PATH" >&2; exit 1; }
mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/cache/huggingface"
module load apptainer 2>/dev/null || true

apptainer exec --nv --cleanenv \
    --bind "$PROJECT_DIR/src:/workspace/src:ro" \
    --env PYTHONPATH=/workspace/src \
    --env HF_HOME=/workspace/hf_cache \
    --env CUDA_VISIBLE_DEVICES=0 \
    --bind "$PROJECT_DIR/cache/huggingface:/workspace/hf_cache" \
    "$SIF_PATH" \
    /workspace/.venv/bin/python -c '
import flash_attn
import torch
from flash_attn import flash_attn_func
from psi.config import config as psi_config

assert torch.cuda.is_available()
device = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
assert capability == (9, 0), f"expected H100 sm_90, got {device} sm_{capability[0]}{capability[1]}"
query = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
output = flash_attn_func(query, query, query)
assert output.isfinite().all()
print(f"device={device}")
print(f"capability={capability}")
print(f"torch={torch.__version__}")
print(f"flash_attn={flash_attn.__version__}")
print(f"flash_output_shape={tuple(output.shape)}")
print(f"psi_config={psi_config.__file__}")
'