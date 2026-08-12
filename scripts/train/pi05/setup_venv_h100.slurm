#!/bin/bash
#SBATCH --job-name=pi05-venv
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Creates .venv-openpi (python 3.10, CUDA 12 wheels) following FINETUNE_PI05.md.
set -euo pipefail
ROOT=/raid/user_marcospaulo/Psi0
cd "$ROOT"

export UV_CACHE_DIR="$ROOT/cache/uv/cache"
export UV_PYTHON_INSTALL_DIR="$ROOT/cache/uv/python"
export UV_INSTALL_DIR="$ROOT/cache/uv/bin"
export PATH="$ROOT/cache/uv/bin:$PATH"

if ! command -v uv >/dev/null; then
    mkdir -p "$UV_INSTALL_DIR"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_DIR" UV_NO_MODIFY_PATH=1 sh
fi
uv --version

if [[ ! -d .venv-openpi ]]; then
    uv venv .venv-openpi --python 3.10
fi

export VIRTUAL_ENV="$ROOT/.venv-openpi"
uv pip install -e .
uv pip install -e src/openpi/openpi-client
GIT_LFS_SKIP_SMUDGE=1 uv pip install -r baselines/pi05/requirements-openpi.cu12.txt

# openpi requires patched transformers modules
cp -r src/openpi/models_pytorch/transformers_replace/* \
    .venv-openpi/lib/python3.10/site-packages/transformers/

.venv-openpi/bin/python - <<'EOF'
import os
os.environ.setdefault("PSI_HOME", "/raid/user_marcospaulo/Psi0")
os.environ.setdefault("DATA_HOME", "/raid/user_marcospaulo/Psi0/data")
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
import openpi.training.config as c
print("openpi config import OK, configs:", len(c._CONFIGS))
print("g1totemix_pi05:", c.get_config("g1totemix_pi05").batch_size)
EOF
echo "venv OK"
