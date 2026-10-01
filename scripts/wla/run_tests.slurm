#!/bin/bash
#SBATCH --job-name=f2a_tests
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
#
# Testes CPU-only do adaptador WLA (tests/wla_adapter) no venv do WLA.
set -euo pipefail

PSI0=/raid/user_marcospaulo/Psi0
source "$PSI0/scripts/wla/env_raid.sh"
cd "$PSI0"
source "$WLA_REPO/.venv/bin/activate"
export PYTHONPATH="$PSI0/src:$WLA_REPO"  # editable install do WLA tem MAPPING vazio

python -c "import pytest" 2>/dev/null || uv pip install --python "$WLA_REPO/.venv/bin/python" pytest
python -m pytest tests/wla_adapter -q -p no:cacheprovider
