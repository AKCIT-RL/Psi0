#!/bin/bash
#SBATCH --job-name=f2a_fk_fixtures
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
#
# F2a — fixtures de FK (pinocchio) para tests/wla_adapter/test_fk.py. CPU-only.
set -euo pipefail

PSI0=/raid/user_marcospaulo/Psi0
source "$PSI0/scripts/wla/env_raid.sh"
cd "$PSI0"

VENV="$PSI0/_scratch/.venv-fk"
if [[ ! -x "$VENV/bin/python" ]]; then
    uv venv --python 3.12 "$VENV"
fi
uv pip install --python "$VENV/bin/python" pin numpy

"$VENV/bin/python" scripts/wla/f2a_fk_fixtures.py --out "$PSI0/_scratch/fixtures/fk_pinocchio.npz"
