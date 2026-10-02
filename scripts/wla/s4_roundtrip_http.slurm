#!/usr/bin/env bash
#SBATCH --job-name=s4-roundtrip-http
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
# CPU-only: round-trip Ψ0 -> adaptador -> Ψ0 via HTTP (servidor --oracle adapter). SPLIT=train|val (padrão val).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/wla/env_raid.sh
source "$WLA_REPO/.venv/bin/activate"
export PYTHONPATH="$PWD/src:$WLA_REPO"
SPLIT=${SPLIT:-val}; LIMIT=${LIMIT:-}
OUT=$WLA_EXP/s4_roundtrip_${SPLIT}
python scripts/wla/s4_roundtrip_http.py --split "$SPLIT" ${LIMIT:+--limit $LIMIT} --out "$OUT"
git rev-parse HEAD > "$OUT/git_commit.txt"
echo S4_DONE
