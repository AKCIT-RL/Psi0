#!/usr/bin/env bash
#SBATCH --job-name=psi0-sif-pull
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/psi0-sif-pull-%j.out
#SBATCH --error=logs/psi0-sif-pull-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
IMAGE_REF="${IMAGE_REF:-docker://rafaeljose/industrial_humanoids:psi0-train.gr00t_devel}"
SIF_PATH="${SIF_PATH:-$PROJECT_DIR/containers/industrial_humanoids_psi0-train.gr00t_devel.sif}"

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$PROJECT_DIR/cache/apptainer}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$PROJECT_DIR/cache/apptainer/tmp}"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/containers" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
module load apptainer 2>/dev/null || true
command -v apptainer >/dev/null

echo "Image source: $IMAGE_REF"
echo "SIF path:     $SIF_PATH"
echo "Node:         ${SLURM_NODELIST:-unknown}"

if [[ ! -s "$SIF_PATH" ]]; then
    apptainer pull "$SIF_PATH" "$IMAGE_REF"
fi

apptainer inspect "$SIF_PATH" >/dev/null
sha256sum "$SIF_PATH" | tee "$SIF_PATH.sha256"