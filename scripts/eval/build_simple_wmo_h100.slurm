#!/usr/bin/env bash
#SBATCH --job-name=simple-wmo-build
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/simple-wmo-build-%j.out
#SBATCH --error=logs/simple-wmo-build-%j.err

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIMPLE="$ROOT/third_party/SIMPLE"
EXPECTED_SIMPLE_COMMIT="${SIMPLE_COMMIT:-599d6c0e3364b31095f1a59fdbfa4ea507ae90c9}"

[[ "$(git -C "$SIMPLE" rev-parse HEAD)" == "$EXPECTED_SIMPLE_COMMIT" ]] \
    || { echo "ERROR: unexpected SIMPLE commit" >&2; exit 1; }
! git -C "$SIMPLE" submodule status --recursive | grep -Eq '^[-+U]'

mkdir -p "$ROOT/logs" "$SIMPLE/.uv-cache" "$ROOT/data"
export DATA_DIR="$ROOT/data"
export TORCH_CUDA_ARCH_LIST="9.0+PTX"
export TIMEZONE="UTC"

cd "$SIMPLE"
docker buildx bake --allow=network.host isaac-sim

cd "$ROOT"
docker build -f eval/docker/Dockerfile.simple-wmo-eval \
    -t simple-teleoperation:251025-wmo-eval .

docker image inspect simple-teleoperation:251025-wmo-eval \
    --format '{{.Id}}' | tee eval/results/simple-wmo-image-id.txt