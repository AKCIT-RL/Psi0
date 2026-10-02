#!/usr/bin/env bash
#SBATCH --job-name=s1-pull-sif
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
# CPU-only: pull the public SIMPLE image (CUDA devel + prebuilt cuRobo + minimal sonic stack) as .sif on /raid.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/wla/env_raid.sh
export APPTAINER_CACHEDIR=$RAID/cache/apptainer APPTAINER_TMPDIR=$RAID/cache/apptainer/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$RAID/containers"
SIF=$RAID/containers/simple-260829.sif
[[ -s $SIF ]] || apptainer pull "$SIF" docker://ghcr.io/physical-superintelligence-lab/simple:260829
ls -lh "$SIF"
echo S1_SIF_DONE
