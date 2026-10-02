#!/usr/bin/env bash
#SBATCH --job-name=s1-sandbox
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
# CPU-only: .sif -> sandbox (sem squashfuse neste cluster, cada `exec` do .sif reconverteria 17GB). Depois limpa o cache do apptainer.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/wla/env_raid.sh
export APPTAINER_CACHEDIR=$RAID/cache/apptainer APPTAINER_TMPDIR=$RAID/cache/apptainer/tmp
SB=$RAID/containers/simple-260829.sandbox
[[ -d $SB/workspace ]] || apptainer build --sandbox --fix-perms "$SB" "$RAID/containers/simple-260829.sif"
du -sh "$SB"
rm -rf "$RAID/cache/apptainer"/*
df -h /raid | tail -1
echo SANDBOX_DONE
