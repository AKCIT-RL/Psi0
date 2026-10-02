#!/usr/bin/env bash
#SBATCH --job-name=s1-submodules
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
# CPU-only: init SIMPLE submodules over HTTPS (all public; SSH keys have no access). LFS smudge skipped.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/wla/env_raid.sh
export GIT_LFS_SKIP_SMUDGE=1 GIT_TERMINAL_PROMPT=0
cd third_party/SIMPLE
git -c "url.https://github.com/.insteadOf=git@github.com:" submodule update --init \
    third_party/gear_sonic third_party/decoupled_wbc third_party/unitree_sdk2_python \
    third_party/televuer third_party/openpi-client third_party/usd2mjcf
git submodule status
du -sh third_party/*
echo S1_SUBMODULES_DONE
