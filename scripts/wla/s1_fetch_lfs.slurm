#!/usr/bin/env bash
#SBATCH --job-name=s1-fetch-lfs
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
# CPU + rede: substitui ponteiros LFS do checkout do SIMPLE pelo conteúdo real (sem git-lfs no host).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/wla/env_raid.sh
python3 scripts/wla/s1_fetch_lfs.py third_party/SIMPLE https://github.com/AKCIT-RL/SIMPLE.git --skip-prefix third_party
echo LFS_DONE
