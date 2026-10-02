#!/usr/bin/env bash
#SBATCH --job-name=s2-import-check
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
# CPU-only: imports do simulador + cliente WLA dentro do container (sem criar contexto GL).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/wla/env_raid.sh
S=$PWD/third_party/SIMPLE; SB=$RAID/containers/simple-260829.sandbox
export APPTAINER_TMPDIR=$RAID/cache/apptainer/tmp; mkdir -p "$APPTAINER_TMPDIR"
apptainer exec --cleanenv -B "$RAID:$RAID" -B "$S/src:/workspace/simple/src" -B "$S/third_party/gear_sonic:/workspace/simple/third_party/gear_sonic" \
  -B "$S/third_party/decoupled_wbc:/workspace/simple/third_party/decoupled_wbc" --env MUJOCO_GL=egl --env HF_HUB_OFFLINE=1 --pwd /workspace/simple "$SB" \
  /workspace/simple/.venv/bin/python - <<'PY'
import importlib, traceback
for m in ["simple.envs", "simple.baselines.wla_decoupled_wbc", "simple.cli.eval_decoupled_wbc", "gear_sonic.utils.mujoco_sim.configs",
          "decoupled_wbc.control.policy.wbc_policy_factory", "lerobot.datasets.lerobot_dataset"]:
    try:
        importlib.import_module(m); print("OK  ", m)
    except Exception as e:
        print("FAIL", m, type(e).__name__, str(e)[:200]); traceback.print_exc(limit=3)
import gymnasium as gym
print("env registrada:", "simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0" in gym.registry)
from simple.baselines.wla_decoupled_wbc import upsample_chunk
import numpy as np
c = np.zeros((30, 36)); c[:, 35] = np.linspace(3.0, -3.0, 30)
print("upsample", upsample_chunk(c, 30).shape)
PY
echo IMPORT_CHECK_DONE
