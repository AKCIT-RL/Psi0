#!/usr/bin/env bash
#SBATCH --job-name=s2-plan-check
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
# CPU-only: LeRobot do SIMPLE (0.3.3) abre as raízes val/test (subset com índices originais) e o plano do --eval-config.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/wla/env_raid.sh
S=$PWD/third_party/SIMPLE; SB=$RAID/containers/simple-260829.sandbox
export APPTAINER_TMPDIR=$RAID/cache/apptainer/tmp; mkdir -p "$APPTAINER_TMPDIR"
apptainer exec --cleanenv -B "$RAID:$RAID" -B "$S/src:/workspace/simple/src" -B "$S/third_party/gear_sonic:/workspace/simple/third_party/gear_sonic" \
  -B "$S/third_party/decoupled_wbc:/workspace/simple/third_party/decoupled_wbc" --env HF_HUB_OFFLINE=1 --env MUJOCO_GL=egl --env WLA_EXP=$WLA_EXP --pwd /workspace/simple "$SB" \
  /workspace/simple/.venv/bin/python - <<'PY'
import json, os, traceback
from simple.cli import eval_decoupled_wbc as E
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from simple.datasets.lerobot import get_episode_lerobot, _load_episode_prompts
root = os.environ["WLA_EXP"] + "/s0_eval_data/roots"
for split in ("val", "test"):
    path = f"{root}/{split}"
    el, dropped = E._eligible_episodes(path)
    print(split, "eligible", len(el), "dropped", dropped, "first", el[:5])
    try:
        ds = LeRobotDataset(repo_id="simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0", root=path)
        print(split, "LeRobotDataset OK; num_episodes", ds.num_episodes, "keys sample", list(ds.meta.episodes)[:3])
        for idx in el[:2]:
            conf, _ = get_episode_lerobot(ds, idx, load_frames=False)
            print(split, idx, "env_conf uid", conf["uid"], "target", conf.get("target_tote_key"), "dr keys", sorted(conf["dr_state_dict"]))
    except Exception:
        traceback.print_exc()
    print("prompts", list(_load_episode_prompts(path).items())[:2])
plan = E._build_episode_plan([("val", f"{root}/val", "recorded", {})], 5, 0)
print("plan", [(p[0], p[2]) for p in plan])
PY
echo PLAN_CHECK_DONE
