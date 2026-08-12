#!/usr/bin/env bash
#SBATCH --job-name=psi0-hf-upload
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/psi0-hf-upload-%j.out
#SBATCH --error=logs/psi0-hf-upload-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIF_PATH="${SIF_PATH:-$PROJECT_DIR/containers/industrial_humanoids_psi0-train.gr00t_devel.sif}"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/secrets/psi0.env}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to the run directory under .runs/finetune}"
HF_REPO_ID="${HF_REPO_ID:?Set HF_REPO_ID to the target model repo, e.g. agentereal/psi0-g1totemix}"
HF_CACHE_HOST="$PROJECT_DIR/cache/huggingface"

for required in "$SIF_PATH" "$ENV_FILE" "$RUN_DIR/checkpoints/best/model.safetensors"; do
    [[ -e "$required" ]] || { echo "ERROR: required path not found: $required" >&2; exit 1; }
done

mkdir -p "$PROJECT_DIR/logs" "$HF_CACHE_HOST"
module load apptainer 2>/dev/null || true

apptainer exec --cleanenv \
    --bind "$ENV_FILE:/workspace/.env:ro" \
    --bind "$RUN_DIR:/workspace/run:ro" \
    --bind "$HF_CACHE_HOST:/workspace/hf_cache" \
    --env HF_HOME=/workspace/hf_cache \
    --env HF_REPO_ID="$HF_REPO_ID" \
    "$SIF_PATH" \
    bash -lc '
        set -euo pipefail
        set -a; source /workspace/.env; set +a
        source /workspace/.venv/bin/activate
        python - <<PY
import json, os
from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO_ID"]
api = HfApi()
api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)

cfg = json.load(open("/workspace/run/run_config.json"))
card = f"""---
library_name: safetensors
tags:
- robotics
- vla
- psi0
- lerobot
---

# {repo_id.split("/")[-1]}

Psi0 (VLA) fine-tuned on G1ToteMix-psi0 (308 episodes / 329,930 frames), the merge of:
- AKCITWMOPOC/G1PickUpToteFromShelfToDeskPsi0 @ 40ae4f97d1ba9422769a6a220547df7ae70e6761 (258 eps)
- lGabrielJJ/G1WholebodyLocomotionPickTotesShelfToTableTeleop @ 219d98f9228f9d2db41d723a56744f3b9093353a (50 eps)

Best checkpoint selected by early stopping (val err_l1_hand_joints = 0.10966 @ step 160001,
patience 15 x 2500 steps, median window 5, val_episode_fraction 0.1, seed 42 split).

- W&B run: https://wandb.ai/ih-akcit/psi-h100/runs/x4qzclk0
- Base VLM: USC-PSI-Lab/psi-model psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k
- Action header: psi0/postpre.1by1.pad36.2601131206.ckpt.he30k
- action_dim 36, chunk 30, flow scheduler (1000 steps), lr 1e-4, batch 8, 1x H100
"""
api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                repo_id=repo_id, repo_type="model")
api.upload_file(path_or_fileobj="/workspace/run/run_config.json",
                path_in_repo="run_config.json", repo_id=repo_id, repo_type="model")
api.upload_folder(
    folder_path="/workspace/run/checkpoints/best",
    repo_id=repo_id, repo_type="model",
    ignore_patterns=["optimizer.bin", "random_states*", "scheduler.bin"],
    commit_message="best checkpoint (early stopping, step 160001, err_l1_hand_joints 0.10966)",
)
info = api.model_info(repo_id)
print("uploaded:", repo_id, "sha:", info.sha)
print("files:", [s.rfilename for s in info.siblings])
PY
    '
