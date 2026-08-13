#!/usr/bin/env bash
#SBATCH --job-name=pi05-hf-upload
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pi05-hf-upload-%j.out
#SBATCH --error=logs/pi05-hf-upload-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
set -a; source secrets/psi0.env; set +a
export HF_HOME="$ROOT/cache/huggingface"

RUN_DIR="${RUN_DIR:-$ROOT/.runs/openpi-05/g1totemix_pi05/pi05-g1totemix-h100-2608121916}"
HF_REPO_ID="${HF_REPO_ID:-AKCITWMOPOC/pi05-g1totemix}"
[[ -f "$RUN_DIR/best/model.safetensors" ]] || { echo "ERROR: no best at $RUN_DIR" >&2; exit 1; }

# Snapshot to avoid racing the trainer's atomic best replacement.
SNAP=$(mktemp -d "$ROOT/cache/pi05-best-snap.XXXX")
trap 'rm -rf "$SNAP"' EXIT
cp -a "$RUN_DIR/best/." "$SNAP/"

HF_REPO_ID="$HF_REPO_ID" "$ROOT/.venv-openpi/bin/python" - "$SNAP" <<'PY'
import json, os, sys
from huggingface_hub import HfApi

snap = sys.argv[1]
state = json.load(open(os.path.join(snap, "early_stopping_state.json")))
repo = os.environ["HF_REPO_ID"]
api = HfApi()
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
info = api.upload_folder(
    repo_id=repo,
    folder_path=snap,
    commit_message=(
        f"best @ step {state['global_step']} "
        f"(err_l1_hand_joints={state['val_metrics']['err_l1_hand_joints']:.5f})"
    ),
)
print("UPLOADED", repo, info.oid)
PY
