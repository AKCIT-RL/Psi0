#!/usr/bin/env bash
#SBATCH --job-name=psi0-totemix-merge
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/psi0-totemix-merge-%j.out
#SBATCH --error=logs/psi0-totemix-merge-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIF_PATH="${SIF_PATH:-$PROJECT_DIR/containers/industrial_humanoids_psi0-train.gr00t_devel.sif}"
SOURCE_A="G1PickUpToteFromShelfToDesk-psi0"
SOURCE_B="G1WholebodyLocomotionPickTotesShelfToTableTeleop-psi0"
OUTPUT_NAME="G1ToteMix-psi0"
DATA_ROOT="$PROJECT_DIR/data"

for required in "$SIF_PATH" \
    "$DATA_ROOT/$SOURCE_A/meta/info.json" \
    "$DATA_ROOT/$SOURCE_B/meta/info.json"; do
    [[ -s "$required" ]] || { echo "ERROR: required file not found: $required" >&2; exit 1; }
done

mkdir -p "$PROJECT_DIR/logs"
module load apptainer 2>/dev/null || true

apptainer exec --cleanenv \
    --bind "$PROJECT_DIR/scripts:/workspace/scripts:ro" \
    --bind "$DATA_ROOT:/workspace/data" \
    --env SOURCE_A="$SOURCE_A" \
    --env SOURCE_B="$SOURCE_B" \
    --env OUTPUT_NAME="$OUTPUT_NAME" \
    "$SIF_PATH" \
    bash -lc '
        set -euo pipefail
        source /workspace/.venv/bin/activate
        out="/workspace/data/$OUTPUT_NAME"

        # remap corrupted teleop task 1 ("\x1b[C") -> task 0 in source A
        python /workspace/scripts/data/merge_lerobot_datasets.py \
            --source "/workspace/data/$SOURCE_A" \
            --source "/workspace/data/$SOURCE_B" \
            --output "$out" \
            --remap-task 0:1=0

        python /workspace/scripts/data/calc_modality_stats.py --task-dir "$out"
        cp "$out/meta/stats.json" "$out/meta/stats_psi0.json"

        python /workspace/scripts/validate_lerobot_modality.py --expect-psi0 "$out"

        python - "$out" <<PY
import json, sys, glob, os
out = sys.argv[1]
info = json.load(open(f"{out}/meta/info.json"))
eps = [json.loads(l) for l in open(f"{out}/meta/episodes.jsonl")]
assert info["total_episodes"] == 308 == len(eps), (info["total_episodes"], len(eps))
assert info["total_frames"] == 329930 == sum(e["length"] for e in eps)
assert [e["episode_index"] for e in eps] == list(range(308))
parquets = sorted(glob.glob(f"{out}/data/chunk-*/episode_*.parquet"))
videos = sorted(glob.glob(f"{out}/videos/chunk-*/egocentric/episode_*.mp4"))
assert len(parquets) == 308 and len(videos) == 308, (len(parquets), len(videos))
tasks = [json.loads(l) for l in open(f"{out}/meta/tasks.jsonl")]
assert all("\x1b" not in t["task"] for t in tasks), tasks
print("tasks:", tasks)
print("FINAL CHECKS OK: 308 episodes, 329930 frames, contiguous indices, videos present")
PY
        echo "output=$out"
    '
