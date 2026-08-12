#!/bin/bash
# finetune-lerobot-psi0.sh — Fine-tune PSI-0 on any LeRobot v2.1 dataset (real/teleop format)
#
# Usage:
#   ./scripts/train/psi0/finetune-lerobot-psi0.sh <dataset_dir> [exp_name]
#
# Examples:
#   ./scripts/train/psi0/finetune-lerobot-psi0.sh pick_cylinder_manipulation_psi
#   ./scripts/train/psi0/finetune-lerobot-psi0.sh /home/marcos/hfm/data/real/Pick_bottle pick-bottle
#
# Optional env vars:
#   CUDA_VISIBLE_DEVICES   GPUs to use (default: all available)
#   TARGET_EPOCHS          Training epochs over dataset (default: 50)
#   TRAIN_BATCH_SIZE       Batch size per GPU (default: 16)
#   VLM_CKPT_PATH          Override VLM backbone checkpoint path
#   ACTION_CKPT_PATH       Override action header checkpoint path
#   WANDB_DISABLED         Set to 1 to disable wandb logging
#   RESUME_FROM_CHECKPOINT Resume from an existing checkpoint path
#   EARLY_STOPPING         Set to 1 to enable early stopping on validation metric
#   EARLY_STOPPING_PATIENCE   Validation rounds without improvement (default: 15)
#   EARLY_STOPPING_MIN_STEPS  Do not stop before this global step (default: 0)
#   EARLY_STOPPING_METRIC     Metric name (default: auto -> err_l1_hand_joints)

set -euo pipefail

# ── 0. Source .env ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$REPO_ROOT/.env"
    set +a
else
    echo "[ERROR] .env file not found at $REPO_ROOT/.env"
    echo "        Copy .env.sample to .env and configure it first."
    exit 1
fi

# ── 1. Args ───────────────────────────────────────────────────────────────────
if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 <dataset_dir> [exp_name]"
    echo "  dataset_dir  Path to LeRobot dataset root (absolute or relative to repo)"
    echo "  exp_name     Optional experiment name (default: auto from dataset name)"
    exit 1
fi

DATASET_ARG="$1"

# Resolve dataset path: try as-is, then relative to repo root
if [[ -d "$DATASET_ARG" ]]; then
    DATASET_PATH="$(cd "$DATASET_ARG" && pwd)"
else
    DATASET_PATH="$REPO_ROOT/$DATASET_ARG"
fi

if [[ ! -d "$DATASET_PATH" ]]; then
    echo "[ERROR] Dataset directory not found: $DATASET_ARG"
    exit 1
fi

DATASET_NAME="$(basename "$DATASET_PATH")"
DATA_ROOT="$(dirname "$DATASET_PATH")"

# Auto exp name: first-second words of dataset name (lowercase, hyphenated)
DATASET_LOWER="${DATASET_NAME,,}"
DATASET_WORDS=( $(echo "$DATASET_LOWER" | tr '_' ' ') )
if [[ "${#DATASET_WORDS[@]}" -ge 2 ]]; then
    DEFAULT_EXP="${DATASET_WORDS[0]}-${DATASET_WORDS[1]}"
else
    DEFAULT_EXP="${DATASET_WORDS[0]}"
fi
EXP="${2:-$DEFAULT_EXP}"

echo "════════════════════════════════════════════════"
echo " PSI-0 Fine-tuning Launcher"
echo " Dataset : $DATASET_PATH"
echo " Repo ID : $DATASET_NAME"
echo " Exp name: $EXP"
echo "════════════════════════════════════════════════"

# ── 2. Validate dataset structure ─────────────────────────────────────────────
echo ""
echo "[1/5] Validating dataset..."

REQUIRED_PATHS=(
    "data"
    "videos"
    "meta/info.json"
    "meta/stats_psi0.json"
    "meta/episodes.jsonl"
    "meta/tasks.jsonl"
)

for rel in "${REQUIRED_PATHS[@]}"; do
    if [[ ! -e "$DATASET_PATH/$rel" ]]; then
        echo "[ERROR] Missing required dataset file/dir: $DATASET_PATH/$rel"
        echo "        Ensure your dataset was exported in LeRobot v2.1 PSI format."
        echo "        The stats_psi0.json can be generated with:"
        echo "          python scripts/data/calc_modality_stats.py --task-dir $DATASET_PATH"
        exit 1
    fi
done

# Read info.json
INFO_JSON="$DATASET_PATH/meta/info.json"
TOTAL_FRAMES=$(python3 -c "import json; d=json.load(open('$INFO_JSON')); print(d['total_frames'])")
TOTAL_EPISODES=$(python3 -c "import json; d=json.load(open('$INFO_JSON')); print(d['total_episodes'])")
FPS=$(python3 -c "import json; d=json.load(open('$INFO_JSON')); print(d['fps'])")

echo "    Episodes : $TOTAL_EPISODES"
echo "    Frames   : $TOTAL_FRAMES"
echo "    FPS      : $FPS"

if [[ "$TOTAL_EPISODES" -lt 5 ]]; then
    echo ""
    echo "[WARN] Only $TOTAL_EPISODES episode(s) found. PSI fine-tuning typically needs"
    echo "       20-50+ episodes for robust results. Proceeding anyway..."
    echo ""
fi

# ── 3. Activate virtual environment ───────────────────────────────────────────
echo "[2/5] Activating Python environment..."

if [[ -f "$REPO_ROOT/.venv-psi/bin/activate" ]] && "$REPO_ROOT/.venv-psi/bin/python" -c "import torch" &>/dev/null; then
    # shellcheck source=/dev/null
    source "$REPO_ROOT/.venv-psi/bin/activate"
    echo "    Using: .venv-psi"
elif [[ -f "$REPO_ROOT/.venv/bin/activate" ]] && "$REPO_ROOT/.venv/bin/python" -c "import torch" &>/dev/null; then
    # shellcheck source=/dev/null
    source "$REPO_ROOT/.venv/bin/activate"
    echo "    Using: .venv"
else
    echo "[ERROR] No Python virtual environment found."
    echo "        Create one with:"
    echo "          uv venv .venv-psi --python 3.10"
    echo "          source .venv-psi/bin/activate"
    echo "          GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups --index-strategy unsafe-best-match --active"
    exit 1
fi

# ── 4. Locate pre-trained checkpoints ─────────────────────────────────────────
echo "[3/5] Locating PSI-0 pre-trained checkpoints..."

PSI_CKPT_DIR="${PSI_HOME:-/home/marcos/hfm}/cache/checkpoints/psi0"

VLM_REMOTE="psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"
ACTION_REMOTE="psi0/postpre.1by1.pad36.2601131206.ckpt.he30k"

VLM_CKPT="${VLM_CKPT_PATH:-$PSI_CKPT_DIR/pre.fast.1by1.2601091803.ckpt.ego200k.he30k}"
ACTION_CKPT="${ACTION_CKPT_PATH:-$PSI_CKPT_DIR/postpre.1by1.pad36.2601131206.ckpt.he30k}"

_check_or_download() {
    local path="$1"
    local remote="$2"  # e.g. "psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"
    local label="$3"
    # parent of PSI_CKPT_DIR = $PSI_HOME/cache/checkpoints
    local hf_local_dir
    hf_local_dir="$(dirname "$PSI_CKPT_DIR")"

    if _checkpoint_complete "$path" "$label"; then
        echo "    $label: $path ✓"
        return 0
    fi

    echo ""
    echo "[WARN] $label not found at: $path"
    echo "       Attempting download from HuggingFace (USC-PSI-Lab/psi-model)..."
    echo "       Remote path: $remote"
    echo ""

    mkdir -p "$hf_local_dir"

    if command -v hf &>/dev/null; then
        hf download USC-PSI-Lab/psi-model \
            --include "${remote}/**" \
            --local-dir="$hf_local_dir" \
            --repo-type=model
    elif command -v huggingface-cli &>/dev/null; then
        huggingface-cli download USC-PSI-Lab/psi-model \
            --include "${remote}/**" \
            --local-dir="$hf_local_dir" \
            --repo-type=model
    else
        echo "[ERROR] Cannot download $label — neither 'hf' nor 'huggingface-cli' found."
        echo "        Download manually with:"
        echo "          hf download USC-PSI-Lab/psi-model \\"
        echo "            --include \"${remote}/**\" \\"
        echo "            --local-dir=\$PSI_HOME/cache/checkpoints \\"
        echo "            --repo-type=model"
        exit 1
    fi

    if ! _checkpoint_complete "$path" "$label"; then
        echo "[ERROR] Download appeared to succeed but $path still not found."
        echo "        Expected the model at: $path"
        exit 1
    fi
    echo "    $label: downloaded ✓"
}

_checkpoint_complete() {
    local path="$1"
    local label="$2"
    if [[ "$label" == "VLM backbone" ]]; then
        compgen -G "$path/model*.safetensors" >/dev/null \
            || compgen -G "$path/pytorch_model*.bin" >/dev/null
    else
        [[ -s "$path/action_header.safetensors" ]]
    fi
}

_check_or_download "$VLM_CKPT"    "$VLM_REMOTE"    "VLM backbone"
_check_or_download "$ACTION_CKPT" "$ACTION_REMOTE" "Action header"

# ── 5. Compute hyperparameters ─────────────────────────────────────────────────
echo "[4/5] Computing training hyperparameters..."

TARGET_EPOCHS="${TARGET_EPOCHS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
VAL_EPISODE_FRACTION="${VAL_EPISODE_FRACTION:-}"

# With a val split, epochs are counted over the training episodes only.
# Mirrors the seed-42 sampling in psi.config.data_lerobot.LerobotDataConfig.
TRAIN_FRAMES="$TOTAL_FRAMES"
if [[ -n "$VAL_EPISODE_FRACTION" ]]; then
    TRAIN_FRAMES=$(python3 - "$DATASET_PATH" "$VAL_EPISODE_FRACTION" <<'PYEOF'
import json, random, sys
path, frac = sys.argv[1], float(sys.argv[2])
lengths = {}
with open(path + "/meta/episodes.jsonl") as f:
    for line in f:
        ep = json.loads(line)
        lengths[ep["episode_index"]] = ep["length"]
total = len(lengths)
num_val = max(1, round(total * frac))
val = set(random.Random(42).sample(range(total), num_val))
print(sum(n for i, n in lengths.items() if i not in val))
PYEOF
)
    echo "    Val split         : $VAL_EPISODE_FRACTION of episodes (train frames: $TRAIN_FRAMES of $TOTAL_FRAMES)"
fi

# Detect GPU count
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | awk 'BEGIN{i=0} {printf "%s%d",(i?",":""),i; i++}' || echo "0")}"
NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# steps = (frames / (batch_per_gpu * n_gpus)) * epochs
EFFECTIVE_BATCH=$(( TRAIN_BATCH_SIZE * NPROC_PER_NODE ))
COMPUTED_MAX_TRAINING_STEPS=$(python3 -c "
frames = $TRAIN_FRAMES
epochs = $TARGET_EPOCHS
batch  = $EFFECTIVE_BATCH
steps  = max(1000, int(frames / batch * epochs))
print(steps)
")

MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-$COMPUTED_MAX_TRAINING_STEPS}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-$(python3 -c "print(max(200, int($MAX_TRAINING_STEPS / 10)))")}"
VALIDATION_STEPS="${VALIDATION_STEPS:-$(python3 -c "print(max(100, int($MAX_TRAINING_STEPS / 20)))")}"
VAL_NUM_BATCHES="${VAL_NUM_BATCHES:-20}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"

echo "    GPUs              : $NPROC_PER_NODE (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "    Effective batch   : $EFFECTIVE_BATCH ($TRAIN_BATCH_SIZE per GPU × $NPROC_PER_NODE GPU(s))"
echo "    Target epochs     : $TARGET_EPOCHS"
echo "    Max training steps: $MAX_TRAINING_STEPS"
echo "    Checkpointing     : every $CHECKPOINTING_STEPS steps"
echo "    Validation        : every $VALIDATION_STEPS steps"
echo "    Validation batches: $VAL_NUM_BATCHES"

# WandB / logging
LOG_BACKEND="wandb"
if [[ "${WANDB_DISABLED:-0}" == "1" ]] || [[ -z "${WANDB_API_KEY:-}" ]]; then
    LOG_BACKEND="tensorboard"
    echo "    Logging           : tensorboard (wandb disabled or no API key)"
else
    echo "    Logging           : wandb (project=${WANDB_PROJECT:-psi}, entity=${WANDB_ENTITY:-})"
fi

# ── 6. Launch training ────────────────────────────────────────────────────────
echo ""
echo "[5/5] Launching training..."
echo ""

ulimit -n 65535 2>/dev/null || true

# Disable DeepSpeed JIT CUDA op compilation (no CUDA toolkit / nvcc required)
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1

cd "$REPO_ROOT"

# Avoid initializing NCCL for a single GPU (required on GB10/DGX Spark).
# For multiple GPUs, use torchrun as before.
LAUNCHER=(python scripts/train.py)
if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
    if command -v torchrun &>/dev/null; then
        LAUNCHER=(torchrun)
    elif [[ -x "$(dirname "$(command -v python)")/torchrun" ]]; then
        LAUNCHER=("$(dirname "$(command -v python)")/torchrun")
    else
        LAUNCHER=(python -m torch.distributed.run)
    fi
    LAUNCHER+=(--nproc_per_node="$NPROC_PER_NODE" --master_port=29500 scripts/train.py)
fi
echo "    Using launcher: ${LAUNCHER[*]}"
echo ""

EXTRA_TRAIN_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    EXTRA_TRAIN_ARGS+=(--train.resume-from-checkpoint="$RESUME_FROM_CHECKPOINT")
    echo "    Resuming from: $RESUME_FROM_CHECKPOINT"
fi
if [[ -n "$VAL_EPISODE_FRACTION" ]]; then
    EXTRA_TRAIN_ARGS+=(--data.val_episode_fraction="$VAL_EPISODE_FRACTION")
fi
# gaussian state-noise augmentation (train only; val stays clean)
if [[ -n "${STATE_NOISE_STD:-}" ]]; then
    EXTRA_TRAIN_ARGS+=(--data.transform.field.state-noise-std="$STATE_NOISE_STD")
    echo "    State noise std   : $STATE_NOISE_STD (joints)"
fi
if [[ -n "${STATE_NOISE_STD_WAIST:-}" ]]; then
    EXTRA_TRAIN_ARGS+=(--data.transform.field.state-noise-std-waist="$STATE_NOISE_STD_WAIST")
    echo "    State noise waist : $STATE_NOISE_STD_WAIST (rpy)"
fi
# early stopping (tyro bool: bare flag sets true, --train.no-early-stopping sets false)
if [[ "${EARLY_STOPPING:-0}" == "1" ]]; then
    EXTRA_TRAIN_ARGS+=(--train.early-stopping)
    if [[ -n "${EARLY_STOPPING_PATIENCE:-}" ]]; then
        EXTRA_TRAIN_ARGS+=(--train.early-stopping-patience="$EARLY_STOPPING_PATIENCE")
    fi
    if [[ -n "${EARLY_STOPPING_MIN_STEPS:-}" ]]; then
        EXTRA_TRAIN_ARGS+=(--train.early-stopping-min-steps="$EARLY_STOPPING_MIN_STEPS")
    fi
    if [[ -n "${EARLY_STOPPING_METRIC:-}" ]]; then
        EXTRA_TRAIN_ARGS+=(--train.early-stopping-metric="$EARLY_STOPPING_METRIC")
    fi
    echo "    Early stopping    : enabled (patience=${EARLY_STOPPING_PATIENCE:-15}, min_steps=${EARLY_STOPPING_MIN_STEPS:-0}, metric=${EARLY_STOPPING_METRIC:-auto})"
fi

"${LAUNCHER[@]}" \
    finetune_real_psi0_config \
    --seed=292285 \
    --exp="$EXP" \
    --train.name=finetune \
    --train.data_parallel=ddp \
    --train.mixed_precision=bf16 \
    --train.train_batch_size="$TRAIN_BATCH_SIZE" \
    --train.max_checkpoints_to_keep=5 \
    --train.gradient_accumulation_steps=1 \
    --train.learning_rate=1e-4 \
    --train.max_training_steps="$MAX_TRAINING_STEPS" \
    --train.warmup_ratio=None \
    --train.warmup_steps="$WARMUP_STEPS" \
    --train.checkpointing_steps="$CHECKPOINTING_STEPS" \
    --train.validation_steps="$VALIDATION_STEPS" \
    --train.val_num_batches="$VAL_NUM_BATCHES" \
    --train.max_grad_norm=1.0 \
    --train.lr_scheduler_type=cosine \
    --train.lr_scheduler_kwargs.weight_decay=1e-6 \
    --train.lr_scheduler_kwargs.betas 0.95 0.999 \
    --log.report_to="$LOG_BACKEND" \
    --wandb.project="${WANDB_PROJECT:-psi}" \
    --data.root_dir="$DATA_ROOT" \
    --data.train_repo_ids="$DATASET_NAME" \
    --data.transform.repack.pad-action-dim=36 \
    --data.transform.repack.pad-state-dim=36 \
    --data.transform.field.stat-path=meta/stats_psi0.json \
    --data.transform.field.stat-action-key=action \
    --data.transform.field.stat-state-key=states \
    --data.transform.field.action_norm_type=bounds \
    --data.transform.field.no-use-norm-mask \
    --data.transform.field.normalize-state \
    --data.transform.field.pad-action-dim=36 \
    --data.transform.field.pad-state-dim=36 \
    --data.transform.model.img-aug \
    --data.transform.model.resize.size 240 320 \
    --data.transform.model.center_crop.size 240 320 \
    --model.model_name_or_path="$VLM_CKPT" \
    --model.pretrained-action-header-path="$ACTION_CKPT" \
    --model.noise-scheduler=flow \
    --model.train-diffusion-steps=1000 \
    --model.n_conditions=0 \
    --model.action-chunk-size=30 \
    --model.action-dim=36 \
    --model.action-exec-horizon=30 \
    --model.observation-horizon=1 \
    --model.odim=36 \
    --model.view_feature_dim=2048 \
    --model.no-tune-vlm \
    --model.no-use_film \
    --model.no-combined_temb \
    --model.rtc \
    --model.max-delay=8 \
    "${EXTRA_TRAIN_ARGS[@]}"
