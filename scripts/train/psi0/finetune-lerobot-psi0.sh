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

if [[ -f "$REPO_ROOT/.venv-psi/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$REPO_ROOT/.venv-psi/bin/activate"
    echo "    Using: .venv-psi"
elif [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
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

    if [[ -d "$path" ]] || [[ -f "$path" ]]; then
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

    if [[ ! -d "$path" ]] && [[ ! -f "$path" ]]; then
        echo "[ERROR] Download appeared to succeed but $path still not found."
        echo "        Expected the model at: $path"
        exit 1
    fi
    echo "    $label: downloaded ✓"
}

_check_or_download "$VLM_CKPT"    "$VLM_REMOTE"    "VLM backbone"
_check_or_download "$ACTION_CKPT" "$ACTION_REMOTE" "Action header"

# ── 5. Compute hyperparameters ─────────────────────────────────────────────────
echo "[4/5] Computing training hyperparameters..."

TARGET_EPOCHS="${TARGET_EPOCHS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"

# Detect GPU count
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | awk 'BEGIN{i=0} {printf "%s%d",(i?",":""),i; i++}' || echo "0")}"
NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# steps = (frames / (batch_per_gpu * n_gpus)) * epochs
EFFECTIVE_BATCH=$(( TRAIN_BATCH_SIZE * NPROC_PER_NODE ))
MAX_TRAINING_STEPS=$(python3 -c "
frames = $TOTAL_FRAMES
epochs = $TARGET_EPOCHS
batch  = $EFFECTIVE_BATCH
steps  = max(1000, int(frames / batch * epochs))
print(steps)
")

CHECKPOINTING_STEPS=$(python3 -c "print(max(200, int($MAX_TRAINING_STEPS / 10)))")
VALIDATION_STEPS=$(python3 -c "print(max(100, int($MAX_TRAINING_STEPS / 20)))")

echo "    GPUs              : $NPROC_PER_NODE (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "    Effective batch   : $EFFECTIVE_BATCH ($TRAIN_BATCH_SIZE per GPU × $NPROC_PER_NODE GPU(s))"
echo "    Target epochs     : $TARGET_EPOCHS"
echo "    Max training steps: $MAX_TRAINING_STEPS"
echo "    Checkpointing     : every $CHECKPOINTING_STEPS steps"
echo "    Validation        : every $VALIDATION_STEPS steps"

# WandB / logging
LOG_BACKEND="wandb"
if [[ "${WANDB_DISABLED:-0}" == "1" ]] || [[ -z "${WANDB_API_KEY:-}" ]]; then
    LOG_BACKEND="tensorboard"
    echo "    Logging           : tensorboard (wandb disabled or no API key)"
else
    echo "    Logging           : wandb (project=psi, entity=${WANDB_ENTITY:-})"
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

# Resolve torchrun: prefer venv-local binary, fall back to python -m torch.distributed.run
if command -v torchrun &>/dev/null; then
    TORCHRUN="torchrun"
elif [[ -x "$(dirname "$(command -v python)")/torchrun" ]]; then
    TORCHRUN="$(dirname "$(command -v python)")/torchrun"
else
    TORCHRUN="python -m torch.distributed.run"
fi
echo "    Using launcher: $TORCHRUN"
echo ""

$TORCHRUN \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port=29500 \
    scripts/train.py \
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
    --train.warmup_steps=500 \
    --train.checkpointing_steps="$CHECKPOINTING_STEPS" \
    --train.validation_steps="$VALIDATION_STEPS" \
    --train.val_num_batches=20 \
    --train.max_grad_norm=1.0 \
    --train.lr_scheduler_type=cosine \
    --train.lr_scheduler_kwargs.weight_decay=1e-6 \
    --train.lr_scheduler_kwargs.betas 0.95 0.999 \
    --log.report_to="$LOG_BACKEND" \
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
    --model.max-delay=8
