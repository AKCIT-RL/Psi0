#!/bin/bash
# PSI-0 Training Entrypoint
#
# Usage inside container:
#   /workspace/docker/entrypoint.sh <dataset_dir_or_repo_id> [exp_name]
#
# Arguments:
#   dataset_dir_or_repo_id  Path inside /workspace/data/ OR a HuggingFace dataset repo_id
#   exp_name                Optional experiment name (default: auto)
#
# Environment variables:
#   WANDB_API_KEY         WandB API key (required for wandb logging)
#   HF_TOKEN              HuggingFace token (required for private/gated datasets)
#   CUDA_VISIBLE_DEVICES  GPUs to use (default: all)
#   TARGET_EPOCHS         Training epochs (default: 50)
#   TRAIN_BATCH_SIZE      Batch size per GPU (default: 16)
#   VLM_CKPT_PATH         Override VLM backbone checkpoint path
#   ACTION_CKPT_PATH      Override action header checkpoint path

set -euo pipefail

# train.py requires a .env file to exist (it calls load_dotenv with assert)
if [[ ! -f /workspace/.env ]]; then
    cat > /workspace/.env <<'EOF'
# PSI-0 Docker environment
PSI_HOME=/workspace/checkpoints
DATA_HOME=/workspace/data
HF_HOME=/workspace/hf_cache
EOF
    [[ -n "${WANDB_API_KEY:-}" ]] && echo "WANDB_API_KEY=${WANDB_API_KEY}" >> /workspace/.env
    [[ -n "${WANDB_ENTITY:-}" ]]  && echo "WANDB_ENTITY=${WANDB_ENTITY}"   >> /workspace/.env
fi

DATASET_ARG="${1:-}"
EXP_NAME="${2:-}"

if [[ -z "$DATASET_ARG" ]]; then
    echo "Usage: $0 <dataset_dir_or_hf_repo_id> [exp_name]"
    echo ""
    echo "Examples:"
    echo "  # Local dataset (mounted at /workspace/data/)"
    echo "  $0 pick_cylinder_manipulation_psi"
    echo ""
    echo "  # HuggingFace dataset subfolder (namespace/repo_name/subfolder/dataset_name)"
    echo "  $0 USC-PSI-Lab/psi-data/simple/G1WholebodyXMovePick-v0"
    exit 1
fi

# ── Resolve dataset path ────────────────────────────────────────────────────────
if [[ -d "/workspace/data/$DATASET_ARG" ]]; then
    DATASET_PATH="/workspace/data/$DATASET_ARG"
elif [[ -d "/workspace/data/real/$DATASET_ARG" ]]; then
    DATASET_PATH="/workspace/data/real/$DATASET_ARG"
elif [[ -d "$DATASET_ARG" ]]; then
    DATASET_PATH="$(cd "$DATASET_ARG" && pwd)"
else
    # Treat as HuggingFace repo_id — format: namespace/repo_name[/subfolder/dataset_name]
    echo "[INFO] Dataset not found locally. Attempting HuggingFace download..."
    echo "       arg: $DATASET_ARG"

    # Split into repo_id (first two components) and optional subfolder (remainder)
    HF_REPO_ID="$(echo "$DATASET_ARG" | cut -d'/' -f1-2)"
    HF_SUBFOLDER="$(echo "$DATASET_ARG" | cut -d'/' -f3-)"

    DATASET_NAME=$(basename "$DATASET_ARG")
    DEST="/workspace/data/$DATASET_NAME"
    mkdir -p "$DEST"

    HF_TOKEN_ARG=""
    if [[ -n "${HF_TOKEN:-}" ]]; then
        HF_TOKEN_ARG="--token=$HF_TOKEN"
    fi

    if [[ -n "$HF_SUBFOLDER" ]]; then
        # Datasets are stored as zips: simple/<dataset_name>.zip
        STAGING="/workspace/data/.hf_staging_$DATASET_NAME"
        mkdir -p "$STAGING"
        hf download "$HF_REPO_ID" \
            --repo-type dataset \
            --include "${HF_SUBFOLDER}.zip" \
            --local-dir "$STAGING" \
            $HF_TOKEN_ARG
        ZIP_FILE="$STAGING/${HF_SUBFOLDER}.zip"
        if [[ ! -f "$ZIP_FILE" ]]; then
            echo "[ERROR] Expected zip not found: $ZIP_FILE"
            rm -rf "$STAGING"
            exit 1
        fi
        echo "[INFO] Extracting $(basename "$ZIP_FILE")..."
        unzip -qo "$ZIP_FILE" -d "$STAGING"
        # The zip extracts to a subdirectory named after the dataset; move it to DEST
        EXTRACTED="$STAGING/$DATASET_NAME"
        if [[ -d "$EXTRACTED" ]]; then
            rm -rf "$DEST"
            mv "$EXTRACTED" "$DEST"
        else
            rm -rf "$DEST"
            mv "$STAGING" "$DEST"
        fi
        rm -rf "$STAGING"
    else
        hf download "$HF_REPO_ID" \
            --repo-type dataset \
            --local-dir "$DEST" \
            $HF_TOKEN_ARG
    fi

    DATASET_PATH="$DEST"
fi

DATASET_NAME="$(basename "$DATASET_PATH")"
DATA_ROOT="$(dirname "$DATASET_PATH")"

echo "════════════════════════════════════════════════"
echo " PSI-0 Training Container"
echo " Dataset : $DATASET_PATH"
echo " Data root: $DATA_ROOT"
echo "════════════════════════════════════════════════"

# ── Validate required metadata ─────────────────────────────────────────────────
for f in "data" "videos" "meta/info.json" "meta/stats_psi0.json" "meta/episodes.jsonl" "meta/tasks.jsonl"; do
    if [[ ! -e "$DATASET_PATH/$f" ]]; then
        echo "[ERROR] Missing: $DATASET_PATH/$f"
        exit 1
    fi
done

# Generate episodes_stats.jsonl if missing
if [[ ! -f "$DATASET_PATH/meta/episodes_stats.jsonl" ]]; then
    echo "[INFO] Generating episodes_stats.jsonl..."
    python /workspace/docker/gen_episodes_stats.py "$DATASET_PATH"
fi

# ── Locate PSI checkpoints ─────────────────────────────────────────────────────
PSI_CKPT_DIR="${PSI_HOME}/cache/checkpoints/psi0"
VLM_CKPT="${VLM_CKPT_PATH:-$PSI_CKPT_DIR/pre.fast.1by1.2601091803.ckpt.ego200k.he30k}"
ACTION_CKPT="${ACTION_CKPT_PATH:-$PSI_CKPT_DIR/postpre.1by1.pad36.2601131206.ckpt.he30k}"

_download_ckpt() {
    local path="$1" remote="$2" label="$3"
    if [[ -d "$path" ]] || [[ -f "$path" ]]; then
        echo "    $label: $path ✓"
        return
    fi
    echo "[INFO] Downloading $label from HuggingFace..."
    mkdir -p "$(dirname "$path")"
    hf download USC-PSI-Lab/psi-model \
        --include "${remote}/**" \
        --local-dir="$(dirname "$PSI_CKPT_DIR")" \
        --repo-type=model
    echo "    $label: downloaded ✓"
}

_download_ckpt "$VLM_CKPT"    "psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"    "VLM backbone"
_download_ckpt "$ACTION_CKPT" "psi0/postpre.1by1.pad36.2601131206.ckpt.he30k"      "Action header"

# ── Compute hyperparameters ────────────────────────────────────────────────────
TARGET_EPOCHS="${TARGET_EPOCHS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | awk 'BEGIN{i=0}{printf "%s%d",(i?",":""),i;i++}' || echo "0")}"
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
EFFECTIVE_BATCH=$(( TRAIN_BATCH_SIZE * NPROC ))
TOTAL_FRAMES=$(python -c "import json; print(json.load(open('$DATASET_PATH/meta/info.json'))['total_frames'])")
MAX_STEPS=$(python -c "print(max(1000, int($TOTAL_FRAMES / $EFFECTIVE_BATCH * $TARGET_EPOCHS)))")
CKPT_STEPS=$(python -c "print(max(200, int($MAX_STEPS / 10)))")
VAL_STEPS=$(python -c "print(max(100, int($MAX_STEPS / 20)))")

# Auto exp name
LOWER="${DATASET_NAME,,}"
WORDS=( $(echo "$LOWER" | tr '_' ' ') )
DEFAULT_EXP="${WORDS[0]}-${WORDS[1]:-run}"
EXP="${EXP_NAME:-$DEFAULT_EXP}"

LOG_BACKEND="wandb"
if [[ -z "${WANDB_API_KEY:-}" ]] || [[ "${WANDB_DISABLED:-}" == "true" ]]; then
    LOG_BACKEND="tensorboard"
fi

echo "    GPUs       : $NPROC ($CUDA_VISIBLE_DEVICES)"
echo "    Frames     : $TOTAL_FRAMES"
echo "    Max steps  : $MAX_STEPS"
echo "    Batch size : $EFFECTIVE_BATCH"
echo "    Logging    : $LOG_BACKEND"
echo "    Exp name   : $EXP"
echo ""

# ── Launch training ────────────────────────────────────────────────────────────
ulimit -n 65535 2>/dev/null || true

torchrun \
    --nproc_per_node="$NPROC" \
    --master_port=29500 \
    /workspace/scripts/train.py \
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
    --train.max_training_steps="$MAX_STEPS" \
    --train.warmup_ratio=None \
    --train.warmup_steps=500 \
    --train.checkpointing_steps="$CKPT_STEPS" \
    --train.validation_steps="$VAL_STEPS" \
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
