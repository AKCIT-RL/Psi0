#!/bin/bash

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

source .venv-psi/bin/activate

NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
ulimit -n 65535
echo "Training with $NPROC_PER_NODE GPUs"

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <repo> <task> [exp]"
    echo "Example: $0 G1WholebodyBendPick-v0-psi0 bend-pick my-exp"
    exit 1
fi

export repo="$1"
export task="$2"
task_words=$(echo "$task" | tr '[:upper:]' '[:lower:]' | tr '_' ' ')
default_exp=$(echo "$task_words" | awk '{if (NF>=2) print $1 "-" $2; else print $1}')
export exp=${3:-$default_exp}

echo "Task: $task"
echo "Experiment name: $exp"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-40000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-10000}"
VALIDATION_STEPS="${VALIDATION_STEPS:-500}"

PSI_CKPT_DIR="${PSI_HOME:-/root/.cache/checkpoints/psi0}"
VLM_REMOTE="psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"
ACTION_REMOTE="psi0/postpre.1by1.pad36.2601131206.ckpt.he30k"

VLM_CKPT="${VLM_CKPT_PATH:-$PSI_CKPT_DIR/pre.fast.1by1.2601091803.ckpt.ego200k.he30k}"
ACTION_CKPT="${ACTION_CKPT_PATH:-$PSI_CKPT_DIR/postpre.1by1.pad36.2601131206.ckpt.he30k}"

_check_or_download() {
    local path="$1"
    local remote="$2"
    local label="$3"
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

_check_or_download "$VLM_CKPT" "$VLM_REMOTE" "VLM backbone"
_check_or_download "$ACTION_CKPT" "$ACTION_REMOTE" "Action header"



args="
finetune_simple_psi0_config \
--seed=292285 \
--exp=$exp \
--train.name=finetune \
--train.data_parallel=ddp \
--train.mixed_precision=bf16 \
--train.train_batch_size=$TRAIN_BATCH_SIZE \
--train.max_checkpoints_to_keep=5 \
--train.gradient_accumulation_steps=1 \
--train.learning_rate=1e-4 \
--train.max_training_steps=$MAX_TRAINING_STEPS \
--train.warmup_ratio=None \
--train.warmup_steps=$WARMUP_STEPS \
--train.checkpointing_steps=$CHECKPOINTING_STEPS \
--train.validation_steps=$VALIDATION_STEPS \
--train.val_num_batches=20 \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_type=cosine \
--train.lr_scheduler_kwargs.weight_decay=1e-6 \
--train.lr_scheduler_kwargs.betas 0.95 0.999 \
--log.report_to=wandb \
--data.root_dir=$PSI_HOME/$repo/$task \
--data.train-repo-ids=$task \
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
--data.transform.model.resize.size 180 320 \
--data.transform.model.center_crop.size 180 320 \
--model.model_name_or_path=$VLM_CKPT \
--model.pretrained-action-header-path=$ACTION_CKPT \
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
"

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=29500 scripts/train.py \
    ${args}

