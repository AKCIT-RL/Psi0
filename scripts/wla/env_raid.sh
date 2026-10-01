# Source in every WLA job: keeps all caches/data on /raid (cluster rule) and loads HF auth.
export RAID=/raid/user_marcospaulo
export WLA_REPO=$RAID/Psi0/third_party/unifolm-wla
export WLA_DATA_ROOT=$RAID/datasets/unifolm
export WLA_MODELS=$RAID/models/unifolm-wla
export PSI0_DATA=$RAID/datasets/psi0
export WLA_EXP=$RAID/experiments/wla
export WLA_CKPT=$RAID/checkpoints/wla

export HF_HOME=$RAID/cache/huggingface
export UV_CACHE_DIR=$RAID/cache/uv
export PIP_CACHE_DIR=$RAID/cache/pip
export TORCH_HOME=$RAID/cache/torch
export XDG_CACHE_HOME=$RAID/cache/xdg
export TRITON_CACHE_DIR=$RAID/cache/triton
export WANDB_DIR=$RAID/wandb
export HF_HUB_ENABLE_HF_TRANSFER=0

if [[ -r $RAID/secrets/hf_token ]]; then
    HF_TOKEN=$(<"$RAID/secrets/hf_token")
    export HF_TOKEN
fi
if [[ -r $RAID/secrets/wandb.env ]]; then
    set -a; source "$RAID/secrets/wandb.env"; set +a
fi

mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
    "$WLA_DATA_ROOT" "$WLA_MODELS" "$PSI0_DATA" "$WLA_EXP" "$WLA_CKPT" "$RAID/slurm_logs"
