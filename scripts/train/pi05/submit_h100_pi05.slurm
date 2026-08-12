#!/bin/bash
#SBATCH --job-name=pi05-g1totemix
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=196G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --signal=B:SIGUSR1@300

# Fine-tune pi05 on G1ToteMix-psi0 with validation + early stopping + W&B.
# Env overrides: CONFIG, EXP_NAME, SMOKE=1 (few steps), EXTRA_ARGS.
set -euo pipefail
ROOT=/raid/user_marcospaulo/Psi0
cd "$ROOT"

CONFIG=${CONFIG:-g1totemix_pi05}
EXP_NAME=${EXP_NAME:-pi05-g1totemix-h100-$(date +%y%m%d%H%M)}

set -a; source secrets/psi0.env; set +a
# host paths (secrets file uses container paths)
export PSI_HOME="$ROOT"
export DATA_HOME="$ROOT/data"
export HF_HOME="$ROOT/cache/huggingface"
export WANDB_CACHE_DIR="$ROOT/cache/wandb"
export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false

# train_pytorch.py asserts a .env at cwd
[[ -e .env ]] || ln -s secrets/psi0.env .env

source .venv-openpi/bin/activate

# norm stats expected at <dataset>/norm_stats.json
if [[ ! -f data/G1ToteMix-psi0/norm_stats.json ]]; then
    python src/openpi/rewrite_norm_stats.py --task_path="$ROOT/data/G1ToteMix-psi0"
fi

NPROC=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' '\n' | wc -l)
echo "config=$CONFIG exp=$EXP_NAME gpus=$NPROC cuda=$CUDA_VISIBLE_DEVICES"

RESUME_ARGS=()
CKPT_DIR=".runs/openpi-05/$CONFIG/$EXP_NAME"
if compgen -G "$CKPT_DIR/[0-9]*" >/dev/null; then
    RESUME_ARGS+=(--resume)
    echo "resuming from $CKPT_DIR"
fi

SMOKE_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
    SMOKE_ARGS+=(--num_train_steps=30 --val_interval=20 --val_num_batches=2 --log_interval=5 --save_interval=20)
fi

ulimit -n 65535
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" src/openpi/train_pytorch.py \
    "$CONFIG" \
    --exp_name="$EXP_NAME" \
    --save_interval=2500 \
    --checkpoint_base_dir=.runs/openpi-05 \
    --num_workers=4 \
    "${RESUME_ARGS[@]}" "${SMOKE_ARGS[@]}" ${EXTRA_ARGS:-}
