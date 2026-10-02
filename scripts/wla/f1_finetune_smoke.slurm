#!/bin/bash
#SBATCH --job-name=f1_smoke
#SBATCH --partition=b200n1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#
# F1 — smoke test do fine-tune oficial (frozen VLM) do UnifoLM-WLA-1.0-Base
# no WBT Shelf Organizing: valida recipe + checkpoint/resume.
#   max_train_steps=300, save_interval=100 -> checkpoints steps_100/200/300.
# REENTRANTE: trainer.is_resume=true faz o trainer procurar o último
# steps_<N>_model.safetensors em $WLA_CKPT/f1_smoke/checkpoints e retomar
# (_get_latest_checkpoint em trainer_utils/trainer_tools.py); se não houver,
# cai no pretrained_checkpoint do release (1a execução).
set -uo pipefail

PSI0=/raid/user_marcospaulo/Psi0
source "$PSI0/scripts/wla/env_raid.sh"
module load cuda12.6/toolkit/12.6.3
export CUDA_HOME=/cm/shared/apps/cuda12.6/toolkit/12.6.3
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

RUN_ROOT_DIR="$WLA_CKPT"
RUN_ID=f1_smoke
OUTPUT_DIR="$RUN_ROOT_DIR/$RUN_ID"
FINAL_MODEL="$OUTPUT_DIR/final_model/model.safetensors"
mkdir -p "$OUTPUT_DIR"

# Proveniência (só na 1a vez, para não sobrescrever em requeues).
if [[ ! -f "$OUTPUT_DIR/git_commit.txt" ]]; then
  {
    echo "psi0: $(git -C "$PSI0" rev-parse HEAD) ($(git -C "$PSI0" rev-parse --abbrev-ref HEAD))"
    echo "unifolm-wla: $(git -C "$WLA_REPO" rev-parse HEAD)"
    echo "dataset: $WLA_DATA_ROOT/UnifoLM_WBT_Dataset (200 ep)"
  } > "$OUTPUT_DIR/git_commit.txt"
  cp "$PSI0/configs/wla/unitree_wbt_shelf.yaml" "$OUTPUT_DIR/data_config.yaml"
fi

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export WANDB_MODE=disabled
export NUM_PROCESSES=1

cd "$WLA_REPO"
source .venv/bin/activate

# --- Requeue handler -------------------------------------------------------
# Slurm envia SIGUSR1 300s antes do time limit (#SBATCH --signal). O trainer
# não tem handler próprio: repassamos o sinal (o python morre após o último
# checkpoint salvo em save_interval) e re-enfileiramos se não terminou.
TRAIN_PID=""
on_preempt() {
  echo "[f1_smoke] sinal recebido; repassando SIGUSR1 ao trainer (pid=$TRAIN_PID)"
  if [[ -n "$TRAIN_PID" ]]; then
    kill -USR1 "$TRAIN_PID" 2>/dev/null
    wait "$TRAIN_PID" 2>/dev/null
  fi
  if [[ ! -f "$FINAL_MODEL" ]]; then
    echo "[f1_smoke] treino incompleto; scontrol requeue $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
  fi
  exit 0
}
trap on_preempt SIGUSR1 SIGTERM

# Launcher oficial: examples/unifolm_wla/train_files/run_finetune_mmdit_frozen_vlm.sh
# (replicado aqui com paths do cluster; overrides via dotlist -> OmegaConf.merge).
.venv/bin/accelerate launch \
  --config_file unifolm_wla/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  unifolm_wla/training/train_unifolm_wla.py \
  --config_yaml unifolm_wla/config/training/mmdit_finetune_frozen_vlm.yaml \
  --framework.qwenvl.base_vlm "$WLA_MODELS/UnifoLM-WLA-1.0-Base/tokenizer" \
  --trainer.pretrained_checkpoint "$WLA_MODELS/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors" \
  --datasets.vla_data.data_config_path "$PSI0/configs/wla/unitree_wbt_shelf.yaml" \
  --run_root_dir "$RUN_ROOT_DIR" \
  --run_id "$RUN_ID" \
  --trainer.is_resume true \
  --trainer.max_train_steps 300 \
  --trainer.save_interval 100 \
  --trainer.logging_frequency 10 &
TRAIN_PID=$!
wait "$TRAIN_PID"
STATUS=$?

echo "[f1_smoke] trainer saiu com status $STATUS"
if [[ -f "$FINAL_MODEL" ]]; then
  echo "[f1_smoke] final_model presente: treino completo."
else
  echo "[f1_smoke] AVISO: sem final_model (status=$STATUS)."
fi
exit "$STATUS"
