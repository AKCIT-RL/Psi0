#!/bin/bash
#SBATCH --job-name=f5_psi0
#SBATCH --partition=b200n1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#
# F5 — fine-tune do action expert (VLM congelada) no Ψ0 convertido.
# Só deve ser submetido depois da conversão e da F3. Retoma do último ckpt.
set -uo pipefail
PSI0=/raid/user_marcospaulo/Psi0
source "$PSI0/scripts/wla/env_raid.sh"
module load cuda12.6/toolkit/12.6.3
export CUDA_HOME=/cm/shared/apps/cuda12.6/toolkit/12.6.3
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
RUN_ROOT_DIR="$WLA_CKPT"
RUN_ID=f5_psi0_tote
OUTPUT_DIR="$RUN_ROOT_DIR/$RUN_ID"
FINAL_MODEL="$OUTPUT_DIR/final_model/model.safetensors"
mkdir -p "$OUTPUT_DIR" "$WLA_EXP/f5_psi0_tote"
if [[ ! -f "$OUTPUT_DIR/git_commit.txt" ]]; then
  {
    echo "psi0: $(git -C "$PSI0" rev-parse HEAD)"
    echo "unifolm-wla: $(git -C "$WLA_REPO" rev-parse HEAD)"
    echo "dataset: $WLA_DATA_ROOT/Psi0_Tote_Dataset/G1ToteMix_psi0_trainval"
  } > "$OUTPUT_DIR/git_commit.txt"
  cp "$PSI0/configs/wla/psi0_tote.yaml" "$OUTPUT_DIR/data_config.yaml"
  cp "$OUTPUT_DIR/git_commit.txt" "$WLA_EXP/f5_psi0_tote/git_commit.txt"
  cp "$PSI0/configs/wla/psi0_tote.yaml" "$WLA_EXP/f5_psi0_tote/config.yaml"
fi
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
# Forge self-hosted. Projeto novo, só do WLA. Entity explícita (não herdar a do psi0).
export WANDB_BASE_URL=https://forge.coreweave.com/wandb
export WANDB_ENTITY=akcit_industrial_humanoids
export WANDB_PROJECT=wla
export WANDB_MODE=offline  # Forge: POST /wandb/graphql dá 405 (URL da API incorreta); sincronizar depois com wandb sync
export WLA_WANDB_RUN_FILE="$WLA_EXP/f5_psi0_tote/wandb_run.txt"
export NUM_PROCESSES=1
cd "$WLA_REPO"
source .venv/bin/activate
TRAIN_PID=""
on_preempt() {
  if [[ -n "$TRAIN_PID" ]]; then kill -USR1 "$TRAIN_PID" 2>/dev/null; wait "$TRAIN_PID" 2>/dev/null; fi
  if [[ ! -f "$FINAL_MODEL" ]]; then scontrol requeue "$SLURM_JOB_ID"; fi
  exit 0
}
trap on_preempt SIGUSR1 SIGTERM
.venv/bin/accelerate launch \
  --config_file unifolm_wla/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  unifolm_wla/training/train_unifolm_wla.py \
  --config_yaml unifolm_wla/config/training/mmdit_finetune_frozen_vlm.yaml \
  --framework.qwenvl.base_vlm "$WLA_MODELS/UnifoLM-WLA-1.0-Base/tokenizer" \
  --trainer.pretrained_checkpoint "$WLA_MODELS/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors" \
  --datasets.vla_data.data_config_path "$PSI0/configs/wla/psi0_tote.yaml" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_project "$WANDB_PROJECT" \
  --run_root_dir "$RUN_ROOT_DIR" \
  --run_id "$RUN_ID" \
  --trainer.is_resume true \
  --trainer.max_train_steps 20000 \
  --trainer.save_interval 2000 \
  --trainer.logging_frequency 20 &
TRAIN_PID=$!
wait "$TRAIN_PID"
STATUS=$?
echo "[f5] status=$STATUS final=$([[ -f $FINAL_MODEL ]] && echo yes || echo no)"
exit "$STATUS"
