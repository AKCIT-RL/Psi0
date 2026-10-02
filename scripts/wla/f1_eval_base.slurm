#!/bin/bash
#SBATCH --job-name=f1_eval_base
#SBATCH --partition=b200n1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
#
# F1 — baseline oficial UnifoLM-WLA-1.0-Base: eval de 1 episódio completo
# (chunk-by-chunk) do WBT Shelf Organizing com o script oficial
# examples/unifolm_wla/eval_files/unitree/eval_local_episode.py.
set -euo pipefail

PSI0=/raid/user_marcospaulo/Psi0
source "$PSI0/scripts/wla/env_raid.sh"
module load cuda12.6/toolkit/12.6.3
export CUDA_HOME=/cm/shared/apps/cuda12.6/toolkit/12.6.3

DATA_CONFIG="$PSI0/configs/wla/unitree_wbt_shelf.yaml"
EXP_DIR="$WLA_EXP/f1_eval_base"
SAVE_DIR="$EXP_DIR/ep0"
mkdir -p "$SAVE_DIR"

# Proveniência do experimento.
{
  echo "psi0: $(git -C "$PSI0" rev-parse HEAD) ($(git -C "$PSI0" rev-parse --abbrev-ref HEAD))"
  echo "unifolm-wla: $(git -C "$WLA_REPO" rev-parse HEAD)"
} > "$EXP_DIR/git_commit.txt"
cp "$DATA_CONFIG" "$EXP_DIR/unitree_wbt_shelf.yaml"
cp "$0" "$EXP_DIR/f1_eval_base.slurm"

cd "$WLA_REPO"
source .venv/bin/activate

# --source_name: dataset_statistics.json do release usa a chave 'UnifoLM_WBT'
# (sem sufixo '_Dataset'); sem isso o eval cairia no fallback para a 1a chave.
# --base_vlm: config.yaml do release tem base_vlm relativo ('./tokenizer');
# apontamos para o tokenizer absoluto (só arquitetura+tokenizer são lidos).
.venv/bin/python -m examples.unifolm_wla.eval_files.unitree.eval_local_episode \
  --ckpt_path "$WLA_MODELS/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors" \
  --data_config_path "$DATA_CONFIG" \
  --source_name UnifoLM_WBT \
  --base_vlm "$WLA_MODELS/UnifoLM-WLA-1.0-Base/tokenizer" \
  --episode_idx 0 \
  --save_dir "$SAVE_DIR" 2>&1 | tee "$EXP_DIR/eval_stdout.log"

# Parse do sumário impresso pelo script oficial -> metrics.json
.venv/bin/python - "$EXP_DIR/eval_stdout.log" "$EXP_DIR/metrics.json" <<'EOF'
import json, re, sys

log_path, out_path = sys.argv[1], sys.argv[2]
text = open(log_path).read()

m = re.search(r"Episode (\d+) \| (\d+) frames \| (\d+) chunks", text)
metrics = {
    "episode_idx": int(m.group(1)),
    "frames": int(m.group(2)),
    "chunks": int(m.group(3)),
}

def parse_block(header, key):
    blk = re.search(header + r":\n((?:    dim .*\n)+)", text)
    out = {}
    if blk:
        for line in blk.group(1).strip().splitlines():
            dm = re.match(r"\s*dim\s+(\d+)\s+(\S+)\s+([\d.]+)", line)
            if dm:
                out[dm.group(2)] = float(dm.group(3))
    metrics[key] = out

parse_block(r"Per-dim mean L1 \(absolute EE\)", "abs_ee_l1_per_dim")
parse_block(r"Per-dim mean L1 \(relative EE\)", "rel_ee_l1_per_dim")
for pat, key in [(r"Overall abs EE L1: ([\d.]+)", "overall_abs_ee_l1"),
                 (r"Overall rel EE L1: ([\d.]+)", "overall_rel_ee_l1")]:
    mm = re.search(pat, text)
    metrics[key] = float(mm.group(1)) if mm else None

with open(out_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"metrics.json written: {out_path}")
print(json.dumps(metrics, indent=2))
EOF
