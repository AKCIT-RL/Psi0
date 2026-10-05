#!/usr/bin/env bash
# Eval closed-loop do WLA no SIMPLE SEM Slurm: servidor do modelo no host (venv do WLA) + simulador em Docker (imagem SIMPLE).
# Equivalente a scripts/wla/simple_eval.slurm (modo model), pensado para uma GPU com RT cores (p.ex. RTX 4090) e render Isaac.
#
# Variáveis (todas opcionais, exceto o que o md de handoff manda definir):
#   WLA_HOME   raiz dos artefatos do zip (padrão: $HOME/wla_data)  -> checkpoints/, experiments/, simple_data/, models/
#   RUN        nome da run (padrão: eval_<data>)            -> $WLA_HOME/experiments/wla/$RUN
#   SPLIT      val|test (padrão val; test SÓ no relatório final, config congelada)   N (padrão 1)   SEED (padrão 0)
#   SIM_MODE   mujoco_isaac (padrão, render igual ao dataset) | mujoco (CPU, só diagnóstico)
#   ATTN       sdpa (padrão) | flash_attention_2 (se flash-attn estiver instalado)
#   IMAGE      imagem do simulador (padrão ghcr.io/physical-superintelligence-lab/simple:260829)
#   PORT       porta do servidor (padrão 22085)    WANDB_SYNC=0 desliga o envio ao W&B
set -euo pipefail
cd "$(dirname "$0")/../.."
PSI0=$PWD; SIMPLE=$PSI0/third_party/SIMPLE; WLA_REPO=$PSI0/third_party/unifolm-wla
WLA_HOME=${WLA_HOME:-$HOME/wla_data}; SPLIT=${SPLIT:-val}; N=${N:-1}; SEED=${SEED:-0}; PORT=${PORT:-22085}
SIM_MODE=${SIM_MODE:-mujoco_isaac}; ATTN=${ATTN:-sdpa}; RUN=${RUN:-eval_$(date +%Y%m%d_%H%M%S)}
IMAGE=${IMAGE:-ghcr.io/physical-superintelligence-lab/simple:260829}
CKPT=$WLA_HOME/checkpoints/wla/f5_psi0_tote/final_model/model.safetensors
BASE_VLM=$WLA_HOME/models/UnifoLM-WLA-1.0-Base/tokenizer
EXP=$WLA_HOME/experiments/wla; OUT=$EXP/$RUN; mkdir -p "$OUT" "$WLA_HOME/simple_data" "$WLA_HOME/cache/simple_home"
for f in "$CKPT" "$BASE_VLM/tokenizer.json" "$EXP/s0_eval_data/roots/$SPLIT/meta/info.json" "$WLA_REPO/.venv/bin/python" "$SIMPLE/third_party/gear_sonic"; do
  [[ -e $f ]] || { echo "FALTA: $f (ver docs/wla/HANDOFF_4090.md)"; exit 2; }
done
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "imagem ausente: docker pull $IMAGE"; exit 2; }

{ echo "psi0: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"; echo "simple: $(git -C "$SIMPLE" rev-parse HEAD)"
  echo "unifolm-wla: $(git -C "$WLA_REPO" rev-parse HEAD)"; echo "image: $IMAGE"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } > "$OUT/git_commit.txt"
cp "$0" "$OUT/docker_eval.sh"
env | grep -E "^(RUN|SPLIT|N|SEED|SIM_MODE|ATTN|IMAGE)=" > "$OUT/config.env" || true
cat > "$OUT/eval_config.yaml" <<EOF
datasets:
  - name: $SPLIT
    path: $EXP/s0_eval_data/roots/$SPLIT
    prompt: recorded
EOF

# ---- servidor do modelo (host, venv do WLA) ----
( source "$WLA_REPO/.venv/bin/activate"; export PYTHONPATH="$PSI0/src:$WLA_REPO" HF_HUB_OFFLINE=1
  exec python scripts/wla/simple_wla_server.py --port "$PORT" --ckpt_path "$CKPT" --source_name Psi0_Tote_Dataset --profile psi0_tote \
       --base_vlm "$BASE_VLM" --use_bf16 --attn_impl "$ATTN" --debug_dir "$OUT/server_debug" ) > "$OUT/server.log" 2>&1 &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 5; done
curl -sf "http://127.0.0.1:$PORT/health" || { echo "servidor não subiu"; tail -30 "$OUT/server.log"; exit 1; }

# ---- simulador (Docker; código/submódulos do checkout por cima da imagem; mesmos caminhos dentro e fora p/ o eval_config) ----
# --network host: o cliente enxerga o servidor em 127.0.0.1. NVIDIA_DRIVER_CAPABILITIES=all: Vulkan/RT para o render Isaac.
docker run --rm --gpus all --network host --ipc host \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=Y \
  -e HOME=/root -e HF_HUB_OFFLINE=${OFFLINE:-1} -e SIMPLE_EVAL_EPISODE_ID= -e WLA_EXEC_STEPS=${EXEC_STEPS:-30} \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  -v "$WLA_HOME/simple_data:/workspace/simple/data" -v "$WLA_HOME/cache/simple_home:/root" \
  -v "$SIMPLE/src:/workspace/simple/src" -v "$SIMPLE/third_party/gear_sonic:/workspace/simple/third_party/gear_sonic" \
  -v "$SIMPLE/third_party/decoupled_wbc:/workspace/simple/third_party/decoupled_wbc" \
  -v "$EXP:$EXP" -w /workspace/simple "$IMAGE" bash -c '
    set -o pipefail
    /workspace/simple/.venv/bin/eval-decoupled-wbc simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0 wla_decoupled_wbc '"$SPLIT"' \
      --eval-config '"$OUT"'/eval_config.yaml --num-episodes '"$N"' --seed '"$SEED"' --host 127.0.0.1 --port '"$PORT"' \
      --sim-mode '"$SIM_MODE"' --headless --eval-dir '"$OUT"'/eval --success-criteria 0.9 --save-video 2>&1 | tee '"$OUT"'/sim.log
    rc=$?; chown -R '"$(id -u):$(id -g)"' '"$OUT"' /workspace/simple/data /root 2>/dev/null; exit $rc' || echo "simulador terminou com erro (ver sim.log)"

# ---- metrics.json a partir de eval_stats.txt ----
python3 - "$OUT" <<'PY'
import json, re, sys
out = sys.argv[1]
res = {}
for l in open(f"{out}/eval/eval_stats.txt"):
    m = re.match(r"(\S+__episode_\d+): (True|False)", l)
    if m: res[m.group(1)] = m.group(2) == "True"
json.dump({"n": len(res), "successes": sum(res.values()), "success_rate": (sum(res.values()) / len(res)) if res else None, "episodes": res},
          open(f"{out}/metrics.json", "w"), indent=2)
print(open(f"{out}/metrics.json").read())
PY
echo EVAL_DONE "$OUT"

# ---- W&B (api.wandb.ai, projeto wla-simple-eval); precisa de WANDB_API_KEY no ambiente ----
if [[ ${WANDB_SYNC:-1} == 1 ]]; then
  (unset WANDB_BASE_URL; "$WLA_REPO/.venv/bin/python" scripts/wla/wandb_sync.py simple "$OUT" --project wla-simple-eval) || echo "wandb sync falhou (não fatal)"
fi
