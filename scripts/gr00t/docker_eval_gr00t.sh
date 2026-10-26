#!/usr/bin/env bash
# Eval closed-loop de um sub-task GR00T N1.7 (approach|grasp) no SIMPLE: servidor no host (venv do GR00T) + simulador em Docker (render Isaac).
# Variáveis: TASK=approach|grasp (obrigatória)  RUN  SPLIT(val)  N(1)  SEED(0)  SIM_MODE(mujoco_isaac)  PORT(5556)  PKG(~/gr00t_x/gr00t_subtask_pkg)
#            GENV(~/gr00t_env/.venv)  WLA_HOME(~/wla_data; reaproveita roots/ e simple_data/)  IMAGE
set -euo pipefail
cd "$(dirname "$0")/../.."
PSI0=$PWD; SIMPLE=$PSI0/third_party/SIMPLE
TASK=${TASK:?defina TASK=approach|grasp}; WLA_HOME=${WLA_HOME:-$HOME/wla_data}; SPLIT=${SPLIT:-val}; N=${N:-1}; SEED=${SEED:-0}
PORT=${PORT:-5556}; SIM_MODE=${SIM_MODE:-mujoco_isaac}; RUN=${RUN:-gr00t_${TASK}_$(date +%Y%m%d_%H%M%S)}
PKG=${PKG:-$HOME/gr00t_x/gr00t_subtask_pkg}; GENV=${GENV:-$HOME/gr00t_env/.venv}
IMAGE=${IMAGE:-ghcr.io/physical-superintelligence-lab/simple:260829}
case $TASK in
  approach) PROMPT="walk to the shelf and stop centered in front of the blue tote with the arms raised." ;;
  grasp)    PROMPT="grasp the blue tote." ;;
  *) echo "TASK inválida"; exit 2 ;;
esac
EXP=$WLA_HOME/experiments/wla; OUT=$EXP/$RUN; mkdir -p "$OUT" "$WLA_HOME/cache/simple_home"
for f in "$PKG/weights/$TASK/model-00001-of-00002.safetensors" "$GENV/bin/python" "$EXP/s0_eval_data/roots/$SPLIT/meta/info.json" "$SIMPLE/third_party/gear_sonic"; do
  [[ -e $f ]] || { echo "FALTA: $f"; exit 2; }
done
{ echo "task: $TASK"; echo "prompt: $PROMPT"; echo "psi0: $(git rev-parse HEAD)"; echo "simple: $(git -C "$SIMPLE" rev-parse HEAD)"
  echo "image: $IMAGE"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } > "$OUT/git_commit.txt"
env | grep -E "^(TASK|RUN|SPLIT|N|SEED|SIM_MODE|PORT)=" > "$OUT/config.env" || true
cat > "$OUT/eval_config.yaml" <<EOT
datasets:
  - name: $SPLIT
    path: $EXP/s0_eval_data/roots/$SPLIT
    prompt: recorded
EOT

( source "$GENV/bin/activate"; export HF_HOME=$PKG/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH="$PKG/code:$PKG/code/gr00t"
  exec python "$PSI0/scripts/gr00t/serve_subtask.py" --embodiment-tag UNITREE_G1_N1D7 --model-path "$PKG/weights/$TASK" \
       --device cuda:0 --port "$PORT" --host 127.0.0.1 --force-prompt "$PROMPT" --seed "${SERVER_SEED:--1}" --debug-dir "$OUT/server_debug" ) > "$OUT/server.log" 2>&1 &
SRV_PID=$!; trap 'kill $SRV_PID 2>/dev/null || true' EXIT
for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; kill -0 $SRV_PID 2>/dev/null || break; sleep 5; done
curl -sf "http://127.0.0.1:$PORT/health" || { echo "servidor não subiu"; tail -30 "$OUT/server.log"; exit 1; }

docker run --rm --gpus all --network host --ipc host \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=Y \
  -e HOME=/root -e HF_HUB_OFFLINE=1 -e SIMPLE_EVAL_EPISODE_ID= -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  -v "$WLA_HOME/simple_data:/workspace/simple/data" -v "$WLA_HOME/cache/simple_home:/root" \
  -v "$SIMPLE/src:/workspace/simple/src" -v "$SIMPLE/third_party/gear_sonic:/workspace/simple/third_party/gear_sonic" \
  -v "$SIMPLE/third_party/decoupled_wbc:/workspace/simple/third_party/decoupled_wbc" \
  -v "$EXP:$EXP" -w /workspace/simple "$IMAGE" bash -c '
    set -o pipefail
    /workspace/simple/.venv/bin/eval-decoupled-wbc simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0 gr00t_n16_decoupled_wbc '"$SPLIT"' \
      --eval-config '"$OUT"'/eval_config.yaml --num-episodes '"$N"' --seed '"$SEED"' --host 127.0.0.1 --port '"$PORT"' \
      --sim-mode '"$SIM_MODE"' --headless --eval-dir '"$OUT"'/eval --success-criteria 0.9 --save-video 2>&1 | tee '"$OUT"'/sim.log
    rc=$?; chown -R '"$(id -u):$(id -g)"' '"$OUT"' /workspace/simple/data /root 2>/dev/null; exit $rc' || echo "simulador terminou com erro (ver sim.log)"
echo EVAL_DONE "$OUT"
