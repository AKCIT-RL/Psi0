#!/usr/bin/env bash
# [PLANO_3_FASES Fase 1] Cadeia approach -> grasp -> carry (3 modelos em UM servidor serve_chain.py, troca de GPU sob demanda). MAX_STEPS(1200) = --max-episode-steps; NUM_SAMPLES="5,5,1" (por fase); portas PORT_A/PORT_G/PORT_C.
# Original (chain8): Eval closed-loop da CADEIA approach -> grasp (GR00T N1.7) no SIMPLE: dois servidores no host + simulador em Docker (render Isaac).
# A troca approach->grasp é feita pelo agente (gr00t_chain_decoupled_wbc.py) por distância à caixa-alvo (oracle do simulador).
# EXEC_H=<n> trunca cada chunk aos primeiros n passos (replaneja mais vezes). Variáveis: RUN  SPLIT(val)  N(1)  SEED(0)  CHAIN_MODE(chain|approach_only)  CHAIN_X_MIN/CHAIN_X_MAX/CHAIN_Y_MAX/CHAIN_K  PORT_A(5556) PORT_G(5557)
set -euo pipefail
cd "$(dirname "$0")/../.."
PSI0=$PWD; SIMPLE=$PSI0/third_party/SIMPLE
WLA_HOME=${WLA_HOME:-$HOME/wla_data}; SPLIT=${SPLIT:-val}; N=${N:-1}; SEED=${SEED:-0}; PORT_A=${PORT_A:-5556}; PORT_G=${PORT_G:-5557}; PORT_C=${PORT_C:-5558}; MAX_STEPS=${MAX_STEPS:-1200}; NUM_SAMPLES=${NUM_SAMPLES:-5,5,1}
SIM_MODE=${SIM_MODE:-mujoco_isaac}; RUN=${RUN:-chain_$(date +%Y%m%d_%H%M%S)}; CHAIN_MODE=${CHAIN_MODE:-chain}
PKG=${PKG:-$HOME/gr00t_x/gr00t_subtask_pkg}; PKG3=${PKG3:-$HOME/gr00t_x/phase3/gr00t_phase3_pkg}; GENV=${GENV:-$HOME/gr00t_env/.venv}
IMAGE=${IMAGE:-ghcr.io/physical-superintelligence-lab/simple:260829}
EXP=$WLA_HOME/experiments/wla; OUT=$EXP/$RUN; mkdir -p "$OUT" "$WLA_HOME/cache/simple_home"
P_A="walk to the shelf and stop centered in front of the blue tote with the arms raised."; P_G="grasp the blue tote."; P_C="carry the blue tote to the table and put it down."
for f in "$PKG/weights/approach/model-00001-of-00002.safetensors" "$PKG/weights/grasp/model-00001-of-00002.safetensors" "$PKG3/weights/phase3/model-00001-of-00002.safetensors" "$GENV/bin/python" "$EXP/s0_eval_data/roots/$SPLIT/meta/info.json"; do
  [[ -e $f ]] || { echo "FALTA: $f"; exit 2; }
done
{ echo "max_steps: $MAX_STEPS num_samples: $NUM_SAMPLES head2: thr=${CHAIN_HEAD2_THR:-0.5} meanw=${CHAIN_HEAD2_MEANW:-15} k=${CHAIN_HEAD2_K:-5} min_grasp=${CHAIN_MIN_GRASP:-100}"; echo "trigger: ${CHAIN_TRIGGER:-oracle} cal_a=${CHAIN_CAL_A:-} cal_b=${CHAIN_CAL_B:-} req_stop=${CHAIN_REQ_STOP:-0}"; echo "mode: $CHAIN_MODE x_min=${CHAIN_X_MIN:-} x_max=${CHAIN_X_MAX:-} y_max=${CHAIN_Y_MAX:-} k=${CHAIN_K:-10}"; echo "psi0: $(git rev-parse HEAD)"
  echo "simple: $(git -C "$SIMPLE" rev-parse HEAD)"; echo "image: $IMAGE"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } > "$OUT/git_commit.txt"
env | grep -E "^(RUN|SPLIT|N|SEED|SIM_MODE|CHAIN_[A-Z_]+)=" > "$OUT/config.env" || true
cat > "$OUT/eval_config.yaml" <<EOT
datasets:
  - name: $SPLIT
    path: $EXP/s0_eval_data/roots/$SPLIT
    prompt: recorded
EOT

( source "$GENV/bin/activate"; export HF_HOME=$PKG/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH="$PKG3/code:$PKG3/code/gr00t:$PSI0/scripts/gr00t"
  exec python "$PSI0/scripts/gr00t/serve_chain.py" --paths "approach=$PKG/weights/approach,grasp=$PKG/weights/grasp,phase3=$PKG3/weights/phase3" \
       --ports "$PORT_A,$PORT_G,$PORT_C" --num-samples "$NUM_SAMPLES" --seed "${SERVER_SEED:--1}" --debug-dir "$OUT/server_debug" \
       --prompts-json "$(python -c 'import json,sys;print(json.dumps(dict(approach=sys.argv[1],grasp=sys.argv[2],phase3=sys.argv[3])))' "$P_A" "$P_G" "$P_C")" ) > "$OUT/server_chain.log" 2>&1 &
PID_S=$!
trap 'kill $PID_S 2>/dev/null || true' EXIT
for p in $PORT_A $PORT_G $PORT_C; do
  for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$p/health" >/dev/null && break; sleep 5; done
  curl -sf "http://127.0.0.1:$p/health" >/dev/null || { echo "servidor :$p não subiu"; tail -20 "$OUT"/server_chain.log; exit 1; }
done

docker run --rm --gpus all --network host --ipc host \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=Y \
  -e HOME=/root -e HF_HUB_OFFLINE=1 -e SIMPLE_EVAL_EPISODE_ID= -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  -e CHAIN_MODE="$CHAIN_MODE" -e CHAIN_PORT_GRASP="$PORT_G" -e CHAIN_PORT_CARRY="$PORT_C" -e CHAIN_HEAD2_JSON=/workspace/carry_head_lin.json -e CHAIN_HEAD2_MODE="${CHAIN_HEAD2_MODE:-head}" -e CHAIN_HEAD2_THR="${CHAIN_HEAD2_THR:-0.5}" -e CHAIN_HEAD2_MEANW="${CHAIN_HEAD2_MEANW:-15}" -e CHAIN_HEAD2_K="${CHAIN_HEAD2_K:-5}" -e CHAIN_MIN_GRASP="${CHAIN_MIN_GRASP:-100}" -v "$PSI0/scripts/gr00t/carry_head_lin.json:/workspace/carry_head_lin.json:ro" -e CHAIN_ROOTS="$EXP/s0_eval_data/roots" -e CHAIN_LOG="$OUT/chain/chain.csv" \
  -e CHAIN_TRIGGER="${CHAIN_TRIGGER:-oracle}" -e CHAIN_CAL_A="${CHAIN_CAL_A:-0}" -e CHAIN_CAL_B="${CHAIN_CAL_B:-0}" -e CHAIN_REQ_STOP="${CHAIN_REQ_STOP:-0}" -e CHAIN_STOP_V="${CHAIN_STOP_V:-0.1}" -e CHAIN_X_SMOOTH="${CHAIN_X_SMOOTH:-5}" \
  -e CHAIN_HEAD_JSON=/workspace/switch_head_lin.json -e CHAIN_HEAD_THR="${CHAIN_HEAD_THR:-0.7}" -e CHAIN_HEAD_MEANW="${CHAIN_HEAD_MEANW:-0}" -e CHAIN_MIN_AREA="${CHAIN_MIN_AREA:-0}" -e CHAIN_CX_DEV="${CHAIN_CX_DEV:-0}" -e CHAIN_ASSIST="${CHAIN_ASSIST:-0}" -e CHAIN_SEARCH="${CHAIN_SEARCH:-0}" -e CHAIN_SEARCH_VY="${CHAIN_SEARCH_VY:-0.3}" -v "$PSI0/scripts/gr00t/chain_assist.py:/workspace/simple/src/simple/baselines/chain_assist.py:ro" -e CHAIN_ASSIST_YMAX="${CHAIN_ASSIST_YMAX:-110}" -e CHAIN_ASSIST_VX="${CHAIN_ASSIST_VX:-0.2}" -v "$PSI0/scripts/gr00t/switch_head_lin.json:/workspace/switch_head_lin.json:ro" \
  -e CHAIN_X_MIN="${CHAIN_X_MIN:-0}" -e CHAIN_X_MAX="${CHAIN_X_MAX:-0}" -e CHAIN_Y_MAX="${CHAIN_Y_MAX:-0}" -e CHAIN_K="${CHAIN_K:-10}" \
  -v "$WLA_HOME/simple_data:/workspace/simple/data" -v "$WLA_HOME/cache/simple_home:/root" \
  -v "$SIMPLE/src:/workspace/simple/src" -v "$SIMPLE/third_party/gear_sonic:/workspace/simple/third_party/gear_sonic" \
  -v "$SIMPLE/third_party/decoupled_wbc:/workspace/simple/third_party/decoupled_wbc" \
  -v "$PSI0/scripts/gr00t/gr00t_chain4_decoupled_wbc.py:/workspace/simple/src/simple/baselines/gr00t_chain4_decoupled_wbc.py:ro" \
  -v "$EXP:$EXP" -w /workspace/simple "$IMAGE" bash -c '
    set -o pipefail
    /workspace/simple/.venv/bin/eval-decoupled-wbc simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0 gr00t_chain4_decoupled_wbc '"$SPLIT"' \
      --eval-config '"$OUT"'/eval_config.yaml --num-episodes '"$N"' --seed '"$SEED"' --max-episode-steps '"$MAX_STEPS"' --host 127.0.0.1 --port '"$PORT_A"' \
      --sim-mode '"$SIM_MODE"' --headless --eval-dir '"$OUT"'/eval --success-criteria 0.9 --save-video 2>&1 | tee '"$OUT"'/sim.log
    rc=$?; chown -R '"$(id -u):$(id -g)"' '"$OUT"' /workspace/simple/data /root 2>/dev/null; exit $rc' || echo "simulador terminou com erro (ver sim.log)"
echo EVAL_DONE "$OUT"
