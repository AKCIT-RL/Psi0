#!/usr/bin/env bash

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
SIMPLE=${SIMPLE_DIR:-$ROOT/third_party/SIMPLE}
BUNDLE_DIR=${BUNDLE_DIR:-$ROOT/eval/bundles/simple-eval-wmo-totes-20260807}
PSI0_RUN_DIR=${PSI0_RUN_DIR:-$BUNDLE_DIR/psi0/run}
HF_CACHE_DIR=${HF_CACHE_DIR:-$BUNDLE_DIR/hf_cache}
DATASET_RELATIVE_PATH=${DATASET_RELATIVE_PATH:-evals/wmo-totes-source/raw/level-0}
DATASET=$SIMPLE/data/$DATASET_RELATIVE_PATH
CONTAINER_DATASET=/workspace/SIMPLE/data/$DATASET_RELATIVE_PATH
RESULT_BASE=$ROOT/eval/results/wmo-totes-20260807
RUN_ID=${RUN_ID:-$(date -u '+%Y%m%dT%H%M%SZ')}
RESULT_DIR=$RESULT_BASE/$RUN_ID
LOGS=$RESULT_DIR/logs
TASK=simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0
POLICY=psi0_decoupled_wbc
SIMPLE_COMMIT=4502e56fbd13501af48ec528dd79957054325472
DATASET_COMMIT=d51d88f6af25f93734502d69fdce90d3bc903af3
CHECKPOINT=${CHECKPOINT:-220801}
PORT=${PORT:-22085}
EPISODES=${EPISODES:-10}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-2400}
SMOKE_ONLY=${SMOKE_ONLY:-0}
IMAGE=${IMAGE:-simple-teleoperation:251025-wmo-eval}
LDP=/isaac-sim/extscache/omni.sensors.nv.camera-0.20.1-coreapi+lx64.r/bin
ISAAC_BACKGROUND_USD=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Simple_Warehouse/warehouse.usd

mkdir -p "$LOGS"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/orchestrator.log"; }
fail() { log "ERRO: $*"; exit 1; }

port_up() { ss -ltn "sport = :$PORT" | grep -q "$PORT"; }
wait_server() {
    for _ in $(seq 1 180); do
        curl -fsS "http://127.0.0.1:$PORT/health" 2>/dev/null \
            | grep -q '"status":"ok"' && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || return 1
        sleep 10
    done
    return 1
}

stop_server() {
    if [[ -n ${SERVER_PID:-} ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=
}

stop_container() {
    if [[ -n ${ACTIVE_PROJECT:-} ]]; then
        docker ps -q --filter "label=com.docker.compose.project=$ACTIVE_PROJECT" \
            | xargs -r docker stop >/dev/null 2>&1 || true
    fi
    ACTIVE_PROJECT=
}

cleanup() {
    stop_container
    stop_server
}
trap cleanup EXIT INT TERM

count_episodes() {
    local stats=$1
    [[ -f $stats ]] || { echo 0; return; }
    awk '/^run:/{count=0} /^episode_/{count++} END{print count+0}' "$stats"
}

start_server() {
    (
        cd "$ROOT" || exit 1
        export HF_HOME=$HF_CACHE_DIR
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
        source .venv-psi/bin/activate
        exec uv run --active --group psi --group serve serve_psi0 \
            --host=0.0.0.0 --port="$PORT" --policy=psi0 \
            --run-dir="$PSI0_RUN_DIR" --ckpt-step="$CHECKPOINT" \
            --action-exec-horizon=24 --rtc
    ) >> "$LOGS/policy-server.log" 2>&1 &
    SERVER_PID=$!
}

run_eval() {
    local phase=$1 episodes=$2
    local project="wmo-${RUN_ID,,}-$phase"
    project=${project//[^a-z0-9_-]/-}
    ACTIVE_PROJECT=$project
    (
        cd "$SIMPLE" || exit 1
        GPUs=0 docker compose \
            -f docker-compose.yml -f "$ROOT/eval/docker-compose.wmo.yml" \
            -p "$project" run --rm --no-deps -T \
            -v "$ROOT/eval:/workspace/eval" \
            -v "$ROOT/eval/overlays/wmo_totes/lerobot.py:/workspace/SIMPLE/src/simple/datasets/lerobot.py:ro" \
            -e LD_LIBRARY_PATH="/usr/local/cuda-12.8/targets/x86_64-linux/lib:/isaac-sim/extscache/omni.kit.streamsdk.plugins-6.1.7+106.2.0.lx64.r/bin:$LDP" \
            -e SIMPLE_ISAAC_BACKGROUND_USD="$ISAAC_BACKGROUND_USD" \
            --entrypoint "uv run --no-sync eval-decoupled-wbc" eval \
            "$TASK" "$POLICY" level-0 \
            --host=localhost --port="$PORT" \
            --sim-mode=mujoco_isaac --headless \
            --max-episode-steps="$MAX_EPISODE_STEPS" \
            --data-format=lerobot \
            --data-dir="$CONTAINER_DATASET" \
            --eval-dir="/workspace/eval/results/wmo-totes-20260807/$RUN_ID/$phase" \
            --num-episodes="$episodes" --episode-start=0 --num-workers=1
    ) >> "$LOGS/eval-$phase.log" 2>&1
    local status=$?
    ACTIVE_PROJECT=
    return $status
}

write_metadata() {
    local image_id main_commit
    image_id=$(docker image inspect "$IMAGE" --format '{{.Id}}')
    main_commit=$(git -C "$ROOT" rev-parse HEAD)
    cat > "$RESULT_DIR/metadata.json" <<EOF
{
  "evaluation_scope": "train-set closed-loop evaluation; not holdout/generalization",
  "task": "$TASK",
  "policy": "$POLICY",
  "checkpoint": "ckpt_$CHECKPOINT",
  "action_exec_horizon": 24,
  "rtc": true,
  "episodes": $EPISODES,
    "max_episode_steps": $MAX_EPISODE_STEPS,
  "dataset": "lGabrielJJ/G1WholebodyLocomotionPickTotesShelfToTableTeleop/raw/level-0",
  "dataset_commit": "$DATASET_COMMIT",
  "main_repository_commit": "$main_commit",
  "simple_commit": "$SIMPLE_COMMIT",
    "docker_image": "$IMAGE",
  "docker_image_id": "$image_id",
    "isaac_background_usd": "$ISAAC_BACKGROUND_USD",
  "policy_port": $PORT,
  "result_directory": "$RESULT_DIR"
}
EOF
}

summarize() {
    local stats=$RESULT_DIR/main/eval_stats.txt
    local episodes successes rate
    episodes=$(count_episodes "$stats")
    successes=$(awk '/^run:/{count=0} /^episode_/{if ($2=="True") count++} END{print count+0}' "$stats")
    rate=$(awk -v successes="$successes" -v episodes="$episodes" 'BEGIN { if (episodes) printf "%.2f%%", 100*successes/episodes; else print "n/a" }')
    cat > "$RESULT_DIR/SUMMARY.md" <<EOF
# WMO totes train-set evaluation

- Task: $TASK
- Checkpoint: ckpt_$CHECKPOINT
- SIMPLE: $SIMPLE_COMMIT
- Dataset: lGabrielJJ/G1WholebodyLocomotionPickTotesShelfToTableTeleop raw, commit $DATASET_COMMIT
- Scope: train-set closed-loop evaluation; not holdout/generalization
- Episodes: $episodes
- Max episode steps: $MAX_EPISODE_STEPS
- Successes: $successes
- Success rate: $rate
- Logs: $LOGS
- Videos: $RESULT_DIR/main/$POLICY/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0/level-0
EOF
}

log "Iniciando WMO totes run_id=$RUN_ID"
[[ $(git -C "$SIMPLE" rev-parse HEAD) == "$SIMPLE_COMMIT" ]] \
    || fail "commit SIMPLE diferente do esperado"
[[ -s $PSI0_RUN_DIR/checkpoints/ckpt_$CHECKPOINT/model.safetensors ]] \
    || fail "checkpoint ausente"
[[ -d $HF_CACHE_DIR ]] || fail "cache Hugging Face ausente: $HF_CACHE_DIR"
[[ $(find "$DATASET/data" -name 'episode_*.parquet' | wc -l) -eq 50 ]] \
    || fail "dataset sem 50 parquets"
[[ $(find "$DATASET/videos" -name 'episode_*.mp4' | wc -l) -eq 50 ]] \
    || fail "dataset sem 50 videos"
docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || fail "imagem $IMAGE ausente"
docker info >/dev/null 2>&1 || fail "Docker indisponivel"
nvidia-smi -L | grep -q 'GPU 0' || fail "GPU 0 indisponivel"
port_up && fail "porta $PORT ja esta em uso"
write_metadata

start_server
wait_server || fail "servidor Psi0 nao ficou pronto; veja $LOGS/policy-server.log"
log "Servidor Psi0 pronto na porta $PORT"

if [[ $(count_episodes "$RESULT_DIR/smoke/eval_stats.txt") -lt 1 ]]; then
    log "Iniciando smoke de um episodio"
    run_eval smoke 1 || fail "smoke falhou; veja $LOGS/eval-smoke.log"
fi
[[ $(count_episodes "$RESULT_DIR/smoke/eval_stats.txt") -ge 1 ]] \
    || fail "smoke terminou sem estatistica de episodio"
find "$RESULT_DIR/smoke" -path '*/episode_0/*.mp4' -size +0c | grep -q . \
    || fail "smoke terminou sem video"
if [[ $SMOKE_ONLY == 1 ]]; then
    log "Smoke concluido; aguardando validacao visual antes do eval principal"
    exit 0
fi
log "Smoke validado; iniciando $EPISODES episodios"

if [[ $(count_episodes "$RESULT_DIR/main/eval_stats.txt") -lt $EPISODES ]]; then
    run_eval main "$EPISODES" || fail "eval principal falhou; veja $LOGS/eval-main.log"
fi
[[ $(count_episodes "$RESULT_DIR/main/eval_stats.txt") -ge $EPISODES ]] \
    || fail "eval principal terminou sem $EPISODES resultados"
summarize
log "Avaliacao concluida; resumo em $RESULT_DIR/SUMMARY.md"