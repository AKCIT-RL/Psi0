#!/usr/bin/env bash
# Closed-loop evaluation in SIMPLE. Starts the policy server in the background, waits until it is
# healthy, runs eval-decoupled-wbc against it, and stops the server when the client exits. Both
# run in one container (the `eval` image), talking over localhost.
#
#   eval.sh --model gr00t_n1d7_finetune_output_carry_totes \
#           --env-id simple/<Task>-v0 \
#           --data-dir carry_totes_heldout \
#           --num-episodes 20
#
# The episodes of --data-dir (a LeRobot root under /home/ovx/data, or an absolute path) provide the
# starting states. --eval-config takes a YAML listing several instead; paths inside it are
# container paths (/home/ovx/data/...). Anything after `--` goes to eval-decoupled-wbc.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() {
    cat >&2 <<'EOF'
usage: eval.sh --model NAME --env-id ENV (--data-dir DIR | --eval-config YAML) [options] [-- client args]

  --model NAME                model directory under /home/ovx/checkpoints (or an absolute path)
  --env-id ENV                SIMPLE environment, e.g. simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0
  --data-dir DIR              LeRobot root with the starting episodes (under /home/ovx/data, or absolute)
  --eval-config YAML          several roots instead (under /home/ovx/data, or absolute); see SIMPLE docs
  --policy NAME               module under simple.baselines (default gr00t_n16_decoupled_wbc; the
                              plain gr00t_n16 is the agent for the non-WBC eval CLI)
  --num-episodes N            default 20
  --success-criteria X        default 0.9
  --sim-mode MODE             default mujoco_isaac (Isaac rendering: needs an RTX GPU); mujoco renders
                              with MuJoCo only -- use whatever renderer the training videos came from
  --max-episode-steps N       default: the task's own limit
  --seed N                    episode sampling seed for --eval-config (default 0)
  --name NAME                 output group under /home/ovx/evals (default: the model directory name)
  --no-video                  do not save episode videos
  --embodiment-tag TAG        default G1_LOCO_DOWNSTREAM
  --action-exec-horizon N     truncate each returned action chunk to N steps
  --server-timeout S          seconds to wait for the server to load (default 900)
EOF
}

# eval-decoupled-wbc imports simple.baselines.<policy> as given: gr00t_n16 would load the non-WBC
# agent, although SIMPLE's eval docs show it.
MODEL=""; ENV_ID=""; DATA_DIR=""; EVAL_CONFIG=""; POLICY=gr00t_n16_decoupled_wbc; NUM_EPISODES=20
SUCCESS=0.9; SIM_MODE=mujoco_isaac; MAX_EP_STEPS=""; SEED=0; NAME=""; VIDEO=1
TAG=G1_LOCO_DOWNSTREAM; EXEC_HORIZON=""; SERVER_TIMEOUT=900; EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --model)               MODEL="$2"; shift 2 ;;
        --env-id)              ENV_ID="$2"; shift 2 ;;
        --data-dir)            DATA_DIR="$2"; shift 2 ;;
        --eval-config)         EVAL_CONFIG="$2"; shift 2 ;;
        --policy)              POLICY="$2"; shift 2 ;;
        --num-episodes)        NUM_EPISODES="$2"; shift 2 ;;
        --success-criteria)    SUCCESS="$2"; shift 2 ;;
        --sim-mode)            SIM_MODE="$2"; shift 2 ;;
        --max-episode-steps)   MAX_EP_STEPS="$2"; shift 2 ;;
        --seed)                SEED="$2"; shift 2 ;;
        --name)                NAME="$2"; shift 2 ;;
        --no-video)            VIDEO=0; shift ;;
        --embodiment-tag)      TAG="$2"; shift 2 ;;
        --action-exec-horizon) EXEC_HORIZON="$2"; shift 2 ;;
        --server-timeout)      SERVER_TIMEOUT="$2"; shift 2 ;;
        --)                    shift; EXTRA=("$@"); break ;;
        -h|--help)             usage; exit 0 ;;
        *)                     usage; die "unknown option: $1" ;;
    esac
done
[ -n "${MODEL}" ]  || { usage; die "--model is required"; }
[ -n "${ENV_ID}" ] || { usage; die "--env-id is required"; }
if { [ -n "${DATA_DIR}" ] && [ -n "${EVAL_CONFIG}" ]; } || { [ -z "${DATA_DIR}" ] && [ -z "${EVAL_CONFIG}" ]; }; then
    usage; die "pass exactly one of --data-dir and --eval-config"
fi
[ -x "${SIMPLE_BIN}/eval-decoupled-wbc" ] || die "SIMPLE is not in this image -- evaluate with the eval target"

resolve_data() {
    case "$1" in
        /*) printf '%s' "$1" ;;
        *)  printf '%s' "${DATA_ROOT}/$1" ;;
    esac
}

setup_caches

# SIMPLE resolves its assets under its own data/ directory, which is part of the mounted checkout:
# whatever it downloads lands in the working copy and is reused by later runs.
[ -w "${SIMPLE_ROOT}/data" ] \
    || log "WARNING: ${SIMPLE_ROOT}/data is read-only; SIMPLE cannot download missing assets"

MODEL_PATH="$(resolve_ckpt "${MODEL}")"
[ -f "${MODEL_PATH}/config.json" ] || die "${MODEL_PATH}/config.json missing -- not a model directory"
NAME="${NAME:-$(basename "${MODEL_PATH}")}"
OUT="${EVAL_ROOT}/${NAME}/$(date +%Y%m%d-%H%M%S)${SLURM_JOB_ID:+-${SLURM_JOB_ID}}"
mkdir -p "${OUT}"

# What produced this evaluation. Written before the server starts, so a run that dies still says
# what it was. SIMPLE's own eval_stats.txt holds the result; this holds the origin.
write_provenance "${OUT}/ovx_eval.jsonl" \
    "model=${MODEL_PATH}" "env_id=${ENV_ID}" "policy=${POLICY}" "sim_mode=${SIM_MODE}" \
    "num_episodes=${NUM_EPISODES}" "success_criteria=${SUCCESS}" "seed=${SEED}" \
    "data_dir=${DATA_DIR}" "eval_config=${EVAL_CONFIG}" "embodiment_tag=${TAG}" \
    "max_episode_steps=${MAX_EP_STEPS}" "action_exec_horizon=${EXEC_HORIZON}" \
    "server_timeout=${SERVER_TIMEOUT}" "save_video=${VIDEO}" "extra_args=${EXTRA[*]:-}" \
    "slurm_job_id=${SLURM_JOB_ID:-}" \
    "psi0_code=$(repo_state "${PSI0_ROOT}")" "simple_code=$(repo_state "${SIMPLE_ROOT}")" \
    "simple_submodules=$(submodule_states "${SIMPLE_ROOT}")"

PORT=$(free_port "$(port_base 21000)")
serve_args=(--model "${MODEL_PATH}" --host 127.0.0.1 --port "${PORT}" --embodiment-tag "${TAG}")
[ -z "${EXEC_HORIZON}" ] || serve_args+=(--action-exec-horizon "${EXEC_HORIZON}")

log "image:  $(build_info)"
log "model:  ${MODEL_PATH}"
log "output: ${OUT}  (server log: server.log)"
"${PSI0_ROOT}/ovx/bin/serve.sh" "${serve_args[@]}" > "${OUT}/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true' EXIT

# Loading the checkpoint and the Cosmos backbone takes minutes. A server that dies meanwhile must
# fail the job now, not after the client gives up.
deadline=$(( SECONDS + SERVER_TIMEOUT ))
until curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        tail -n 40 "${OUT}/server.log" >&2
        die "policy server exited before becoming healthy"
    fi
    if [ "${SECONDS}" -ge "${deadline}" ]; then
        tail -n 40 "${OUT}/server.log" >&2
        die "policy server not healthy after ${SERVER_TIMEOUT}s"
    fi
    sleep 5
done
log "policy server healthy on 127.0.0.1:${PORT}"

simple_env
cd "${OUT}"

client=("${SIMPLE_BIN}/eval-decoupled-wbc" "${ENV_ID}" "${POLICY}" train
        --host 127.0.0.1 --port "${PORT}"
        --data-format lerobot
        --eval-dir "${OUT}"
        --num-episodes "${NUM_EPISODES}"
        --success-criteria "${SUCCESS}"
        --sim-mode "${SIM_MODE}"
        --headless)
if [ -n "${EVAL_CONFIG}" ]; then
    client+=(--eval-config "$(resolve_data "${EVAL_CONFIG}")" --seed "${SEED}")
else
    client+=(--data-dir "$(resolve_data "${DATA_DIR}")")
fi
[ -z "${MAX_EP_STEPS}" ] || client+=(--max-episode-steps "${MAX_EP_STEPS}")
[ "${VIDEO}" = 1 ] || client+=(--no-save-video)

log "client: ${client[*]} ${EXTRA[*]:-}"
"${client[@]}" "${EXTRA[@]}"
