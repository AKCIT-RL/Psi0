#!/usr/bin/env bash
# Serve a GR00T N1.7 checkpoint over HTTP for SIMPLE (POST /act, GET /health).
#
#   serve.sh --model gr00t_n1d7_finetune_output_carry_totes [--port 5555]
#
# This is src/gr00t/gr00t/deploy/gr00t_serve_simple.py, the server the SIMPLE gr00t_n16* agents
# talk to. eval.sh starts it on localhost; run it alone to evaluate from another machine.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() {
    cat >&2 <<'EOF'
usage: serve.sh --model NAME [options] [-- server args]

  --model NAME                model directory under /home/ovx/checkpoints (or an absolute path)
  --port N                    default 5555
  --host ADDR                 default 0.0.0.0
  --embodiment-tag TAG        default G1_LOCO_DOWNSTREAM (the tag the specialists trained with)
  --action-exec-horizon N     truncate each returned action chunk to N steps
EOF
}

MODEL=""; PORT=5555; HOST=0.0.0.0; TAG=G1_LOCO_DOWNSTREAM; EXEC_HORIZON=""; EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --model)               MODEL="$2"; shift 2 ;;
        --port)                PORT="$2"; shift 2 ;;
        --host)                HOST="$2"; shift 2 ;;
        --embodiment-tag)      TAG="$2"; shift 2 ;;
        --action-exec-horizon) EXEC_HORIZON="$2"; shift 2 ;;
        --)                    shift; EXTRA=("$@"); break ;;
        -h|--help)             usage; exit 0 ;;
        *)                     usage; die "unknown option: $1" ;;
    esac
done
[ -n "${MODEL}" ] || { usage; die "--model is required"; }

setup_caches
gr00t_env

MODEL_PATH="$(resolve_ckpt "${MODEL}")"
[ -f "${MODEL_PATH}/config.json" ] || die "${MODEL_PATH}/config.json missing -- not a model directory"
ensure_processor_at_root "${MODEL_PATH}"

args=(--model-path "${MODEL_PATH}" --embodiment-tag "${TAG}" --host "${HOST}" --port "${PORT}")
[ -z "${EXEC_HORIZON}" ] || args+=(--action-exec-horizon "${EXEC_HORIZON}")

log "image: $(build_info)"
log "serving ${MODEL_PATH} on ${HOST}:${PORT} (embodiment ${TAG})"
exec "${GR00T_PY}" -m gr00t.deploy.gr00t_serve_simple "${args[@]}" "${EXTRA[@]}"
