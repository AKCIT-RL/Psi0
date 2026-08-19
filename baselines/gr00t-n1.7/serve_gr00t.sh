#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 <embodiment_tag> <model_path> [port] [command]"
    echo "  <embodiment_tag> : Embodiment tag (required, e.g. G1_LOCO_DOWNSTREAM)"
    echo "  <model_path>     : Model path (required)"
    echo "  [port]           : Port (default: 22085)"
    echo "  [command]        : run|start|stop|status|logs (default: run)"
    exit 1
}

if [ "$#" -lt 2 ]; then
    usage
fi

embodiment_tag="$1"
model_path="$2"
port="${3:-22085}"
command="${4:-run}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

run_dir="logs/gr00t"
mkdir -p "$run_dir"
safe_tag="${embodiment_tag//\//_}"
pid_file="$run_dir/${safe_tag}_${port}.pid"
log_file="$run_dir/${safe_tag}_${port}.log"

is_running() {
    if [ -f "$pid_file" ]; then
        local pid
        pid="$(cat "$pid_file")"
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

start_server() {
    if is_running; then
        echo "Server already running (pid $(cat "$pid_file"))."
        echo "Log: $log_file"
        exit 0
    fi

    echo "Starting GR00T on GPU $CUDA_VISIBLE_DEVICES (port $port)"
    echo "with model: $model_path"
    echo "with embodiment: $embodiment_tag"
    nohup python -m gr00t.deploy.gr00t_serve_simple \
        --host 0.0.0.0 \
        --port "$port" \
        --device "cuda:$CUDA_VISIBLE_DEVICES" \
        --use-sim-policy-wrapper \
        --strict \
        --model-path "$model_path" \
        --embodiment-tag "$embodiment_tag" \
        >"$log_file" 2>&1 &

    echo $! >"$pid_file"
    echo "Started with pid $(cat "$pid_file")"
    echo "Log: $log_file"
}

stop_server() {
    if ! is_running; then
        echo "Server is not running."
        rm -f "$pid_file"
        exit 0
    fi

    local pid
    pid="$(cat "$pid_file")"
    echo "Stopping server pid $pid"
    kill -TERM "$pid" || true
    rm -f "$pid_file"
}

status_server() {
    if is_running; then
        echo "Server is running (pid $(cat "$pid_file"))."
    else
        echo "Server is not running."
    fi
    echo "Log: $log_file"
}

show_logs() {
    if [ ! -f "$log_file" ]; then
        echo "No log file yet: $log_file"
        exit 1
    fi
    tail -f "$log_file"
}

case "$command" in
run)
    echo "Serving GR00T on GPU $CUDA_VISIBLE_DEVICES"
    echo "with model: $model_path"
    echo "with embodiment: $embodiment_tag"
    python -m gr00t.deploy.gr00t_serve_simple \
        --host 0.0.0.0 \
        --port "$port" \
        --device "cuda:$CUDA_VISIBLE_DEVICES" \
        --use-sim-policy-wrapper \
        --strict \
        --model-path "$model_path" \
        --embodiment-tag "$embodiment_tag"
    ;;
start)
    start_server
    ;;
stop)
    stop_server
    ;;
status)
    status_server
    ;;
logs)
    show_logs
    ;;
*)
    usage
    ;;
esac
