#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 <run_dir> <ckpt_step> [port] [command]"
    echo "  <run_dir>     : PSI run directory (required)"
    echo "  <ckpt_step>   : Checkpoint step, e.g. latest or 40000 (required)"
    echo "  [port]        : Port to serve on (default: 22085)"
    echo "  [command]     : run|start|stop|status|logs (default: run)"
    exit 1
}

if [ "$#" -lt 2 ]; then
    usage
fi

source .venv-psi/bin/activate

RUN_DIR="$1"
CKPT_STEP="$2"
PORT="${3:-22085}"
COMMAND="${4:-run}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CUDA_EXTRA="${PSI_CUDA_EXTRA:-cuda13}"

safe_run_dir="$(basename "$RUN_DIR" | tr '/' '_')"
state_dir="logs/psi0"
mkdir -p "$state_dir"
pid_file="$state_dir/${safe_run_dir}_${PORT}.pid"
log_file="$state_dir/${safe_run_dir}_${PORT}.log"

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

run_cmd() {
    uv run --active --extra "$CUDA_EXTRA" --no-default-groups --group psi --group serve serve_psi0 \
        --host 0.0.0.0 \
        --port "$PORT" \
        --policy=psi0 \
        --run-dir="$RUN_DIR" \
        --ckpt-step="$CKPT_STEP" \
        --action-exec-horizon=24 \
        --rtc
}

start_server() {
    if is_running; then
        echo "Server already running (pid $(cat "$pid_file"))."
        echo "Log: $log_file"
        exit 0
    fi

    echo "Starting PSI0 server on GPU $CUDA_VISIBLE_DEVICES (port $PORT)"
    echo "run_dir: $RUN_DIR"
    echo "ckpt_step: $CKPT_STEP"
    echo "uv extra: $CUDA_EXTRA"

    nohup bash -lc "source .venv-psi/bin/activate && CUDA_VISIBLE_DEVICES=\"$CUDA_VISIBLE_DEVICES\" PSI_CUDA_EXTRA=\"$CUDA_EXTRA\" $(printf '%q ' "$0" "$RUN_DIR" "$CKPT_STEP" "$PORT" run)" >"$log_file" 2>&1 &

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

case "$COMMAND" in
run)
    echo "Serving PSI0 on GPU $CUDA_VISIBLE_DEVICES"
    echo "run_dir: $RUN_DIR"
    echo "ckpt_step: $CKPT_STEP"
    echo "uv extra: $CUDA_EXTRA"
    run_cmd
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
