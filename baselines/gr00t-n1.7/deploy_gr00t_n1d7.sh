#!/usr/bin/env bash
set -euo pipefail

# Serve a GR00T N1.7 checkpoint as a policy server.
# AutoModel.from_pretrained detects model_type from config.json automatically.
#
# Usage:
#   ./deploy_gr00t_n1d7.sh --model-path /path/to/checkpoint --embodiment-tag UNITREE_G1_N1D7
#   ./deploy_gr00t_n1d7.sh --model-path nvidia/GR00T-N1.7-3B --embodiment-tag UNITREE_G1_N1D7

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${VIRTUAL_ENV:-}/bin/python"
if [[ -z "${VIRTUAL_ENV:-}" || ! -x "$PYTHON_BIN" ]]; then
  if [[ -x ".venv-gr00t/bin/python" ]]; then
    PYTHON_BIN=".venv-gr00t/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

PYTHONPATH=src:src/gr00t \
  "$PYTHON_BIN" \
  src/gr00t/gr00t/eval/run_gr00t_server.py \
  "$@"
