#!/usr/bin/env bash
set -euo pipefail

# Serve a GR00T N1.7 checkpoint as a policy server.
# AutoModel.from_pretrained detects model_type from config.json automatically.
#
# Usage:
#   ./deploy_gr00t_n1d7.sh --model-path /path/to/checkpoint --embodiment-tag UNITREE_G1_N1D7
#   ./deploy_gr00t_n1d7.sh --model-path nvidia/GR00T-N1.7-3B --embodiment-tag UNITREE_G1_N1D7

cd "$(git rev-parse --show-toplevel)"

PYTHONPATH=src:src/gr00t \
  src/gr00t/.venv/bin/python \
  src/gr00t/gr00t/eval/run_gr00t_server.py \
  "$@"
