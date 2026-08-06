#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PYTHONPATH=src:src/gr00t \
  src/gr00t/.venv/bin/python \
  baselines/gr00t-n1.6/openloop_eval.py "$@"
# NOTE: eval_simple uses the N1.6 launcher (model-agnostic via registry).
# For SimplerEnv eval run:
#   python baselines/gr00t-n1.6/eval_simple.py \
#     --preset baselines/gr00t-n1.7/presets/eval/simple_local.yaml \
#     [--model-path /path/to/checkpoint] [--num-episodes 20] [--dry-run]
