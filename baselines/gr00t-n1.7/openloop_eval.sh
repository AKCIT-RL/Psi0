#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PYTHONPATH=src:src/gr00t \
  src/gr00t/.venv/bin/python \
  baselines/gr00t-n1.6/openloop_eval.py "$@"
