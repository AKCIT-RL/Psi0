# Psi0 Setup

This folder contains the current setup notes for the Psi0 environment and the canonical links to the training and deployment scripts.

## Environment

Psi0 uses the shared repo root environment with the Psi, serve, and viz groups:

```bash
uv venv .venv-psi --python 3.10
source .venv-psi/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync \
  --group serve \
  --group viz \
  --group psi \
  --index-strategy unsafe-best-match \
  --active
```

If you want SIMPLE evaluation in the same workspace, use the fuller path described in [examples/quick_start/psi.md](../../examples/quick_start/psi.md).

## Training

The canonical fine-tuning launcher and dataset preparation notes live in [scripts/train/psi0/README.md](../../scripts/train/psi0/README.md).

The most common entrypoint is:

```bash
bash scripts/train/psi0/finetune-lerobot-psi0.sh <dataset_dir> [exp_name]
```

## Serving

The current deployment wrapper is [scripts/deploy/serve_psi0_simple.sh](../../scripts/deploy/serve_psi0_simple.sh).

It supports the following commands:

```bash
bash scripts/deploy/serve_psi0_simple.sh <run_dir> <ckpt_step> [port] [run|start|stop|status|logs]
```

Use `run` for a foreground server or `start` to background it and keep a pid/log file under `logs/psi0/`.
