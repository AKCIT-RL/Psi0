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

### LeRobot Fine-Tuning

The LeRobot fine-tuning wrapper is:

```bash
bash scripts/train/psi0/finetune-lerobot-psi0.sh <dataset_dir> [exp_name]
```

`<dataset_dir>` can be an absolute path or a repo-relative dataset directory. `exp_name` is optional and defaults to a lowercased, hyphenated version of the dataset name.

The script arguments and environment variables are documented in [scripts/train/psi0/README.md](../../scripts/train/psi0/README.md).

### SIMPLE Fine-Tuning

For SIMPLE simulation tasks, use:

```bash
bash scripts/train/psi0/finetune-simple-psi0.sh <repo> <task> [exp]
```

That wrapper takes these positional arguments:

- `<repo>`: the data root under `PSI_HOME`, for example `data` or `data/simple`.
- `<task>`: the SIMPLE task name.
- `[exp]`: optional experiment name; defaults to a lowercased name derived from `<task>`.

The wrapper reads these environment variables:

- `CUDA_VISIBLE_DEVICES`: GPU list used to size DDP; defaults to `0,1,2,3,4,5,6,7`.
- `OMP_NUM_THREADS`: host-side thread count; defaults to `4`.
- `TRAIN_BATCH_SIZE`: batch size passed to the training config; defaults to `16`.
- `MAX_TRAINING_STEPS`: maximum training steps; defaults to `40000`.
- `WARMUP_STEPS`: warmup steps; defaults to `1000`.
- `CHECKPOINTING_STEPS`: checkpoint cadence; defaults to `10000`.
- `VALIDATION_STEPS`: validation cadence; defaults to `500`.
- `PSI_HOME`: root directory used to resolve checkpoints and dataset paths.
- `VLM_CKPT_PATH`: override for the VLM backbone checkpoint.
- `ACTION_CKPT_PATH`: override for the action-header checkpoint.

The script auto-downloads missing checkpoints from `USC-PSI-Lab/psi-model` when possible and uses `--data.root_dir=$PSI_HOME/$repo/$task` for the dataset path.

## Serving

The current deployment wrapper is [scripts/deploy/serve_psi0_simple.sh](../../scripts/deploy/serve_psi0_simple.sh).

It supports the following commands:

```bash
bash scripts/deploy/serve_psi0_simple.sh <run_dir> <ckpt_step> [port] [run|start|stop|status|logs]
```

Use `run` for a foreground server or `start` to background it and keep a pid/log file under `logs/psi0/`.
