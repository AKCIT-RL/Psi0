# PSI-0 Training Scripts

Scripts for training and fine-tuning the $\Psi_0$ model.

---

## `finetune-lerobot-psi0.sh` — Automated Fine-tuning on LeRobot Datasets

Generic script for fine-tuning PSI-0 on any dataset in **LeRobot v2.1 PSI real** format (data captured via teleoperation on the G1 robot).

### Usage

```bash
# Basic — point to the dataset directory
./scripts/train/psi0/finetune-lerobot-psi0.sh <dataset_dir> [exp_name]

# Example with a locally captured pick_cylinder dataset
./scripts/train/psi0/finetune-lerobot-psi0.sh pick_cylinder_manipulation_psi

# With custom experiment name
./scripts/train/psi0/finetune-lerobot-psi0.sh pick_cylinder_manipulation_psi pick-cylinder-v1

# Dataset at an absolute path
./scripts/train/psi0/finetune-lerobot-psi0.sh /path/to/data/real/Pick_bottle_and_turn_and_pour_into_cup

# Customize GPUs and number of epochs
CUDA_VISIBLE_DEVICES=0 TARGET_EPOCHS=100 ./scripts/train/psi0/finetune-lerobot-psi0.sh pick_cylinder_manipulation_psi
```

### What the script does

The script runs 5 phases automatically:

**[1/5] Dataset validation**
- Checks that the directory contains `data/`, `videos/`, `meta/info.json`, `meta/stats_psi0.json`, `meta/episodes.jsonl`, and `meta/tasks.jsonl`
- Reads `total_episodes` and `total_frames` from `info.json`
- Warns if the number of episodes is less than 5 (overfitting risk)

**[2/5] Python environment activation**
- Tries `.venv-psi` first; falls back to `.venv`
- Both are created by `uv sync` in the repo

**[3/5] Locating pre-trained weights**
- Looks for checkpoints in `$PSI_HOME/cache/checkpoints/psi0/`:
  - VLM backbone: `pre.fast.1by1.2601091803.ckpt.ego200k.he30k`
  - Action header: `postpre.1by1.pad36.2601131206.ckpt.he30k`
- If not found, **downloads automatically** from HuggingFace (`USC-PSI-Lab/psi-model`)
- Paths can be overridden via `VLM_CKPT_PATH` and `ACTION_CKPT_PATH`

**[4/5] Automatic hyper-parameter computation**
- `max_training_steps = max(1000, total_frames / (batch_size × n_gpus) × TARGET_EPOCHS)`
- `checkpointing_steps = max_training_steps / 10`
- `validation_steps = max_training_steps / 20`
- Detects number of GPUs via `CUDA_VISIBLE_DEVICES`

**[5/5] Training launch**
- Uses `torchrun` if available in PATH, otherwise `python -m torch.distributed.run`
- All model and transform args are inherited from the canonical `finetune-real-psi0.sh` script

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | all GPUs | GPUs to use (e.g. `"0"`, `"0,1"`) |
| `TARGET_EPOCHS` | `50` | Number of epochs over the dataset |
| `TRAIN_BATCH_SIZE` | `16` | Batch size per GPU |
| `VLM_CKPT_PATH` | auto (`$PSI_HOME/...`) | Override VLM backbone checkpoint path |
| `ACTION_CKPT_PATH` | auto (`$PSI_HOME/...`) | Override action header checkpoint path |
| `WANDB_DISABLED` | not set | Set to `1` to use TensorBoard instead |

### Expected dataset format

The dataset must follow the **LeRobot v2.1** format:

```
<dataset_dir>/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       └── egocentric/
│           ├── episode_000000.mp4
│           └── ...
└── meta/
    ├── info.json          # total_frames, fps, robot_type, etc.
    ├── episodes.jsonl     # episode indices and instructions
    ├── tasks.jsonl        # task list
    └── stats_psi0.json    # normalization statistics (min/max/mean/std)
```

Required parquet columns:
- `observation.images.egocentric` — egocentric image (video)
- `states` — robot state (joints)
- `action` — actions (wrist xyz/rpy + gripper)
- `task` — text instruction (automatically injected by LeRobot via `tasks.jsonl`)

If `meta/stats_psi0.json` does not exist, generate it with:
```bash
python scripts/data/calc_modality_stats.py --task-dir <dataset_dir>
```

### Example output

```
════════════════════════════════════════════════
 PSI-0 Fine-tuning Launcher
 Dataset : /home/user/Psi0/pick_cylinder_manipulation_psi
 Repo ID : pick_cylinder_manipulation_psi
 Exp name: pick-cylinder
════════════════════════════════════════════════

[1/5] Validating dataset...
    Episodes : 3
    Frames   : 10299
    FPS      : 30

[WARN] Only 3 episode(s) found. PSI fine-tuning typically needs
       20-50+ episodes for robust results. Proceeding anyway...

[2/5] Activating Python environment...
    Using: .venv
[3/5] Locating PSI-0 pre-trained checkpoints...
    VLM backbone: .../cache/checkpoints/psi0/pre.fast... ✓
    Action header: .../cache/checkpoints/psi0/postpre... ✓
[4/5] Computing training hyperparameters...
    GPUs              : 1 (CUDA_VISIBLE_DEVICES=0)
    Effective batch   : 16 (16 per GPU × 1 GPU(s))
    Target epochs     : 50
    Max training steps: 32184
    Checkpointing     : every 3218 steps
    Validation        : every 1609 steps
    Logging           : wandb (project=psi)
[5/5] Launching training...
    Using launcher: python -m torch.distributed.run
```

---

## Running with Docker

The `psi0-train` image bundles the entire environment (PyTorch 2.7, flash-attn, DeepSpeed) with no host venv setup required.

### Build the image

```bash
# From the repo root
docker build -t psi0-train -f docker/Dockerfile .
```

### Local dataset (mounted via volume)

```bash
docker run --gpus all --rm \
  -v $(pwd)/my_dataset:/workspace/data/my_dataset \
  -v /path/to/checkpoints:/workspace/checkpoints/cache/checkpoints \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  psi0-train my_dataset
```

### HuggingFace dataset (auto-download)

```bash
docker run --gpus all --rm \
  -v /path/to/checkpoints:/workspace/checkpoints/cache/checkpoints \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e HF_TOKEN=$HF_TOKEN \
  psi0-train <hf_org>/<dataset_repo_id>
```

### Container environment variables

| Variable | Default | Description |
|---|---|---|
| `WANDB_API_KEY` | — | WandB key; if absent, falls back to TensorBoard |
| `HF_TOKEN` | — | HF token for private datasets |
| `CUDA_VISIBLE_DEVICES` | all | GPUs to use |
| `TARGET_EPOCHS` | `50` | Training epochs |
| `TRAIN_BATCH_SIZE` | `16` | Batch size per GPU |
| `VLM_CKPT_PATH` | auto | Override VLM checkpoint path |
| `ACTION_CKPT_PATH` | auto | Override action header checkpoint path |
| `WANDB_DISABLED` | — | Set to `true` to force TensorBoard |

### Expected volumes

| Container path | What to mount |
|---|---|
| `/workspace/data/<dataset>` | LeRobot dataset directory |
| `/workspace/checkpoints/cache/checkpoints` | `$PSI_HOME/cache/checkpoints` from host |
| `/workspace/hf_cache` *(optional)* | HF cache to avoid re-downloads |

---

## Other scripts

| Script | Description |
|---|---|
| `finetune-real-psi0.sh` | Canonical fine-tuning (requires hardcoded task name, dataset in `$DATA_HOME/real/`) |
| `finetune-simple-psi0.sh` | Fine-tuning on SIMPLE datasets (simulation) |
| `posttrain-he-psi0.sh` | Post-training on the Humanoid Everyday dataset |
| `pretrain-egodex-psi0-fast.sh` | Pre-training on EgoDex |
