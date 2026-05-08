# GR00T N1.6 — Fine-Tuning Guide

Step-by-step guide to fine-tune GR00T N1.6 on a real-robot LeRobot-format dataset (Unitree G1 or similar).

---

## Requirements

- NVIDIA GPU with at least 24 GB VRAM (tested on RTX 4090)
- Dataset in LeRobot format with a `meta/modality.json` describing state/action/video keys
- Base model checkpoint (GR00T-N1.6-3B)

---

## 1. Install the environment

```bash
cd /path/to/Psi0/src/gr00t
uv sync
```

Install `bitsandbytes` for 8-bit Adam (required to fit in 24 GB VRAM):

```bash
/path/to/Psi0/src/gr00t/.venv/bin/python -m ensurepip
/path/to/Psi0/src/gr00t/.venv/bin/python -m pip install bitsandbytes
```

---

## 2. Download the base model

Place the GR00T-N1.6-3B checkpoint at a local path, e.g.:

```
<checkpoints_dir>/GR00T-N1.6-3B/
  config.json
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
  model.safetensors.index.json
  processor_config.json
  statistics.json
  embodiment_id.json
```

---

## 3. Prepare the dataset

Your dataset must follow the LeRobot v2 format:

```
<dataset_root>/
  meta/
    info.json
    modality.json      ← describes state/action/video keys and dimensions
    episodes.jsonl
    stats.json
  data/chunk-000/
    episode_000000.parquet
    ...
  videos/chunk-000/egocentric/
    episode_000000.mp4
    ...
```

The `modality.json` for the G1 loco-manipulation embodiment must have:

- **state keys**: `left_hand`, `right_hand`, `left_arm`, `right_arm`, `rpy`, `height`
- **action keys**: `left_hand`, `right_hand`, `left_arm`, `right_arm`, `rpy`, `height`, `torso_vx`, `torso_vy`, `torso_vyaw`, `target_yaw`
- **video key**: `rs_view` → mapped from `observation.images.egocentric`

---

## 4. (Optional) Login to Weights & Biases

```bash
/path/to/Psi0/src/gr00t/.venv/bin/wandb login
```

---

## 5. Run fine-tuning

Run from the repository root (`/path/to/Psi0`):

```bash
cd /path/to/Psi0

nohup bash -c '
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME=/path/to/hfm/cache \
CUDA_VISIBLE_DEVICES=0 \
DATASET_PATH=/path/to/dataset \
SIMPLE_DATASET_PATH=/path/to/dataset \
PYTHONPATH=/path/to/Psi0/src:/path/to/Psi0/src/gr00t \
NO_ALBUMENTATIONS_UPDATE=1 \
/path/to/Psi0/src/gr00t/.venv/bin/python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port=29501 \
  /path/to/Psi0/src/gr00t/gr00t/experiment/launch_finetune.py \
  --base-model-path /path/to/GR00T-N1.6-3B \
  --dataset-path /path/to/dataset \
  --embodiment-tag G1_LOCO_DOWNSTREAM \
  --modality-config-path src/gr00t/gr00t/configs/modality/g1_locomanip.py \
  --num-gpus 1 \
  --output-dir /path/to/output/checkpoints \
  --save-steps 10000 \
  --save-total-limit 4 \
  --max-steps 50000 \
  --warmup-ratio 0.05 \
  --weight-decay 1e-05 \
  --learning-rate 0.0001 \
  --global-batch-size 4 \
  --gradient-accumulation-steps 6 \
  --gradient-checkpointing \
  --dataloader-num-workers 2 \
  --eval-strategy steps \
  --eval-steps 1000 \
  --val-split 0.1 \
  --use-wandb \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
' > /tmp/gr00t_train.log 2>&1 &

echo "Training PID: $!"
```

Monitor progress:

```bash
tail -f /tmp/gr00t_train.log
```

### Key flags explained

| Flag | Default | Description |
|------|---------|-------------|
| `--global-batch-size 4` | 24 | Per-device batch size; reduced to fit 24 GB VRAM |
| `--gradient-accumulation-steps 6` | 1 | Effective batch = 4 × 6 = 24 |
| `--gradient-checkpointing` | off | Reduces activation memory at cost of ~20% slower training |
| `--eval-strategy steps` | no | Evaluate on held-out val episodes every N steps |
| `--eval-steps 1000` | 500 | Evaluation frequency |
| `--val-split 0.1` | 0.1 | Last 10% of episodes held out for validation |
| `--use-wandb` | off | Log metrics to Weights & Biases |

> **Memory note**: the optimizer state (Adam) for 1.6 B trainable parameters requires ~13 GB in fp32.
> The launcher sets `optim=adamw_bnb_8bit` (8-bit Adam via bitsandbytes) to keep total VRAM under 24 GB.

### Environment variables required

| Variable | Purpose |
|----------|---------|
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Avoids CUDA memory fragmentation |
| `HF_HOME` | Redirect HuggingFace cache to a writable directory |
| `DATASET_PATH` | Path read by `g1_locomanip.py` to load `meta/modality.json` |
| `NO_ALBUMENTATIONS_UPDATE` | Suppress albumentations update check |
| `PYTHONPATH` | Expose `src/` and `src/gr00t/` to Python |

---

## 6. Using the preset launcher (alternative)

You can also use the preset-based launcher from the repo root:

```bash
DATASET_PATH=/path/to/dataset \
/path/to/Psi0/.venv/bin/python3 baselines/gr00t-n1.6/finetune_gr00t.py \
  --preset finetune_simple \
  --dataset-path /path/to/dataset \
  --base-model-path /path/to/GR00T-N1.6-3B \
  --output-dir /path/to/output/checkpoints \
  --cuda-visible-devices 0 \
  --num-gpus 1
```

Add `--dry-run` to print the full command without executing it.

Preset files live under `baselines/gr00t-n1.6/presets/train/`. Copy and edit `finetune_simple.yaml` to create your own preset.

---

## 7. Resume from checkpoint

Training resumes automatically if `--output-dir` already contains a checkpoint. Just re-run the same command.

---

## 8. Checkpoints

Checkpoints are saved at every `--save-steps` steps under `--output-dir`:

```
<output-dir>/
  checkpoint-10000/
  checkpoint-20000/
  ...
  experiment_cfg/
  processor/
```

The `processor/` directory is needed for inference and deployment.
