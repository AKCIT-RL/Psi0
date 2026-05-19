# GR00T N1.7 — Fine-Tuning Guide

Step-by-step guide to fine-tune GR00T N1.7 (Qwen3-VL / Cosmos-Reason2 backbone)
on a real-robot LeRobot-format dataset (Unitree G1 or similar).

---

## Requirements

- NVIDIA GPU with at least 24 GB VRAM (tested on RTX 4090)
- Dataset in LeRobot format with a `meta/modality.json` describing state/action/video keys
- Base model checkpoint (GR00T-N1.7-3B)
- `transformers >= 4.57.3` (already set in `src/gr00t/pyproject.toml`)

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

Place the GR00T-N1.7-3B checkpoint at a local path, e.g.:

```
<checkpoints_dir>/GR00T-N1.7-3B/
  config.json              ← must contain "model_type": "Gr00tN1d7"
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
  model.safetensors.index.json
  processor_config.json
  statistics.json
  embodiment_id.json
```

Or download directly from the HF Hub:

```bash
huggingface-cli download nvidia/GR00T-N1.7-3B \
  --local-dir <checkpoints_dir>/GR00T-N1.7-3B
```

---

## 3. Prepare the dataset

Your dataset must follow the LeRobot v2 format (same as N1.6):

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

The `modality.json` for the G1 loco-manipulation embodiment is identical to N1.6:

- **state keys**: `left_hand`, `right_hand`, `left_arm`, `right_arm`, `rpy`, `height`
- **action keys**: `left_hand`, `right_hand`, `left_arm`, `right_arm`, `rpy`, `height`, `torso_vx`, `torso_vy`, `torso_vyaw`, `target_yaw`
- **video key**: `rs_view`

Use `UNITREE_G1_N1D7` as the embodiment tag (maps to
`"unitree_g1_full_body_with_waist_height_nav_cmd"` in the modality config registry).

---

## 4. (Optional) Login to Weights & Biases

```bash
/path/to/Psi0/src/gr00t/.venv/bin/wandb login
```

---

## 5. Run fine-tuning

### Option A — Preset launcher (recommended)

```bash
cd /path/to/Psi0

python3 baselines/gr00t-n1.7/launch_finetune_n1d7.py \
  --preset finetune_simple \
  --base-model-path /path/to/GR00T-N1.7-3B \
  --dataset-path /path/to/dataset \
  --output-dir /path/to/output/checkpoints
```

Add `--dry-run` to inspect the generated command without running it.

### Option B — Direct torchrun (background, with logging)

```bash
cd /path/to/Psi0

nohup bash -c '
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME=/path/to/hfm/cache \
CUDA_VISIBLE_DEVICES=0 \
NO_ALBUMENTATIONS_UPDATE=1 \
PYTHONPATH=/path/to/Psi0/src:/path/to/Psi0/src/gr00t \
/path/to/Psi0/src/gr00t/.venv/bin/python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port=29502 \
  /path/to/Psi0/baselines/gr00t-n1.7/launch_finetune_n1d7_inner.py \
  --base-model-path /path/to/GR00T-N1.7-3B \
  --dataset-path /path/to/dataset \
  --embodiment-tag UNITREE_G1_N1D7 \
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
' > /tmp/gr00t_n1d7_train.log 2>&1 &

echo "Training PID: $!"
```

Monitor progress:

```bash
tail -f /tmp/gr00t_n1d7_train.log
```

### Key flags explained

| Flag | Default | Description |
|------|---------|-------------|
| `--global-batch-size 4` | 24 | Per-device batch size; reduced to fit 24 GB VRAM |
| `--gradient-accumulation-steps 6` | 1 | Effective batch = 4 × 6 = 24 |
| `--gradient-checkpointing` | off | Reduces activation memory at cost of ~20% slower training |
| `--tune-llm` | false | Fine-tune the Qwen3-VL language backbone (needs more VRAM) |
| `--tune-visual` | false | Fine-tune the Qwen3-VL visual encoder |
| `--eval-strategy steps` | no | Evaluate on held-out val episodes every N steps |
| `--val-split 0.1` | 0.1 | Last 10% of episodes held out for validation |
| `--use-wandb` | off | Log to Weights & Biases (project: `finetune-gr00t-n1d7`) |

> **Memory note**: by default only the diffusion action head and projector are
> trained (backbone frozen). This fits comfortably in 24 GB.
> Enabling `--tune-llm` requires ≥40 GB VRAM or stronger gradient checkpointing.

### Environment variables required

| Variable | Purpose |
|----------|---------|
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Avoids CUDA memory fragmentation |
| `HF_HOME` | Redirect HuggingFace cache to a writable directory |
| `NO_ALBUMENTATIONS_UPDATE` | Suppress albumentations update check |
| `PYTHONPATH` | Expose `src/` and `src/gr00t/` to Python |

---

## 6. Differences from N1.6

| Aspect | N1.6 | N1.7 |
|--------|------|------|
| Backbone | `nvidia/Eagle-Block2A-2B-v2` | `nvidia/Cosmos-Reason2-2B` (Qwen3-VL) |
| `backbone_model_type` | `"eagle"` | `"qwen"` |
| `eagle_collator` | required | not used |
| Embodiment tag | `UNITREE_G1` | `UNITREE_G1_N1D7` |
| `transformers` version | 4.51.x | ≥ 4.57.3 |
| Mask-guided BG suppression | no | yes (`masks` field in `VLAStepData`) |

---

## 7. Resume from checkpoint

Training resumes automatically if `--output-dir` already contains a checkpoint.
Just re-run the same command.

---

## 8. Checkpoints

Checkpoints are saved at every `--save-steps` steps under `--output-dir`:

```
<output-dir>/
  checkpoint-10000/
    config.json            ← model_type: "Gr00tN1d7"
    model-*.safetensors
    processor/
  checkpoint-20000/
  ...
  experiment_cfg/
  processor/
```

The `processor/` directory is needed for inference and deployment.
