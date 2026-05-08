## GR00T Baselines

Use the canonical launchers in this directory instead of adding more one-off shell scripts.

---

### Training with Docker

> 📄 For a complete step-by-step guide (environment setup, memory optimizations, WandB, dataset format): [finetune_gr00t.md](finetune_gr00t.md)

**1. Build the image** (once):

```bash
docker build -t gr00t-train src/gr00t/
```

**2. Run fine-tuning:**

```bash
docker run --rm --gpus all \
  -v $PSI_HOME/hfm:/hfm \
  -v $PSI_HOME/src:/workspace/src \
  -e HF_HOME=/hfm/cache \
  -e DATASET_PATH=/hfm/data/real/<your_task> \
  -e SIMPLE_DATASET_PATH=/hfm/data/real/<your_task> \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NO_ALBUMENTATIONS_UPDATE=1 \
  gr00t-train \
  python -m torch.distributed.run --nproc_per_node=1 --master_port=29501 \
    /workspace/src/gr00t/gr00t/experiment/launch_finetune.py \
    --base-model-path /hfm/checkpoints/GR00T-N1.6-3B \
    --dataset-path /hfm/data/real/<your_task> \
    --embodiment-tag G1_LOCO_DOWNSTREAM \
    --modality-config-path /workspace/src/gr00t/gr00t/configs/modality/g1_locomanip.py \
    --num-gpus 1 \
    --output-dir /hfm/checkpoints/gr00t_finetune_output \
    --save-steps 10000 --save-total-limit 4 --max-steps 50000 \
    --warmup-ratio 0.05 --weight-decay 1e-05 --learning-rate 0.0001 \
    --global-batch-size 4 --gradient-accumulation-steps 6 \
    --gradient-checkpointing \
    --dataloader-num-workers 2 \
    --eval-strategy steps --eval-steps 1000 --val-split 0.1 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
```

> **Notes:**
> - `-v .../hfm:/hfm` mounts checkpoints, HF cache, and datasets
> - `-v .../src:/workspace/src` mounts source code — local changes are reflected without rebuild
> - The Dockerfile is at `src/gr00t/Dockerfile`
> - To run in background: add `-d` and redirect logs with `2>&1 | tee /tmp/gr00t_train.log`

---

Train or pretrain with a YAML preset:

```bash
python3 baselines/gr00t-n1.6/finetune_gr00t.py --preset finetune_simple
bash baselines/gr00t-n1.6/pretrain_gr00t.sh --preset pretrain_g1_ee --dry-run
```

Run SIMPLE eval through the Python API:

```bash
.venv/bin/python baselines/gr00t-n1.6/eval_simple.py --preset simple_local
.venv/bin/python baselines/gr00t-n1.6/eval_simple.py --preset simple_local --dry-run
```

Preset files live under:

- `baselines/gr00t-n1.6/presets/train`
- `baselines/gr00t-n1.6/presets/eval`
- `pretrain_gr00t.sh` for preset-based pretraining
- `sim_eval.sh` for SIMPLE eval
- `deploy_gr00t_simple.sh` for server-only deployment
