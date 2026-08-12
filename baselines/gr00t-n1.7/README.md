## GR00T N1.7 Baseline

GR00T N1.7 uses a **Qwen3-VL-2B (Cosmos-Reason2-2B)** vision-language backbone
instead of Eagle. The action head architecture is identical to N1.6.

> 📄 For a complete step-by-step guide (environment setup, memory optimizations,
> WandB, dataset format): [finetune_gr00t_n1d7.md](finetune_gr00t_n1d7.md)

---

### Requirements

- Requires `transformers >= 4.57.3` (already pinned in `src/gr00t/pyproject.toml`)
- Same GPU/VRAM profile as N1.6 (tested on RTX 4090, 24 GB)

---

### Training

```bash
# preset-based (recommended)
python3 baselines/gr00t-n1.7/launch_finetune_n1d7.py \
  --preset finetune_simple \
  --dataset-path /path/to/your/lerobot/dataset \
  --base-model-path /path/to/GR00T-N1.7-3B \
  --output-dir ./checkpoints/gr00t_n1d7_finetune

# dry-run to inspect the generated command
python3 baselines/gr00t-n1.7/launch_finetune_n1d7.py --preset finetune_simple --dry-run
```

Preset files live under `baselines/gr00t-n1.7/presets/train/`.

---

### Deployment

`AutoModel.from_pretrained` detects the model type from `config.json`
automatically — the same server binary works for both N1.6 and N1.7.

```bash
./baselines/gr00t-n1.7/deploy_gr00t_n1d7.sh \
  --model-path ./checkpoints/gr00t_n1d7_finetune/checkpoint-10000 \
  --embodiment-tag UNITREE_G1_N1D7 \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5555 \
  --strict
```

Or directly from the HF Hub:

```bash
./baselines/gr00t-n1.7/deploy_gr00t_n1d7.sh \
  --model-path nvidia/GR00T-N1.7-3B \
  --embodiment-tag UNITREE_G1_N1D7 \
  --device cuda:0 --port 5555
```

---

### Open-loop evaluation

Start the server in one terminal, then:

```bash
./baselines/gr00t-n1.7/openloop_eval.sh \
  --dataset-path /path/to/your/lerobot/dataset \
  --modality-config-path src/gr00t/gr00t/configs/modality/g1_locomanip.py \
  --episode-index 0
```
