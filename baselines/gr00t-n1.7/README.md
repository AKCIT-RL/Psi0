## GR00T N1.7 Baseline

GR00T N1.7 uses a Qwen3-VL-2B backbone and the current repo adds a dedicated serving wrapper for the N1.7 checkpoint path.

> 📄 For the detailed fine-tuning and memory-optimization guide, see [finetune_gr00t_n1d7.md](finetune_gr00t_n1d7.md).

### Environment

```bash
uv venv .venv-gr00t --python 3.10
source .venv-gr00t/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --directory src/gr00t --extra cuda12 --active
```

### Training

```bash
python3 baselines/gr00t-n1.7/launch_finetune_n1d7.py \
  --preset finetune_simple \
  --dataset-path /path/to/your/lerobot/dataset \
  --base-model-path /path/to/GR00T-N1.7-3B \
  --output-dir ./checkpoints/gr00t_n1d7_finetune
```

Use `--dry-run` to inspect the generated command before launching.

### Serving

```bash
./baselines/gr00t-n1.7/serve_gr00t.sh <embodiment_tag> <model_path> [port] [run|start|stop|status|logs]
```

`model_path` can be a local checkpoint path or a Hub model name. The wrapper defaults to `cuda:0` and writes logs under `logs/gr00t/` when you use `start`.

### Open-loop evaluation

```bash
./baselines/gr00t-n1.7/openloop_eval.sh \
  --dataset-path /path/to/your/lerobot/dataset \
  --modality-config-path src/gr00t/gr00t/configs/modality/g1_locomanip.py \
  --episode-index 0
```
