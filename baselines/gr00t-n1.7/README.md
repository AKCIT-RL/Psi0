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

The launcher command is:

```bash
python3 baselines/gr00t-n1.7/launch_finetune_n1d7.py \
  --preset finetune_simple \
  --dataset-path /path/to/your/lerobot/dataset \
  --base-model-path /path/to/GR00T-N1.7-3B \
  --output-dir ./checkpoints/gr00t_n1d7_finetune
```

For a quick check, you can also run:

```bash
python3 baselines/gr00t-n1.7/launch_finetune_n1d7.py --preset finetune_simple --dry-run
```

### Script Arguments

- `--preset`: preset name or YAML path; defaults to `finetune_simple`.
- `--base-model-path`: overrides `model.base_model_path` from the preset.
- `--dataset-path`: overrides `dataset.path` from the preset.
- `--output-dir`: overrides `training.output_dir` from the preset.
- `--embodiment-tag`: overrides `dataset.embodiment_tag` from the preset.
- `--cuda-visible-devices`: overrides `runtime.cuda_visible_devices`.
- `--num-gpus`: overrides `training.num_gpus`.
- `--dry-run`: prints the resolved distributed command without launching it.

### Preset Fields

The launcher resolves the selected YAML preset and reads these sections:

- `runtime.cuda_visible_devices`: sets `CUDA_VISIBLE_DEVICES` and the default process count.
- `runtime.master_port`: forwarded to `torch.distributed.run`.
- `model.base_model_path`: the GR00T checkpoint or Hub model to fine-tune.
- `dataset.path`: dataset path passed to the inner launcher.
- `dataset.embodiment_tag`: embodiment tag passed to the inner launcher.
- `dataset.modality_config_path`: optional modality config path.
- `training.*`: training flags forwarded to the inner launcher, including `num_gpus`.
- `env.*`: extra environment variables injected into the subprocess environment.

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: set automatically from `runtime.cuda_visible_devices`; can be overridden by the shell or `--cuda-visible-devices`.
- Any key under `env` in the selected preset is added to the subprocess environment.

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
