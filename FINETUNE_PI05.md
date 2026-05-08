# Fine-tuning π₀.₅ on Psi0

Step-by-step guide to adapt the π₀.₅ VLA model to new loco-manipulation tasks on the G1 robot.

---

## Prerequisites

- `PSI_HOME` environment variable pointing to the repo root (e.g. `/home/user/Psi0`)
- `DATA_HOME` environment variable pointing to the data directory (e.g. `/home/user/hfm/data/real`)
- `uv` installed

Load environment variables:
```bash
cd $PSI_HOME && source .env
```

---

## 1. Set Up the Environment

π₀.₅ uses a **separate** virtualenv from the main Psi0 environment:

```bash
uv venv .venv-openpi --python 3.10
source .venv-openpi/bin/activate

VIRTUAL_ENV=.venv-openpi uv pip install -e .
VIRTUAL_ENV=.venv-openpi uv pip install -e src/openpi/openpi-client
VIRTUAL_ENV=.venv-openpi GIT_LFS_SKIP_SMUDGE=1 uv pip install -r baselines/pi05/requirements-openpi.txt
```

Apply required patches to the `transformers` library:

```bash
cp -r src/openpi/models_pytorch/transformers_replace/* \
    .venv-openpi/lib/python3.10/site-packages/transformers/
```

---

## 2. Download Pre-trained Weights

```bash
hf download USC-PSI-Lab/psi-model \
    --local-dir=$PSI_HOME/cache/checkpoints \
    --include="openpi/pi05_droid/*" \
    --repo-type=model
```

Weights are saved to `$PSI_HOME/cache/checkpoints/openpi/pi05_droid/`.

---

## 3. Download Task Data

For tasks from the Psi0 SIMPLE suite:

```bash
export task=G1WholebodyXMovePick-v0

hf download USC-PSI-Lab/psi-data simple/$task.zip \
    --local-dir=$PSI_HOME/data \
    --repo-type=dataset

unzip "$PSI_HOME/data/simple/$task.zip" -d "$PSI_HOME/data/simple"
```

For your own real-world data (LeRobot format), just point to the local directory — see step 4.

---

## 4. Create a `TrainConfig` for Your Task

Edit [`src/openpi/training/config.py`](src/openpi/training/config.py) and add an entry to the `_CONFIGS` list.
Copy an existing block (e.g. `Pick_bottle_and_turn_and_pour_into_cup`) and adjust:

```python
TrainConfig(
    name="my_task",                # unique identifier — used for training and serving
    project_name="psi",
    num_workers=8,
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=36,             # G1 uses 36 DOF (original DROID uses 32)
        action_horizon=16,         # action chunk length
        max_token_len=250,         # max tokens (250 for dual-arm robots)
    ),
    data=LeRobotHFMDataConfig(
        repo_id=f"{os.environ['DATA_HOME']}/my_task",  # local dataset path
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_droid/params"
    ),
    num_train_steps=40_000,
    batch_size=128,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=1e-4,
        decay_steps=40_000,
        decay_lr=1e-8,
    ),
    pytorch_weight_path=f"{os.environ['PSI_HOME']}/cache/checkpoints/openpi/pi05_droid",
    policy_metadata={"dataset": "my_task"},
    checkpoint_base_dir=".runs/openpi-05",
),
```

> **Note on `action_dim=36`:** π₀.₅ was pre-trained with 32 dimensions (DROID). The repository automatically adapts action projection layer weights via padding when loading the checkpoint.

---

## 5. Compute Normalization Statistics

Option 1 — official script (slower, more accurate):
```bash
source .venv-openpi/bin/activate
python src/openpi/compute_norm_stats.py --config-name my_task
```

Option 2 — rewrite pre-computed Psi0 stats (faster, minimal difference):
```bash
python src/openpi/rewrite_norm_stats.py --task_path=$PSI_HOME/data/simple/my_task
```

---

## 6. Train

```bash
source .venv-openpi/bin/activate
CUDA_VISIBLE_DEVICES=0 bash baselines/pi05/train_pi05.sh my_task
```

For multiple GPUs, list them comma-separated:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash baselines/pi05/train_pi05.sh my_task
```

The script uses `torchrun` and automatically detects the number of GPUs from `CUDA_VISIBLE_DEVICES`.
Checkpoints are saved to `.runs/openpi-05/my_task/`.

---

## 7. Serve the Model

```bash
# Arguments: <task> [ckpt_step] [port]
CUDA_VISIBLE_DEVICES=0 bash baselines/pi05/serve_pi05.sh my_task 40000 9000
```

---

## 8. Open-Loop Evaluation (on training data)

```bash
source .venv-openpi/bin/activate
python baselines/pi05/eval_openloop.py --port=9000 --task=my_task
```

---

## References

- [`baselines/pi05/README.md`](baselines/pi05/README.md) — detailed baseline documentation
- [`src/openpi/training/config.py`](src/openpi/training/config.py) — all existing `TrainConfig` entries
- [`baselines/pi05/train_pi05.sh`](baselines/pi05/train_pi05.sh) — training script
- [`baselines/pi05/serve_pi05.sh`](baselines/pi05/serve_pi05.sh) — serving script
