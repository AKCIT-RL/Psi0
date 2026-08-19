## OpenPI $\pi_{0.5}$

### Environment

The current environment is Python 3.10 on CUDA 12.

```bash
uv venv .venv-openpi --python 3.10
source .venv-openpi/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --directory baselines/pi05 --extra cuda12 --active
VIRTUAL_ENV=.venv-openpi uv pip install -e . --torch-backend=cu128
VIRTUAL_ENV=.venv-openpi uv pip install -e src/openpi/openpi-client --torch-backend=cu128
VIRTUAL_ENV=.venv-openpi GIT_LFS_SKIP_SMUDGE=1 uv pip install -r baselines/pi05/requirements-openpi.cu12.txt --torch-backend=cu128
cp -r src/openpi/models_pytorch/transformers_replace/* \
  .venv-openpi/lib/python3.10/site-packages/transformers/
```

### Training

```bash
bash baselines/pi05/train_pi05.sh <task>
```

`<task>` is the OpenPI task/config key passed through to `src/openpi/train_pytorch.py` as both the task selector and the default experiment name.

The wrapper is also commonly launched like this when you want a fixed GPU list and gradient checkpointing disabled:

```bash
OPENPI_GRAD_CHECKPOINTING=0 CUDA_VISIBLE_DEVICES=0,1,2,3 bash baselines/pi05/train_pi05.sh "$task"
```

### Script Arguments

- `<task>`: required training target, for example `G1WholebodyXMovePickBetweenTablesTeleop-v0`.

### Environment Variables

The wrapper uses these variables before launching `torchrun`:

- `CUDA_VISIBLE_DEVICES`: GPU list for DDP; defaults to `0,1,2,3,4,5,6,7`.
- `OMP_NUM_THREADS`: host-side thread count; defaults to `2`.
- `TORCH_NCCL_BLOCKING_WAIT`: NCCL wait mode; defaults to `1`.
- `TORCH_NCCL_ASYNC_ERROR_HANDLING`: async NCCL error handling; defaults to `1`.
- `NCCL_DEBUG`: NCCL logging level; defaults to `WARN`.
- `TORCH_DISTRIBUTED_DEBUG`: distributed debug level; defaults to `INFO`.
- `OPENPI_GRAD_CHECKPOINTING`: toggles gradient checkpointing in `src/openpi/train_pytorch.py`; defaults to `1`. Set to `0` to disable.
- `NCCL_RAS_ENABLE`: NCCL RAS toggle; defaults to `0`.
- `NCCL_IB_DISABLE`: InfiniBand transport toggle; defaults to `1`.
- `NCCL_P2P_DISABLE`: peer-to-peer transport toggle; defaults to `1`.

### Training Flags

The launcher forwards these training values to `src/openpi/train_pytorch.py`:

- `--exp_name=<task>`: derived from the task name.
- `--save_interval=10000`: checkpoint cadence.
- `--checkpoint_base_dir=.runs/openpi-05`: output directory.
- `--batch_size=32`: per-step batch size.

The training implementation also reads `OPENPI_GRAD_CHECKPOINTING` inside [src/openpi/train_pytorch.py](../../src/openpi/train_pytorch.py) to decide whether to enable or disable gradient checkpointing.

### Serving

```bash
bash baselines/pi05/serve_pi05.sh <task> [model_path] [port] [run|start|stop|status|logs]
```

`model_path` defaults to `nvidia/openpi05-3B`. Use `start` to background the server and write logs under `logs/pi05/`.

### Open-loop eval

```bash
python baselines/pi05/eval_openloop.py --port=<port> --task=<task>
```


### Eval in SIMPLE

TODO: migrate following instructions using SIMPLE third_party

```
cd <project root of SIMPLE>
source .venv/bin/activate
```

```
export task=G1WholebodyXMovePick-v0
```

Download eval data and extract it:
```
hf download USC-PSI-Lab/psi-data \
	simple-eval/$task.zip \
	--local-dir=data/evals \
	--repo-type=dataset

unzip data/evals/simple-eval/$task.zip -d data/evals/simple-eval
```
Now start SIMPLE eval in the SIMPLE environment:

> We provide three domain randomization levels: `level-0`, `level-1`, `level-2` for each task

```
export dr=level-0
```
We use two different entrypoints for evaluating different tasks:

set entrypoint and agent to `eval_decoupled_wbc.py` and `pi05_decoupled_wbc` if the evaluating task ends with `Teleop`, which means the task data is collected using teleoperation:
```
export entry=eval_decoupled_wbc.py
export agent=pi05_decoupled_wbc
```

and set entrypoint and agent to `eval.py` and `pi05` if the evaluating task ends with `MP`, which means the task data is generated using CuRobo Motion planning:
```
export entry=eval.py
export entry=pi05
```

```
python src/simple/cli/$entry \
	simple/$task \
	$agent \
	$dr \
	--host=localhost \
	--port=9000 \
	--sim-mode=mujoco_isaac \
	--no-headless \
	--data-format=lerobot \
	--data-dir=data/evals/simple-eval/$task/$dr
```
