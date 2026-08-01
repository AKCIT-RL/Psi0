# Local SIMPLE evaluation workspace

This directory is the handoff point for policies trained on the DGX Spark and the local
output root for SIMPLE evaluations.

## Drop checkpoints here

Copy each complete training run into exactly one policy inbox:

```text
eval/weights/
├── psi0/incoming/<run-name>/
└── dp/incoming/<run-name>/
```

Keep the original run structure. In particular, include the `checkpoints` directory,
run configuration, training arguments, normalization statistics, and any policy assets.
Do not copy only `model.safetensors` or a `.pth` file when the loader depends on the rest
of the run.

Example transfer destinations from the training machine:

```bash
rsync -a --info=progress2 /path/to/psi0-run/ \
  <eval-host>:/home/marcos_paulo/Documents/Psi0/eval/weights/psi0/incoming/psi0-run/

rsync -a --info=progress2 /path/to/dp-run/ \
  <eval-host>:/home/marcos_paulo/Documents/Psi0/eval/weights/dp/incoming/dp-run/
```

Weights are ignored by Git. Evaluation artifacts belong under:

```text
eval/results/<policy>/<task>/<timestamp>/
```

Each result directory should contain the exact commands and metadata needed to identify
the main repository commit, SIMPLE commit, Docker image ID, checkpoint, dataset, task,
episode count, success metric, logs, and videos.

## Upload evaluation videos to W&B

Keep evaluation runs separate from training history. The uploader creates one W&B run
per policy configuration, logs both camera videos in an episode table, writes aggregate
metrics to the run summary, and stores telemetry and logs in a `simple-eval` artifact.

Validate every expected episode, video, stats file, and telemetry file without network
access before uploading:

```bash
python scripts/eval/upload_simple_eval_to_wandb.py \
  --manifest eval/wandb/screwdriver-20260729.json \
  --dry-run
```

After configuring `WANDB_API_KEY`, upload all runs from the repository root:

```bash
python scripts/eval/upload_simple_eval_to_wandb.py \
  --manifest eval/wandb/screwdriver-20260729.json
```

Use `--run psi0_h24` to upload one manifest entry. Run IDs are deterministic and use
W&B resume mode; completed uploads are skipped unless `--force` is provided. Videos are
stored in the W&B table and are not duplicated in the telemetry artifact.

## SIMPLE checkout

Initialize the configured `industrial_env` submodule from the repository root:

```bash
git submodule sync -- third_party/SIMPLE
git submodule update --init --recursive --remote third_party/SIMPLE
```

Then read these local sources before running Docker:

```bash
sed -n '1,240p' third_party/SIMPLE/docs/source/tutorials/docker.md
git -C third_party/SIMPLE log -10 --stat
```

The tutorial and Compose files at the checked-out commit are authoritative. Commands in
older Psi0 READMEs may refer to the pre-`industrial_env` environment.

## Verified industrial environment

The `industrial_env` checkout uses `docker-compose.yml` with host networking and the
`simple-teleoperation:${DATE}` image. Run Compose from `third_party/SIMPLE` so its
`.env`, build context, data mount, and `.uv-cache` resolve consistently.
Include `../../eval/docker-compose.yml` to mount this evaluation workspace at
`/workspace/eval` inside the eval container.

Two evaluation paths are available:

| Task family | CLI | Psi0 agent | DP agent |
|---|---|---|---|
| Motion-planned tasks such as `*MP-v0` | `eval` | `psi0` | `dp` |
| Teleoperated Sonic tasks such as `*Teleop-v0` | `eval-decoupled-wbc` | `psi0_decoupled_wbc` | `dp_decoupled_wbc` |

The Compose `eval` service invokes the first CLI. For a decoupled-WBC evaluation,
override the service entrypoint instead of using the generic one:

```bash
cd third_party/SIMPLE

# MP task
docker compose -f docker-compose.yml -f ../../eval/docker-compose.yml run --rm eval \
  simple/<MP-task>-v0 psi0 train \
  --host=localhost --port=22085 \
  --sim-mode=mujoco_isaac --headless \
  --data-format=lerobot --data-dir=data/<dataset> \
  --eval-dir=/workspace/eval/results \
  --num-episodes=1

# Teleop / industrial task
docker compose -f docker-compose.yml -f ../../eval/docker-compose.yml \
  run --rm --entrypoint 'uv run --no-sync eval-decoupled-wbc' eval \
  simple/G1IndustrialSortingTeleop-v0 psi0_decoupled_wbc train \
  --host=localhost --port=22085 \
  --sim-mode=mujoco_isaac --headless \
  --data-format=lerobot --data-dir=data/<dataset> \
  --eval-dir=/workspace/eval/results \
  --num-episodes=1
```

Host networking makes `localhost` inside these containers reach the policy server on
the host. Start and verify that server before running either command. Replace the Psi0
agent with the corresponding DP agent when evaluating a Diffusion Policy run.

At SIMPLE commit `9a18b604c37c743cf59fb29e504f8ba6f4e1255d`, the local
`simple-teleoperation:latest` image built on 2026-06-01 passes the Compose CUDA test but
does not contain `G1IndustrialSortingTeleop-v0`. Rebuild `isaac-sim` from the current
checkout before evaluating that task. Do not rebuild automatically when disk capacity
is tight; inspect `docker system df` and `df -h .` first.