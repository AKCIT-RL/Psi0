# Project Instructions

## Project
Psi0 (humanoid VLA) fork. Active work (branch `feat/wla-psi0`): fine-tuned UnifoLM-WLA checkpoint F5 (G1 tote shelf→table), evaluated closed-loop in SIMPLE. Other dirs (`src/psi`, `real`, `baselines`) are upstream Psi0 code.

## Stack
Python (uv), PyTorch, UnifoLM-WLA (`third_party/unifolm-wla`), SIMPLE sim (`third_party/SIMPLE`, runs in Docker with Isaac RTX render), LeRobot v2.1 data.

## Structure
- `docs/wla/` — plan, blocker, `HANDOFF_4090.md` (eval runbook); read before WLA work
- `src/wla_adapter/`, `tests/wla_adapter/` — Psi0↔WLA↔SIMPLE conversion (CPU tests)
- `scripts/wla/` — eval runner, model server, diagnostics
- `~/wla_data` (`WLA_HOME`) — checkpoint, eval roots, assets, outside the repo
- `_scratch/` — git-ignored scratch/logs

## Commands
Env: `cd third_party/unifolm-wla && uv sync`
Adapter tests: `PYTHONPATH=src third_party/unifolm-wla/.venv/bin/python -m pytest tests/wla_adapter -q`
Closed-loop eval: `RUN=<name> SPLIT=val N=1 SEED=0 WANDB_SYNC=0 bash scripts/wla/docker_eval.sh`
GR00T sub-tasks (approach/grasp): see `docs/wla/GR00T_SUBTASK_EVAL.md`; `TASK=approach|grasp RUN=<n> N=1 bash scripts/gr00t/docker_eval_gr00t.sh`, chain: `scripts/gr00t/docker_eval_chain.sh`; copy videos to `./videos/<run>/` with `scripts/gr00t/collect_videos.sh <run>`
Deployable approach→grasp switch (image-only head + optional assist/consensus): `docker_eval_chain8.sh`, config and results in `docs/wla/GR00T_SUBTASK_EVAL.md`; `val3` root (VAL eps 9,12,24) via `scripts/gr00t/make_subroot.py`. Don't edit a runner script while it is running (bash reads it incrementally); copy it instead.

3-phase chain (approach → grasp → carry/place) plan and rules for the user's "loop" command: `docs/wla/PLANO_3_FASES.md` (follow it in order; update its "Registro" table).

## Project-specific constraints
- This machine (RTX 4090, local Docker) overrides the Slurm/cluster rules in AGENTS.md; the rest of AGENTS.md applies.
- Eval must use `SIM_MODE=mujoco_isaac`. MuJoCo/Mesa render is diagnostic only, never report it as the model's result.
- TEST split untouched until the user asks for the final run; never print/commit secrets; no `git push` or commits unless asked.
- Report failures as failures; if Isaac doesn't start or the robot stays still, stop and report, don't tune model/adapter.
- Third-party submodules carry local patches (`docs/wla/patches/`); don't commit inside them.

## Expensive operations
Run a 1-episode eval and its sanity checks before multi-episode or W&B runs.

# Compact instructions
Preserve: objective, decisions/constraints, files changed, validation results, errors, next steps. Discard repetitive exploration.
