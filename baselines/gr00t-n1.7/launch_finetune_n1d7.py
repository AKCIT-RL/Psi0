#!/usr/bin/env python3
# Launch finetuning for GR00T N1.7 (Qwen3-VL / Cosmos-Reason2 backbone).
# Drop-in equivalent of baselines/gr00t-n1.6/finetune_gr00t.py for N1.7.

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PRESET_ROOT = SCRIPT_DIR / "presets" / "train"
GR00T_PYTHON = REPO_ROOT / "src/gr00t/.venv/bin/python"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_preset(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate.resolve()
    preset_path = PRESET_ROOT / f"{name_or_path}.yaml"
    if preset_path.exists():
        return preset_path.resolve()
    raise FileNotFoundError(f"Preset not found: {name_or_path}")


def _load_preset(path: Path) -> dict[str, Any]:
    preset = _load_yaml(path)
    extends = preset.pop("extends", None)
    if extends is None:
        return preset

    extend_list = extends if isinstance(extends, list) else [extends]
    merged: dict[str, Any] = {}
    for entry in extend_list:
        parent_path = _resolve_preset(
            str((path.parent / entry).resolve() if not Path(entry).is_absolute() else Path(entry))
        )
        merged = _deep_merge(merged, _load_preset(parent_path))
    return _deep_merge(merged, preset)


def _nproc_from_visible_devices(value: str) -> int:
    return len([part.strip() for part in value.split(",") if part.strip()])


def _flag_name(key: str) -> str:
    return f"--{key.replace('_', '-')}"


def _append_args(cmd: list[str], args_map: dict[str, Any]) -> None:
    for key, value in args_map.items():
        if value is None:
            continue
        flag = _flag_name(key)
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if isinstance(value, list):
            cmd.append(flag)
            cmd.extend(str(item) for item in value)
            continue
        cmd.extend([flag, str(value)])


def _build_cmd(preset: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    runtime = preset.get("runtime", {})
    model = preset.get("model", {})
    dataset = preset.get("dataset", {})
    training = preset.get("training", {})
    augmentation = preset.get("augmentation", {})
    extra_env = preset.get("env", {})

    cuda_devices: str = runtime.get("cuda_visible_devices", "0")
    nproc: int = training.pop("num_gpus", _nproc_from_visible_devices(cuda_devices))
    master_port: int = runtime.get("master_port", 29502)

    python = str(GR00T_PYTHON)
    launcher_script = str(REPO_ROOT / "baselines/gr00t-n1.7/launch_finetune_n1d7_inner.py")

    cmd: list[str] = [
        python,
        "-m", "torch.distributed.run",
        "--nproc_per_node", str(nproc),
        "--master_port", str(master_port),
        launcher_script,
    ]

    _append_args(cmd, {"base_model_path": model.get("base_model_path")})
    _append_args(cmd, {"dataset_path": dataset.get("path")})
    _append_args(cmd, {"embodiment_tag": dataset.get("embodiment_tag")})
    if dataset.get("modality_config_path"):
        _append_args(cmd, {"modality_config_path": dataset.get("modality_config_path")})

    training_args = {k: v for k, v in training.items()}
    training_args["num_gpus"] = nproc
    _append_args(cmd, training_args)

    # Color jitter augmentation
    color_jitter = augmentation.get("color_jitter", {})
    if color_jitter:
        cmd.append("--color-jitter-params")
        for k, v in color_jitter.items():
            cmd.extend([k, str(v)])

    env: dict[str, str] = {**os.environ, **extra_env, "CUDA_VISIBLE_DEVICES": cuda_devices}
    return cmd, env


def main() -> None:
    parser = argparse.ArgumentParser(description="GR00T N1.7 fine-tune launcher")
    parser.add_argument("--preset", default="finetune_simple", help="Preset name or path")
    parser.add_argument("--base-model-path", help="Override model.base_model_path")
    parser.add_argument("--dataset-path", help="Override dataset.path")
    parser.add_argument("--output-dir", help="Override training.output_dir")
    parser.add_argument("--embodiment-tag", help="Override dataset.embodiment_tag")
    parser.add_argument("--cuda-visible-devices", help="Override runtime.cuda_visible_devices")
    parser.add_argument("--num-gpus", type=int, help="Override training.num_gpus")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    args = parser.parse_args()

    preset_path = _resolve_preset(args.preset)
    preset = _load_preset(preset_path)

    # Apply CLI overrides
    if args.base_model_path:
        preset.setdefault("model", {})["base_model_path"] = args.base_model_path
    if args.dataset_path:
        preset.setdefault("dataset", {})["path"] = args.dataset_path
    if args.output_dir:
        preset.setdefault("training", {})["output_dir"] = args.output_dir
    if args.embodiment_tag:
        preset.setdefault("dataset", {})["embodiment_tag"] = args.embodiment_tag
    if args.cuda_visible_devices:
        preset.setdefault("runtime", {})["cuda_visible_devices"] = args.cuda_visible_devices
    if args.num_gpus:
        preset.setdefault("training", {})["num_gpus"] = args.num_gpus

    cmd, env = _build_cmd(preset)

    print("Command:")
    print(" \\\n  ".join(cmd))
    if args.dry_run:
        return

    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
