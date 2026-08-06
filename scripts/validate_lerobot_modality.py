#!/usr/bin/env python3
"""Validate that a LeRobot dataset's meta/modality.json matches its actual data.

modality.json declares each state/action joint group as a slice of a parquet column.
numpy truncates out-of-range slices silently, so a mismatched modality.json does not
crash training -- it feeds the model short or empty joint groups, converges to a
plausible-looking loss, and only fails at deployment (typically as a robot that
collapses, because the base-height / navigate slots were filled with whatever
happened to sit at those indices).

This script catches that before a training run is launched.

Usage:
    python scripts/validate_lerobot_modality.py <dataset_path> [<dataset_path> ...]
    python scripts/validate_lerobot_modality.py --expect-psi0 <dataset_path>

Exit code 0 if every dataset is consistent, 1 otherwise.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


DEFAULT_COLUMN = {"state": "observation.state", "action": "action"}

# The 36-dim action / 32-dim state layout the psi0 + gr00t SIMPLE deployment path
# assumes (see gr00t/deploy/gr00t_serve_simple.py::_action_to_psi_format and
# simple/baselines/*_decoupled_wbc.py). Only checked with --expect-psi0.
PSI0_STATE_DIMS = {
    "left_hand": 7,
    "right_hand": 7,
    "left_arm": 7,
    "right_arm": 7,
    "rpy": 3,
    "height": 1,
}
PSI0_ACTION_DIMS = {
    **PSI0_STATE_DIMS,
    "torso_vx": 1,
    "torso_vy": 1,
    "torso_vyaw": 1,
    "target_yaw": 1,
}


def _first_parquet(dataset_path: Path) -> Path:
    matches = sorted(glob.glob(str(dataset_path / "data" / "*" / "*.parquet")))
    if not matches:
        raise FileNotFoundError(f"no parquet files under {dataset_path / 'data'}")
    return Path(matches[0])


def _col_dim(value) -> int | None:
    """Width of a parquet cell, or None if it is non-numeric.

    LeRobot stores shape-[1] features as bare scalars rather than 1-element arrays
    (e.g. teleop.base_height_command). The gr00t loader passes such columns through
    without slicing (LeRobotEpisodeLoader._extract_joint_groups), so their effective
    width is 1 -- not "invalid".
    """
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.number):
        return None
    return int(arr.shape[-1]) if arr.ndim >= 1 else 1


def validate(dataset_path: Path, expect_psi0: bool = False) -> list[str]:
    """Return a list of problems; empty means the dataset is consistent."""
    problems: list[str] = []

    modality_path = dataset_path / "meta" / "modality.json"
    if not modality_path.exists():
        return [f"missing {modality_path}"]
    modality = json.loads(modality_path.read_text())

    parquet_path = _first_parquet(dataset_path)
    df = pd.read_parquet(parquet_path)
    row = df.iloc[0]

    stats_path = dataset_path / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    if stats is None:
        problems.append("meta/stats.json is missing (gr00t requires it for normalization)")

    # --- state / action slices -------------------------------------------------
    for modality_type in ("state", "action"):
        groups = modality.get(modality_type, {})
        if not groups:
            problems.append(f"modality.json has no '{modality_type}' section")
            continue

        for name, info in groups.items():
            start, end = info["start"], info["end"]
            column = info.get("original_key") or DEFAULT_COLUMN[modality_type]
            declared = end - start

            if column not in df.columns:
                problems.append(
                    f"{modality_type}.{name}: column '{column}' not in parquet "
                    f"(available: {sorted(df.columns)})"
                )
                continue

            actual_dim = _col_dim(row[column])
            if actual_dim is None:
                problems.append(f"{modality_type}.{name}: column '{column}' is a scalar")
                continue

            if start < 0 or end > actual_dim or declared <= 0:
                got = max(0, min(end, actual_dim) - start)
                problems.append(
                    f"{modality_type}.{name}: declares {column}[{start}:{end}] "
                    f"({declared} dims) but '{column}' has {actual_dim} dims "
                    f"-> would silently yield {got} dims"
                )
                continue

            if stats is not None:
                if column not in stats:
                    problems.append(
                        f"{modality_type}.{name}: '{column}' has no entry in stats.json"
                    )
                elif len(stats[column]["mean"]) != actual_dim:
                    problems.append(
                        f"{modality_type}.{name}: stats.json['{column}'] has "
                        f"{len(stats[column]['mean'])} dims but the parquet column has "
                        f"{actual_dim} -- stats.json is stale"
                    )

    # --- video / annotation ----------------------------------------------------
    info_path = dataset_path / "meta" / "info.json"
    features = json.loads(info_path.read_text()).get("features", {}) if info_path.exists() else {}
    for name, entry in modality.get("video", {}).items():
        original_key = entry.get("original_key", f"observation.images.{name}")
        if features and original_key not in features:
            problems.append(
                f"video.{name}: original_key '{original_key}' not in info.json features "
                f"(available: {[k for k in features if 'image' in k or 'rgb' in k]})"
            )
    for name, entry in modality.get("annotation", {}).items():
        original_key = entry.get("original_key", name)
        if original_key not in df.columns:
            problems.append(f"annotation.{name}: column '{original_key}' not in parquet")

    # --- degenerate (constant) action channels ---------------------------------
    # Not fatal, but a constant channel means the policy can never learn to move it.
    if stats is not None:
        for name, info in modality.get("action", {}).items():
            column = info.get("original_key") or DEFAULT_COLUMN["action"]
            if column not in stats:
                continue
            lo = np.asarray(stats[column]["min"][info["start"] : info["end"]], dtype=float)
            hi = np.asarray(stats[column]["max"][info["start"] : info["end"]], dtype=float)
            if lo.size and np.allclose(lo, hi):
                problems.append(
                    f"NOTE action.{name}: constant across the dataset (value {lo[0]:.4g}); "
                    f"the policy cannot learn to vary it"
                )

    # --- optional psi0 layout check --------------------------------------------
    if expect_psi0:
        for modality_type, expected in (("state", PSI0_STATE_DIMS), ("action", PSI0_ACTION_DIMS)):
            declared = {k: v["end"] - v["start"] for k, v in modality.get(modality_type, {}).items()}
            if declared != expected:
                problems.append(
                    f"--expect-psi0: {modality_type} layout is {declared}, expected {expected}"
                )
            if list(modality.get(modality_type, {})) != list(expected):
                # Not a correctness problem: training concatenates groups in
                # modality.json order and inference re-splits them in that same order
                # (both read the order from the processor config saved in the
                # checkpoint), while the deploy server assembles the psi0 vector by
                # group NAME (gr00t_serve_simple.py::_action_to_psi_format). Order is
                # therefore self-consistent. Reported only as a readability note.
                problems.append(
                    f"NOTE {modality_type} key order is "
                    f"{list(modality.get(modality_type, {}))}, conventionally "
                    f"{list(expected)}; harmless, but easier to review if it matches"
                )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_paths", nargs="+", type=Path)
    parser.add_argument(
        "--expect-psi0",
        action="store_true",
        help="also assert the 32-dim state / 36-dim action psi0 layout used by the SIMPLE deploy path",
    )
    args = parser.parse_args()

    failed = False
    for dataset_path in args.dataset_paths:
        print(f"=== {dataset_path}")
        try:
            problems = validate(dataset_path, expect_psi0=args.expect_psi0)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            failed = True
            continue

        hard = [p for p in problems if not p.startswith("NOTE")]
        for problem in problems:
            print(f"  {'note: ' if problem.startswith('NOTE') else 'FAIL: '}{problem.removeprefix('NOTE ')}")
        if hard:
            failed = True
        elif not problems:
            print("  OK")
        else:
            print("  OK (with notes)")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
