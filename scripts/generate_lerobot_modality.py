#!/usr/bin/env python3
"""Generate a meta/modality.json for a LeRobot dataset from what the dataset actually contains.

modality.json maps each state/action joint group to a slice of a parquet column. Writing
it by hand -- or copying one from another dataset -- is how you end up training on a
layout that does not match the data: numpy truncates out-of-range slices silently, so the
run converges on garbage and only fails at deployment.

This script inspects the dataset (info.json features + a real parquet file), recognises
the schema, and emits a modality.json that is correct *by construction* for that schema.

WHAT IT CAN AND CANNOT INFER
----------------------------
Dimensions, column names, video keys and annotation keys are read from the data and are
therefore reliable. Physical SEMANTICS (which slice is "left_arm") cannot be read from a
parquet file -- it lives in the robot definition. This script therefore only assigns
semantics when it positively recognises a known schema; for anything else it emits a
scaffold with explicit TODOs instead of guessing. It never invents a layout silently.

Usage:
    python scripts/generate_lerobot_modality.py <dataset_path>              # print
    python scripts/generate_lerobot_modality.py <dataset_path> --write      # write file
    python scripts/generate_lerobot_modality.py <dataset_path> --write --force
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Known schemas
# --------------------------------------------------------------------------------------
# Unitree G1 whole-body, 43 dof: WHOLE_BODY_JOINTS in simple/robots/g1_sonic.py
#   left_leg(6) right_leg(6) waist(3) left_arm(7) right_arm(7) left_hand(7) right_hand(7)
G1_WHOLEBODY_DOF = 43
G1_STATE_BLOCKS = {"legs": 0, "waist": 12, "left_arm": 15, "right_arm": 22,
                   "left_hand": 29, "right_hand": 36}

# psi0 / gr00t SIMPLE deployment layout: 32-dim state, 36-dim action.
# Consumed by gr00t/deploy/gr00t_serve_simple.py and simple/baselines/*_decoupled_wbc.py.
PSI0_STATE = [("left_hand", 7), ("right_hand", 7), ("left_arm", 7),
              ("right_arm", 7), ("rpy", 3), ("height", 1)]
PSI0_ACTION = PSI0_STATE + [("torso_vx", 1), ("torso_vy", 1),
                            ("torso_vyaw", 1), ("target_yaw", 1)]
PSI0_STATE_DIM = sum(d for _, d in PSI0_STATE)    # 32
PSI0_ACTION_DIM = sum(d for _, d in PSI0_ACTION)  # 36

# Action groups that are velocity/rate commands rather than absolute targets.
NON_ABSOLUTE = {"torso_vx", "torso_vy", "torso_vyaw"}


def _entry(start: int, end: int, original_key: str, absolute: bool = True) -> dict:
    return {
        "start": start,
        "end": end,
        "rotation_type": None,
        "absolute": absolute,
        "dtype": "float32",
        "original_key": original_key,
    }


def _consecutive(groups: list[tuple[str, int]], column: str) -> dict:
    out, cursor = {}, 0
    for name, dim in groups:
        out[name] = _entry(cursor, cursor + dim, column, absolute=name not in NON_ABSOLUTE)
        cursor += dim
    return out


# --------------------------------------------------------------------------------------
# Dataset inspection
# --------------------------------------------------------------------------------------
def inspect(dataset_path: Path) -> dict:
    """Read the dataset and return the facts a modality.json must be built from."""
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise SystemExit(f"missing {info_path}")
    info = json.loads(info_path.read_text())

    parquets = sorted(glob.glob(str(dataset_path / "data" / "*" / "*.parquet")))
    if not parquets:
        raise SystemExit(f"no parquet files under {dataset_path / 'data'}")
    df = pd.read_parquet(parquets[0])
    row = df.iloc[0]

    # LeRobot stores shape-[1] features as bare scalars in parquet, not 1-element
    # arrays. The gr00t loader handles that (see LeRobotEpisodeLoader._extract_joint_groups:
    # non-ndarray columns bypass slicing entirely), so treat them as width-1 columns
    # rather than dropping them -- teleop.base_height_command is exactly such a column.
    vector_cols: dict[str, int] = {}
    scalar_cols: list[str] = []
    for col in df.columns:
        arr = np.asarray(row[col])
        if not np.issubdtype(arr.dtype, np.number):
            scalar_cols.append(col)
        elif arr.ndim >= 1:
            vector_cols[col] = int(arr.shape[-1])
        else:
            vector_cols[col] = 1
            scalar_cols.append(col)  # also record it as scalar-typed

    video_keys = [k for k, v in info.get("features", {}).items()
                  if isinstance(v, dict) and v.get("dtype") == "video"]

    return {
        "info": info,
        "df": df,
        "parquet": Path(parquets[0]),
        "vector_cols": vector_cols,
        "scalar_cols": scalar_cols,
        "video_keys": video_keys,
        "n_episodes": info.get("total_episodes"),
    }


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    return np.vstack([np.asarray(x, dtype=np.float32) for x in df[name]])


def detect_action_joint_blocks(df: pd.DataFrame, lag: int = 5) -> dict[str, int] | None:
    """Empirically locate the arm blocks inside a 43-dim `action` column.

    The raw SIMPLE teleop schema does NOT use the same joint ordering for
    `observation.state` and `action` (action carries an unused left-hand slot at 22:29 and
    puts right_arm at 29:36). Rather than assume either layout, correlate each candidate
    action block against the known state blocks and report what actually matches.

    Returns the detected action block offsets, or None if nothing matches confidently.
    """
    state, action = _col(df, "observation.state"), _col(df, "action")
    if state.shape[1] != G1_WHOLEBODY_DOF or action.shape[1] != G1_WHOLEBODY_DOF:
        return None
    a, s = action[:-lag], state[lag:]

    def score(a_base: int, s_base: int, n: int) -> float:
        corrs = []
        for k in range(n):
            aj, sk = a[:, a_base + k], s[:, s_base + k]
            if aj.std() < 1e-6 or sk.std() < 1e-6:
                continue
            corrs.append(abs(np.corrcoef(aj, sk)[0, 1]))
        return float(np.median(corrs)) if corrs else 0.0

    blocks = {"legs": 0, "waist": 12, "left_arm": 15}
    # right_arm sits either directly after left_arm (22) or after a left-hand slot (29).
    # Decide by which candidate tracks the measured right arm, and require a margin over
    # the runner-up. Note the slot at 22 is NOT necessarily zero: single-arm tasks leave
    # the left hand idle, but a handover task commands it.
    cand = {off: score(off, G1_STATE_BLOCKS["right_arm"], 7) for off in (22, 29)}
    best = max(cand, key=cand.get)
    runner_up = max(v for k, v in cand.items() if k != best)
    if cand[best] < 0.8 or (cand[best] - runner_up) < 0.15:
        return None
    blocks["right_arm"] = best
    blocks["left_hand"] = 29 if best == 22 else 22
    blocks["right_hand"] = 36
    blocks["_confidence"] = cand
    return blocks


# --------------------------------------------------------------------------------------
# Schema recognition
# --------------------------------------------------------------------------------------
def build(facts: dict) -> tuple[dict, str, list[str]]:
    """Return (modality_dict, schema_name, notes)."""
    cols = facts["vector_cols"]
    df = facts["df"]
    notes: list[str] = []

    video_key = facts["video_keys"][0] if facts["video_keys"] else None
    if len(facts["video_keys"]) > 1:
        notes.append(
            f"multiple video features {facts['video_keys']}; using '{video_key}'. "
            f"Add the others by hand if the policy consumes more than one view."
        )
    annotation_col = "task_index" if "task_index" in df.columns else None

    def wrap(state: dict, action: dict) -> dict:
        out = {"state": state, "action": action}
        if video_key:
            out["video"] = {"rs_view": {"original_key": video_key}}
        else:
            notes.append("no video feature found in info.json; 'video' section omitted")
        if annotation_col:
            out["annotation"] = {"human.task_description": {"original_key": annotation_col}}
        else:
            notes.append("no 'task_index' column; 'annotation' section omitted")
        return out

    # --- Schema A: already in psi0 layout -------------------------------------------
    if cols.get("states") == PSI0_STATE_DIM and cols.get("action") == PSI0_ACTION_DIM:
        return (
            wrap(_consecutive(PSI0_STATE, "states"), _consecutive(PSI0_ACTION, "action")),
            "psi0 (states=32, action=36)",
            notes,
        )

    # --- Schema B: raw whole-body decoupled-WBC teleop --------------------------------
    if (cols.get("observation.state") == G1_WHOLEBODY_DOF
            and cols.get("action") == G1_WHOLEBODY_DOF
            and "teleop.navigate_command" in cols
            and "teleop.base_height_command" in cols):
        blocks = detect_action_joint_blocks(df)
        notes.append(
            "SCHEMA NOT DIRECTLY TRAINABLE for the psi0/gr00t SIMPLE deployment path. "
            "The 36-dim psi0 action vector needs teleop.base_height_command and "
            "teleop.navigate_command merged into the joint targets, which modality.json "
            "cannot express (it slices one column per group). Convert the dataset first:\n"
            "    python third_party/SIMPLE/scripts/postprocess_psi0_teleop_wbc.py \\\n"
            "        --sim-root <this dataset> --out-dir <converted>\n"
            "then run this generator on the converted dataset."
        )
        if blocks:
            notes.append(
                f"detected action joint blocks by correlation: right_arm at "
                f"{blocks['right_arm']}, unused hand slot at {blocks['left_hand']} "
                f"(median |r| per candidate: "
                f"{ {k: round(v, 3) for k, v in blocks['_confidence'].items()} })"
            )
        else:
            notes.append("could not confirm action joint ordering by correlation")

        # Emit the honest RAW description of this dataset (not a psi0 layout).
        state = {}
        for name, base in G1_STATE_BLOCKS.items():
            dim = 15 if name == "legs" else (3 if name == "waist" else 7)
            if name == "legs":
                state["left_leg"] = _entry(0, 6, "observation.state")
                state["right_leg"] = _entry(6, 12, "observation.state")
                continue
            state[name] = _entry(base, base + dim, "observation.state")
        action = {}
        if blocks:
            action["left_leg"] = _entry(0, 6, "action")
            action["right_leg"] = _entry(6, 12, "action")
            action["waist"] = _entry(12, 15, "action")
            action["left_arm"] = _entry(15, 22, "action")
            action["right_arm"] = _entry(blocks["right_arm"], blocks["right_arm"] + 7, "action")
            action["right_hand"] = _entry(36, 43, "action")
        action["base_height_command"] = _entry(0, 1, "teleop.base_height_command")
        action["navigate_command"] = _entry(0, 4, "teleop.navigate_command", absolute=False)
        return wrap(state, action), "raw whole-body teleop (43 dof) -- NEEDS CONVERSION", notes

    # --- Schema C: unknown ------------------------------------------------------------
    notes.append(
        "UNRECOGNISED SCHEMA -- the sections below are a scaffold, not a working config. "
        "Each candidate column is emitted as a single whole-column group. Split them into "
        "real joint groups yourself; the dims are correct, the SEMANTICS are placeholders."
    )
    state_col = next((c for c in ("observation.state", "states", "state") if c in cols), None)
    action_col = next((c for c in ("action", "actions") if c in cols), None)
    if state_col is None or action_col is None:
        raise SystemExit(
            f"cannot find a state and an action column.\n"
            f"  vector columns: {json.dumps(cols, indent=4)}\n"
            f"Pass a dataset with recognisable state/action columns, or write "
            f"modality.json by hand and check it with scripts/validate_lerobot_modality.py"
        )
    return (
        wrap(
            {"TODO_state_group": _entry(0, cols[state_col], state_col)},
            {"TODO_action_group": _entry(0, cols[action_col], action_col)},
        ),
        f"unknown (state='{state_col}'[{cols[state_col]}], action='{action_col}'[{cols[action_col]}])",
        notes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--write", action="store_true", help="write meta/modality.json")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing modality.json (a .bak copy is kept)")
    args = parser.parse_args()

    facts = inspect(args.dataset_path)
    modality, schema, notes = build(facts)

    print(f"dataset          : {args.dataset_path}")
    print(f"episodes         : {facts['n_episodes']}")
    print(f"sampled parquet  : {facts['parquet'].name}")
    print(f"vector columns   : {json.dumps(facts['vector_cols'])}")
    print(f"video features   : {facts['video_keys']}")
    print(f"detected schema  : {schema}")
    for note in notes:
        print("\n[!] " + note)
    print("\n" + "-" * 70)
    print(json.dumps(modality, indent=4))

    scaffold = any("UNRECOGNISED" in n or "NEEDS CONVERSION" in n or
                   "NOT DIRECTLY TRAINABLE" in n for n in notes)

    if args.write:
        target = args.dataset_path / "meta" / "modality.json"
        if scaffold:
            print(f"\nRefusing to write {target}: the generated config is a scaffold, not a "
                  f"usable layout (see the notes above). Resolve those first.")
            return 1
        if target.exists() and not args.force:
            print(f"\n{target} already exists. Re-run with --force to overwrite "
                  f"(a .bak copy will be kept).")
            return 1
        if target.exists():
            backup = target.with_suffix(".json.bak")
            shutil.copyfile(target, backup)
            print(f"\nexisting config backed up to {backup}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(modality, indent=4))
        print(f"wrote {target}")
        print(f"\nNow verify it against the data:")
        print(f"  python scripts/validate_lerobot_modality.py {args.dataset_path} --expect-psi0")
    elif not scaffold:
        print("\n(dry run -- pass --write to save it to meta/modality.json)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
