#!/usr/bin/env python3
"""Turn the SIMPLE teleop .zip archives into validated, trainable psi0 datasets.

This automates the manual runbook (docs/runbook_modality.md) end to end: extract,
locate the real dataset root, detect the schema, convert when needed, generate
modality.json, validate, sanity-check, and only then reclaim disk.

WHY IT IS NOT A THREE-LINE FOR LOOP
-----------------------------------
The archives are not uniform. Assuming "<name>/level-0/" -- the shape you get from
the Handover archive -- silently produces the wrong thing for six of the ten:

  <name>/level-0/           BendPickAndPlace, CloseDoor, Handover, XMovePick
  <name>/                   TabletopGraspMP, LocomotionPickBetweenTables.zip.1
  nested .zip archives      OpenOven (2 inner), OpenFaucet (3 inner)
  deep, renamed root        LocomotionPickBetweenTables.zip -> teleop_G1_wholebody_
                            loco_between_tables/teleop_decoupled_wbc/simple/...Sonic-v0/level-0
  two datasets per archive  XMoveBendPick -> teleop_decoupled_wbc/  and  teleop_decoupled_wbc3/

Worse, OpenOven's two inner archives both unpack to the *same* path
("G1WholebodyOpenOvenTeleop-v0/level-0"), so extracting them into one directory makes
the second silently clobber the first. That is exactly what happened by hand: the
resulting directory holds 104 episodes from OpenOven2 and none from OpenOven3.

So: never build the path, always *find* it (every directory holding meta/info.json),
and give every inner archive its own extraction directory.

SAFETY
------
Nothing is deleted until its successor exists and has been proven good. The raw
extracted tree is removed only after the converted dataset passes both
validate_lerobot_modality.py --expect-psi0 and the physical sanity check. The .zip
archives are never touched -- they remain the only reproducible source.

Usage:
    python scripts/prepare_simple_datasets.py --list
    python scripts/prepare_simple_datasets.py                    # all untrained
    python scripts/prepare_simple_datasets.py --only G1WholebodyCloseDoorTeleop-v0
    python scripts/prepare_simple_datasets.py --include-trained --force
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

# Directory layout (all relative to the project root unless overridden on the CLI).
ZIP_DIR = PROJECT_DIR / "data" / "simple" / "simple-teleop"
RAW_DIR = PROJECT_DIR / "data" / "simple" / "simple-teleop"
OUT_DIR = PROJECT_DIR / "data" / "simple" / "simple-converted"
STAGING_DIR = PROJECT_DIR / "data" / "simple" / ".staging"

GENERATOR = PROJECT_DIR / "scripts" / "generate_lerobot_modality.py"
VALIDATOR = PROJECT_DIR / "scripts" / "validate_lerobot_modality.py"
CONV_WBC = PROJECT_DIR / "third_party" / "SIMPLE" / "scripts" / "postprocess_psi0_teleop_wbc.py"
CONV_SONIC = PROJECT_DIR / "third_party" / "SIMPLE" / "scripts" / "postprocess_psi0_sonic.py"

# A dataset directory name looks like "G1WholebodyOpenOvenTeleop2-v0".
ENV_NAME_RE = re.compile(r"^G1\w*-v\d+$")

# Unitree G1 whole-body joint count, same constant the generator checks against.
G1_WHOLEBODY_DOF = 43

# Datasets already fine-tuned, skipped unless --include-trained.
#
# "G1WholebodyOpenOvenTeleop2-v0" is listed because checkpoints/..._openoven was trained
# on the by-hand extraction of OpenOven.zip, whose 104 episodes came entirely from the
# OpenOven2 inner archive (OpenOven3 was overwritten). OpenOven3's ~45 episodes have
# never been trained on, so they are NOT excluded here.
ALREADY_TRAINED = {
    "G1WholebodyHandoverTeleop-v0",
    "G1WholebodyOpenOvenTeleop-v0",
    "G1WholebodyOpenOvenTeleop2-v0",
}

# Dataset directory name -> run slug used for checkpoints/gr00t_n1d7_finetune_output_<slug>.
# Explicit rather than derived: the existing runs use inconsistent conventions
# ("openoven" but "bend_pick_and_place", "xmove_pick" but "tabletop_grasp_mp") and no
# single camel-to-snake rule reproduces all of them. camel_to_snake() is the fallback
# for names not listed here.
RUN_SLUGS = {
    "G1WholebodyHandoverTeleop-v0": "handover",
    "G1WholebodyOpenOvenTeleop-v0": "openoven",
    "G1WholebodyOpenOvenTeleop2-v0": "openoven2",
    "G1WholebodyOpenOvenTeleop3-v0": "openoven3",
    "G1WholebodyOpenFaucetTeleop-v0": "open_faucet",
    "G1WholebodyOpenFaucetTeleop2-v0": "open_faucet2",
    "G1WholebodyOpenFaucetTeleop3-v0": "open_faucet3",
    "G1WholebodyBendPickAndPlaceTeleop-v0": "bend_pick_and_place",
    "G1WholebodyCloseDoorTeleop-v0": "close_door",
    "G1WholebodyTabletopGraspMP-v0": "tabletop_grasp_mp",
    "G1WholebodyXMovePickTeleop-v0": "xmove_pick",
    "G1WholebodyLocomotionPickBetweenTablesTeleop-v0": "loco_between_tables",
    "G1WholebodyLocomotionPickBetweenTablesSonic-v0": "loco_between_tables_sonic",
    "G1WholebodyXMoveBendPickTeleop-v0__teleop_decoupled_wbc": "xmove_bend_pick_wbc",
    "G1WholebodyXMoveBendPickTeleop-v0__teleop_decoupled_wbc3": "xmove_bend_pick_wbc3",
    "G1IndustrialSortingTeleop-psi0": "industrial_sorting",
    "G1IndustrialScrewdriverToToteTeleop-v0": "screwdriver_to_tote",
    "G1IndustrialScrewdriverToToteTeleop-v0-raw": "screwdriver_to_tote_raw",
    # Same 50 trajectories in both, bit-for-bit identical actions; only the rendered
    # video differs. Kept as two runs on purpose, to compare the visual domains.
    "G1WholebodyLocomotionPickTotesShelfToTableTeleop__raw": "totes_shelf_to_table_raw",
    "G1WholebodyLocomotionPickTotesShelfToTableTeleop__render": "totes_shelf_to_table_render",
    # The dataset author's own psi0 conversion, published 2026-08-09. Differs from what
    # postprocess_psi0_teleop_wbc.py produces: the right-hand finger block is permuted
    # [4,5,6,0,1,2,3] (thumb moved to the front) and states.rpy differs by up to 0.12 rad.
    # See the note in docs/runbook_modality.md about the unreordered right hand.
    "G1WholebodyLocomotionPickTotesShelfToTableTeleopPsi0": "totes_shelf_to_table_psi0",
}


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def log(msg: str, indent: int = 0) -> None:
    print(("  " * indent) + msg, flush=True)


def camel_to_snake(name: str) -> str:
    """G1WholebodyBendPickAndPlaceTeleop-v0 -> bend_pick_and_place (fallback only)."""
    core = re.sub(r"^G1(Wholebody)?", "", name)
    core = re.sub(r"(Teleop\d*|Sonic|MP)?(-v\d+|-psi0)$", "", core)
    core = re.sub(r"(?<!^)(?=[A-Z])", "_", core).lower()
    core = re.sub(r"[^a-z0-9]+", "_", core)
    return re.sub(r"_+", "_", core).strip("_") or name.lower()


def run_slug(name: str) -> str:
    return RUN_SLUGS.get(name) or camel_to_snake(name)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def run_cmd(cmd: list[str], cwd: Path = PROJECT_DIR,
            extra_env: dict[str, str] | None = None) -> tuple[int, str]:
    """Run a subprocess, stream nothing, return (returncode, combined output)."""
    env = None
    if extra_env:
        env = {**os.environ, **extra_env}
    proc = subprocess.run(
        [str(c) for c in cmd], cwd=str(cwd), text=True, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


# --------------------------------------------------------------------------------------
# Extraction and root discovery
# --------------------------------------------------------------------------------------

def extract_recursive(archive: Path, dest: Path, depth: int = 0) -> None:
    """Extract `archive` into `dest`, then extract any nested archives it contains.

    Each nested archive gets its own sibling directory named after the archive stem, so
    two inner archives that unpack to the same relative path cannot overwrite each other.
    """
    if depth > 4:
        raise RuntimeError(f"nested archives deeper than 4 levels: {archive}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        # Refuse absolute paths and .. traversal rather than trusting the archive.
        for member in zf.namelist():
            if member.startswith("/") or ".." in Path(member).parts:
                raise RuntimeError(f"unsafe path in {archive.name}: {member}")
        zf.extractall(dest)

    for inner in sorted(dest.rglob("*.zip")):
        if not inner.is_file():
            continue
        log(f"nested archive: {inner.relative_to(dest)}", 2)
        extract_recursive(inner, inner.parent / inner.stem, depth + 1)
        inner.unlink()  # the copy inside the staging tree; the original is untouched


def find_roots(base: Path) -> list[Path]:
    """Every directory that holds meta/info.json -- i.e. every real LeRobot dataset."""
    return sorted({p.parent.parent for p in base.rglob("meta/info.json")})


# Path segments that describe structure, not identity. A dataset called "level-0" or
# "raw" tells you nothing about which recording it is.
GENERIC_SEGMENTS = {"level-0", "level-1", "level-2", "data", "meta", "videos",
                    "raw", "render", "simple", "train", "test"}


def base_name(root: Path) -> str:
    """Nearest ancestor segment that looks like an env name ('.../<env>/level-0' -> env).

    Falling back to root.name alone is not enough: a repo laid out as raw/level-0 and
    render/level-0 would name both datasets "level-0". So when no segment matches the env
    pattern, walk up to the first segment that actually identifies something, and let
    name_roots() disambiguate the pair with the raw/render segment.
    """
    parts = [root.name, *reversed(root.parts[:-1])]
    for part in parts:
        if ENV_NAME_RE.match(part):
            return part
    for part in parts:
        if part.lower() not in GENERIC_SEGMENTS:
            return part
    return root.name


def name_roots(roots: list[Path], staging: Path) -> dict[Path, str]:
    """Assign a unique, deterministic name to each discovered dataset root.

    On a collision (two roots in the same archive sharing a base name), disambiguate with
    the first path segment where the two differ. If that segment is itself an env name --
    the OpenOven case, where the discriminator is the inner archive name -- use it as the
    name outright instead of concatenating.
    """
    names: dict[Path, str] = {r: base_name(r) for r in roots}
    by_name: dict[str, list[Path]] = {}
    for root, name in names.items():
        by_name.setdefault(name, []).append(root)

    for name, group in by_name.items():
        if len(group) < 2:
            continue
        rels = {r: r.relative_to(staging).parts for r in group}
        # First index at which the group's paths diverge.
        idx = 0
        while len({rels[r][idx] if idx < len(rels[r]) else None for r in group}) == 1:
            idx += 1
        for root in group:
            disc = rels[root][idx]
            names[root] = disc if ENV_NAME_RE.match(disc) else f"{name}__{disc}"
    return names


# --------------------------------------------------------------------------------------
# Pipeline steps
# --------------------------------------------------------------------------------------

SONIC_SOURCE = PROJECT_DIR / "third_party" / "SIMPLE" / "src" / "simple" / "robots" / "g1_sonic.py"


def build_sonic_shim(cache: Path) -> Path:
    """Make `from simple.robots.g1_sonic import WHOLE_BODY_JOINTS` importable.

    The sonic post-processor needs one constant from that module, but importing it drags
    in mujoco, curobo and the rest of the simulator -- which is not, and should not be,
    installed in the training venv.

    Hard-coding the joint list here would be the very mistake this pipeline exists to
    prevent (a layout copied by hand that no later step contradicts). So instead: parse
    the real source with ast, keep only the module-level *_JOINTS assignments, and re-emit
    them as a minimal module. Single source of truth, and it fails loudly if the upstream
    file is restructured.
    """
    import ast

    if not SONIC_SOURCE.exists():
        raise RuntimeError(f"cannot build the sonic shim: {SONIC_SOURCE} is missing")

    tree = ast.parse(SONIC_SOURCE.read_text())
    kept = [node for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.endswith("_JOINTS")]
    if not kept:
        raise RuntimeError(f"no *_JOINTS assignments found in {SONIC_SOURCE}")

    source = ast.unparse(ast.Module(body=kept, type_ignores=[]))
    namespace: dict = {}
    exec(compile(source, str(SONIC_SOURCE), "exec"), namespace)  # noqa: S102
    joints = namespace.get("WHOLE_BODY_JOINTS")
    if not isinstance(joints, list) or len(joints) != G1_WHOLEBODY_DOF:
        raise RuntimeError(
            f"extracted WHOLE_BODY_JOINTS is not a list of {G1_WHOLEBODY_DOF} names "
            f"(got {type(joints).__name__} of length {len(joints) if joints else 0}) -- "
            f"{SONIC_SOURCE} changed shape; fix the shim rather than guessing")

    root = cache / "simple"
    (root / "robots").mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text("")
    (root / "robots" / "__init__.py").write_text("")
    (root / "robots" / "g1_sonic.py").write_text(
        f'"""Generated by scripts/prepare_simple_datasets.py from\n'
        f'{SONIC_SOURCE.relative_to(PROJECT_DIR)} -- joint name constants only.\n'
        f'Do not edit; it is rewritten on every run."""\n\n' + source + "\n")
    return cache


def read_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def video_keys(info: dict) -> list[str]:
    return [k for k in info.get("features", {}) if k.startswith("observation.images.")]


def detect_schema(dataset: Path) -> tuple[str, str]:
    """Run the generator in read-only mode and return (schema line, full output)."""
    code, out = run_cmd([sys.executable, GENERATOR, dataset])
    match = re.search(r"^detected schema\s*:\s*(.+)$", out, re.M)
    if not match:
        raise RuntimeError(f"generator produced no 'detected schema' line (exit {code}):\n{out}")
    return match.group(1).strip(), out


def convert(root: Path, out: Path, info: dict, sonic: bool) -> None:
    """Run the appropriate SIMPLE post-processor. Raises on failure."""
    script = CONV_SONIC if sonic else CONV_WBC
    keys = video_keys(info)
    if not keys:
        raise RuntimeError(f"{root}: info.json declares no observation.images.* feature")
    if len(keys) > 1:
        log(f"WARNING: several video features {keys}; using {keys[0]}", 2)

    episodes = int(info.get("total_episodes", 0))
    cmd = [sys.executable, script, "--sim-root", root, "--out-dir", out,
           "--video-key", keys[0]]
    extra_env = None
    if sonic:
        # The sonic post-processor defaults to --total_episodes 99 and truncates
        # silently past that. Pass the real count so nothing is dropped.
        cmd += ["--total_episodes", str(max(episodes, 1))]
        shim = build_sonic_shim(PROJECT_DIR / "cache" / "sonic_shim")
        extra_env = {"PYTHONPATH": str(shim)}

    log(f"converting with {script.name} ({episodes} episodes, video key {keys[0]})", 2)
    code, out_txt = run_cmd(cmd, extra_env=extra_env)
    if code != 0:
        raise RuntimeError(f"conversion failed (exit {code}):\n{out_txt}")
    for line in out_txt.splitlines():
        if line.strip().startswith(("right arm", "left hand", "layout check", "Done:")):
            log(line.strip(), 3)


def write_modality(dataset: Path) -> None:
    code, out = run_cmd([sys.executable, GENERATOR, dataset, "--write", "--force"])
    if code != 0:
        raise RuntimeError(f"modality generation failed (exit {code}):\n{out}")


def validate(dataset: Path) -> list[str]:
    """Gate: raises unless the validator exits 0. Returns the note lines."""
    code, out = run_cmd([sys.executable, VALIDATOR, dataset, "--expect-psi0"])
    notes = [l.strip() for l in out.splitlines() if l.strip().startswith("note:")]
    if code != 0:
        raise RuntimeError(f"validation FAILED (exit {code}):\n{out}")
    for note in notes:
        log(note, 3)
    return notes


# Torso height in metres. PLAUSIBLE is where a G1 normally operates; a value outside
# IMPLAUSIBLE cannot be a height at all (it is a joint angle, i.e. a broken conversion).
HEIGHT_PLAUSIBLE = (0.40, 0.80)
HEIGHT_IMPLAUSIBLE = (0.30, 0.90)
# Fraction of frames allowed outside IMPLAUSIBLE before it stops being an outlier and
# starts being a systematic error.
HEIGHT_OUTLIER_BUDGET = 0.02


def sanity_check(dataset: Path) -> tuple[dict, list[str]]:
    """The physical checks from runbook section 5, as a gate rather than a suggestion.

    Catches a conversion that produced structurally valid but physically wrong data --
    e.g. a height channel holding joint angles instead of metres.

    Distinguishes a broken conversion from a strange episode. A conversion error moves
    the whole distribution (the median leaves the plausible band); a bad recording leaves
    a handful of frames out of range. The first fails; the second is returned as a
    warning naming the episodes, because refusing a 39-episode dataset over 33 frames
    would just teach you to pass --force.

    Returns (measurements, warnings) and raises on a real failure.
    """
    files = sorted(glob.glob(str(dataset / "data" / "*" / "*.parquet")))
    if not files:
        raise RuntimeError(f"{dataset}: no parquet files")

    def col(frame: pd.DataFrame, name: str) -> np.ndarray:
        return np.vstack([np.asarray(x, dtype=np.float32) for x in frame[name]])

    worst_feedback = 0.0
    actions, states = [], []
    odd_episodes: list[str] = []
    for path in files:
        frame = pd.read_parquet(path)
        a, s = col(frame, "action"), col(frame, "states")
        # Per episode: concatenating first would compare one episode's last frame with
        # the next episode's first and raise a false alarm.
        worst_feedback = max(worst_feedback, float(np.abs(s[1:, 31] - a[:-1, 31]).max()))
        h = a[:, 31]
        if (h < HEIGHT_IMPLAUSIBLE[0]).any() or (h > HEIGHT_IMPLAUSIBLE[1]).any():
            odd_episodes.append(Path(path).name)
        actions.append(a)
        states.append(s)
    action = np.concatenate(actions)
    state = np.concatenate(states)
    height = action[:, 31]
    out_of_band = float(np.mean((height < HEIGHT_IMPLAUSIBLE[0]) | (height > HEIGHT_IMPLAUSIBLE[1])))

    result = {
        "episodes": len(files),
        "frames": int(len(action)),
        "action_dim": int(action.shape[1]),
        "states_dim": int(state.shape[1]),
        "height_min": float(height.min()),
        "height_median": float(np.median(height)),
        "height_max": float(height.max()),
        "height_frames_out_of_band": out_of_band,
        "height_odd_episodes": odd_episodes,
        "torso_vx_absmax": float(np.abs(action[:, 32]).max()),
        "torso_vy_absmax": float(np.abs(action[:, 33]).max()),
        "height_feedback_worst": worst_feedback,
        "nan_or_inf": bool(np.isnan(action).any() or np.isinf(action).any()
                           or np.isnan(state).any() or np.isinf(state).any()),
    }

    problems, warnings = [], []
    if result["action_dim"] != 36 or result["states_dim"] != 32:
        problems.append(f"wrong dims: action={result['action_dim']} states={result['states_dim']}")
    if result["nan_or_inf"]:
        problems.append("NaN or Inf present")
    if not (HEIGHT_IMPLAUSIBLE[0] <= result["height_median"] <= HEIGHT_IMPLAUSIBLE[1]):
        problems.append(
            f"median action.height is {result['height_median']:.3f} m, outside "
            f"{HEIGHT_IMPLAUSIBLE[0]}..{HEIGHT_IMPLAUSIBLE[1]} -- that is a joint angle, "
            f"not a height: the conversion is wrong")
    elif out_of_band > HEIGHT_OUTLIER_BUDGET:
        problems.append(
            f"{out_of_band:.1%} of frames have action.height outside "
            f"{HEIGHT_IMPLAUSIBLE[0]}..{HEIGHT_IMPLAUSIBLE[1]} m -- too many to be outliers")
    elif odd_episodes:
        warnings.append(
            f"action.height leaves {HEIGHT_IMPLAUSIBLE[0]}..{HEIGHT_IMPLAUSIBLE[1]} m in "
            f"{len(odd_episodes)}/{len(files)} episode(s) ({out_of_band:.2%} of frames, "
            f"min {result['height_min']:.3f}): {', '.join(odd_episodes[:5])}"
            f"{' ...' if len(odd_episodes) > 5 else ''}")
    elif not (HEIGHT_PLAUSIBLE[0] <= result["height_min"]
              and result["height_max"] <= HEIGHT_PLAUSIBLE[1]):
        warnings.append(
            f"action.height spans {result['height_min']:.3f}..{result['height_max']:.3f} m, "
            f"wider than the usual {HEIGHT_PLAUSIBLE[0]}..{HEIGHT_PLAUSIBLE[1]}")
    if max(result["torso_vx_absmax"], result["torso_vy_absmax"]) > 5.0:
        problems.append(
            f"torso velocity out of range (|vx|max={result['torso_vx_absmax']:.2f}, "
            f"|vy|max={result['torso_vy_absmax']:.2f})")
    if worst_feedback > 1e-3:
        problems.append(f"state.height[t] != action.height[t-1] (worst {worst_feedback:.6f})")

    log(f"episodes={result['episodes']} frames={result['frames']} "
        f"height={result['height_min']:.3f}..{result['height_max']:.3f} "
        f"(median {result['height_median']:.3f}) "
        f"feedback={worst_feedback:.2e} nan={result['nan_or_inf']}", 3)
    for warning in warnings:
        log(f"WARNING: {warning}", 3)
    if problems:
        raise RuntimeError("sanity check failed:\n  - " + "\n  - ".join(problems))
    return result, warnings


# --------------------------------------------------------------------------------------
# Per-dataset driver
# --------------------------------------------------------------------------------------

def prepare_root(root: Path, name: Path | str, source: Path, staging: Path,
                 raw_dir: Path, out_dir: Path, args, source_kind: str = "zip") -> dict:
    """Take one discovered dataset root all the way to a validated psi0 dataset."""
    name = str(name)
    log(f"--- {name}", 1)
    log(f"root in {source_kind}: {root.relative_to(staging)}", 2)

    final = out_dir / name
    if (final / "PROVENANCE.json").exists() and not args.force:
        log("already prepared (PROVENANCE.json present) -- skipping", 2)
        return {"dataset": name, "status": "skipped", "path": str(final)}

    # 1. Flatten the root into the raw area, dropping the level-0 indirection.
    raw = raw_dir / name
    if raw.resolve() == root.resolve():
        # A directory source that already sits where the raw tree belongs: leave it be.
        log(f"raw dataset in place at {raw.relative_to(PROJECT_DIR)} "
            f"({human(dir_size(raw))})", 2)
    else:
        if raw.exists():
            if not args.force:
                raise RuntimeError(f"{raw} already exists; pass --force to replace it")
            shutil.rmtree(raw)
        raw.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root), str(raw))
        log(f"raw dataset -> {raw.relative_to(PROJECT_DIR)} ({human(dir_size(raw))})", 2)

    info = read_info(raw)
    try:
        return _prepare_validated(root, name, source, staging, raw, final, info, args,
                                  source_kind)
    except Exception:
        # A half-written output directory looks like a ready dataset at a glance. Remove
        # it; the raw tree stays behind on purpose, so the failure can be investigated.
        if final.exists() and not (final / "PROVENANCE.json").exists():
            shutil.rmtree(final)
        raise


def _prepare_validated(root: Path, name: str, source: Path, staging: Path,
                       raw: Path | None, final: Path, info: dict, args,
                       source_kind: str = "zip") -> dict:
    # 2. Identify the schema. This decides everything downstream.
    schema, _ = detect_schema(raw)
    log(f"detected schema: {schema}", 2)

    if "NEEDS CONVERSION" in schema:
        # The Sonic recordings need the other post-processor.
        sonic = "Sonic" in name or "Sonic" in str(root)
        if final.exists():
            shutil.rmtree(final)
        final.mkdir(parents=True, exist_ok=True)
        convert(raw, final, info, sonic=sonic)
        converter = (CONV_SONIC if sonic else CONV_WBC).name
        # The converter already writes a byte-identical meta/modality.json (runbook 3).
    elif schema.startswith("psi0"):
        # Already trainable: it *is* the deliverable, so it moves rather than converts.
        if final.exists():
            if not args.force:
                raise RuntimeError(f"{final} already exists; pass --force")
            shutil.rmtree(final)
        log(f"already psi0: moving {raw.relative_to(PROJECT_DIR)} -> "
            f"{final.relative_to(PROJECT_DIR)} (moved, not copied — nothing is lost)", 2)
        shutil.move(str(raw), str(final))
        raw = None
        converter = None
        write_modality(final)
    else:
        raise RuntimeError(
            f"unrecognised schema '{schema}' -- investigate by hand "
            f"(see 'Schema desconhecido' in docs/runbook_modality.md). "
            f"Raw data left at {raw}")

    # 3. The gates. Nothing is deleted before both pass.
    notes = validate(final)
    checks, warnings = sanity_check(final)

    # 4. Record provenance, then reclaim the raw tree.
    provenance = {
        "dataset": name,
        "run_slug": run_slug(name),
        "source_kind": source_kind,
        "source": str(source.relative_to(PROJECT_DIR)),
        "source_sha256": (sha256(source)
                          if source_kind == "zip" and not args.no_hash else None),
        "source_bytes": source.stat().st_size if source_kind == "zip" else None,
        "root_inside_source": str(root.relative_to(staging)),
        "detected_schema": schema,
        "converter": converter,
        "validator_notes": notes,
        "sanity": checks,
        "sanity_warnings": warnings,
        "raw_deleted": False,
        "prepared_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "prepared_on": socket.gethostname(),
    }

    # A .zip source stays on disk, so the extracted tree is redundant and goes. A
    # directory source is the only copy there is -- deleting it would destroy the data,
    # so it is always kept regardless of --keep-raw.
    if raw is not None and not args.keep_raw and source_kind == "zip":
        freed = dir_size(raw)
        shutil.rmtree(raw)
        provenance["raw_deleted"] = True
        log(f"raw tree removed ({human(freed)} reclaimed); the .zip is untouched", 2)
    elif raw is not None and source_kind == "directory":
        log(f"source directory kept at {raw.relative_to(PROJECT_DIR)} "
            f"(no archive to restore it from)", 2)

    (final / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    log(f"OK -> {final.relative_to(PROJECT_DIR)} ({human(dir_size(final))})", 2)
    return {"dataset": name, "status": "ok", "path": str(final),
            "run_slug": run_slug(name), "episodes": checks["episodes"],
            "warnings": warnings}


def process_directory(path: Path, args) -> list[dict]:
    """Prepare a dataset tree that is already extracted -- e.g. pulled from the Hub.

    Same discovery as an archive: find every meta/info.json rather than assuming a
    layout. Nothing is unpacked, and the source is never deleted, because unlike a .zip
    there is no second copy to restore it from.
    """
    log("")
    log(f"=== {path.name}/ (directory, {human(dir_size(path))})")
    if not path.is_dir():
        log("not a directory -- skipping", 1)
        return [{"dataset": path.name, "status": "no-dataset"}]

    roots = find_roots(path)
    if not roots:
        log("no meta/info.json anywhere under this path -- skipping", 1)
        return [{"dataset": path.name, "status": "no-dataset"}]

    # Names resolve against the parent, so a tree that *is* the dataset root keeps its
    # own directory name.
    base = path.parent
    names = name_roots(roots, base)
    log(f"{len(roots)} dataset root(s): {', '.join(sorted(names.values()))}", 1)

    results: list[dict] = []
    for root in roots:
        name = names[root]
        if name in ALREADY_TRAINED and not args.include_trained:
            log(f"--- {name}", 1)
            log("already fine-tuned -- skipping (--include-trained to force)", 2)
            results.append({"dataset": name, "status": "already-trained"})
            continue
        if args.only and name not in args.only:
            results.append({"dataset": name, "status": "not-selected"})
            continue
        try:
            results.append(prepare_root(root, name, path, base,
                                        Path(args.raw_dir), Path(args.out_dir), args,
                                        source_kind="directory"))
        except Exception as exc:
            log(f"FAILED: {exc}", 2)
            results.append({"dataset": name, "status": "failed", "error": str(exc)})
    return results


def process_archive(archive: Path, args) -> list[dict]:
    log("")
    log(f"=== {archive.name} ({human(archive.stat().st_size)})")
    staging = Path(args.staging_dir) / archive.name
    if staging.exists():
        shutil.rmtree(staging)

    results: list[dict] = []
    try:
        log("extracting...", 1)
        extract_recursive(archive, staging)
        roots = find_roots(staging)
        if not roots:
            log("no meta/info.json anywhere in this archive -- skipping", 1)
            return [{"dataset": archive.name, "status": "no-dataset"}]

        names = name_roots(roots, staging)
        log(f"{len(roots)} dataset root(s): {', '.join(sorted(names.values()))}", 1)

        for root in roots:
            name = names[root]
            if name in ALREADY_TRAINED and not args.include_trained:
                log(f"--- {name}", 1)
                log("already fine-tuned -- skipping (--include-trained to force)", 2)
                results.append({"dataset": name, "status": "already-trained"})
                continue
            if args.only and name not in args.only:
                results.append({"dataset": name, "status": "not-selected"})
                continue
            try:
                results.append(prepare_root(root, name, archive, staging,
                                            Path(args.raw_dir), Path(args.out_dir), args))
            except Exception as exc:  # one bad dataset must not stop the rest
                log(f"FAILED: {exc}", 2)
                results.append({"dataset": name, "status": "failed", "error": str(exc)})
    finally:
        if staging.exists() and not args.keep_staging:
            shutil.rmtree(staging)
    return results


# --------------------------------------------------------------------------------------
# Listing
# --------------------------------------------------------------------------------------

def list_archives(archives: list[Path], args) -> None:
    print(f"{'archive':<52} {'size':>8}  contents")
    print("-" * 110)
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
        roots = sorted({n[: -len("meta/info.json")].rstrip("/")
                        for n in names if n.endswith("meta/info.json")})
        inner = [n for n in names if n.endswith(".zip")]
        if roots:
            detail = "; ".join(r or "<root>" for r in roots)
        elif inner:
            detail = f"{len(inner)} nested archive(s): " + ", ".join(Path(i).name for i in inner)
        else:
            detail = "no dataset found"
        print(f"{archive.name:<52} {human(archive.stat().st_size):>8}  {detail}")
    print()
    print("Nested archives are expanded at prepare time, so their datasets do not appear")
    print("above. Run with --list after preparing, or read data/simple/simple-converted/*/")
    print("PROVENANCE.json for what was actually produced.")


# --------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip-dir", default=str(ZIP_DIR))
    parser.add_argument("--raw-dir", default=str(RAW_DIR),
                        help="where extracted raw datasets land before conversion")
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help="where validated psi0 datasets are written")
    parser.add_argument("--staging-dir", default=str(STAGING_DIR))
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="prepare only these dataset names")
    parser.add_argument("--archives", nargs="+", metavar="ZIP",
                        help="process only these archive filenames")
    parser.add_argument("--from-dir", nargs="+", metavar="PATH", default=[],
                        help="prepare an already-extracted dataset tree (no .zip needed). "
                             "The source directory is never deleted.")
    parser.add_argument("--include-trained", action="store_true",
                        help="do not skip datasets that already have a fine-tuning run")
    parser.add_argument("--keep-raw", action="store_true",
                        help="keep the extracted raw tree after a successful conversion")
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--no-hash", action="store_true",
                        help="skip the source .zip sha256 (faster on large archives)")
    parser.add_argument("--force", action="store_true",
                        help="redo datasets that are already prepared")
    parser.add_argument("--list", action="store_true",
                        help="show what each archive contains and exit")
    parser.add_argument("--print-run-slug", metavar="DATASET",
                        help="print the run slug for a dataset name and exit "
                             "(so submit_slurm.sh and this script never disagree)")
    args = parser.parse_args()

    if args.print_run_slug:
        print(run_slug(args.print_run_slug))
        return 0

    for script in (GENERATOR, VALIDATOR, CONV_WBC, CONV_SONIC):
        if not script.exists():
            log(f"ERROR: missing {script}")
            return 1

    directories = [Path(p).resolve() for p in args.from_dir]
    for directory in directories:
        if not directory.is_dir():
            log(f"ERROR: --from-dir path is not a directory: {directory}")
            return 1

    # --from-dir on its own means "just this tree"; the archive scan is skipped so a
    # single new dataset does not drag the whole collection along.
    archives: list[Path] = []
    if not directories or args.archives:
        archives = sorted(p for p in Path(args.zip_dir).iterdir()
                          if p.is_file() and (p.suffix == ".zip" or ".zip." in p.name))
        if args.archives:
            wanted = set(args.archives)
            archives = [a for a in archives if a.name in wanted]
        if not archives and not directories:
            log(f"no archives found in {args.zip_dir}")
            return 1

    if args.list:
        list_archives(archives, args)
        return 0

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for archive in archives:
        results.extend(process_archive(archive, args))
    for directory in directories:
        results.extend(process_directory(directory, args))

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for status in ("ok", "skipped", "already-trained", "not-selected", "no-dataset", "failed"):
        rows = [r for r in results if r["status"] == status]
        if not rows:
            continue
        print(f"\n{status} ({len(rows)}):")
        for row in rows:
            extra = f"  episodes={row['episodes']}" if "episodes" in row else ""
            extra += f"\n      {row['error']}" if row.get("error") else ""
            for warning in row.get("warnings", []):
                extra += f"\n      WARNING: {warning}"
            print(f"  - {row['dataset']}{extra}")

    ready = [r for r in results if r["status"] == "ok"]
    if ready:
        print("\nReady to train:")
        for row in ready:
            print(f"  DATASET_NAME={row['dataset']}  (run slug: {row['run_slug']})")

    failed = [r for r in results if r["status"] == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
