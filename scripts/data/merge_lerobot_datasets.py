#!/usr/bin/env python3
"""Merge two or more LeRobot v2.1 datasets into a single dataset.

Renumbers episode_index / index / task_index in parquets, copies videos,
rewrites meta/episodes.jsonl, meta/episodes_stats.jsonl, meta/tasks.jsonl
(distinct task strings unified) and meta/info.json. Requires identical
meta/modality.json across sources (aborts otherwise).

Optional per-source task remap fixes corrupted task entries before
unification, e.g. --remap-task 0:1=0 remaps task_index 1 -> 0 in source 0.

Usage:
    python scripts/data/merge_lerobot_datasets.py \
        --source /path/dsA --source /path/dsB \
        --output /path/merged [--remap-task SRC:OLD=NEW ...]

Stats are NOT computed here; run scripts/data/calc_modality_stats.py
--task-dir <output> afterwards and copy meta/stats.json to meta/stats_psi0.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, dest="sources",
                        help="source dataset dir (repeat; order defines episode order)")
    parser.add_argument("--output", required=True, help="output dataset dir (must not exist)")
    parser.add_argument("--remap-task", action="append", default=[], dest="remaps",
                        metavar="SRC:OLD=NEW",
                        help="remap task_index OLD->NEW within source SRC before unification")
    args = parser.parse_args()

    sources = [Path(s) for s in args.sources]
    output = Path(args.output)
    if output.exists():
        print(f"ERROR: output already exists: {output}", file=sys.stderr)
        return 1
    for src in sources:
        if not (src / "meta" / "info.json").is_file():
            print(f"ERROR: not a LeRobot dataset: {src}", file=sys.stderr)
            return 1

    remaps: dict[int, dict[int, int]] = {}
    for spec in args.remaps:
        src_part, mapping = spec.split(":", 1)
        old, new = mapping.split("=", 1)
        remaps.setdefault(int(src_part), {})[int(old)] = int(new)

    # --- modality.json must be identical -----------------------------------
    modalities = [json.loads((s / "meta" / "modality.json").read_text()) for s in sources]
    for i, m in enumerate(modalities[1:], start=1):
        if m != modalities[0]:
            print(f"ERROR: modality.json of source {i} ({sources[i]}) differs from source 0",
                  file=sys.stderr)
            return 1

    infos = [json.loads((s / "meta" / "info.json").read_text()) for s in sources]
    for key in ("codebase_version", "robot_type", "fps", "data_path", "video_path", "features"):
        vals = [json.dumps(info.get(key), sort_keys=True) for info in infos]
        if len(set(vals)) != 1:
            print(f"ERROR: info.json field '{key}' differs across sources", file=sys.stderr)
            return 1

    # --- unified task table --------------------------------------------------
    unified_tasks: list[dict] = []          # rows for output tasks.jsonl
    task_text_to_new: dict[str, int] = {}
    src_task_maps: list[dict[int, int]] = []  # per source: old task_index -> new
    for si, src in enumerate(sources):
        rows = read_jsonl(src / "meta" / "tasks.jsonl")
        by_index = {r["task_index"]: r for r in rows}
        table: dict[int, int] = {}
        for old_idx, row in sorted(by_index.items()):
            eff_idx = remaps.get(si, {}).get(old_idx, old_idx)
            eff_row = by_index[eff_idx]
            text = eff_row["task"]
            if text not in task_text_to_new:
                new_idx = len(unified_tasks)
                task_text_to_new[text] = new_idx
                out_row = dict(eff_row)
                out_row["task_index"] = new_idx
                unified_tasks.append(out_row)
            table[old_idx] = task_text_to_new[text]
        src_task_maps.append(table)

    # --- output skeleton -----------------------------------------------------
    chunks_size = infos[0].get("chunks_size", 1000)
    data_path_tpl = infos[0]["data_path"]
    video_path_tpl = infos[0]["video_path"]
    video_keys = [k for k, f in infos[0]["features"].items() if f.get("dtype") == "video"]

    (output / "meta").mkdir(parents=True)

    new_episodes: list[dict] = []
    new_ep_stats: list[dict] = []
    global_ep = 0
    global_frame = 0

    for si, src in enumerate(sources):
        tmap = src_task_maps[si]
        episodes = read_jsonl(src / "meta" / "episodes.jsonl")
        episodes.sort(key=lambda e: e["episode_index"])
        ep_stats = {e["episode_index"]: e for e in read_jsonl(src / "meta" / "episodes_stats.jsonl")}

        for ep in episodes:
            old_ep = ep["episode_index"]
            new_ep = global_ep

            # parquet: rewrite indices
            src_parquet = src / data_path_tpl.format(
                episode_chunk=old_ep // chunks_size, episode_index=old_ep)
            df = pd.read_parquet(src_parquet)
            n = len(df)
            df["episode_index"] = new_ep
            df["index"] = range(global_frame, global_frame + n)
            df["task_index"] = df["task_index"].map(lambda t: tmap[int(t)])
            dst_parquet = output / data_path_tpl.format(
                episode_chunk=new_ep // chunks_size, episode_index=new_ep)
            dst_parquet.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst_parquet)

            # videos
            for vkey in video_keys:
                cam = vkey.removeprefix("observation.images.")
                src_video = src / video_path_tpl.format(
                    episode_chunk=old_ep // chunks_size, episode_index=old_ep,
                    video_key=vkey, camera=cam)
                dst_video = output / video_path_tpl.format(
                    episode_chunk=new_ep // chunks_size, episode_index=new_ep,
                    video_key=vkey, camera=cam)
                if not src_video.is_file():
                    print(f"ERROR: missing video {src_video}", file=sys.stderr)
                    return 1
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_video, dst_video)

            # episodes.jsonl row
            row = dict(ep)
            row["episode_index"] = new_ep
            if isinstance(row.get("tasks"), list):
                row["tasks"] = [tmap[int(t)] if isinstance(t, int) else task_text_to_new[t]
                                for t in row["tasks"]]
            if isinstance(row.get("instruction"), dict) and "task_index" in row["instruction"]:
                nt = tmap[int(row["instruction"]["task_index"])]
                row["instruction"] = {"task_index": nt, "task": unified_tasks[nt]["task"]}
            if "dataset_from_index" in row:
                row["dataset_from_index"] = global_frame
                row["dataset_to_index"] = global_frame + n - 1
            new_episodes.append(row)

            if old_ep in ep_stats:
                srow = dict(ep_stats[old_ep])
                srow["episode_index"] = new_ep
                new_ep_stats.append(srow)

            global_ep += 1
            global_frame += n

    write_jsonl(output / "meta" / "episodes.jsonl", new_episodes)
    write_jsonl(output / "meta" / "episodes_stats.jsonl", new_ep_stats)
    write_jsonl(output / "meta" / "tasks.jsonl", unified_tasks)
    (output / "meta" / "modality.json").write_text(
        json.dumps(modalities[0], indent=4) + "\n")

    for aux in ("lang_map.json", "relative_stats.json"):
        contents = [(s / "meta" / aux).read_text() if (s / "meta" / aux).is_file() else None
                    for s in sources]
        if contents[0] is not None:
            (output / "meta" / aux).write_text(contents[0])

    info = dict(infos[0])
    info["total_episodes"] = global_ep
    info["total_frames"] = global_frame
    info["total_tasks"] = len(unified_tasks)
    info["total_videos"] = global_ep * len(video_keys)
    info["total_chunks"] = (global_ep + chunks_size - 1) // chunks_size
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")

    print(f"merged {len(sources)} sources -> {output}")
    print(f"total_episodes={global_ep} total_frames={global_frame} total_tasks={len(unified_tasks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
