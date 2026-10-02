#!/usr/bin/env python3
"""Raízes LeRobot v2.1 só com os episódios de um split (train/val/test), para o --eval-config do SIMPLE.

O LeRobot do SIMPLE (0.3.3) exige episódios CONTÍGUOS 0..N-1 (com índices esparsos tenta baixar do hub), então os
episódios são renumerados pela posição em sorted(split). Mapa novo->original em meta/orig_index.json e no campo
`orig_index` de cada linha de episodes.jsonl. Parquet reescrito (episode_index, index); vídeos por symlink.
Divisão vinda de docs/wla/split.json (fixada antes de qualquer conversão). Roda no venv do WLA (pyarrow). CPU.
"""
import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/raid/user_marcospaulo/datasets/psi0/G1ToteMix-psi0")
ap.add_argument("--split_json", default=str(Path(__file__).resolve().parents[2] / "docs/wla/split.json"))
ap.add_argument("--split", required=True, choices=["train", "val", "test"])
ap.add_argument("--out", required=True)
a = ap.parse_args()

src, out = Path(a.src), Path(a.out)
eps = sorted(json.load(open(a.split_json))[a.split])
new_of = {e: i for i, e in enumerate(eps)}
(out / "meta").mkdir(parents=True, exist_ok=True)
for m in src.joinpath("meta").iterdir():
    if m.name not in ("episodes.jsonl", "episodes_stats.jsonl", "info.json"):
        dst = out / "meta" / m.name
        if not dst.exists():
            os.symlink(m, dst)
json.dump(eps, open(out / "meta" / "orig_index.json", "w"))

rows = {e: json.loads(l) for l in open(src / "meta/episodes.jsonl") if l.strip() for e in [json.loads(l)["episode_index"]] if e in new_of}
assert len(rows) == len(eps)
starts, cur = {}, 0
for e in eps:
    starts[e] = cur
    cur += rows[e]["length"]
frames = cur

with open(out / "meta/episodes.jsonl", "w") as f:
    for e in eps:
        r = dict(rows[e])
        span = r.get("dataset_to_index", 0) - r.get("dataset_from_index", 0)
        r.update(orig_index=e, episode_index=new_of[e], dataset_from_index=starts[e], dataset_to_index=starts[e] + span)
        f.write(json.dumps(r) + "\n")
with open(out / "meta/episodes_stats.jsonl", "w") as f:
    for l in open(src / "meta/episodes_stats.jsonl"):
        if l.strip():
            r = json.loads(l)
            if r["episode_index"] in new_of:
                r["orig_index"], r["episode_index"] = r["episode_index"], new_of[r["episode_index"]]
                f.write(json.dumps(r) + "\n")

info = json.load(open(src / "meta/info.json"))
info.update(total_episodes=len(eps), total_frames=frames, total_videos=len(eps), splits={a.split: f"0:{len(eps)}"})
json.dump(info, open(out / "meta/info.json", "w"), indent=4)

(out / "data/chunk-000").mkdir(parents=True, exist_ok=True)
(out / "videos/chunk-000/egocentric").mkdir(parents=True, exist_ok=True)
for e in eps:
    n = new_of[e]
    t = pq.read_table(src / f"data/chunk-000/episode_{e:06d}.parquet")
    L = t.num_rows
    assert L == rows[e]["length"], (e, L, rows[e]["length"])
    t = t.set_column(t.schema.get_field_index("episode_index"), "episode_index", pa.array([n] * L, pa.int64()))
    t = t.set_column(t.schema.get_field_index("index"), "index", pa.array(range(starts[e], starts[e] + L), pa.int64()))
    pq.write_table(t, out / f"data/chunk-000/episode_{n:06d}.parquet")
    dst = out / f"videos/chunk-000/egocentric/episode_{n:06d}.mp4"
    if not dst.exists():
        os.symlink(src / f"videos/chunk-000/egocentric/episode_{e:06d}.mp4", dst)
print(f"{a.split}: {len(eps)} episódios (renumerados 0..{len(eps) - 1}), {frames} frames -> {out}")
