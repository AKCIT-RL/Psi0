#!/usr/bin/env python3
"""Raízes LeRobot v2.1 só com os episódios de um split (val/test), para o --eval-config do SIMPLE.

Mantém os índices ORIGINAIS dos episódios (nomes de arquivo, episodes.jsonl) e usa symlinks para
dados e vídeos. episodes.jsonl/episodes_stats.jsonl são filtrados; info.json recebe totais do subset.
A divisão vem de docs/wla/split.json (fixada antes de qualquer conversão). CPU-only, só stdlib.
"""
import argparse
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/raid/user_marcospaulo/datasets/psi0/G1ToteMix-psi0")
ap.add_argument("--split_json", default=str(Path(__file__).resolve().parents[2] / "docs/wla/split.json"))
ap.add_argument("--split", required=True, choices=["train", "val", "test"])
ap.add_argument("--out", required=True)
a = ap.parse_args()

src, out = Path(a.src), Path(a.out)
eps = sorted(json.load(open(a.split_json))[a.split])
keep = set(eps)
(out / "meta").mkdir(parents=True, exist_ok=True)
for m in src.joinpath("meta").iterdir():
    if m.name not in ("episodes.jsonl", "episodes_stats.jsonl", "info.json"):
        dst = out / "meta" / m.name
        if not dst.exists():
            os.symlink(m, dst)

frames = 0
for name in ("episodes.jsonl", "episodes_stats.jsonl"):
    rows = [json.loads(l) for l in open(src / "meta" / name) if l.strip()]
    rows = [r for r in rows if r["episode_index"] in keep]
    assert len(rows) == len(eps), (name, len(rows), len(eps))
    if name == "episodes.jsonl":
        frames = sum(r["length"] for r in rows)
    with open(out / "meta" / name, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

info = json.load(open(src / "meta" / "info.json"))
info.update(total_episodes=len(eps), total_frames=frames, total_videos=len(eps), splits={a.split: f"0:{len(eps)}"})
json.dump(info, open(out / "meta" / "info.json", "w"), indent=4)

for e in eps:
    for rel in (f"data/chunk-000/episode_{e:06d}.parquet", f"videos/chunk-000/egocentric/episode_{e:06d}.mp4"):
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.symlink(src / rel, dst)
print(f"{a.split}: {len(eps)} episódios, {frames} frames -> {out}")
