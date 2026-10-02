#!/usr/bin/env python3
"""Ação Ψ0 @30 FPS (a mesma reamostragem do conversor, wla_adapter.write_dataset) por episódio -> <out>/<name>__episode_<idx>.npy.

Alimenta o servidor --oracle (S3 replay original / S5 replay via adaptador). Roda no venv do WLA (pyarrow). CPU.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PSI0 = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PSI0 / "src"), str(PSI0 / "third_party/unifolm-wla")]
from wla_adapter.write_dataset import read_episode_raw, resample_episode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/raid/user_marcospaulo/datasets/psi0/G1ToteMix-psi0")
ap.add_argument("--split", required=True, choices=["train", "val", "test"])
ap.add_argument("--name", required=True, help="nome da fonte no --eval-config (prefixo do episode id)")
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()
eps = sorted(json.load(open(PSI0 / "docs/wla/split.json"))[a.split])[: a.limit]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
# o eval do SIMPLE vê os episódios renumerados 0..N-1 (s0_make_split_roots): nome = posição em sorted(split)
for n, e in enumerate(eps):
    act = resample_episode(read_episode_raw(Path(a.src), e))["action"].astype(np.float32)
    np.save(out / f"{a.name}__episode_{n}.npy", act)
print(f"{a.split}: {len(eps)} episódios -> {out}; exemplo shape {act.shape}")
