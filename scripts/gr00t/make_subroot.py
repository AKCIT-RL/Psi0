"""Cria uma raiz LeRobot v2.1 só com alguns episódios de VAL, renumerados 0..n-1 (o LeRobot do SIMPLE exige índices contíguos).
Uso: make_subroot.py <nome> <ep> [<ep> ...]   (ep = índice em roots/val; saída: roots/<nome>; task ids viram <nome>__episode_<0..n-1>)"""
import json, os, shutil, sys
import pandas as pd

name, eps = sys.argv[1], [int(x) for x in sys.argv[2:]]
base = os.path.expanduser("~/wla_data/experiments/wla/s0_eval_data/roots")
src, dst = f"{base}/val", f"{base}/{name}"
shutil.rmtree(dst, ignore_errors=True)
for d in ("meta", "data/chunk-000", "videos/chunk-000/egocentric"): os.makedirs(f"{dst}/{d}")
rows = {int(json.loads(l)["episode_index"]): json.loads(l) for l in open(f"{src}/meta/episodes.jsonl")}
stats = {int(json.loads(l)["episode_index"]): json.loads(l) for l in open(f"{src}/meta/episodes_stats.jsonl")}
orig = json.load(open(f"{src}/meta/orig_index.json"))
off, ep_out, st_out, orig_out = 0, [], [], []
for new, e in enumerate(eps):
    df = pd.read_parquet(f"{src}/data/chunk-000/episode_{e:06d}.parquet"); n = len(df)
    df["episode_index"] = new; df["index"] = range(off, off + n)
    df.to_parquet(f"{dst}/data/chunk-000/episode_{new:06d}.parquet")
    shutil.copy(f"{src}/videos/chunk-000/egocentric/episode_{e:06d}.mp4", f"{dst}/videos/chunk-000/egocentric/episode_{new:06d}.mp4")
    r = dict(rows[e]); r.update(episode_index=new, dataset_from_index=off, dataset_to_index=off + n - 1); ep_out.append(r)
    s = dict(stats[e]); s["episode_index"] = new; st_out.append(s); orig_out.append(orig[e]); off += n
w = lambda p, lines: open(f"{dst}/meta/{p}", "w").write("".join(json.dumps(x) + "\n" for x in lines))
w("episodes.jsonl", ep_out); w("episodes_stats.jsonl", st_out); json.dump(orig_out, open(f"{dst}/meta/orig_index.json", "w"))
info = json.load(open(f"{src}/meta/info.json")); info.update(total_episodes=len(eps), total_frames=off, total_videos=len(eps), splits={name: f"0:{len(eps)}"})
json.dump(info, open(f"{dst}/meta/info.json", "w"), indent=4)
for f in os.listdir(f"{src}/meta"):
    if not os.path.exists(f"{dst}/meta/{f}"): shutil.copy(f"{src}/meta/{f}", f"{dst}/meta/{f}")
print("ok", dst, "orig val indices", eps, "->", list(range(len(eps))), "frames", off)
