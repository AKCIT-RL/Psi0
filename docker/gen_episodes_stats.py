"""
Generate meta/episodes_stats.jsonl for a LeRobot v2.1 dataset.
Required by the custom lerobot fork used by PSI-0.
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

dataset_dir = Path(sys.argv[1])
episodes_file = dataset_dir / "meta" / "episodes.jsonl"
out_file = dataset_dir / "meta" / "episodes_stats.jsonl"

episodes = [json.loads(l) for l in episodes_file.read_text().splitlines() if l.strip()]

with open(out_file, "w") as f:
    for ep in episodes:
        idx = ep["episode_index"]
        chunk = idx // 1000
        parquet = dataset_dir / f"data/chunk-{chunk:03d}/episode_{idx:06d}.parquet"
        df = pd.read_parquet(parquet)
        n = len(df)
        stats = {}
        for col in df.columns:
            sample = df[col].iloc[0]
            if isinstance(sample, (list, np.ndarray)):
                arr = np.vstack(df[col].apply(np.array).values).astype(np.float32)
            elif isinstance(sample, bool):
                continue
            elif isinstance(sample, (int, float, np.integer, np.floating)):
                arr = df[col].values.astype(np.float32).reshape(-1, 1)
            else:
                continue
            stats[col] = {
                "mean":  arr.mean(axis=0).tolist(),
                "std":   arr.std(axis=0).tolist(),
                "min":   arr.min(axis=0).tolist(),
                "max":   arr.max(axis=0).tolist(),
                "count": [n],
            }
        f.write(json.dumps({"episode_index": idx, "stats": stats}) + "\n")
        print(f"  episode {idx}: {n} frames")

print(f"Wrote {out_file}")
