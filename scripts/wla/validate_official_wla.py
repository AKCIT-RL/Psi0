"""F2a — golden test: pipeline independente (wla_adapter.pipeline) vs dataset oficial do WLA.

Compara action/state/state_unnorm/action_mask/state_mask em índices fixos (seed 0) e o
round-trip invert_action(action oficial) vs valores crus do parquet. Só CPU, sem vídeo.
Uso: PYTHONPATH=src python scripts/wla/validate_official_wla.py [--out DIR]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES, STATE_SLICES
from unifolm_wla.dataloader.multi_source_dataset.config import load_config
from unifolm_wla.dataloader.multi_source_dataset.single_source_dataset import create_single_source_dataset

from wla_adapter import pipeline as P
from wla_adapter.geometry import rotation_angle
from unifolm_wla.dataloader.multi_source_dataset.se3_utils import pose_to_se3_from_format

REPO = Path(__file__).resolve().parents[2]
TOL = 1e-5


def stat(d, name, v):
    v = np.asarray(v, np.float64)
    r = d.setdefault(name, {"max": 0.0, "sum": 0.0, "n": 0})
    r["max"] = max(r["max"], float(v.max()))
    r["sum"] += float(v.sum())
    r["n"] += v.size


def finish(d):
    return {k: {"max": v["max"], "mean": v["sum"] / max(v["n"], 1), "n": v["n"]} for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/wla/unitree_wbt_shelf.yaml"))
    ap.add_argument("--out", default=os.path.join(os.environ.get("WLA_EXP", "/raid/user_marcospaulo/experiments/wla"), "f2_validation"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_cfg = yaml.safe_load(open(args.config))
    for d in raw_cfg["datasets"]:
        d["image_keys"] = []
    scratch_cfg = REPO / "_scratch/f2a_golden_noimg.yaml"
    yaml.safe_dump(raw_cfg, open(scratch_cfg, "w"))

    cfg = load_config(scratch_cfg)
    ds_cfg = [d for d in cfg.datasets if d.enabled][0]
    src = create_single_source_dataset(ds_cfg, cfg)
    assert len(src._task_dirs) == 1
    data_dir = src._task_dirs[0]
    ns = P.load_norm_stats(ds_cfg.precollected_stats_path)

    cols = P.load_wbt_columns(data_dir, P.ACTION_KEYS + P.STATE_KEYS)
    N = len(cols["index"])
    assert N == len(src), (N, len(src))
    assert np.array_equal(cols["index"], np.arange(N)), "índice global != ordem das linhas"
    starts, ends = P.episode_bounds(cols["episode_index"])
    ep_ids = np.unique(cols["episode_index"])
    E = len(ep_ids)

    # 300 índices estratificados por episódio (todos os 200 + 100 sorteados) + 60 de borda (janela com padding).
    rng = np.random.default_rng(0)
    ep_pick = np.concatenate([np.arange(E), rng.choice(E, 100)])
    ep_first = np.array([np.flatnonzero(cols["episode_index"] == e)[0] for e in ep_ids])
    ep_len = np.array([(cols["episode_index"] == e).sum() for e in ep_ids])
    main_idx = ep_first[ep_pick] + (rng.random(len(ep_pick)) * ep_len[ep_pick]).astype(int)
    edge_ep = rng.choice(E, 30, replace=False)
    edge_idx = np.concatenate([
        ep_first[edge_ep] + ep_len[edge_ep] - 1,
        ep_first[edge_ep] + ep_len[edge_ep] - 1 - rng.integers(0, P.H - 1, 30),
    ])
    groups = {"main": main_idx, "edge": edge_idx}

    a_st, s_st, su_st, rt_st, mag = {}, {}, {}, {}, {}
    mask_ok = True
    n = 0
    for gname, idxs in groups.items():
        for idx in idxs:
            idx = int(idx)
            off = src[idx]
            ours = P.build_sample(cols, idx, starts[idx], ends[idx], ns)
            da = np.abs(off["action"].numpy().astype(np.float64) - ours["action"])
            for sn, sl in SLICES.items():
                stat(a_st, sn, da[:, sl].max(axis=0))
            stat(a_st, "ALL", da.max())
            oa = np.abs(off["action"].numpy().astype(np.float64))
            for sn, sl in SLICES.items():
                stat(mag, sn, oa[:, sl].max(axis=0))
            ds = np.abs(off["state"].numpy().astype(np.float64) - ours["state"])
            dsu = np.abs(off["state_unnorm"].numpy().astype(np.float64) - ours["state_unnorm"])
            for sn, sl in STATE_SLICES.items():
                stat(s_st, sn, ds[sl])
                stat(su_st, sn, dsu[sl])
            stat(s_st, "ALL", ds.max())
            stat(su_st, "ALL", dsu.max())
            mask_ok &= bool(np.array_equal(off["action_mask"].numpy(), ours["action_mask"]))
            mask_ok &= bool(np.array_equal(off["state_mask"].numpy(), ours["state_mask"]))
            mask_ok &= bool(np.array_equal(off["action_norm_offset"].numpy(), ns.action_offset))
            mask_ok &= bool(np.array_equal(off["action_norm_scale"].numpy(), ns.action_scale))

            # round-trip: inversa do action oficial vs crus futuros do parquet
            inv = P.invert_action(off["action"].numpy(), off["state_unnorm"].numpy(), ns, cols[P.S_BASE_POSE][idx])
            widx = np.clip(idx + np.arange(P.H), starts[idx], ends[idx] - 1)
            for side in ("left", "right"):
                T_raw = pose_to_se3_from_format(cols[P.A_EE[side]][widx], "xyz_rpy")
                stat(rt_st, f"{side}_ee_pos_m", np.linalg.norm(inv[f"{side}_ee_T"][:, :3, 3] - T_raw[:, :3, 3], axis=-1))
                stat(rt_st, f"{side}_ee_rot_rad", rotation_angle(inv[f"{side}_ee_T"][:, :3, :3], T_raw[:, :3, :3]))
                stat(rt_st, f"{side}_fig6d", np.abs(inv[f"{side}_fig6d"] - cols[P.A_FIG[side]][widx]))
                stat(rt_st, f"{side}_leg", np.abs(inv[f"{side}_leg"] - cols[P.A_LEG[side]][widx]))
            stat(rt_st, "waist", np.abs(inv["waist"] - cols[P.A_WAIST][widx]))
            stat(rt_st, "base_command", np.abs(inv["base_command"] - cols[P.A_BASE_CMD][widx]))
            Tb_raw = pose_to_se3_from_format(cols[P.A_BASE_POSE][widx], "xyz_quat")
            stat(rt_st, "base_pose_pos_m", np.linalg.norm(inv["base_pose_T"][:, :3, 3] - Tb_raw[:, :3, 3], axis=-1))
            stat(rt_st, "base_pose_rot_rad", rotation_angle(inv["base_pose_T"][:, :3, :3], Tb_raw[:, :3, :3]))
            n += 1

    rep = {
        "n_samples": n,
        "n_main": len(main_idx),
        "n_edge": len(edge_idx),
        "tolerance": TOL,
        "action": finish(a_st),
        "official_action_abs_magnitude (anti-vacuidade; max/mean de |action oficial|)": finish(mag),
        "state": finish(s_st),
        "state_unnorm": finish(su_st),
        "masks_and_norm_vectors_identical": mask_ok,
        "action_mask_sum": int(P.action_mask().sum()),
        "state_mask_sum": int(P.state_mask().sum()),
        "roundtrip_wbt": finish(rt_st),
    }
    for sec in ("action", "state", "state_unnorm", "roundtrip_wbt"):
        for k, v in rep[sec].items():
            v["status"] = "PASS" if v["max"] < TOL else "FAIL"
    rep["status"] = {
        "action": rep["action"]["ALL"]["status"],
        "state": rep["state"]["ALL"]["status"],
        "state_unnorm": rep["state_unnorm"]["ALL"]["status"],
        "masks": "PASS" if mask_ok else "FAIL",
        "roundtrip_wbt": "PASS" if all(v["status"] == "PASS" for v in rep["roundtrip_wbt"].values()) else "FAIL",
    }
    rep["overall"] = "PASS" if all(v == "PASS" for v in rep["status"].values()) else "FAIL"
    json.dump(rep, open(out_dir / "golden_report.json", "w"), indent=1)
    print(json.dumps(rep["status"]), "overall:", rep["overall"])
    for sec in ("action", "state", "state_unnorm", "roundtrip_wbt"):
        for k, v in rep[sec].items():
            print(f"{sec:14s} {k:20s} max={v['max']:.3e} mean={v['mean']:.3e} {v['status']}")
    sys.exit(0 if rep["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
