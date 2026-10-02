"""F2b — conversor Ψ0 (G1ToteMix-psi0, LeRobot v2.1 @50FPS) -> dataset canônico WLA
(LeRobot v3.0 @30FPS, schema WBT), escrito com a API do lerobot 0.5.0 (D1).

Uso (sempre via sbatch — ver scripts/wla/f2b_convert.slurm):
  python -m wla_adapter.write_dataset --split train,val --out $WLA_DATA_ROOT/Psi0_Tote_Dataset/G1ToteMix_psi0_trainval \
      --stats-out $WLA_DATA_ROOT/stats_psi0_tote_train
  python -m wla_adapter.write_dataset --episodes 0,1,2 --out ...   # ids ORIGINAIS do Ψ0
Episódios `test` de docs/wla/split.json são RECUSADOS, a menos de --allow-test (Fase 3).

Por episódio: parquet v2.1 -> reamostragem linear 50->30 (resample.py) -> chaves WBT
(convert_state/convert_action; FK das juntas reamostradas, nunca o contrário) ->
vídeo = frame de origem mais próximo de t_j, re-encodado pelo lerobot.
Guarda colunas cruas reamostradas psi0.{action(36),states(32),hand_joints(14),
arm_joints(14),leg_joints(15)} para os testes de round-trip, e
meta/psi0_episode_map.json (episode_index novo -> original).
Stats (D4): apenas dos episódios train DESTA execução (--stats-out).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
SPLIT_PATH = REPO / "docs/wla/split.json"
TASKS_PATH = "meta/tasks.jsonl"

# Colunas cruas Ψ0 reamostradas gravadas no dataset (para os testes R2/R3/R5).
RAW_FEATURES = {
    "psi0.action": 36,
    "psi0.states": 32,
    "psi0.hand_joints": 14,
    "psi0.arm_joints": 14,
    "psi0.leg_joints": 15,
}
VIDEO_KEY = "observation.images.egocentric"
VIDEO_SHAPE = (360, 640, 3)  # H, W, C


def repo_version() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def converter_hash() -> str:
    """SHA256 dos fontes do conversor (rastreabilidade do conversion_version)."""
    h = hashlib.sha256()
    for p in sorted((REPO / "src/wla_adapter").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def resolve_episodes(args) -> list[int]:
    split = json.load(open(SPLIT_PATH))
    test_ids = set(split["test"])
    if args.episodes is not None:
        eps = [int(x) for x in args.episodes.split(",") if x.strip()]
    else:
        eps = []
        for name in args.split.split(","):
            name = name.strip()
            if name not in ("train", "val", "test"):
                raise ValueError(f"split desconhecido: {name!r}")
            eps += split[name]
    bad = sorted(set(eps) & test_ids)
    if bad and not args.allow_test:
        raise SystemExit(
            f"RECUSADO: episódios de TEST no pedido: {bad[:10]}{'...' if len(bad) > 10 else ''} "
            "(regra 1; use --allow-test apenas na Fase 3)"
        )
    if len(set(eps)) != len(eps):
        raise SystemExit("episódios duplicados no pedido")
    return eps


def load_task(src: Path) -> str:
    with open(src / TASKS_PATH) as f:
        return json.loads(f.readline())["task"]


def read_episode_raw(src: Path, ep: int) -> dict[str, np.ndarray]:
    """Colunas cruas do parquet v2.1 (50 FPS) como float64."""
    t = pq.read_table(src / f"data/chunk-000/episode_{ep:06d}.parquet")
    out = {}
    for col, key in [("observation.hand_joints", "hand_joints"), ("observation.arm_joints", "arm_joints"),
                     ("observation.leg_joints", "leg_joints"), ("states", "states"), ("action", "action")]:
        arr = t[col].combine_chunks()
        out[key] = np.stack(arr.to_numpy(zero_copy_only=False)).astype(np.float64)
    return out


def resample_episode(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """50 -> 30 FPS: linear em todos os canais; target_yaw (action[35]) com unwrap."""
    from .resample import resample_angle, resample_linear

    out = {k: resample_linear(raw[k]) for k in ("hand_joints", "arm_joints", "leg_joints", "states")}
    a_lin = resample_linear(raw["action"][:, :35])
    a_yaw = resample_angle(raw["action"][:, 35:36])
    out["action"] = np.concatenate([a_lin, a_yaw], axis=1)
    return out


def decode_video(path: Path) -> np.ndarray:
    """(N,H,W,3) uint8 via PyAV (torchcodec sem FFmpeg do sistema neste nó)."""
    import av

    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def build_features() -> dict:
    from .convert_action import ACTION_OUT_KEYS, A_BASE_CMD, A_EE, A_FIG, A_WAIST
    from .convert_state import STATE_OUT_KEYS

    names6 = ["x", "y", "z", "roll", "pitch", "yaw"]
    feats = {
        VIDEO_KEY: {"dtype": "video", "shape": list(VIDEO_SHAPE), "names": ["height", "width", "channel"]},
        A_EE.format(side="left"): {"dtype": "float32", "shape": [6], "names": names6},
        A_EE.format(side="right"): {"dtype": "float32", "shape": [6], "names": names6},
        A_FIG.format(side="left"): {"dtype": "float32", "shape": [6], "names": ["thumb_oc", "thumb_lat", "index", "middle", "ring", "little"]},
        A_FIG.format(side="right"): {"dtype": "float32", "shape": [6], "names": ["thumb_oc", "thumb_lat", "index", "middle", "ring", "little"]},
        A_WAIST: {"dtype": "float32", "shape": [3], "names": ["yaw", "roll", "pitch"]},
        A_BASE_CMD: {"dtype": "float32", "shape": [4], "names": ["vx", "vy", "angle_z", "height"]},
    }
    from .convert_state import S_EE, S_FIG, S_LEG, S_WAIST

    for side in ("left", "right"):
        feats[S_EE.format(side=side)] = {"dtype": "float32", "shape": [6], "names": names6}
        feats[S_FIG.format(side=side)] = {"dtype": "float32", "shape": [6], "names": ["thumb_oc", "thumb_lat", "index", "middle", "ring", "little"]}
        feats[S_LEG.format(side=side)] = {"dtype": "float32", "shape": [6], "names": ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]}
    feats[S_WAIST] = {"dtype": "float32", "shape": [3], "names": ["yaw", "roll", "pitch"]}
    for k, d in RAW_FEATURES.items():
        feats[k] = {"dtype": "float32", "shape": [d]}
    assert set(feats) == {VIDEO_KEY, *ACTION_OUT_KEYS, *STATE_OUT_KEYS, *RAW_FEATURES}
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train,val", help="train,val (padrão) | train | val | ...")
    ap.add_argument("--episodes", default=None, help="ids ORIGINAIS Ψ0 separados por vírgula (sobrepõe --split)")
    ap.add_argument("--allow-test", action="store_true", help="permite episódios test (só Fase 3)")
    ap.add_argument("--src", default=os.path.join(os.environ.get("PSI0_DATA", "/raid/user_marcospaulo/datasets/psi0"), "G1ToteMix-psi0"))
    ap.add_argument("--out", default=os.path.join(os.environ.get("WLA_DATA_ROOT", "/raid/user_marcospaulo/datasets/unifolm"), "Psi0_Tote_Dataset/G1ToteMix_psi0_trainval"))
    ap.add_argument("--stats-out", default=None, help="dir de stats D4 (default: <WLA_DATA_ROOT>/stats_psi0_tote_train); vazio = não calcula")
    ap.add_argument("--no-stats", action="store_true")
    ap.add_argument("--no-video", action="store_true", help="pula vídeo (debug rápido; dataset NÃO serve p/ treino)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} já existe (use --overwrite)")
        import shutil

        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    episodes = resolve_episodes(args)
    task = load_task(src)

    from lerobot.datasets import utils as lerobot_utils
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # lerobot 0.5 compara shape tuple com list ([6] != (6,)). Só neste processo.
    _orig_np = lerobot_utils.validate_feature_numpy_array

    def _shape_as_tuple(name, expected_dtype, expected_shape, value):
        return _orig_np(name, expected_dtype, tuple(expected_shape), value)

    lerobot_utils.validate_feature_numpy_array = _shape_as_tuple

    from .convert_action import action_keys
    from .convert_state import state_keys
    from .resample import nearest_source_indices

    features = build_features()
    if args.no_video:
        features.pop(VIDEO_KEY)
    ds = LeRobotDataset.create(
        repo_id=f"psi0/{out.name}", fps=30, features=features, root=out,
        robot_type="g1", use_videos=not args.no_video, video_backend="pyav",
    )

    ep_map: dict[int, int] = {}
    train_eps_for_stats: list[dict[str, np.ndarray]] = []
    train_ids = set(json.load(open(SPLIT_PATH))["train"])
    n_frames = 0
    for new_idx, ep in enumerate(episodes):
        raw = read_episode_raw(src, ep)
        raw30 = resample_episode(raw)
        keys = {**state_keys(raw30), **action_keys(raw30)}
        m = len(raw30["action"])
        if not args.no_video:
            frames = decode_video(src / f"videos/chunk-000/egocentric/episode_{ep:06d}.mp4")
            vidx = nearest_source_indices(len(raw["action"]))
            if len(frames) != len(raw["action"]):
                raise RuntimeError(f"ep {ep}: vídeo {len(frames)} frames != parquet {len(raw['action'])}")
        for j in range(m):
            frame = {k: keys[k][j] for k in keys}
            for rk in RAW_FEATURES:
                frame[rk] = raw30[rk.removeprefix("psi0.")][j].astype(np.float32)
            if not args.no_video:
                frame[VIDEO_KEY] = frames[vidx[j]]
            frame["task"] = task
            ds.add_frame(frame)
        ds.save_episode()
        ep_map[new_idx] = ep
        n_frames += m
        if ep in train_ids:
            train_eps_for_stats.append(keys)
        print(f"[{new_idx + 1}/{len(episodes)}] ep original {ep}: {len(raw['action'])} -> {m} frames", flush=True)
        del raw, raw30, keys
        if not args.no_video:
            del frames

    ds.finalize()
    with open(out / "meta/psi0_episode_map.json", "w") as f:
        json.dump({"new_to_original": {str(k): v for k, v in ep_map.items()},
                   "src": str(src), "split_path": str(SPLIT_PATH)}, f, indent=1)

    stats_dir = None
    if not args.no_stats:
        stats_dir = args.stats_out or os.path.join(
            os.environ.get("WLA_DATA_ROOT", "/raid/user_marcospaulo/datasets/unifolm"), "stats_psi0_tote_train")
        from .normalization import write_stats

        n_val = sum(1 for e in episodes if e not in train_ids)
        write_stats(
            stats_dir, train_eps_for_stats,
            dataset_version=f"agentereal/G1ToteMix-psi0 v2.1 308ep@50fps -> {out.name} {len(episodes)}ep@30fps",
            conversion_version=f"git:{repo_version()} converter_sha256:{converter_hash()}",
        )
        print(f"stats (D4) de {len(train_eps_for_stats)} episódios train -> {stats_dir} "
              f"({n_val} ep val/test convertidos NÃO entraram nos stats)")

    print(json.dumps({"out": str(out), "episodes": len(episodes), "frames": n_frames,
                      "stats_dir": stats_dir}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
