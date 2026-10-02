#!/usr/bin/env python3
"""Dry-run do servidor do baseline WLA-Base (Policy.act, modo modelo) com um chunk REAL de um episódio de VAL.

Valida de ponta a ponta, sem simulador: carregamento do checkpoint/tokenizer/norm source, montagem do estado WLA
(FK/fig6d a partir de joint_qpos na ordem do SIMPLE), imagem (frame do vídeo), máscaras do perfil, forward,
desnormalização, EE rel->abs, IK e ação Ψ0 36D. Em CPU só valida o caminho (lento); na GPU use o job simple_eval.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PSI0 = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PSI0 / "src"), str(PSI0 / "third_party/unifolm-wla"), str(PSI0 / "scripts/wla")]
from simple_wla_server import Policy  # noqa: E402
from wla_adapter.geometry import fk_ee, rotation_angle  # noqa: E402
from wla_adapter.write_dataset import read_episode_raw, resample_episode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt_path", required=True)
ap.add_argument("--base_vlm", required=True)
ap.add_argument("--source_name", default="UnifoLM_WBT")
ap.add_argument("--profile", default="wbt")
ap.add_argument("--device", default="cpu")
ap.add_argument("--src", default="/raid/user_marcospaulo/datasets/psi0/G1ToteMix-psi0")
ap.add_argument("--split", default="val")
ap.add_argument("--t", type=int, default=300, help="passo a 30 FPS")
ap.add_argument("--out", required=True)
a = ap.parse_args()

ep = sorted(json.load(open(PSI0 / "docs/wla/split.json"))[a.split])[0]
r = resample_episode(read_episode_raw(Path(a.src), ep))
t = a.t
qpos = np.concatenate([r["leg_joints"][t, 0:12], r["leg_joints"][t, 12:15], r["arm_joints"][t], r["hand_joints"][t]]).astype(np.float32)[None]
import av  # noqa: E402
want = int(round(t * 50 / 30))
with av.open(str(Path(a.src) / f"videos/chunk-000/egocentric/episode_{ep:06d}.mp4")) as c:
    for i, f in enumerate(c.decode(video=0)):
        if i == want:
            img = f.to_ndarray(format="rgb24")
            break
print("episódio", ep, "t", t, "imagem", img.shape, img.dtype, "qpos", qpos.shape, flush=True)

args = argparse.Namespace(oracle=None, profile=a.profile, instruction="pick up the blue tote from the shelf and bring it to the table.",
                          ckpt_path=a.ckpt_path, base_vlm=a.base_vlm, source_name=a.source_name, image_size=[336, 448],
                          use_bf16=a.device == "cuda", device=a.device, debug_dir=None)
t0 = time.time()
pol = Policy(args)
print(f"modelo carregado em {time.time() - t0:.1f}s; fonte de normalização com {len(pol.stats.action_offset)}D ação", flush=True)
req = {"image": {"rgb_head_stereo_left": img}, "instruction": args.instruction,
       "state": {"joint_qpos": qpos, "base_quat": np.array([[1, 0, 0, 0]], np.float32), "yaw": np.array([0.0], np.float32)}}
res = {"episode": ep, "t": t, "profile": a.profile, "device": a.device}
try:
    t0 = time.time()
    A = pol.act(req)
    res["forward_s"] = time.time() - t0
    res["shape"] = list(A.shape)
    res["finite"] = bool(np.isfinite(A).all())
    gt = r["action"][t:t + 30]
    for i, side in enumerate(("left", "right")):
        sl = slice(14 + 7 * i, 21 + 7 * i)
        Tp = fk_ee(A[:, [30, 28, 29]].astype(np.float64), A[:, sl].astype(np.float64), side)
        Tg = fk_ee(gt[:, [30, 28, 29]], gt[:, sl], side)
        res[f"{side}_ee_pos_err_vs_gt_m_mean"] = float(np.linalg.norm(Tp[:, :3, 3] - Tg[:, :3, 3], axis=1).mean())
    res["action_range"] = [float(A.min()), float(A.max())]
    print("OK", json.dumps(res), flush=True)
except Exception:
    res["error"] = traceback.format_exc()
    print("FALHOU", res["error"], flush=True)
Path(a.out).mkdir(parents=True, exist_ok=True)
json.dump(res, open(Path(a.out) / "dryrun.json", "w"), indent=2)
