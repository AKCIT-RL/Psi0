#!/usr/bin/env python3
"""Round-trip ação Ψ0 -> adaptador (servidor --oracle adapter, via HTTP/HttpActionClient real do SIMPLE) -> ação Ψ0.

Sem simulador e sem modelo (CPU). Para cada episódio do split, a cada --stride passos (30 FPS), envia
gt_action (chunk de 30 ações Ψ0) + estado medido (joint_qpos na ordem do SIMPLE) ao servidor e compara a
resposta com o original. Valida em uma passada: protocolo JSON/numpy, ordem de juntas do estado, IK com seed
do estado medido e a equivalência em task-space (EE por FK), mãos (juntas e classe aberto/fechado), cintura,
base e target_yaw. Regra de ouro 1: use --split train ou val; test só no relatório final congelado.
"""
import argparse
import importlib.util
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

PSI0 = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PSI0 / "src"), str(PSI0 / "third_party/unifolm-wla"), str(PSI0 / "scripts/wla")]

from simple_wla_server import Policy, make_handler  # noqa: E402
from wla_adapter.geometry import fk_ee, rotation_angle  # noqa: E402
from wla_adapter.hands import closure  # noqa: E402
from wla_adapter.write_dataset import read_episode_raw, resample_episode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/raid/user_marcospaulo/datasets/psi0/G1ToteMix-psi0")
ap.add_argument("--split", required=True, choices=["train", "val", "test"])
ap.add_argument("--limit", type=int, default=None, help="n episódios")
ap.add_argument("--stride", type=int, default=30)
ap.add_argument("--port", type=int, default=22185)
ap.add_argument("--out", required=True)
a = ap.parse_args()

# cliente HTTP real do SIMPLE (carregado por caminho: não importa o pacote simple inteiro)
spec = importlib.util.spec_from_file_location("simple_client", PSI0 / "third_party/SIMPLE/src/simple/baselines/client.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

srv_args = argparse.Namespace(oracle="adapter", profile="psi0_tote", instruction="", debug_dir=None)
srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(Policy(srv_args)))
threading.Thread(target=srv.serve_forever, daemon=True).start()
client = sc.HttpActionClient("127.0.0.1", a.port)

eps = sorted(json.load(open(PSI0 / "docs/wla/split.json"))[a.split])[: a.limit]
H = 30
rows = []
for e in eps:
    r = resample_episode(read_episode_raw(Path(a.src), e))
    act, arm, hand, leg = r["action"], r["arm_joints"], r["hand_joints"], r["leg_joints"]
    n = len(act)
    for t in range(0, n, a.stride):
        win = act[t:t + H]
        if len(win) < H:
            win = np.concatenate([win, np.repeat(win[-1:], H - len(win), axis=0)])
        # joint_qpos na ordem do SIMPLE: pernas(12), cintura(3), braço esq(7), dir(7), mão esq(7), dir(7)
        qpos = np.concatenate([leg[t, 0:12], leg[t, 12:15], arm[t], hand[t]]).astype(np.float32)[None]
        pred, err, _ = client.query_action({}, "x", {"joint_qpos": qpos}, {}, history={}, dataset="simple",
                                           gt_action=win.astype(np.float32))
        assert err == 0.0 and pred.shape == (H, 36), (err, pred.shape)
        rec = {"ep": e, "t": t}
        for i, side in enumerate(("left", "right")):
            sl = slice(14 + 7 * i, 21 + 7 * i)
            Tg = fk_ee(win[:, [30, 28, 29]], win[:, sl], side)
            Tp = fk_ee(pred[:, [30, 28, 29]].astype(np.float64), pred[:, sl].astype(np.float64), side)
            rec[f"{side}_ee_pos_mm"] = float(np.linalg.norm(Tp[:, :3, 3] - Tg[:, :3, 3], axis=1).max() * 1e3)
            rec[f"{side}_ee_rot_deg"] = float(np.degrees(rotation_angle(Tp[:, :3, :3], Tg[:, :3, :3]).max()))
            rec[f"{side}_arm_joint_maxabs_rad"] = float(np.abs(pred[:, sl] - win[:, sl]).max())
        rec["hand_joint_maxabs_rad"] = float(np.abs(pred[:, 0:14] - win[:, 0:14]).max())
        rec["waist_maxabs"] = float(np.abs(pred[:, 28:31] - win[:, 28:31]).max())
        rec["base_cmd_maxabs"] = float(np.abs(pred[:, 31:35] - win[:, 31:35]).max())
        d = (pred[:, 35] - win[:, 35] + np.pi) % (2 * np.pi) - np.pi
        rec["target_yaw_maxabs"] = float(np.abs(d).max())
        # classe aberto/fechado (fecho > 0.5) por mão: sim = ordem simétrica (ação -> assimétrica na esquerda)
        from wla_adapter.hands import action_hands_to_symmetric
        cg = np.concatenate([closure(action_hands_to_symmetric(win[:, 0:14])[:, 7 * i:7 * i + 7], s)
                             for i, s in enumerate(("left", "right"))], axis=1)
        cp = np.concatenate([closure(action_hands_to_symmetric(pred[:, 0:14].astype(np.float64))[:, 7 * i:7 * i + 7], s)
                             for i, s in enumerate(("left", "right"))], axis=1)
        rec["hand_class_mismatch"] = float(((cg.mean(1) > 0.5) != (cp.mean(1) > 0.5)).mean())
        rows.append(rec)
    print(f"ep {e}: {len(rows)} chunks acumulados", flush=True)
srv.shutdown()

keys = [k for k in rows[0] if k not in ("ep", "t")]
summary = {k: {"mean": float(np.mean([r[k] for r in rows])), "p99": float(np.percentile([r[k] for r in rows], 99)),
               "max": float(np.max([r[k] for r in rows]))} for k in keys}
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
json.dump({"split": a.split, "episodes": eps, "chunks": len(rows), "stride": a.stride, "summary": summary},
          open(out / "report.json", "w"), indent=2)
json.dump(rows, open(out / "chunks.json", "w"))
print(json.dumps(summary, indent=2))
