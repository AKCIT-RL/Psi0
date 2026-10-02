#!/usr/bin/env python3
"""Executado x pretendido: o robô (SONIC) está executando o que o modelo pediu?
Lê <run>/server_debug/step_*.npz (state 60D medido, action 36D do chunk) e, para cada chunk i, compara a pose do EE
pretendida em h=29 (~1 s) com a medida no chunk i+1 (mesma FK). Também: fração do deslocamento pretendido realizada,
amplitude prevista por chunk e fecho da mão pedido x medido. Sem GPU.
"""
import glob, json, sys
from pathlib import Path
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "src")]
from wla_adapter.geometry import fk_ee  # noqa: E402

run = Path(sys.argv[1]); fs = sorted(glob.glob(str(run / "server_debug/step_*.npz")))
S = np.array([np.load(f)["state"] for f in fs], np.float64); A = np.array([np.load(f)["action"] for f in fs], np.float64)
XYZ = {"left": slice(0, 3), "right": slice(16, 19)}
res = {"n_chunks": len(fs)}
for side, i0 in (("left", 14), ("right", 21)):
    err, frac, amp = [], [], []
    for i in range(len(fs) - 1):
        if np.abs(S[i + 1, XYZ[side]] - S[i, XYZ[side]]).max() > 0.6:  # troca de episódio/reset
            continue
        T = fk_ee(A[i][:, [30, 28, 29]], A[i][:, i0:i0 + 7], side)[:, :3, 3]
        want, got = T[-1] - S[i, XYZ[side]], S[i + 1, XYZ[side]] - S[i, XYZ[side]]
        err.append(np.linalg.norm(T[-1] - S[i + 1, XYZ[side]]) * 100)
        amp.append(np.linalg.norm(T[-1] - T[0]) * 100)
        if np.linalg.norm(want) > 0.02: frac.append(float(got @ want / (want @ want)))
    res[side] = {"intended_vs_measured_cm_mean": float(np.mean(err)), "p90": float(np.percentile(err, 90)),
                 "chunk_amplitude_cm_mean": float(np.mean(amp)), "progress_fraction_median": float(np.median(frac)) if frac else None,
                 "n_pairs": len(err)}
cmd = np.max(A[:, :, 7:14], axis=(1, 2)); res["right_hand_cmd_max_joint_rad"] = float(cmd.max())
res["right_hand_measured_closure_min"] = float(S[:, 26:32].min())
res["base_vx_cmd_mean"] = float(A[:, :, 32].mean())
print(json.dumps(res, indent=1)); json.dump(res, open(run / "tracking.json", "w"), indent=1)
