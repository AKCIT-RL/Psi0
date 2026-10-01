"""Gera fixtures de FK com pinocchio (lib independente) para tests/wla_adapter/test_fk.py.

Roda no venv `_scratch/.venv-fk` (pin + numpy). Não importa wla_adapter.
Uso: python scripts/wla/f2a_fk_fixtures.py [--n 1000] [--out _scratch/fixtures/fk_pinocchio.npz]
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pinocchio as pin

REPO = Path(__file__).resolve().parents[2]
URDF = REPO / "real/assets/g1/g1_body29_hand14.urdf"
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default=str(REPO / "_scratch/fixtures/fk_pinocchio.npz"))
    args = ap.parse_args()

    limits = {}
    for j in ET.parse(URDF).getroot().findall("joint"):
        lim = j.find("limit")
        if j.get("type") == "revolute" and lim is not None:
            limits[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))

    model = pin.buildModelFromUrdf(str(URDF))
    data = model.createData()
    rng = np.random.default_rng(0)
    out = {"pin_version": np.array(pin.__version__)}
    q_waist = np.stack([rng.uniform(*limits[n], args.n) for n in WAIST], axis=1)
    out["q_waist_yrp"] = q_waist

    def place(q, fid):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return data.oMf[fid].homogeneous

    for side in ("left", "right"):
        names = [f"{side}_{a}_joint" for a in ARM]
        q_arm = np.stack([rng.uniform(*limits[n], args.n) for n in names], axis=1)
        out[f"q_arm_{side}"] = q_arm
        tips = {tip: model.getFrameId(f"{side}_{tip}_link") for tip in ("wrist_yaw", "hand_palm")}
        torso = model.getFrameId("torso_link")
        res = {(tip, base): np.zeros((args.n, 4, 4)) for tip in tips for base in ("pelvis", "torso_link")}
        for i in range(args.n):
            q = pin.neutral(model)
            for k, n in enumerate(WAIST + names):
                q[model.joints[model.getJointId(n)].idx_q] = (q_waist[i, k] if k < 3 else q_arm[i, k - 3])
            T_torso = place(q, torso)
            for tip, fid in tips.items():
                T = data.oMf[fid].homogeneous
                res[(tip, "pelvis")][i] = T
                res[(tip, "torso_link")][i] = np.linalg.inv(T_torso) @ T
        for (tip, base), v in res.items():
            out[f"T_{side}_{tip}_{base}"] = v

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print("fixtures:", args.out, {k: v.shape for k, v in out.items() if v.ndim})


if __name__ == "__main__":
    main()
