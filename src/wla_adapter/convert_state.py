"""Estado Ψ0 (reamostrado a 30 FPS) -> chaves WBT do dataset convertido (D2).

Entrada `raw`: colunas Ψ0 já reamostradas (resample.py), float64:
  hand_joints (M,14) ordem simétrica; arm_joints (M,14) ordem URDF; leg_joints (M,15)
  = [left_leg6, right_leg6, waist yaw,roll,pitch].
Saída: chaves de estado do schema WBT (conversion_spec.md §4.2):
  observation.state.{left,right}_ee_pose_gripper_base  xyz_rpy, FK base pelvis + offset E
  observation.state.{left,right}_fig6d                 Dex3 -> fig6d (1=aberto)
  observation.state.waist_state_joint                  leg_joints[12:15] (yaw,roll,pitch)
  observation.state.{left,right}_leg                   leg_joints[0:6] / [6:12]
"""

import numpy as np

from .geometry import fk_ee, se3_to_xyz_rpy
from .hands import joints_to_fig6d

S_EE = "observation.state.{side}_ee_pose_gripper_base"
S_FIG = "observation.state.{side}_fig6d"
S_WAIST = "observation.state.waist_state_joint"
S_LEG = "observation.state.{side}_leg"

STATE_OUT_KEYS = [
    S_EE.format(side="left"), S_EE.format(side="right"),
    S_FIG.format(side="left"), S_FIG.format(side="right"),
    S_WAIST, S_LEG.format(side="left"), S_LEG.format(side="right"),
]


def ee_pose_xyz_rpy(q_waist_yrp: np.ndarray, q_arm: np.ndarray, side: str) -> np.ndarray:
    """(M,3)+(M,7) -> (M,6) xyz_rpy do EE gripper_base (base pelvis, offset E da F2a)."""
    return se3_to_xyz_rpy(fk_ee(q_waist_yrp, q_arm, side))


def state_keys(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    leg = np.asarray(raw["leg_joints"], dtype=np.float64)
    arm = np.asarray(raw["arm_joints"], dtype=np.float64)
    hand = np.asarray(raw["hand_joints"], dtype=np.float64)
    q_waist = leg[:, 12:15]  # (yaw, roll, pitch) — ordem URDF, verificada na F2a
    for i, side in enumerate(("left", "right")):
        out[S_EE.format(side=side)] = ee_pose_xyz_rpy(q_waist, arm[:, 7 * i:7 * i + 7], side)
        out[S_FIG.format(side=side)] = joints_to_fig6d(hand[:, 7 * i:7 * i + 7], side)
        out[S_LEG.format(side=side)] = leg[:, 6 * i:6 * i + 6]
    out[S_WAIST] = q_waist
    return {k: v.astype(np.float32) for k, v in out.items()}
