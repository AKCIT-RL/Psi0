"""Ação Ψ0 (36D, reamostrada a 30 FPS) -> chaves WBT de ação do dataset convertido (D2).

Fontes (conversion_spec.md §3/§4.1):
  action[0:14]  alvos de mão, layout ASSIMÉTRICO (esq t3,m2,i2; dir t3,i2,m2) -> reordenar
  action[14:28] alvos de braço (ordem URDF; esq [14:21], dir [21:28])
  action[[30,28,29]] alvo de cintura (yaw,roll,pitch)  (colunas crus: roll,pitch,yaw em [28:31])
  action[31] altura; [32] vx; [33] vy; [34] vyaw
Saída:
  action.{left,right}_ee_pose_gripper_base  xyz_rpy, FK das juntas-ALVO + offset E
  action.{left,right}_fig6d                 Dex3 alvo -> fig6d (1=aberto)
  action.waist_action_joint                 (yaw,roll,pitch)
  action.base_command                       [vx, vy, vyaw, height]  (dims vx=0,vy=1,vw=2,height=3)
"""

import numpy as np

from .convert_state import ee_pose_xyz_rpy
from .hands import action_hands_to_symmetric, joints_to_fig6d

A_EE = "action.{side}_ee_pose_gripper_base"
A_FIG = "action.{side}_fig6d"
A_WAIST = "action.waist_action_joint"
A_BASE_CMD = "action.base_command"

ACTION_OUT_KEYS = [
    A_EE.format(side="left"), A_EE.format(side="right"),
    A_FIG.format(side="left"), A_FIG.format(side="right"),
    A_WAIST, A_BASE_CMD,
]

# colunas do action Ψ0 -> (yaw, roll, pitch)
WAIST_YRP_COLS = (30, 28, 29)
# action.base_command = [vx, vy, vyaw(angle_z), height]
BASE_CMD_COLS = (32, 33, 34, 31)


def action_keys(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    a = np.asarray(raw["action"], dtype=np.float64)
    q_waist = a[:, list(WAIST_YRP_COLS)]
    hands_sym = action_hands_to_symmetric(a[:, 0:14])
    out: dict[str, np.ndarray] = {}
    for i, side in enumerate(("left", "right")):
        out[A_EE.format(side=side)] = ee_pose_xyz_rpy(q_waist, a[:, 14 + 7 * i:14 + 7 * i + 7], side)
        out[A_FIG.format(side=side)] = joints_to_fig6d(hands_sym[:, 7 * i:7 * i + 7], side)
    out[A_WAIST] = q_waist
    out[A_BASE_CMD] = a[:, list(BASE_CMD_COLS)]
    return {k: v.astype(np.float32) for k, v in out.items()}
