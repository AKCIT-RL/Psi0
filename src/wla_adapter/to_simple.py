"""Saída do WLA (resultado de pipeline.invert_action) -> ação Ψ0 de 36D que o cliente SIMPLE executa.

Inverso de convert_action.action_keys (conversion_spec.md §4.1). Layout Ψ0 (36D):
  [0:14]  alvos de mão, ASSIMÉTRICO (esq t3,m2,i2 | dir t3,i2,m2)
  [14:21] braço esq (ordem URDF)   [21:28] braço dir
  [28:31] cintura (roll, pitch, yaw)
  [31] altura  [32] vx  [33] vy  [34] vyaw  [35] target_yaw (heading absoluto, integrado de vyaw)

EE absoluto -> juntas por IK numérica (geometry.ik), com seed = solução do passo anterior
(no 1º passo, as juntas medidas). O braço tem 7 DoF para uma pose de 6: |q - q_dataset| não
vai a zero, então a equivalência verificada é em task-space (FK(IK(T)) == T), não em juntas.
"""

import numpy as np

from .convert_action import BASE_CMD_COLS, WAIST_YRP_COLS
from .geometry import ik
from .hands import fig6d_to_joints, symmetric_to_action_hands

PSI0_ACTION_DIM = 36
ARM_COLS = {"left": slice(14, 21), "right": slice(21, 28)}


def integrate_target_yaw(vyaw: np.ndarray, yaw0: float, dt: float) -> np.ndarray:
    """target_yaw[k] = yaw0 + sum_{i<=k} vyaw[i]*dt, enrolado em (-pi, pi]."""
    yaw = yaw0 + np.cumsum(np.asarray(vyaw, np.float64)) * dt
    return (yaw + np.pi) % (2 * np.pi) - np.pi


def to_psi0_action(
    inv: dict[str, np.ndarray],
    q_arm_seed: np.ndarray,
    yaw0: float = 0.0,
    dt: float = 1.0 / 30.0,
    ik_kwargs: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """inv: saída de pipeline.invert_action (H passos). q_arm_seed (14,) = [esq 7, dir 7].

    Retorna (A (H,36) float64, info) com info = {"pos_err": (H,2), "rot_err": (H,2), "converged": (H,2)}.
    """
    kw = dict(max_iter=100, tol_pos=1e-6, tol_rot=1e-6) | (ik_kwargs or {})
    H = inv["waist"].shape[0]
    A = np.zeros((H, PSI0_ACTION_DIM))
    waist_yrp = np.asarray(inv["waist"], np.float64)
    A[:, list(WAIST_YRP_COLS)] = waist_yrp
    sym = np.concatenate([fig6d_to_joints(inv["left_fig6d"], "left"),
                          fig6d_to_joints(inv["right_fig6d"], "right")], axis=-1)
    A[:, 0:14] = symmetric_to_action_hands(sym)
    bc = np.asarray(inv["base_command"], np.float64)  # [vx, vy, vyaw, height]
    A[:, list(BASE_CMD_COLS)] = bc
    A[:, 35] = integrate_target_yaw(bc[:, 2], yaw0, dt)

    info = {k: np.zeros((H, 2)) for k in ("pos_err", "rot_err")} | {"converged": np.zeros((H, 2), bool)}
    for i, side in enumerate(("left", "right")):
        q = np.asarray(q_arm_seed, np.float64)[7 * i:7 * i + 7].copy()
        for k in range(H):
            _, q, r = ik(inv[f"{side}_ee_T"][k], side, q, q_waist0=waist_yrp[k], **kw)
            A[k, ARM_COLS[side]] = q
            info["pos_err"][k, i], info["rot_err"][k, i] = r["pos_err"], r["rot_err"]
            info["converged"][k, i] = r["converged"]
    return A, info
