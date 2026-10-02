"""FK do G1 em numpy puro (URDF) + helpers SE(3) na convenção do WLA.

Requer o venv do WLA (importa se3_utils do submódulo, sem modificá-lo).
Convenções (docs/wla/conversion_spec.md §1.6): xyz_rpy = Rotation.from_euler("xyz"),
rot6d = [R00,R10,R20,R01,R11,R21], rotvec do scipy.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from unifolm_wla.dataloader.multi_source_dataset.se3_utils import (  # noqa: F401
    matrix_to_rot6d,
    matrix_to_rotvec,
    pose_to_se3,
    pose_to_se3_from_format,
    rotvec_to_matrix,
    rpy_to_matrix,
    se3_inverse,
    se3_to_xyz_rotvec,
)

URDF_PATH = Path(__file__).resolve().parents[2] / "real/assets/g1/g1_body29_hand14.urdf"

WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
ARM_JOINT_NAMES = (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_roll", "wrist_pitch", "wrist_yaw",
)


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    type: str
    origin: np.ndarray  # (4,4) parent->joint frame
    axis: np.ndarray  # (3,)
    lower: float
    upper: float


@lru_cache(maxsize=None)
def load_urdf(path: str = str(URDF_PATH)) -> dict[str, Joint]:
    """Mapeia link filho -> junta (a árvore do G1 tem um pai por link)."""
    joints = {}
    for el in ET.parse(path).getroot().findall("joint"):
        o = el.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
        rpy = np.array([float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()])
        ax = el.find("axis")
        axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.array([0.0, 0.0, 1.0])
        lim = el.find("limit")
        lower = float(lim.get("lower", -np.inf)) if lim is not None else -np.inf
        upper = float(lim.get("upper", np.inf)) if lim is not None else np.inf
        j = Joint(
            name=el.get("name"),
            parent=el.find("parent").get("link"),
            child=el.find("child").get("link"),
            type=el.get("type"),
            origin=pose_to_se3(xyz, rpy_to_matrix(rpy)),
            axis=axis / np.linalg.norm(axis),
            lower=lower,
            upper=upper,
        )
        joints[j.child] = j
    return joints


def arm_joint_names(side: str) -> list[str]:
    return [f"{side}_{n}_joint" for n in ARM_JOINT_NAMES]


def joint_limits(side: str, path: str = str(URDF_PATH)) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) de [waist yaw,roll,pitch, braço 7 (ordem URDF)]."""
    by_name = {j.name: j for j in load_urdf(path).values()}
    names = list(WAIST_JOINTS) + arm_joint_names(side)
    return (np.array([by_name[n].lower for n in names]), np.array([by_name[n].upper for n in names]))


def _axis_rotation(axis: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rodrigues: (...,) ângulos -> (...,4,4) transformação homogênea."""
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    s, c = np.sin(q)[..., None, None], np.cos(q)[..., None, None]
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    T = np.zeros(q.shape + (4, 4))
    T[..., :3, :3] = R
    T[..., 3, 3] = 1.0
    return T


def chain_joints(base_link: str, tip_link: str, path: str = str(URDF_PATH)) -> list[Joint]:
    joints = load_urdf(path)
    out, link = [], tip_link
    while link != base_link:
        if link not in joints:
            raise ValueError(f"{tip_link!r} não é descendente de {base_link!r}")
        out.append(joints[link])
        link = joints[link].parent
    return out[::-1]


def fk_chain(q: dict[str, np.ndarray], base_link: str, tip_link: str, path: str = str(URDF_PATH)) -> np.ndarray:
    """Pose de `tip_link` no frame de `base_link`; q: nome da junta -> (...,) rad."""
    chain = chain_joints(base_link, tip_link, path)
    shape = np.broadcast_shapes(*[np.shape(q[j.name]) for j in chain if j.type == "revolute"])
    T = np.broadcast_to(np.eye(4), shape + (4, 4)).copy()
    for j in chain:
        T = T @ j.origin
        if j.type == "revolute":
            T = T @ _axis_rotation(j.axis, np.asarray(q[j.name], dtype=np.float64))
        elif j.type != "fixed":
            raise ValueError(f"tipo de junta não suportado: {j.type}")
    return T


def fk(
    q_waist_yrp: np.ndarray,
    q_arm: np.ndarray,
    side: str,
    base_link: str = "pelvis",
    tip_link: str | None = None,
) -> np.ndarray:
    """FK (...,4,4). q_waist_yrp (...,3)=[yaw,roll,pitch]; q_arm (...,7) na ordem do URDF
    [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw].
    tip_link: nome completo ou curto ("wrist_yaw", "hand_palm"); padrão {side}_wrist_yaw_link.
    Com base_link="torso_link" a cintura não entra na cadeia."""
    if side not in ("left", "right"):
        raise ValueError(side)
    if tip_link is None:
        tip_link = f"{side}_wrist_yaw_link"
    elif not tip_link.endswith("_link"):
        tip_link = f"{side}_{tip_link}_link"
    q_waist_yrp, q_arm = np.asarray(q_waist_yrp, dtype=np.float64), np.asarray(q_arm, dtype=np.float64)
    q = {n: q_waist_yrp[..., i] for i, n in enumerate(WAIST_JOINTS)}
    q.update({n: q_arm[..., i] for i, n in enumerate(arm_joint_names(side))})
    return fk_chain(q, base_link, tip_link)


def xyz_rpy_to_se3(pose: np.ndarray) -> np.ndarray:
    return pose_to_se3_from_format(np.asarray(pose), "xyz_rpy")


def se3_to_xyz_rpy(T: np.ndarray) -> np.ndarray:
    shape = T.shape[:-2]
    rpy = Rotation.from_matrix(T[..., :3, :3].reshape(-1, 3, 3)).as_euler("xyz").reshape(*shape, 3)
    return np.concatenate([T[..., :3, 3], rpy], axis=-1)


def xyz_rotvec_to_se3(pose: np.ndarray) -> np.ndarray:
    return pose_to_se3_from_format(np.asarray(pose), "xyz_rvec")


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """Inversa de matrix_to_rot6d (Gram-Schmidt sobre as 2 primeiras colunas)."""
    r6 = np.asarray(r6, dtype=np.float64)
    a1, a2 = r6[..., 0:3], r6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def xyz_rot6d_to_se3(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return pose_to_se3(p[..., :3], rot6d_to_matrix(p[..., 3:9]))


def se3_to_xyz_rot6d(T: np.ndarray) -> np.ndarray:
    return np.concatenate([T[..., :3, 3], matrix_to_rot6d(T[..., :3, :3])], axis=-1)


def rel_to_abs(T_curr: np.ndarray, rel_xyz_rotvec: np.ndarray) -> np.ndarray:
    """T_abs = T_curr @ T_rel (inversa de compute_relative_actions)."""
    return T_curr @ xyz_rotvec_to_se3(rel_xyz_rotvec)


def rotation_angle(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Distância geodésica (rad) entre rotações, (...,3,3)."""
    return np.linalg.norm(matrix_to_rotvec(np.swapaxes(Ra, -1, -2) @ Rb), axis=-1)


# Offset fixo tip(wrist_yaw_link)->EE "gripper_base" (translação no frame do tip, rotação
# identidade), calibrado no WBT oficial na F2a ($WLA_EXP/f2_validation/wbt_facts.json,
# VERIFIED: p99 pos <= 3.9 mm, rot ~0). conversion_spec.md §8 "Resolvido na F2a".
EE_OFFSET_XYZ = {
    "left": np.array([0.10994945822848229, -0.00021625560909691147, 0.0]),
    "right": np.array([0.10983074060032255, 0.0006490814488727838, 0.0]),
}


def ee_offset(side: str) -> np.ndarray:
    """(4,4) transformação fixa tip -> EE (gripper_base) para o lado dado."""
    E = np.eye(4)
    E[:3, 3] = EE_OFFSET_XYZ[side]
    return E


def fk_ee(q_waist_yrp: np.ndarray, q_arm: np.ndarray, side: str) -> np.ndarray:
    """FK até o EE gripper_base: fk(pelvis->{side}_wrist_yaw_link) @ E (...,4,4)."""
    return fk(q_waist_yrp, q_arm, side) @ ee_offset(side)


def ik(
    T_target: np.ndarray,
    side: str,
    q_arm0: np.ndarray,
    q_waist0: np.ndarray | None = None,
    fix_waist: bool = True,
    max_iter: int = 200,
    tol_pos: float = 1e-9,
    tol_rot: float = 1e-9,
    lam: float = 1e-6,
    eps: float = 1e-8,
    max_step: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """IK numpy por damped least squares (Jacobiano numérico) para o braço do G1.

    Resolve q tal que fk_ee(q_waist, q_arm, side) ≈ T_target. Com fix_waist=True
    (padrão de deploy: a cintura vem de action.waist_action_joint) só as 7 juntas do
    braço são otimizadas (redundância 7-DoF -> |q - q'| não vai a zero; verificar EE).
    Seed: q_arm0 (e q_waist0) — usar q do frame anterior no deploy.

    Retorna (q_waist, q_arm, info) com info = {"converged", "iters", "pos_err", "rot_err"}.
    """
    T_target = np.asarray(T_target, dtype=np.float64)
    lo, hi = joint_limits(side)
    q = np.concatenate([np.zeros(3) if q_waist0 is None else np.asarray(q_waist0, np.float64),
                        np.asarray(q_arm0, np.float64)])
    free = np.arange(3, 10) if fix_waist else np.arange(10)
    lo, hi = np.clip(lo, -10, 10), np.clip(hi, -10, 10)

    def err(qv: np.ndarray) -> np.ndarray:
        T_cur = fk_ee(qv[:3], qv[3:], side)
        return se3_to_xyz_rotvec(se3_inverse(T_cur) @ T_target)

    e = err(q)
    info = {"converged": False, "iters": 0, "pos_err": float(np.linalg.norm(e[:3])),
            "rot_err": float(np.linalg.norm(e[3:]))}
    for it in range(max_iter):
        if info["pos_err"] < tol_pos and info["rot_err"] < tol_rot:
            info["converged"] = True
            break
        J = np.empty((6, len(free)))
        for k, i in enumerate(free):
            dq = q.copy()
            dq[i] += eps
            J[:, k] = (err(dq) - e) / eps
        # J = d(err)/dq e err decresce ao aproximar do alvo => J = -J_geo; passo de Gauss-Newton: dq = -J^T (J J^T + lam I)^-1 e
        dq_free = -J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
        nrm = np.linalg.norm(dq_free)
        if nrm > max_step:
            dq_free *= max_step / nrm
        q[free] = np.clip(q[free] + dq_free, lo[free], hi[free])
        e = err(q)
        info.update(iters=it + 1, pos_err=float(np.linalg.norm(e[:3])),
                    rot_err=float(np.linalg.norm(e[3:])))
    return q[:3], q[3:], info
