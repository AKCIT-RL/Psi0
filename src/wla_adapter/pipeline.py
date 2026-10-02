"""Reimplementação independente (numpy) do preprocessing `unitree_fullbody_base` do WLA
a partir das colunas cruas do parquet, mais a inversa `invert_action`.

Usa só funções puras do WLA (se3_utils, stats_utils, action_mapping.SLICES/STATE_SLICES);
NÃO usa a classe do dataset. Comportamento de referência: single_source_dataset.py
(`__getitem__`, `_precompute_normalizers`, `_build_*_norm_vectors`) e lerobot
(`_get_query_indices`: janela t..t+H-1 com clamp ao último frame do episódio).
"""

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES, STATE_SLICES
from unifolm_wla.dataloader.multi_source_dataset.se3_utils import (
    pose_to_se3_from_format,
    pose_to_xyz_rot6d_from_format,
    se3_inverse,
    se3_to_xyz_rotvec,
)
from unifolm_wla.dataloader.multi_source_dataset.stats_utils import (
    get_normalizer,
    load_relative_stats,
    load_stats,
)

from .geometry import rot6d_to_matrix, se3_to_xyz_rpy, xyz_rotvec_to_se3

H = 30
UNIFIED_DIM, STATE_DIM = 54, 60
WLA_STATS_DIR = (
    Path(__file__).resolve().parents[2]
    / "third_party/unifolm-wla/unifolm_wla/dataloader/multi_source_dataset/stats"
)

# Chaves de parquet do `unitree_fullbody_base` (configs/wla/unitree_wbt_shelf.yaml).
A_EE = {"left": "action.left_ee_pose_gripper_base", "right": "action.right_ee_pose_gripper_base"}
S_EE = {"left": "observation.state.left_ee_pose_gripper_base", "right": "observation.state.right_ee_pose_gripper_base"}
A_FIG = {"left": "action.left_fig6d", "right": "action.right_fig6d"}
S_FIG = {"left": "observation.state.left_fig6d", "right": "observation.state.right_fig6d"}
A_LEG = {"left": "action.left_leg", "right": "action.right_leg"}
S_LEG = {"left": "observation.state.left_leg", "right": "observation.state.right_leg"}
A_WAIST, S_WAIST = "action.waist_action_joint", "observation.state.waist_state_joint"
A_BASE_CMD = "action.base_command"
A_BASE_POSE, S_BASE_POSE = "action.action_base_pose", "observation.state.state_base_pose"
S_BASE_ROT = "observation.state.state_base_rot"
# base_command [vx, vy, angle_z, height] -> slots de ação
BASE_CMD_SLOTS = [32, 33, 34, 41]

ACTION_KEYS = [*A_EE.values(), *A_FIG.values(), *A_LEG.values(), A_WAIST, A_BASE_CMD, A_BASE_POSE]
STATE_KEYS = [*S_EE.values(), *S_FIG.values(), *S_LEG.values(), S_WAIST, S_BASE_ROT, S_BASE_POSE]

# Slots ativos (máscara) — exatamente os do config WBT.
ACTIVE_ACTION_SLICES = [
    "left_xyz_rotvec", "left_fig6d", "right_xyz_rotvec", "right_fig6d", "waist_joint",
    "base_vx_vy", "base_vw", "base_rotvec", "height", "left_leg_joint", "right_leg_joint",
]
ACTIVE_STATE_SLICES = [
    "left_xyz_rot6d", "left_fig6d", "right_xyz_rot6d", "right_fig6d", "waist_joint",
    "base_rotvec", "left_leg_joint", "right_leg_joint",
]


@dataclass(frozen=True)
class Profile:
    """Perfil de schema: quais chaves/slots o dataset convertido possui."""

    name: str
    active_action_slices: tuple[str, ...]
    active_state_slices: tuple[str, ...]
    has_action_legs: bool
    has_state_legs: bool
    has_base_pose: bool
    has_base_rot: bool
    default_stats_dir: str | None = None

    @property
    def action_keys(self) -> list[str]:
        keys = [*A_EE.values(), *A_FIG.values(), A_WAIST, A_BASE_CMD]
        if self.has_action_legs:
            keys += [*A_LEG.values()]
        if self.has_base_pose:
            keys += [A_BASE_POSE]
        return keys

    @property
    def state_keys(self) -> list[str]:
        keys = [*S_EE.values(), *S_FIG.values(), S_WAIST]
        if self.has_state_legs:
            keys += [*S_LEG.values()]
        if self.has_base_rot:
            keys += [S_BASE_ROT]
        if self.has_base_pose:
            keys += [S_BASE_POSE]
        return keys


# WBT oficial (200 ep): EE+fig6d+waist+base_command+base_pose+pernas (ação e estado).
WBT = Profile(
    name="wbt",
    active_action_slices=tuple(ACTIVE_ACTION_SLICES),
    active_state_slices=tuple(ACTIVE_STATE_SLICES),
    has_action_legs=True, has_state_legs=True, has_base_pose=True, has_base_rot=True,
    default_stats_dir=str(WLA_STATS_DIR),
)

# Ψ0 convertido (F2b, D2): sem ação de perna, sem base_pose/base_rot (sem odometria/IMU).
# ação ativa = 31/54; estado ativo = 45/60.
PSI0_TOTE = Profile(
    name="psi0_tote",
    active_action_slices=(
        "left_xyz_rotvec", "left_fig6d", "right_xyz_rotvec", "right_fig6d", "waist_joint",
        "base_vx_vy", "base_vw", "height",
    ),
    active_state_slices=(
        "left_xyz_rot6d", "left_fig6d", "right_xyz_rot6d", "right_fig6d", "waist_joint",
        "left_leg_joint", "right_leg_joint",
    ),
    has_action_legs=False, has_state_legs=True, has_base_pose=False, has_base_rot=False,
    default_stats_dir=os.path.join(
        os.environ.get("WLA_DATA_ROOT", "/raid/user_marcospaulo/datasets/unifolm"),
        "stats_psi0_tote_train"),
)


@dataclass(frozen=True)
class NormStats:
    action_offset: np.ndarray  # (54,) float32; x_norm = (x - offset) / scale
    action_scale: np.ndarray
    state_offset: np.ndarray  # (60,)
    state_scale: np.ndarray


def _put(off, sc, sl, o, s):
    o, s = np.atleast_1d(o), np.atleast_1d(s)
    n = min(sl.stop - sl.start, len(o))
    off[sl.start:sl.start + n] = o[:n]
    sc[sl.start:sl.start + n] = s[:n]


def load_norm_stats(stats_dir: str | Path | None = None, profile: Profile = WBT) -> NormStats:
    """norm=minmax_q (stats.json), rel=zscore (relative_stats.json), fig6d=minmax_q."""
    if stats_dir is None:
        stats_dir = profile.default_stats_dir or WLA_STATS_DIR
    stats_dir = Path(stats_dir)
    data = load_stats(stats_dir / "stats.json")
    rel = load_relative_stats(stats_dir / "relative_stats.json")

    ao, asc = np.zeros(UNIFIED_DIM, np.float32), np.ones(UNIFIED_DIM, np.float32)
    for side in ("left", "right"):
        _put(ao, asc, SLICES[f"{side}_xyz_rotvec"], *get_normalizer(rel, f"{side}_ee_pose_gripper_base", "zscore"))
        _put(ao, asc, SLICES[f"{side}_fig6d"], *get_normalizer(data, A_FIG[side], "minmax_q"))
        if profile.has_action_legs:
            _put(ao, asc, SLICES[f"{side}_leg_joint"], *get_normalizer(data, A_LEG[side], "minmax_q"))
    _put(ao, asc, SLICES["waist_joint"], *get_normalizer(data, A_WAIST, "minmax_q"))
    if profile.has_base_pose:
        _put(ao, asc, SLICES["base_rotvec"], *get_normalizer(rel, "action_base_pose", "zscore"))
    bo, bs = get_normalizer(data, A_BASE_CMD, "minmax_q")
    ao[BASE_CMD_SLOTS], asc[BASE_CMD_SLOTS] = bo, bs

    so, ss = np.zeros(STATE_DIM, np.float32), np.ones(STATE_DIM, np.float32)
    for side in ("left", "right"):
        o, s = get_normalizer(data, S_EE[side], "minmax_q")
        _put(so, ss, slice(STATE_SLICES[f"{side}_xyz_rot6d"].start, STATE_SLICES[f"{side}_xyz_rot6d"].start + 3), o[:3], s[:3])  # só xyz
        _put(so, ss, STATE_SLICES[f"{side}_fig6d"], *get_normalizer(data, S_FIG[side], "minmax_q"))
        if profile.has_state_legs:
            _put(so, ss, STATE_SLICES[f"{side}_leg_joint"], *get_normalizer(data, S_LEG[side], "minmax_q"))
    _put(so, ss, STATE_SLICES["waist_joint"], *get_normalizer(data, S_WAIST, "minmax_q"))
    if profile.has_base_rot:
        o, s = get_normalizer(data, S_BASE_ROT, "minmax_q")  # [gravidade(3), omega(3)]: gravidade sem normalizar
        o, s = o.copy(), s.copy()
        o[:3], s[:3] = 0.0, 1.0
        _put(so, ss, STATE_SLICES["base_rotvec"], o, s)
    return NormStats(ao, asc, so, ss)


def _mask(slices: dict, names: list[str], dim: int) -> np.ndarray:
    m = np.zeros(dim, dtype=bool)
    for n in names:
        m[slices[n]] = True
    return m


def action_mask(profile: Profile = WBT) -> np.ndarray:
    return _mask(SLICES, list(profile.active_action_slices), UNIFIED_DIM)


def state_mask(profile: Profile = WBT) -> np.ndarray:
    return _mask(STATE_SLICES, list(profile.active_state_slices), STATE_DIM)


def state_unnorm(row: dict[str, np.ndarray], profile: Profile = WBT) -> np.ndarray:
    """(60,) float32 não normalizado; row: chave de parquet -> vetor do frame atual."""
    s = np.zeros(STATE_DIM, np.float32)
    for side in ("left", "right"):
        s[STATE_SLICES[f"{side}_xyz_rot6d"]] = pose_to_xyz_rot6d_from_format(np.asarray(row[S_EE[side]], np.float32), "xyz_rpy")
        s[STATE_SLICES[f"{side}_fig6d"]] = row[S_FIG[side]]
        if profile.has_state_legs:
            s[STATE_SLICES[f"{side}_leg_joint"]] = row[S_LEG[side]]
    s[STATE_SLICES["waist_joint"]] = row[S_WAIST]
    if profile.has_base_rot:
        s[STATE_SLICES["base_rotvec"]] = row[S_BASE_ROT]
    return s


def _rel(curr_pose, future_poses, fmt):
    """T_curr^-1 @ T_future -> xyz+rotvec (float64)."""
    T_c = pose_to_se3_from_format(np.asarray(curr_pose), fmt)
    T_f = pose_to_se3_from_format(np.asarray(future_poses), fmt)
    return se3_to_xyz_rotvec(se3_inverse(T_c) @ T_f)


def _norm(x, o, s):
    return (x - o) / s


def action_chunk(row: dict[str, np.ndarray], win: dict[str, np.ndarray], ns: NormStats, profile: Profile = WBT) -> np.ndarray:
    """(H,54) float32 normalizado. row: frame atual (estado); win: chave -> (H,D) futuro t..t+H-1."""
    ao, asc = ns.action_offset, ns.action_scale
    A = np.zeros((H, UNIFIED_DIM), np.float32)
    for side in ("left", "right"):
        sl = SLICES[f"{side}_xyz_rotvec"]
        A[:, sl] = _norm(_rel(row[S_EE[side]], win[A_EE[side]], "xyz_rpy"), ao[sl], asc[sl])  # float64 como no oficial
        sl = SLICES[f"{side}_fig6d"]
        A[:, sl] = _norm(win[A_FIG[side]], ao[sl], asc[sl])
        if profile.has_action_legs:
            sl = SLICES[f"{side}_leg_joint"]
            A[:, sl] = _norm(win[A_LEG[side]], ao[sl], asc[sl])
    sl = SLICES["waist_joint"]
    A[:, sl] = _norm(win[A_WAIST], ao[sl], asc[sl])
    if profile.has_base_pose:
        sl = SLICES["base_rotvec"]
        A[:, sl] = _norm(_rel(row[S_BASE_POSE], win[A_BASE_POSE], "xyz_quat"), ao[sl], asc[sl])
    A[:, BASE_CMD_SLOTS] = _norm(win[A_BASE_CMD], ao[BASE_CMD_SLOTS], asc[BASE_CMD_SLOTS])
    return A


def build_sample(
    cols: dict[str, np.ndarray], idx: int, ep_start: int, ep_end: int, ns: NormStats,
    profile: Profile = WBT,
) -> dict[str, np.ndarray]:
    """Amostra no índice global `idx`; episódio = linhas [ep_start, ep_end)."""
    row = {k: cols[k][idx] for k in profile.state_keys}
    widx = np.clip(idx + np.arange(H), ep_start, ep_end - 1)
    win = {k: cols[k][widx] for k in profile.action_keys}
    s_un = state_unnorm(row, profile)
    return {
        "action": action_chunk(row, win, ns, profile),
        "state": _norm(s_un, ns.state_offset, ns.state_scale),
        "state_unnorm": s_un,
        "action_mask": action_mask(profile),
        "state_mask": state_mask(profile),
    }


def invert_action(
    action_norm: np.ndarray,
    state_unnorm_: np.ndarray,
    stats: NormStats,
    base_pose_curr: np.ndarray | None = None,
    profile: Profile = WBT,
) -> dict[str, np.ndarray]:
    """(H,54) normalizado + estado atual não normalizado (60) -> valores absolutos.

    EE: T_abs = T_curr @ T_rel (T_curr vindo de xyz+rot6d do estado). base_pose_curr
    (xyz_quat, 7) é opcional: o estado de 60D não contém state_base_pose.
    """
    a = np.asarray(action_norm, np.float64) * stats.action_scale + stats.action_offset
    out = {}
    for side in ("left", "right"):
        p = np.asarray(state_unnorm_, np.float64)[STATE_SLICES[f"{side}_xyz_rot6d"]]
        T_curr = np.eye(4)
        T_curr[:3, :3], T_curr[:3, 3] = rot6d_to_matrix(p[3:9]), p[:3]
        T_abs = T_curr @ xyz_rotvec_to_se3(a[:, SLICES[f"{side}_xyz_rotvec"]])
        out[f"{side}_ee_T"] = T_abs
        out[f"{side}_ee_xyz_rpy"] = se3_to_xyz_rpy(T_abs)
        out[f"{side}_fig6d"] = a[:, SLICES[f"{side}_fig6d"]]
        if profile.has_action_legs:
            out[f"{side}_leg"] = a[:, SLICES[f"{side}_leg_joint"]]
    out["waist"] = a[:, SLICES["waist_joint"]]
    out["base_command"] = a[:, BASE_CMD_SLOTS]  # [vx, vy, angle_z, height]
    if profile.has_base_pose:
        rel_base = a[:, SLICES["base_rotvec"]]
        out["base_pose_rel"] = rel_base
        if base_pose_curr is not None:
            T_b = pose_to_se3_from_format(np.asarray(base_pose_curr, np.float64), "xyz_quat") @ xyz_rotvec_to_se3(rel_base)
            out["base_pose_T"] = T_b
    return out


def load_wbt_columns(data_dir: str | Path, keys: list[str]) -> dict[str, np.ndarray]:
    """Lê as colunas do parquet v3.0 (ordem dos arquivos = ordem do índice global)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(Path(data_dir).glob("data/*/*.parquet"))
    table = pa.concat_tables([pq.read_table(f, columns=["index", "episode_index", "frame_index", *keys]) for f in files])
    out = {k: table[k].to_numpy() for k in ("index", "episode_index", "frame_index")}
    for k in keys:
        arr = table[k].combine_chunks()
        flat = arr.flatten() if hasattr(arr, "flatten") else arr  # colunas escalares (ex.: int8) não são lista
        out[k] = flat.to_numpy().astype(np.float32).reshape(len(table), -1)
    return out


def episode_bounds(episode_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(start,end) por linha, assumindo episódios contíguos e ordenados."""
    change = np.flatnonzero(np.diff(episode_index)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(episode_index)]])
    lens = ends - starts
    return np.repeat(starts, lens), np.repeat(ends, lens)
