"""Stats no formato do WLA (D4), calculados APENAS em episódios `train` do split.

stats.json: por chave {min,max,mean,std,count,q01,q10,q50,q90,q99} (mesmo schema do
third_party/unifolm-wla/.../stats/stats.json). Chaves: as 6 de ação + 7 de estado do
perfil psi0_tote (convert_action.ACTION_OUT_KEYS, convert_state.STATE_OUT_KEYS).

relative_stats.json: chaves {left,right}_ee_pose_gripper_base com
{min,max,q01,q99,mean,std, global_*} — as duas chaves recebem os MESMOS valores
globais (merge L/R, como o WLA: stats oficiais têm left == right). As ações
relativas são computadas exatamente como o loader: para cada frame t,
janela t..t+H-1 (clamp ao fim do episódio), T_rel = inv(T_state[t]) @ T_action[t+k]
-> xyz+rotvec (se3_utils.compute_relative_actions equivalente).
"""

import json
from pathlib import Path

import numpy as np
from unifolm_wla.dataloader.multi_source_dataset.se3_utils import se3_inverse, se3_to_xyz_rotvec

from .convert_action import A_EE, ACTION_OUT_KEYS
from .convert_state import S_EE, STATE_OUT_KEYS
from .geometry import xyz_rpy_to_se3

H = 30  # chunk_size do WLA (configs/wla/*.yaml): janela de ação t..t+H-1

STATS_KEYS = ACTION_OUT_KEYS + STATE_OUT_KEYS
REL_KEY = "{side}_ee_pose_gripper_base"
QUANTILE_FIELDS = ("q01", "q10", "q50", "q90", "q99")


def _stats_block(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(-1, np.asarray(x).shape[-1])
    d = {
        "min": x.min(axis=0), "max": x.max(axis=0),
        "mean": x.mean(axis=0), "std": x.std(axis=0),
        "q01": np.quantile(x, 0.01, axis=0), "q10": np.quantile(x, 0.10, axis=0),
        "q50": np.quantile(x, 0.50, axis=0), "q90": np.quantile(x, 0.90, axis=0),
        "q99": np.quantile(x, 0.99, axis=0),
    }
    return {k: v.astype(np.float32).tolist() for k, v in d.items()} | {"count": int(x.shape[0])}


def compute_stats(episodes: list[dict[str, np.ndarray]]) -> dict:
    """episodes: lista de dicts chave->(M,D) (saída de convert_state/convert_action)."""
    out = {}
    for key in STATS_KEYS:
        x = np.concatenate([ep[key] for ep in episodes], axis=0)
        out[key] = _stats_block(x)
    return out


def _relative_ee_episode(T_state: np.ndarray, T_action: np.ndarray) -> np.ndarray:
    """(M,6*H) ações relativas do episódio como o loader as vê (janela t..t+H-1, clamp)."""
    m = T_state.shape[0]
    widx = np.clip(np.arange(m)[:, None] + np.arange(H)[None, :], 0, m - 1)
    T_rel = se3_inverse(T_state)[:, None, :, :] @ T_action[widx]  # (M,H,4,4)
    return se3_to_xyz_rotvec(T_rel).reshape(m * H, 6)


def compute_relative_stats(episodes: list[dict[str, np.ndarray]]) -> dict:
    """Merge L/R: ambas as chaves recebem os stats globais do pool concatenado."""
    pool = []
    for ep in episodes:
        for side in ("left", "right"):
            T_s = xyz_rpy_to_se3(np.asarray(ep[S_EE.format(side=side)], np.float64))
            T_a = xyz_rpy_to_se3(np.asarray(ep[A_EE.format(side=side)], np.float64))
            pool.append(_relative_ee_episode(T_s, T_a))
    x = np.concatenate(pool, axis=0)
    blk = _stats_block(x)
    merged = {f"global_{k}": blk[k] for k in ("min", "max", "q01", "q99", "mean", "std")}
    merged |= {k: blk[k] for k in ("min", "max", "q01", "q99", "mean", "std")}
    return {REL_KEY.format(side="left"): merged, REL_KEY.format(side="right"): merged}


def write_stats(
    stats_dir: str | Path,
    episodes: list[dict[str, np.ndarray]],
    dataset_version: str,
    conversion_version: str,
) -> Path:
    """Grava stats.json + relative_stats.json + dataset_version.txt + conversion_version.txt."""
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_dir / "stats.json", "w") as f:
        json.dump(compute_stats(episodes), f)
    with open(stats_dir / "relative_stats.json", "w") as f:
        json.dump(compute_relative_stats(episodes), f)
    (stats_dir / "dataset_version.txt").write_text(dataset_version + "\n")
    (stats_dir / "conversion_version.txt").write_text(conversion_version + "\n")
    return stats_dir
