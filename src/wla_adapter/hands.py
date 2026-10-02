"""Mãos Dex3 (7 juntas/mão) <-> fig6d (6) na convenção WLA: **1 = aberto, menor = fechado**.

Ordem simétrica por mão (igual à coluna observation.hand_joints do Ψ0 e a
LEFT/RIGHT_HAND_JOINTS de third_party/SIMPLE/src/simple/robots/g1_sonic.py:39-40):
    [thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1]

Limites FIXOS por junta vêm dos URDFs real/assets/unitree_hand/unitree_dex3_{left,right}.urdf
(não de estatística de dados). q_open = 0 para todas as juntas (zero do URDF = mão aberta;
close_qpos do SIMPLE, g1_wholebody.py ~linhas 90-100, parte de init_qpos = 0). A direção de
fechamento (qual limite é q_close) segue o sinal de close_qpos do SIMPLE:

  esquerda close_qpos [+0.3523, -0.0964, +0.2790, -0.5058, -1.1950, -0.5389, -0.9835]
    -> thumb_0 upper, thumb_1 lower, thumb_2 upper, index/middle lower (URDF esq. só tem
       faixa negativa para index/middle, confirmando).
  direita  close_qpos [+0.0233, -0.0240, -0.2217, +0.2566, +1.3371, +0.3085, +0.9805]
    -> thumb_0 upper, thumb_1 lower, thumb_2 lower, index/middle upper (sinais OPOSTOS
       à esquerda em thumb_2/index/middle, como nos limites do URDF).

fig6d = [thumb_oc, thumb_lat, index, middle, ring, little] com fecho c_j em [0,1]:
  thumb_oc  = 1 - mean(c thumb_1, c thumb_2)   (oposição)
  thumb_lat = 1 - c thumb_0                    (lateral)
  index     = 1 - mean(c index_0, c index_1)
  middle    = 1 - mean(c middle_0, c middle_1)
  ring = little = middle                       (Dex3 não tem ring/little; duplicata documentada)
Inversa (no manifold de fecho): c por dedo distribuído igualmente entre as 2 juntas;
  q = q_open + c * (q_close - q_open)  -> round-trip joints->fig6d->joints->fig6d exato.

Layout ASSIMÉTRICO de states/action do Ψ0 (conversion_spec.md §3): esq [t3, m2, i2],
dir [t3, i2, m2]. `action_hands_to_symmetric` reordena action[0:14] para a ordem
simétrica antes de mapear (a direita já está na ordem simétrica).
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from .geometry import load_urdf

DEX3_URDF = {
    side: str(Path(__file__).resolve().parents[2] / f"real/assets/unitree_hand/unitree_dex3_{side}.urdf")
    for side in ("left", "right")
}

JOINTS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")

# Direção de fechamento por junta (ver docstring; sinais do close_qpos do SIMPLE).
CLOSE_TOWARD = {
    "left":  ("upper", "lower", "upper", "lower", "lower", "lower", "lower"),
    "right": ("upper", "lower", "lower", "upper", "upper", "upper", "upper"),
}

FIG6D_DIMS = ("thumb_oc", "thumb_lat", "index", "middle", "ring", "little")


@lru_cache(maxsize=None)
def hand_limits(side: str) -> tuple[np.ndarray, np.ndarray]:
    """(q_open, q_close) (7,) na ordem JOINTS, dos limites fixos do URDF Dex3."""
    if side not in ("left", "right"):
        raise ValueError(side)
    by_name = {j.name: j for j in load_urdf(DEX3_URDF[side]).values()}
    q_open = np.zeros(7)
    q_close = np.zeros(7)
    for i, (jn, toward) in enumerate(zip(JOINTS, CLOSE_TOWARD[side])):
        j = by_name[f"{side}_hand_{jn}_joint"]
        lo, hi = min(j.lower, j.upper), max(j.lower, j.upper)
        if not (lo <= 0.0 <= hi):
            raise ValueError(f"{j.name}: 0 (aberto) fora dos limites [{lo}, {hi}]")
        q_close[i] = hi if toward == "upper" else lo
    return q_open, q_close


def closure(q: np.ndarray, side: str) -> np.ndarray:
    """Fecho por junta c (...,7) em [0,1]: 0 = aberto, 1 = fechado (clipado aos limites)."""
    q_open, q_close = hand_limits(side)
    q = np.asarray(q, dtype=np.float64)
    return np.clip((q - q_open) / (q_close - q_open), 0.0, 1.0)


def joints_to_fig6d(q: np.ndarray, side: str) -> np.ndarray:
    """(…,7) juntas Dex3 (ordem simétrica) -> (…,6) fig6d WLA (1=aberto)."""
    c = closure(q, side)
    thumb_lat = 1.0 - c[..., 0]
    thumb_oc = 1.0 - 0.5 * (c[..., 1] + c[..., 2])
    index = 1.0 - 0.5 * (c[..., 3] + c[..., 4])
    middle = 1.0 - 0.5 * (c[..., 5] + c[..., 6])
    return np.stack([thumb_oc, thumb_lat, index, middle, middle, middle], axis=-1)


def fig6d_to_joints(f: np.ndarray, side: str) -> np.ndarray:
    """Inversa no manifold de fecho: (…,6) fig6d -> (…,7) juntas (ring/little ignorados)."""
    f = np.asarray(f, dtype=np.float64)
    q_open, q_close = hand_limits(side)
    c = np.empty(f.shape[:-1] + (7,))
    c[..., 0] = 1.0 - f[..., 1]                       # thumb_lat
    c[..., 1] = c[..., 2] = 1.0 - f[..., 0]           # thumb_oc dividido igualmente
    c[..., 3] = c[..., 4] = 1.0 - f[..., 2]           # index
    c[..., 5] = c[..., 6] = 1.0 - f[..., 3]           # middle
    c = np.clip(c, 0.0, 1.0)
    return q_open + c * (q_close - q_open)


# Reordenação do layout assimétrico do Ψ0 (esq t3,m2,i2 | dir t3,i2,m2) -> simétrico.
_LEFT_ASYM_TO_SYM = [0, 1, 2, 5, 6, 3, 4]
_LEFT_SYM_TO_ASYM = [0, 1, 2, 5, 6, 3, 4]  # a permutação é involutiva (troca i<->m)


def action_hands_to_symmetric(a14: np.ndarray) -> np.ndarray:
    """action[0:14] (esq t3,m2,i2 | dir t3,i2,m2) -> (…,14) simétrico (t3,i2,m2 | t3,i2,m2)."""
    a14 = np.asarray(a14, dtype=np.float64)
    left = a14[..., 0:7][..., _LEFT_ASYM_TO_SYM]
    return np.concatenate([left, a14[..., 7:14]], axis=-1)


def symmetric_to_action_hands(q14: np.ndarray) -> np.ndarray:
    """Inversa de action_hands_to_symmetric (permutação involutiva na esquerda)."""
    q14 = np.asarray(q14, dtype=np.float64)
    left = q14[..., 0:7][..., _LEFT_SYM_TO_ASYM]
    return np.concatenate([left, q14[..., 7:14]], axis=-1)
