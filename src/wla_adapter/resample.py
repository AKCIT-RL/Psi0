"""Reamostragem 50 -> 30 FPS do conversor Ψ0 -> WLA (D1).

Tempos alvo t_j = j/30 dentro do episódio (j <= (N-1)*30/50); interpolação LINEAR dos
canais crus a 50 Hz (juntas e comandos); FK/fig6d só DEPOIS, sobre os valores reamostrados.
target_yaw (ângulo absoluto) usa unwrap -> linear -> wrap. Vídeo: frame de origem mais
próximo de t_j (`nearest_source_indices`).
"""

import numpy as np

SRC_FPS, TGT_FPS = 50.0, 30.0


def target_times(n_src: int, src_fps: float = SRC_FPS, tgt_fps: float = TGT_FPS) -> np.ndarray:
    """(M,) tempos alvo t_j = j/tgt_fps com M = floor((N-1)*tgt/src) + 1 (dentro do episódio)."""
    if n_src < 2:
        raise ValueError(f"episódio curto demais: {n_src} frames")
    m = int(np.floor((n_src - 1) * tgt_fps / src_fps)) + 1
    return np.arange(m, dtype=np.float64) / tgt_fps


def resample_linear(x: np.ndarray, src_fps: float = SRC_FPS, tgt_fps: float = TGT_FPS) -> np.ndarray:
    """Interpolação linear por canal: x (N,...) a src_fps -> (M,...) a tgt_fps."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    t_src = np.arange(n, dtype=np.float64) / src_fps
    t_tgt = target_times(n, src_fps, tgt_fps)
    flat = x.reshape(n, -1)
    out = np.stack([np.interp(t_tgt, t_src, flat[:, c]) for c in range(flat.shape[1])], axis=1)
    return out.reshape((len(t_tgt),) + x.shape[1:])


def resample_angle(x: np.ndarray, src_fps: float = SRC_FPS, tgt_fps: float = TGT_FPS) -> np.ndarray:
    """Linear com unwrap/wrap para canais angulares absolutos (ex.: target_yaw em ±π)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    t_src = np.arange(n, dtype=np.float64) / src_fps
    t_tgt = target_times(n, src_fps, tgt_fps)
    flat = x.reshape(n, -1)
    out = np.stack(
        [np.interp(t_tgt, t_src, np.unwrap(flat[:, c])) for c in range(flat.shape[1])], axis=1
    )
    out = (out + np.pi) % (2 * np.pi) - np.pi
    return out.reshape((len(t_tgt),) + x.shape[1:])


def nearest_source_indices(n_src: int, src_fps: float = SRC_FPS, tgt_fps: float = TGT_FPS) -> np.ndarray:
    """Índice do frame de origem mais próximo de cada t_j (para o vídeo)."""
    m = len(target_times(n_src, src_fps, tgt_fps))
    return np.clip(np.round(np.arange(m) * src_fps / tgt_fps), 0, n_src - 1).astype(np.int64)
