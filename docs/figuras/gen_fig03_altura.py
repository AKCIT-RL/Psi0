#!/usr/bin/env python3
"""Figura 3: canal de altura ao longo dos frames.

Sobrepõe, num único eixo, a série que o treino efetivamente usou (action[31], um
ângulo de junta do braço) e a série correta (teleop.base_height_command, altura da
pelve). O eixo único é deliberado: a separação entre as duas séries é exatamente o
diagnóstico, e um segundo eixo y a esconderia.

Paleta: slots 1 e 2 do palette de referência, sem alteração.
"""

from pathlib import Path
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET = "data/simple_teleop_g1/simple/G1WholebodyOpenOvenTeleop-v0/G1WholebodyOpenOvenTeleop-v0"
EPISODE = 0
OUT = Path("docs/figuras")

THEMES = {
    "light": dict(
        surface="#fcfcfb", text_primary="#0b0b0b", text_secondary="#52514e",
        muted="#8a8984", grid="#e4e3df",
        s1="#2a78d6",   # correto
        s2="#eb6834",   # usado no treino
    ),
    "dark": dict(
        surface="#1a1a19", text_primary="#ffffff", text_secondary="#c3c2b7",
        muted="#8a8984", grid="#33322f",
        s1="#3987e5",
        s2="#d95926",
    ),
}


def load():
    f = sorted(glob.glob(f"{DATASET}/data/*/*.parquet"))[EPISODE]
    df = pd.read_parquet(f)
    action = np.vstack([np.asarray(x, dtype=np.float32) for x in df["action"]])
    height = np.asarray(df["teleop.base_height_command"], dtype=np.float32)
    return action[:, 31], height


def build(mode: str) -> None:
    c = THEMES[mode]
    used, correct = load()
    n = len(used)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=200)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])

    ax.set_axisbelow(True)
    ax.grid(axis="y", color=c["grid"], linewidth=0.8, zorder=0)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["grid"])
    ax.spines["bottom"].set_linewidth(0.8)

    # faixa fisicamente comandável para a pelve
    ax.axhspan(0.40, 0.76, color=c["s1"], alpha=0.07, zorder=1, linewidth=0)
    ax.text(n * 0.985, 0.665, "faixa comandável de altura da pelve",
            color=c["muted"], fontsize=8.5, ha="right", va="center", zorder=4)

    ax.axhline(0.0, color=c["muted"], linewidth=0.8, linestyle=(0, (3, 3)), zorder=2)

    ax.plot(x, correct, color=c["s1"], linewidth=2.0, zorder=3, solid_capstyle="round")
    ax.plot(x, used, color=c["s2"], linewidth=2.0, zorder=3, solid_capstyle="round")

    # rótulos diretos: cada um numa região vazia, imediatamente abaixo da sua série.
    # A cor do texto identifica a série, dispensando linha-guia.
    ax.text(n * 0.022, 0.575, "teleop.base_height_command\ncorreto · 0,74 → 0,42 m",
            color=c["s1"], fontsize=9.5, fontweight="bold", ha="left", va="center", zorder=5)
    ax.text(n * 0.022, -0.245, "action[31]\nusado no treino · −0,07 a +0,28",
            color=c["s2"], fontsize=9.5, fontweight="bold", ha="left", va="center", zorder=5)

    # a distância entre as séries é o diagnóstico
    xg = int(n * 0.80)
    ax.annotate("", xy=(xg, correct[xg]), xytext=(xg, used[xg]),
                arrowprops=dict(arrowstyle="<->", color=c["text_secondary"],
                                linewidth=1.0, shrinkA=1, shrinkB=1, alpha=0.75))
    ax.text(xg - n * 0.015, 0.295,
            "o WBC recebia ≈ 0,07 m\nonde esperava 0,42 m",
            color=c["text_primary"], fontsize=9, ha="right", va="center",
            fontweight="bold", zorder=5)

    ax.set_xlim(0, n - 1)
    ax.set_ylim(-0.40, 0.88)
    ax.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels([f"{v:.1f}".replace(".", ",") for v in ax.get_yticks()])
    ax.tick_params(colors=c["text_secondary"], labelsize=9, length=0)
    ax.set_xlabel("frame do episódio", color=c["text_secondary"], fontsize=9.5, labelpad=8)
    ax.set_ylabel("valor no índice 31 do vetor de ação",
                  color=c["text_secondary"], fontsize=9.5, labelpad=8)

    fig.suptitle("Canal de altura ao longo dos frames", x=0.048, y=0.972,
                 ha="left", color=c["text_primary"], fontsize=15.5, fontweight="bold")
    fig.text(0.048, 0.912,
             "O mesmo canal que o controlador de corpo inteiro lê como altura da pelve, "
             "em dois conteúdos distintos.",
             ha="left", color=c["text_secondary"], fontsize=10)
    fig.text(0.048, 0.052,
             f"OpenOven, episódio {EPISODE} ({n} frames). Ambas as séries ocupam o índice 31 do vetor de ação, "
             "com grandezas físicas diferentes.",
             ha="left", color=c["muted"], fontsize=8)
    fig.text(0.048, 0.020,
             "Eixo único é deliberado: a separação entre as séries é o diagnóstico, e um segundo eixo y a esconderia.",
             ha="left", color=c["muted"], fontsize=8)

    fig.subplots_adjust(left=0.088, right=0.975, top=0.845, bottom=0.165)
    out = OUT / f"fig03_canal_altura_{mode}.png"
    fig.savefig(out, facecolor=c["surface"])
    plt.close(fig)
    print(f"  {out}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for mode in ("light", "dark"):
        build(mode)
