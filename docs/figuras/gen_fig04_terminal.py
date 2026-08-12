#!/usr/bin/env python3
"""Figura 4: saída de terminal do gerador e do validador.

Todas as linhas abaixo são saída real das ferramentas, capturada em disco. As únicas
edições são o encurtamento de caminhos longos (marcados com .../) e a elisão do JSON
do scaffold, indicada explicitamente. Nada foi reescrito.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path("docs/figuras")

# paleta: superfícies do palette de referência + tinta de terminal
BG = "#1a1a19"
CHROME = "#26261f"
INK = "#e8e7e0"
DIM = "#8a8984"
PROMPT = "#1baf7a"   # slot aqua
FAIL = "#e34948"     # status critical
WARN = "#eda100"     # slot amarelo
OK = "#1baf7a"
NOTE = "#3987e5"     # slot azul (dark)

# (texto, cor) — saída literal das ferramentas
LINES = [
    ("$ python scripts/generate_lerobot_modality.py data/.../G1WholebodyOpenOvenTeleop-v0 --write", PROMPT),
    ("dataset          : data/.../G1WholebodyOpenOvenTeleop-v0", INK),
    ("episodes         : 104", INK),
    ("sampled parquet  : episode_000000.parquet", INK),
    ('vector columns   : {"observation.state": 43, "observation.eef_state": 14, "action": 43,', INK),
    ('                    "teleop.navigate_command": 4, "teleop.base_height_command": 1, ...}', INK),
    ("video features   : ['observation.images.ego_view']", INK),
    ("detected schema  : raw whole-body teleop (43 dof) -- NEEDS CONVERSION", WARN),
    ("", INK),
    ("[!] SCHEMA NOT DIRECTLY TRAINABLE for the psi0/gr00t SIMPLE deployment path.", WARN),
    ("    The 36-dim psi0 action vector needs teleop.base_height_command and", DIM),
    ("    teleop.navigate_command merged into the joint targets, which modality.json", DIM),
    ("    cannot express (it slices one column per group). Convert the dataset first.", DIM),
    ("", INK),
    ("[!] detected action joint blocks by correlation: right_arm at 29, unused hand", NOTE),
    ("    slot at 22 (median |r| per candidate: {22: 0.0, 29: 0.97})", NOTE),
    ("", INK),
    ("    [ ... scaffold JSON omitido nesta figura ... ]", DIM),
    ("", INK),
    ("Refusing to write .../meta/modality.json: the generated config is a scaffold,", FAIL),
    ("not a usable layout (see the notes above). Resolve those first.", FAIL),
    ("$ echo $?", PROMPT),
    ("1", INK),
    ("", INK),
    ("$ python scripts/validate_lerobot_modality.py data/.../G1WholebodyOpenOvenTeleop-v0 --expect-psi0", PROMPT),
    ("=== data/.../G1WholebodyOpenOvenTeleop-v0", INK),
    ("  FAIL: state.left_arm: declares observation.eef_state[14:21] (7 dims) but", FAIL),
    ("        'observation.eef_state' has 14 dims -> would silently yield 0 dims", FAIL),
    ("  FAIL: state.right_arm: declares observation.eef_state[21:28] (7 dims) but ...", FAIL),
    ("  FAIL: state.rpy:       declares observation.eef_state[28:31] (3 dims) but ...", FAIL),
    ("  FAIL: state.height:    declares observation.eef_state[31:32] (1 dims) but ...", FAIL),
    ("$ echo $?", PROMPT),
    ("1", INK),
    ("", INK),
    ("$ python scripts/validate_lerobot_modality.py data/.../psi0_converted/... --expect-psi0", PROMPT),
    ("=== data/.../psi0_converted/G1WholebodyOpenOvenTeleop-v0", INK),
    ("  note: action.left_hand: constant across the dataset (value 0)", DIM),
    ("  OK (with notes)", OK),
    ("$ echo $?", PROMPT),
    ("0", INK),
]

TITLE = "Gerador e validador de modality.json"
SUB = "Saída real das ferramentas. O gerador reconhece o schema e recusa-se a escrever uma configuração que não pode justificar."


def font(size: int, bold: bool = False):
    base = "/usr/share/fonts/truetype/dejavu"
    name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    try:
        return ImageFont.truetype(f"{base}/{name}", size)
    except OSError:
        return ImageFont.load_default()


def sans(size: int, bold: bool = False):
    base = "/usr/share/fonts/truetype/dejavu"
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"{base}/{name}", size)
    except OSError:
        return ImageFont.load_default()


def build() -> None:
    fs = 17
    lh = 25
    pad_x, head_h = 34, 104
    mono, mono_b = font(fs), font(fs, bold=True)

    probe = Image.new("RGB", (10, 10))
    d0 = ImageDraw.Draw(probe)
    width = max(d0.textlength(t, font=mono_b) for t, _ in LINES) + pad_x * 2 + 18
    width = int(max(width, 1180))
    height = head_h + lh * len(LINES) + 44

    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)

    # barra de janela
    d.rectangle([0, 0, width, head_h - 34], fill=CHROME)
    for i, col in enumerate(("#e34948", "#eda100", "#1baf7a")):
        d.ellipse([pad_x + i * 22, 20, pad_x + i * 22 + 12, 32], fill=col)
    d.text((pad_x + 92, 18), TITLE, font=sans(16, bold=True), fill=INK)

    d.text((pad_x, head_h - 26), SUB, font=sans(13), fill=DIM)

    y = head_h + 8
    for text, col in LINES:
        if text:
            bold = text.startswith("$") or text.startswith("  FAIL") or text.startswith("Refusing")
            d.text((pad_x, y), text, font=mono_b if bold else mono, fill=col)
        y += lh

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "fig04_ferramentas_terminal.png"
    img.save(out)
    print(f"  {out}  ({width}x{height})")


if __name__ == "__main__":
    build()
