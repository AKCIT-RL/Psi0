#!/usr/bin/env python3
"""Eval OFFLINE (malha aberta, sem simulador): saída do modelo x dado convertido pela nossa transformação.

Para cada ponto de episódio (a cada --stride frames a 30 FPS) usa a MESMA entrada do treino (loader oficial do WLA sobre
o dataset convertido: imagem, estado, máscaras), roda o modelo e compara o chunk previsto (30 passos) com o chunk do
dataset convertido (GT), tudo desnormalizado, no espaço do WLA:
  - mão: posição (cm) e rotação (graus) da pose ABSOLUTA (T_atual @ T_rel) e erro do fig6d + acerto aberto/fechado;
  - cintura (rad); base (vx, vy, vyaw, altura); nL1 = L1 médio normalizado nas 31 dims ativas (métrica comparável).
Referências: "hold" (não mexe; fig6d/cintura = estado atual; velocidades 0) e o WLA-Base (--base).
ATENÇÃO: os episódios de VAL foram VISTOS no treino do F5 (dataset trainval) -> métricas EM AMOSTRA: dizem se o modelo
reproduz o que treinou e como o erro evolui por checkpoint, NÃO medem generalização (só o TEST mede, uma vez, no fim).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PSI0 = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PSI0 / "src"), str(PSI0 / "third_party/unifolm-wla")]

import torch  # noqa: E402
from examples.unifolm_wla.eval_files.unitree.eval_local_episode import (  # noqa: E402
    _build_example_from_sample, _episode_frame_range, _to_numpy, load_model,
)
from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES, STATE_SLICES  # noqa: E402
from unifolm_wla.dataloader.multi_source_dataset.config import load_config  # noqa: E402
from unifolm_wla.dataloader.multi_source_dataset.single_source_dataset import create_single_source_dataset  # noqa: E402

from wla_adapter.geometry import rot6d_to_matrix, rotation_angle, xyz_rotvec_to_se3  # noqa: E402
from wla_adapter.pipeline import PSI0_TOTE, WBT, action_mask, state_mask  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(PSI0 / "configs/wla/psi0_tote.yaml"))
ap.add_argument("--ckpt_dir", default="/raid/user_marcospaulo/checkpoints/wla/f5_psi0_tote")
ap.add_argument("--steps", default="2000,4000,6000,8000,10000,12000,14000,16000,18000,20000")
ap.add_argument("--val_idx", default="12,24,27", help="índices na lista ordenada de VAL (os mesmos episódios do eval fechado)")
ap.add_argument("--stride", type=int, default=30)
ap.add_argument("--base", action="store_true", help="inclui o WLA-Base como referência")
ap.add_argument("--base_dir", default="/raid/user_marcospaulo/models/unifolm-wla/UnifoLM-WLA-1.0-Base")
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
SRC_NAME = "Psi0_Tote_Dataset"
BASE_VLM = f"{a.base_dir}/tokenizer"

# ---------- amostras (uma vez, compartilhadas por todos os modelos) ----------
split = json.load(open(PSI0 / "docs/wla/split.json"))
val_sorted = sorted(split["val"])
orig_eps = [val_sorted[int(i)] for i in a.val_idx.split(",")]
cfg = load_config(a.config)
ds_cfg = [d for d in cfg.datasets if d.enabled][0]
src = create_single_source_dataset(ds_cfg, cfg)
emap = json.load(open("/raid/user_marcospaulo/datasets/unifolm/Psi0_Tote_Dataset/G1ToteMix_psi0_trainval/meta/psi0_episode_map.json"))["new_to_original"]
orig2new = {int(v): int(k) for k, v in emap.items()}
st_ours = json.load(open(f"{a.ckpt_dir}/dataset_statistics.json"))[SRC_NAME]
A_OFF, A_SC = np.array(st_ours["action"]["offset"], np.float64), np.array(st_ours["action"]["scale"], np.float64)

samples = []
for oe in orig_eps:
    new = orig2new[oe]
    off, f0, f1 = _episode_frame_range(src, 0, new)
    for t in range(0, f1 - f0, a.stride):
        s = src[off + f0 + t]
        samples.append({
            "orig_ep": oe, "t": t, "example": _build_example_from_sample(s),
            "state_unnorm": _to_numpy(s["state_unnorm"]).astype(np.float64),
            "gt_norm": _to_numpy(s["action"]).astype(np.float64),
            "amask": _to_numpy(s["action_mask"]).astype(bool),
        })
print(f"{len(samples)} amostras de {len(orig_eps)} episódios de VAL {orig_eps}", flush=True)
ACTIVE = samples[0]["amask"]
GT_UN = [x["gt_norm"] * A_SC + A_OFF for x in samples]


# ---------- métricas ----------
def _T_curr(state, side):
    p = state[STATE_SLICES[f"{side}_xyz_rot6d"]]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rot6d_to_matrix(p[3:9]), p[:3]
    return T


def metrics(pred_un, gt_un, state):
    m = {}
    for side in ("left", "right"):
        Tc = _T_curr(state, side)
        Tp = Tc @ xyz_rotvec_to_se3(pred_un[:, SLICES[f"{side}_xyz_rotvec"]])
        Tg = Tc @ xyz_rotvec_to_se3(gt_un[:, SLICES[f"{side}_xyz_rotvec"]])
        m[f"{side}_pos_cm"] = np.linalg.norm(Tp[:, :3, 3] - Tg[:, :3, 3], axis=1) * 100  # (T,)
        m[f"{side}_rot_deg"] = np.degrees(rotation_angle(Tp[:, :3, :3], Tg[:, :3, :3]))
        fp, fg = pred_un[:, SLICES[f"{side}_fig6d"]], gt_un[:, SLICES[f"{side}_fig6d"]]
        m[f"{side}_fig6d_l1"] = np.abs(fp - fg).mean(axis=1)
        m[f"{side}_hand_class_acc"] = ((fp[:, :4].mean(1) < 0.5) == (fg[:, :4].mean(1) < 0.5)).astype(float)
    m["waist_l1_rad"] = np.abs(pred_un[:, SLICES["waist_joint"]] - gt_un[:, SLICES["waist_joint"]]).mean(axis=1)
    for name, sl in (("vx", 32), ("vy", 33), ("vyaw", 34), ("height", 41)):
        m[f"base_{name}_mae"] = np.abs(pred_un[:, sl] - gt_un[:, sl])
    nl = np.abs((pred_un - A_OFF) / A_SC - (gt_un - A_OFF) / A_SC)[:, ACTIVE]
    m["nL1"] = nl.mean(axis=1)
    return m  # cada valor (T,)


def hold_pred(state):
    p = np.zeros((30, 54))
    for side in ("left", "right"):
        p[:, SLICES[f"{side}_fig6d"]] = state[STATE_SLICES[f"{side}_fig6d"]]
    p[:, SLICES["waist_joint"]] = state[STATE_SLICES["waist_joint"]]
    p[:, 41] = 0.74
    return p


def summarize(per_sample):
    keys = per_sample[0].keys()
    cat = {k: np.stack([m[k] for m in per_sample]) for k in keys}  # (N,T)
    s = {k: float(v.mean()) for k, v in cat.items()}
    for step in (0, 9, 19, 29):
        s[f"pos_cm_h{step}"] = float(0.5 * (cat["left_pos_cm"][:, step].mean() + cat["right_pos_cm"][:, step].mean()))
    s["pos_cm_mean"] = 0.5 * (s["left_pos_cm"] + s["right_pos_cm"])
    s["rot_deg_mean"] = 0.5 * (s["left_rot_deg"] + s["right_rot_deg"])
    s["hand_class_acc_mean"] = 0.5 * (s["left_hand_class_acc"] + s["right_hand_class_acc"])
    s["fig6d_l1_mean"] = 0.5 * (s["left_fig6d_l1"] + s["right_fig6d_l1"])
    return s


def per_episode(ms, tag):
    res = {}
    for oe in orig_eps:
        idx = [i for i, x in enumerate(samples) if x["orig_ep"] == oe]
        res[str(oe)] = summarize([ms[i] for i in idx])
    return res


results = {"note": "EM AMOSTRA: os episódios de VAL foram vistos no treino do F5", "episodes_val_orig": orig_eps,
           "n_samples": len(samples), "models": {}}

# referência trivial
ms = [metrics(hold_pred(x["state_unnorm"]), GT_UN[i], x["state_unnorm"]) for i, x in enumerate(samples)]
results["models"]["hold"] = {"overall": summarize(ms), "by_episode": per_episode(ms, "hold")}
print("hold", {k: round(v, 3) for k, v in results["models"]["hold"]["overall"].items() if k in ("pos_cm_mean", "nL1")}, flush=True)


def run_model(name, ckpt, source, kind):
    t0 = time.time()
    model = load_model(Path(ckpt), BASE_VLM).to(torch.bfloat16).to("cuda").eval()
    st = model.norm_stats[source]
    o_a, s_a = np.array(st["action"]["offset"], np.float64), np.array(st["action"]["scale"], np.float64)
    o_s, s_s = np.array(st["state"]["offset"], np.float64), np.array(st["state"]["scale"], np.float64)
    ms = []
    for i, x in enumerate(samples):
        ex = dict(x["example"])
        if kind == "base":  # estado/máscaras no perfil WBT; IMU (base_rot) não existe no dataset -> gravidade upright
            su = x["state_unnorm"].copy()
            su[STATE_SLICES["base_rotvec"]] = [0, 0, -1, 0, 0, 0]
            ex["state"] = ((su - o_s) / s_s).astype(np.float32)
            ex["state_mask"], ex["action_mask"] = state_mask(WBT), action_mask(WBT)
        with torch.no_grad():
            pred_n = np.asarray(model.predict_action([ex])["normalized_actions"][0], np.float64)
        ms.append(metrics(pred_n * s_a + o_a, GT_UN[i], x["state_unnorm"]))
    del model
    torch.cuda.empty_cache()
    results["models"][name] = {"overall": summarize(ms), "by_episode": per_episode(ms, name), "ckpt": str(ckpt)}
    o = results["models"][name]["overall"]
    print(f"{name}: pos={o['pos_cm_mean']:.2f}cm rot={o['rot_deg_mean']:.1f}° nL1={o['nL1']:.3f} "
          f"hand_acc={o['hand_class_acc_mean']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(results, open(out / "metrics.json", "w"), indent=1)


if a.base:
    run_model("wla_base", f"{a.base_dir}/checkpoints/model.safetensors", "UnifoLM_WBT", "base")
for s in a.steps.split(","):
    run_model(f"steps_{s}", f"{a.ckpt_dir}/checkpoints/steps_{s}_model.safetensors", SRC_NAME, "ft")

# ---------- tabela e curvas ----------
cols = ["pos_cm_mean", "rot_deg_mean", "hand_class_acc_mean", "fig6d_l1_mean", "waist_l1_rad", "base_vx_mae", "base_vy_mae",
        "base_vyaw_mae", "base_height_mae", "nL1", "pos_cm_h0", "pos_cm_h29"]
with open(out / "summary.csv", "w") as f:
    f.write("model," + ",".join(cols) + "\n")
    for n, r in results["models"].items():
        f.write(n + "," + ",".join(f"{r['overall'][c]:.4f}" for c in cols) + "\n")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = [int(s) for s in a.steps.split(",")]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for axx, (c, t) in zip(ax.ravel(), [("pos_cm_mean", "posição da mão (cm)"), ("rot_deg_mean", "rotação da mão (°)"),
                                         ("hand_class_acc_mean", "acerto aberto/fechado"), ("waist_l1_rad", "cintura L1 (rad)"),
                                         ("base_vx_mae", "vx MAE (m/s)"), ("nL1", "nL1 (31 dims, normalizado)")]):
        axx.plot(steps, [results["models"][f"steps_{s}"]["overall"][c] for s in steps], "o-", label="fine-tuned")
        axx.axhline(results["models"]["hold"]["overall"][c], color="gray", ls="--", label="hold")
        if a.base:
            axx.axhline(results["models"]["wla_base"]["overall"][c], color="red", ls=":", label="WLA-Base")
        axx.set_title(t)
        axx.set_xlabel("steps")
    ax[0, 0].legend()
    fig.suptitle("Eval offline em amostra (VAL visto no treino): " + ", ".join(map(str, orig_eps)))
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=110)
except Exception as e:  # noqa: BLE001
    print("sem plot:", e)
print("OFFLINE_DONE")
