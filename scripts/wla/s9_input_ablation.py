#!/usr/bin/env python3
"""Qual entrada faz o modelo ficar parado no simulador? Troca imagem e/ou estado entre dataset (real) e simulador (debug do
servidor de um run closed-loop) e imprime o que o modelo prevê (vx, fecho da mão dir., deslocamento do EE dir. no chunk)."""
import argparse, json, sys
from pathlib import Path
import numpy as np
PSI0 = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PSI0 / "src"), str(PSI0 / "third_party/unifolm-wla")]
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from examples.unifolm_wla.eval_files.unitree.eval_local_episode import (  # noqa: E402
    _build_example_from_sample, _episode_frame_range, _to_numpy, load_model)
from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES  # noqa: E402
from unifolm_wla.dataloader.multi_source_dataset.config import load_config  # noqa: E402
from unifolm_wla.dataloader.multi_source_dataset.single_source_dataset import create_single_source_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--debug_dir", required=True)
ap.add_argument("--val_idx", type=int, default=27)
ap.add_argument("--ts", default="0,15,30,45,60,90")
ap.add_argument("--ckpt", default="/raid/user_marcospaulo/checkpoints/wla/f5_psi0_tote/final_model/model.safetensors")
ap.add_argument("--sim_steps", default="1,10")
a = ap.parse_args()
BASE_VLM = "/raid/user_marcospaulo/models/unifolm-wla/UnifoLM-WLA-1.0-Base/tokenizer"
cfg = load_config(str(PSI0 / "configs/wla/psi0_tote.yaml")); ds_cfg = [d for d in cfg.datasets if d.enabled][0]
src = create_single_source_dataset(ds_cfg, cfg)
D = "/raid/user_marcospaulo/datasets/unifolm/Psi0_Tote_Dataset/G1ToteMix_psi0_trainval/meta/psi0_episode_map.json"
o2n = {int(v): int(k) for k, v in json.load(open(D))["new_to_original"].items()}
oe = sorted(json.load(open(PSI0 / "docs/wla/split.json"))["val"])[a.val_idx]
off, f0, f1 = _episode_frame_range(src, 0, o2n[oe])
model = load_model(Path(a.ckpt), BASE_VLM).to(torch.bfloat16).to("cuda").eval()
st = model.norm_stats["Psi0_Tote_Dataset"]
o_a, s_a = np.array(st["action"]["offset"]), np.array(st["action"]["scale"])
o_s, s_s = np.array(st["state"]["offset"]), np.array(st["state"]["scale"])
sim = {int(k): np.load(Path(a.debug_dir) / f"step_{int(k):05d}.npz") for k in a.sim_steps.split(",")}


def run(ex):
    with torch.no_grad():
        p = np.asarray(model.predict_action([ex])["normalized_actions"][0], np.float64) * s_a + o_a
    r = p[:, SLICES["right_xyz_rotvec"]][:, :3]
    return f"vx={p[:, 32].mean():+.2f} handR_max={p[:, SLICES['right_fig6d']].min():.2f}(1=aberta) EEdir_move={np.linalg.norm(r[-1]-r[0])*100:.1f}cm"


for t in map(int, a.ts.split(",")):
    s = src[off + f0 + t]; ex0 = _build_example_from_sample(s)
    print(f"--- dataset t={t} (ep val[{a.val_idx}])")
    print("  real img + real state :", run(dict(ex0)))
    for k, d in sim.items():
        img, sst = [Image.fromarray(d["image"])], ((d["state"] - o_s) / s_s).astype(np.float32)
        print(f"  sim img  + real state (sim step {k}):", run({**ex0, "image": img}))
        print(f"  real img + sim state  (sim step {k}):", run({**ex0, "state": sst}))
for k, d in sim.items():
    ex = dict(_build_example_from_sample(src[off + f0]))
    ex["image"], ex["state"] = [Image.fromarray(d["image"])], ((d["state"] - o_s) / s_s).astype(np.float32)
    print(f"--- sim step {k} img+state (reprodução do servidor):", run(ex))
