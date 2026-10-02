#!/usr/bin/env python3
"""Envia resultados para o W&B (segredos só via ambiente: WANDB_API_KEY de $RAID/secrets/wandb.env; nunca impressos).

  train   <offline-run-dir> [--project wla]        sincroniza o run offline de treino (wandb sync)
  simple  <run-dir>         [--project wla-simple-eval]   metrics.json + config.env + vídeos de um run de eval do SIMPLE
  offline <dir>             [--project wla]        summary.csv + curves.png do eval offline (s8)
Destino: WANDB_BASE_URL (padrão https://api.wandb.ai) / WANDB_ENTITY (padrão akcit_industrial_humanoids).
"""
import argparse, csv, glob, json, os, re, subprocess, sys
from pathlib import Path

os.environ.pop("WANDB_MODE", None)
os.environ["WANDB_BASE_URL"] = os.environ.get("WLA_WANDB_URL", "https://api.wandb.ai")
ENTITY = os.environ.get("WLA_WANDB_ENTITY", "akcit_industrial_humanoids")  # ignora WANDB_ENTITY de wandb.env (aponta para outro time)

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["train", "simple", "offline"])
ap.add_argument("path")
ap.add_argument("--project")
ap.add_argument("--name")
a = ap.parse_args()
P = Path(a.path)

if a.kind == "train":
    cmd = [sys.executable, "-m", "wandb", "sync", "-e", ENTITY, "-p", a.project or "wla", "--no-mark-synced", str(P)]
    print("+", " ".join(cmd)); sys.exit(subprocess.call(cmd))

import wandb  # noqa: E402

def cfg_env(p):
    d = {}
    if p.exists():
        for l in p.read_text().splitlines():
            if "=" in l:
                k, v = l.split("=", 1); d[k] = v
    return d

if a.kind == "simple":
    run_name = a.name or P.name
    cfg = cfg_env(P / "config.env")
    commit = (P / "git_commit.txt").read_text().strip() if (P / "git_commit.txt").exists() else None
    m = json.load(open(P / "metrics.json"))
    run = wandb.init(entity=ENTITY, project=a.project or "wla-simple-eval", name=run_name, id=re.sub(r"[^A-Za-z0-9_-]", "_", run_name),
                     resume="allow", job_type="simple_closed_loop", config={**cfg, "git_commit": commit},
                     tags=[cfg.get("MODE", ""), cfg.get("SPLIT", "")])
    run.summary.update({"n": m["n"], "successes": m["successes"], "success_rate": m["success_rate"]})
    tbl = wandb.Table(columns=["episode", "success", "video"])
    for ep, ok in m["episodes"].items():
        vids = glob.glob(str(P / "eval" / "**" / f"{ep}" / "**" / "head_stereo_left_*.mp4"), recursive=True) or \
               glob.glob(str(P / "eval" / "**" / f"*{ep}*" / "head_stereo_left_*.mp4"), recursive=True)
        tbl.add_data(ep, ok, wandb.Video(vids[0], format="mp4") if vids else None)
    run.log({"episodes": tbl})
    for f in ("config.env", "metrics.json"):
        if (P / f).exists(): run.save(str(P / f), base_path=str(P))
    print("run:", run.url); run.finish()
else:
    rows = list(csv.DictReader(open(P / "summary.csv")))
    run = wandb.init(entity=ENTITY, project=a.project or "wla", name=a.name or "s8_offline_val3", id="s8_offline_val3", resume="allow",
                     job_type="offline_eval", config={"note": "VAL entrou no treino do F5: métricas em amostra", "dir": str(P)})
    cols = list(rows[0].keys())
    tbl = wandb.Table(columns=cols, data=[[r[c] if c == "model" else float(r[c]) for c in cols] for r in rows])
    run.log({"summary": tbl})
    for r in rows:
        m = re.match(r"steps_(\d+)", r["model"])
        if m:
            run.log({"step": int(m.group(1)), **{f"offline/{c}": float(r[c]) for c in cols[1:]}})
    if (P / "curves.png").exists(): run.log({"curves": wandb.Image(str(P / "curves.png"))})
    print("run:", run.url); run.finish()
