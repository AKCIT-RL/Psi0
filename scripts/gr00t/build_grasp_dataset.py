"""[FASE 2] Tabela por episódio-troca a partir dos CSVs da cadeia (todas as execuções): características de IMAGEM (mancha azul) em janelas após a troca 1->2
e rótulo 'held' privilegiado (pose do simulador): na janela s+400, caixa a <0,8 m da base, deslocada >0,3 m e z>=0,6 (mesma definição de chain_metrics.py).
Saída: _scratch/grasp_ds.json (lista de dicts)."""
import csv, glob, json, os
import numpy as np
runs = glob.glob(os.path.expanduser("~/wla_data/experiments/wla/*/chain/chain_*.csv"))
rows = []
for f in sorted(runs):
    try: r = list(csv.DictReader(open(f)))
    except Exception: continue
    if len(r) < 50 or "areaL" not in r[0]: continue
    sw = [i for i, q in enumerate(r) if q["switch"] == "1"]
    if not sw: continue
    s = sw[0]
    a = lambda k: np.array([float(x[k]) for x in r])
    x, y, tz, bx, by, yaw = a("x"), a("y"), a("tote_z"), a("base_x"), a("base_y"), a("base_yaw")
    wx = bx + np.cos(yaw) * x - np.sin(yaw) * y; wy = by + np.sin(yaw) * x + np.cos(yaw) * y
    W = min(s + 400, len(r) - 1)
    if len(r) - 1 < s + 400: continue            # sem janela completa
    held = bool(np.hypot(x[W], y[W]) < 0.8 and np.hypot(wx[W] - wx[0], wy[W] - wy[0]) > 0.3 and tz[W] >= 0.6)
    d = dict(run=f.split("/")[-3], ep=os.path.basename(f)[6:-4], s=s, held=held, x_sw=float(x[s]), y_sw=float(y[s]))
    for dt in (60, 100, 150, 200, 250, 300):
        j = min(s + dt, len(r) - 1)
        seg = slice(max(j - 10, s), j + 1)
        d[f"cy_{dt}"] = float(np.nanmean(a("cyL")[seg])); d[f"ymax_{dt}"] = float(np.nanmean(a("ymaxL")[seg])); d[f"area_{dt}"] = float(np.nanmean(a("areaL")[seg]))
        d[f"cmdv_{dt}"] = float(np.nanmean(a("cmd_v")[seg]))
    rows.append(d)
json.dump(rows, open(os.path.expanduser("~/Documents/Psi0/_scratch/grasp_ds.json"), "w"))
print(len(rows), "episódios-troca;", sum(r["held"] for r in rows), "held")
