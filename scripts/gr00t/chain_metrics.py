"""Métricas da cadeia approach->grasp a partir do CSV por passo (chain/chain_<ep>.csv). Uso: python chain_metrics.py <run_dir>
Definições (sinal físico, independente do success do SIMPLE, que só vale para 'caixa na mesa'):
  switched      houve a troca approach->grasp
  grasped       após a troca, a caixa ficou a <0.8 m da base nos últimos 100 passos E se deslocou >0.3 m no mundo (saiu da prateleira e está na mão; caixa ainda no alto, z>=0.6 m, não no chão)
  carried_far   grasped E a caixa se deslocou >1.0 m no mundo
  table_dist    distância final da caixa ao centro da mesa (-7.206, 0.625), para contexto (sucesso oficial: ver eval_stats.txt)"""
import csv, glob, json, os, sys
import numpy as np

run = sys.argv[1]
TABLE = np.array([-7.206, 0.625])
out = {}
for f in sorted(glob.glob(f"{run}/chain/chain_*.csv")):
    ep = os.path.basename(f)[6:-4]
    r = list(csv.DictReader(open(f)))
    a = lambda k: np.array([float(x[k]) for x in r])
    x, y, z, bx, by, yaw, tz = a("x"), a("y"), a("z_rel"), a("base_x"), a("base_y"), a("base_yaw"), a("tote_z")
    wx = bx + np.cos(yaw) * x - np.sin(yaw) * y; wy = by + np.sin(yaw) * x + np.cos(yaw) * y
    sw = [i for i, q in enumerate(r) if q["switch"] == "1"]
    # chegou à zona de alcance (oracle, só para avaliar): em algum passo da fase approach a caixa esteve a x in [0.35,0.75] e |y|<=0.35
    ph = np.array([q["phase"] for q in r])
    reached = bool(np.any((x >= 0.35) & (x <= 0.75) & (np.abs(y) <= 0.35) & (ph == "approach")) or sw)
    d = dict(steps=len(r), switch_step=sw[0] if sw else None, reached=reached)
    if sw:
        s0 = sw[0]; d["switch_x"] = round(float(x[s0]), 3); d["switch_y"] = round(float(y[s0]), 3)
        d["false_switch"] = bool(not (0.35 <= x[s0] <= 0.75 and abs(y[s0]) <= 0.35))
    if sw:
        s = sw[0]
        near = np.hypot(x[-100:], y[-100:]).max()
        moved = float(np.hypot(wx[-1] - wx[0], wy[-1] - wy[0]))
        W = min(s + 400, len(r) - 1)   # janela do grasp: o treino do grasp dura ~262 passos; depois (carregar até a mesa) não foi treinado
        nearW = float(np.hypot(x[W], y[W])); movedW = float(np.hypot(wx[W] - wx[0], wy[W] - wy[0])); tzW = float(tz[W])
        d.update(grasped_window=bool(nearW < 0.8 and movedW > 0.3 and tzW >= 0.6), tote_z_window=round(tzW, 3), tote_moved_window_m=round(movedW, 2), tote_to_base_window_m=round(nearW, 2),
                 steps_after_switch=len(r) - s, tote_moved_m=round(moved, 2), robot_moved_after_switch_m=round(float(np.hypot(bx[-1] - bx[s], by[-1] - by[s])), 2),
                 tote_to_base_last100_max_m=round(float(near), 2), tote_z_end=round(float(tz[-1]), 3), tote_z_at_switch=round(float(tz[s]), 3),
                 grasped=bool(near < 0.8 and moved > 0.3 and tz[-1] >= 0.6), carried_far=bool(near < 0.8 and moved > 1.0 and tz[-1] >= 0.6), table_dist_end_m=round(float(np.hypot(*(np.array([wx[-1], wy[-1]]) - TABLE))), 2))
    else:
        d.update(grasped=False, carried_far=False, grasped_window=False, note="approach não atingiu a zona de troca; sem fase grasp")
    out[ep] = d
json.dump(out, open(f"{run}/chain_metrics.json", "w"), indent=1)
print(json.dumps(out, indent=1))
