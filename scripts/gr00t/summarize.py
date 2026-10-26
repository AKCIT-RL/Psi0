"""Resumo comparável de runs da cadeia: python summarize.py <run> [<run> ...] (usa chain_metrics.json; rode chain_metrics.py antes)."""
import json, os, sys
home = os.path.expanduser("~/wla_data/experiments/wla")
print(f"{'run':30s} {'n':>3s} {'chegou':>7s} {'trocou':>7s} {'troca falsa':>12s} {'segura/trocou':>14s} {'segura/chegou':>14s} {'carregou>1m':>12s}")
for run in sys.argv[1:]:
    p = f"{home}/{run}/chain_metrics.json"
    if not os.path.exists(p): print(run, "sem chain_metrics.json"); continue
    d = json.load(open(p)); n = len(d)
    rc = [e for e, v in d.items() if v.get("reached")]; sw = [e for e, v in d.items() if v.get("switch_step") is not None]
    fs = [e for e in sw if d[e].get("false_switch")]; gw = [e for e in sw if d[e].get("grasped_window")]
    gr = [e for e in rc if d[e].get("grasped_window")]; cf = [e for e in sw if d[e].get("carried_far")]
    print(f"{run:30s} {n:3d} {len(rc):3d}/{n:<3d} {len(sw):3d}/{n:<3d} {len(fs):6d}/{len(sw):<5d} {len(gw):7d}/{len(sw):<6d} {len(gr):7d}/{len(rc):<6d} {len(cf):6d}/{len(sw):<5d}")
sys.exit(0)
print(f"{'run':28s} {'n':>3s} {'trocou':>7s} {'grasp(janela400)':>17s} {'grasp(fim)':>11s} {'carregou>1m':>12s}")
for run in sys.argv[1:]:
    p = f"{home}/{run}/chain_metrics.json"
    if not os.path.exists(p): print(run, "sem chain_metrics.json"); continue
    d = json.load(open(p)); n = len(d)
    sw = [e for e, v in d.items() if v.get("switch_step") is not None]
    gw = [e for e in sw if d[e].get("grasped_window")]; ge = [e for e in sw if d[e].get("grasped")]; cf = [e for e in sw if d[e].get("carried_far")]
    print(f"{run:28s} {n:3d} {len(sw):3d}/{n:<3d} {len(gw):3d}/{len(sw):<3d} (cond.) {len(ge):4d}/{len(sw):<3d} {len(cf):6d}/{len(sw):<3d}")
