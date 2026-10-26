"""Verifica um run GR00T sub-task: ação prevista (server_debug) vs estado executado, e movimento de base/braços/mãos.
Uso: python check_run.py <run_dir>   (ação 36D: lh0:7 rh7:14 la14:21 ra21:28 rpy28:31 h31 vx32 vy33 vyaw34 tyaw35;
estado 32D: lh0:7 rh7:14 la14:21 ra21:28 rpyh28:32)"""
import glob, json, sys
import numpy as np

run = sys.argv[1]
fs = sorted(glob.glob(f"{run}/server_debug/step_*.npz"))
D = [np.load(f) for f in fs]
A = [d["action"] for d in D]                   # (T,36)
S = np.stack([np.asarray(d["state"]).reshape(-1)[:32] for d in D])   # (N,32) estado ao pedir cada chunk
print("n_chunks", len(D), "chunk_len", A[0].shape, "prompts_used", {str(d["used_prompt"]) for d in D})
sl = dict(lh=slice(0,7), rh=slice(7,14), la=slice(14,21), ra=slice(21,28))
def rng(x): return float(np.abs(x.max(0)-x.min(0)).max()) if len(x) else 0.0
rows = []
for i, a in enumerate(A):
    r = dict(step=i+1, vx=float(a[:,32].mean()), vy=float(a[:,33].mean()), vyaw=float(a[:,34].mean()),
             height=float(a[:,31].mean()),
             amp_la=rng(a[:,sl['la']]), amp_ra=rng(a[:,sl['ra']]), amp_rh=rng(a[:,sl['rh']]), amp_lh=rng(a[:,sl['lh']]))
    if i+1 < len(S):  # estado no próximo pedido vs ação em t=h do chunk atual (h = nº de passos executados ~ desconhecido) -> usa ação média final do chunk
        for k in ('la','ra','rh','lh'):
            tgt = a[:, sl[k]]
            r[f"cmd_vs_next_state_{k}"] = float(np.abs(S[i+1][sl[k]] - tgt[-1]).mean())
            r[f"moved_{k}"] = float(np.abs(S[i+1][sl[k]] - S[i][sl[k]]).mean())
            r[f"cmd_move_{k}"] = float(np.abs(tgt[-1] - S[i][sl[k]]).mean())
    rows.append(r)
out = dict(per_chunk=rows,
    summary=dict(vx_mean=float(np.mean([r['vx'] for r in rows])), vx_max=float(np.max([r['vx'] for r in rows])),
                 vyaw_absmean=float(np.mean([abs(r['vyaw']) for r in rows])),
                 arm_amp_mean_rad=dict(la=float(np.mean([r['amp_la'] for r in rows])), ra=float(np.mean([r['amp_ra'] for r in rows]))),
                 rh_amp_mean=float(np.mean([r['amp_rh'] for r in rows])),
                 state_total_move_rad=dict(la=float(np.abs(S[-1][sl['la']]-S[0][sl['la']]).mean()), ra=float(np.abs(S[-1][sl['ra']]-S[0][sl['ra']]).mean()))))
json.dump(out, open(f"{run}/check_run.json", "w"), indent=1)
print(json.dumps(out["summary"], indent=1))
tr = [(r['step'], round(r['cmd_move_ra'],3), round(r.get('moved_ra',0),3)) for r in rows if 'moved_ra' in r]
print("right arm: (chunk, commanded_move, executed_move) every 3rd:", tr[::3])
print("vx per chunk:", [round(r['vx'],2) for r in rows])
print("first state:", np.round(S[0],2)); print("last state: ", np.round(S[-1],2))
