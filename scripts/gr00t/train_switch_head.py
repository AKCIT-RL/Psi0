"""Treina/valida a cabeça de conclusão (approach terminou?) em _scratch/cut_feats2.npz, com validação por episódio (GroupKFold).
Compara HistGradientBoosting (teto) com regressão logística padronizada (deployável em numpy). Exporta o logístico em JSON.
Métricas por episódio retido: AUC; erro do instante de troca (primeiro t com p>=thr por K amostras seguidas, em passos relativos ao corte);
falso disparo = troca com t < -FALSE_T passos antes do corte (robô ainda longe)."""
import json, os, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

D = np.load(os.path.expanduser("~/Documents/Psi0/_scratch/cut_feats2.npz"))
X, y, ep, t = D["X"], D["y"], D["ep"], D["t"]
STRIDE, K, FALSE_T = 5, 3, 60
cols = ["area", "cx", "cy", "xmin", "xmax", "ymin", "ymax"] + [f"s{i}" for i in range(32)] + ["pv_x", "pv_y", "pv_yaw", "mean_v"]
sel_sets = {"img+state+vel": list(range(len(cols))), "img+vel": list(range(7)) + [39, 40, 41, 42], "img only": list(range(7))}

def prep_lin(Xa, mu=None, sd=None):
    Z = Xa.copy(); miss = np.isnan(Z[:, :1]) | (Z[:, :1] == 0)           # blob ausente
    Z[np.isnan(Z)] = 0.0
    Z = np.hstack([Z, (Z[:, :1] == 0).astype(float)])
    if mu is None: mu, sd = Z.mean(0), Z.std(0) + 1e-6
    return (Z - mu) / sd, mu, sd

def eval_switch(p, tt, thr):
    ok = np.convolve((p >= thr).astype(int), np.ones(K, int), "valid") >= K
    i = np.argmax(ok) if ok.any() else None
    return None if i is None else int(tt[i])                              # tt relativo ao corte (passos)

def run(name, cols_idx, model_kind):
    gkf = GroupKFold(n_splits=5); aucs, errs, falses, miss_n, tot = [], [], 0, 0, 0
    for tr, te in gkf.split(X, y, ep):
        if model_kind == "gb":
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=4, random_state=0).fit(X[tr][:, cols_idx], y[tr])
            pr = lambda A: m.predict_proba(A[:, cols_idx])[:, 1]
        else:
            Z, mu, sd = prep_lin(X[tr][:, cols_idx]); m = LogisticRegression(C=0.5, max_iter=2000).fit(Z, y[tr])
            pr = lambda A: m.predict_proba(prep_lin(A[:, cols_idx], mu, sd)[0])[:, 1]
        p = pr(X[te]); aucs.append(roc_auc_score(y[te], p))
        for e in np.unique(ep[te]):
            mk = ep[te] == e; tt = t[te][mk]; pp = p[mk]; s = eval_switch(pp, tt, 0.5); tot += 1
            if s is None: miss_n += 1
            else:
                errs.append(s)
                if s < -FALSE_T: falses += 1
    e = np.array(errs)
    print(f"{name:14s} {model_kind:6s} AUC {np.mean(aucs):.3f} | switch error (steps vs cut): median {np.median(e):+.0f} p10 {np.percentile(e,10):+.0f} p90 {np.percentile(e,90):+.0f} | early(<-{FALSE_T}) {falses}/{tot} | never {miss_n}/{tot}")

for nm, idx in sel_sets.items():
    for kind in ("gb", "lin"): run(nm, idx, kind)

# exporta o logístico (img+state+vel) treinado em tudo
idx = sel_sets["img+state+vel"]; Z, mu, sd = prep_lin(X[:, idx]); m = LogisticRegression(C=0.5, max_iter=2000).fit(Z, y)
json.dump(dict(cols_idx=idx, mu=mu.tolist(), sd=sd.tolist(), w=m.coef_[0].tolist(), b=float(m.intercept_[0])), open(os.path.expanduser("~/Documents/Psi0/_scratch/switch_head_lin_v2.json"), "w"))
print("exported linear head")
