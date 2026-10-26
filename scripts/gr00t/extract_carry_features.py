"""[FASE 1 do PLANO_3_FASES] Variante para o corte grasp_end (troca 2->3): janela [end, grasp_end+POST]; rótulo 1 = quadro >= grasp_end; t = i - grasp_end. Saída: _scratch/carry_feats.npz.
Original: Extrai, dos episódios anotados (cuts.json), características por quadro: mancha azul + estado + comando anterior, e o rótulo
(0 = approach, 1 = a partir do corte `end`). Usa quadros [0, end+POST] a cada STRIDE. Saída: _scratch/carry_feats.npz"""
import json, os, sys
import av, numpy as np, pandas as pd

ANN = os.path.expanduser("~/ann"); STRIDE, POST = 5, 60
cuts = json.load(open(f"{ANN}/cuts.json"))["episodes"]

def blob(im):
    R, G, B = [im[..., i].astype(np.int16) for i in range(3)]
    m = (B > 170) & (B - R > 50) & (B - G > 25); n = int(m.sum())
    if n < 30: return [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    ys, xs = np.nonzero(m)
    return [n, xs.mean(), ys.mean(), xs.min(), xs.max(), ys.min(), ys.max()]

X, y, ep_id, t_rel = [], [], [], []
for k, v in cuts.items():
    e, end, ge = int(k), v["end"], v["grasp_end"]
    if end <= 0 or ge <= end: continue
    df = pd.read_parquet(f"{ANN}/data/chunk-000/episode_{e:06d}.parquet")
    S = np.stack(df["states"].values); A = np.stack(df["action"].values)
    want = set(range(end, min(ge + POST, len(df)), STRIDE))
    c = av.open(f"{ANN}/videos/chunk-000/egocentric/episode_{e:06d}.mp4")
    for i, f in enumerate(c.decode(video=0)):
        if i in want:
            prev_v = A[max(i - 1, 0), 32:35]; mean_v = np.abs(A[max(i - 10, 0):i + 1, 32:34]).mean()
            X.append(blob(f.to_ndarray(format="rgb24")) + list(S[i]) + list(prev_v) + [mean_v])
            y.append(int(i >= ge)); ep_id.append(e); t_rel.append(i - ge)
        if i > max(want): break
    c.close()
    print(e, len(X), flush=True)
np.savez(os.path.expanduser("~/Documents/Psi0/_scratch/carry_feats.npz"), X=np.array(X, float), y=np.array(y), ep=np.array(ep_id), t=np.array(t_rel))
print("done", len(X))
