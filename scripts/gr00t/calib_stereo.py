"""Calibra a distância da caixa azul pela imagem (área/centroide) e pelo estéreo (disparidade esquerda-direita) contra o x do oracle.
Usa vídeos head_stereo_{left,right}_*.mp4 dos runs antigos e os CSVs chain/*.csv (fase approach, a cada STRIDE passos)."""
import csv, glob, os, sys
import av, numpy as np

STRIDE = 10
H = os.path.expanduser("~/wla_data/experiments/wla/")

def blob(im):
    R, G, B = [im[..., i].astype(np.int16) for i in range(3)]
    m = (B > 120) & (B - R > 60) & (B - G > 20)
    n = int(m.sum())
    if n < 30: return None
    ys, xs = np.nonzero(m)
    return n, float(xs.mean()), float(ys.mean()), int(xs.min()), int(xs.max()), int(ys.max())

def frames(path, steps):
    want = set(steps); out = {}
    c = av.open(path)
    for i, f in enumerate(c.decode(video=0)):
        if i in want: out[i] = f.to_ndarray(format="rgb24")
        if i > max(steps): break
    c.close(); return out

rows = []
runs = sys.argv[1:] or ["chain_val10_base", "chain_val10_v4_tight", "chain_val10_v2_h16", "chain_val10_base_seed1"]
for run in runs:
    for csvf in sorted(glob.glob(f"{H}{run}/chain/chain_val__episode_*.csv")):
        ep = csvf.split("chain_")[-1][:-4]
        r = list(csv.DictReader(open(csvf)))
        steps = [int(q["step"]) for q in r if q["phase"] == "approach" and int(q["step"]) % STRIDE == 0]
        if not steps: continue
        d = glob.glob(f"{H}{run}/eval/*/*/val/{ep}")
        if not d: continue
        pl = glob.glob(d[0] + "/head_stereo_left*.mp4"); pr = glob.glob(d[0] + "/head_stereo_right*.mp4")
        if not pl or not pr: continue
        L, Rr = frames(pl[0], steps), frames(pr[0], steps)
        for s in steps:
            if s in L and s in Rr:
                bl, br = blob(L[s]), blob(Rr[s])
                if bl and br:
                    q = r[s]
                    rows.append([float(q["x"]), float(q["y"]), bl[0], bl[1], bl[2], bl[5], br[0], br[1], bl[1] - br[1], hash(run) % 1000, int(ep.split("_")[-1]), s])
    print(run, "rows so far", len(rows), flush=True)
A = np.array(rows)
np.save(os.path.expanduser("~/Documents/Psi0/_scratch/stereo_calib.npy"), A)
print("cols: x_oracle y_oracle areaL cxL cyL ymaxL areaR cxR disparity run ep step")
x, y, aL, cxL, cyL, ym, aR, cxR, d = A[:, :9].T
def corr(a, b): return float(np.corrcoef(a, b)[0, 1])
print("n", len(A), "x range", x.min(), x.max())
print("corr(x, area^-0.5)", corr(x, aL ** -0.5), "| corr(x, 1/disp)", corr(x, 1 / np.maximum(d, 1)), "| corr(x, ymaxL)", corr(x, ym))
for name, feat in (("1/disp", 1 / np.maximum(d, 1)), ("area^-.5", aL ** -0.5)):
    k = np.polyfit(feat, x, 1); res = x - np.polyval(k, feat)
    print(f"fit x ~ {k[0]:.3f}*{name}+{k[1]:.3f}  rmse {res.std():.3f} m  | near zone x<0.8: rmse {res[x < 0.8].std():.3f}")
print("disparity percentiles (px) for x in [0.4,0.64]:", np.percentile(d[(x > 0.4) & (x < 0.64)], [10, 50, 90]))
