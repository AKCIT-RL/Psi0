"""Fase 0 (PLANO_3_FASES): testes offline dos 3 checkpoints GR00T (approach, grasp, phase3) sobre quadros reais do dataset anotado.

Saída: _scratch/phase0_results.json  (matriz 3x3, ablações do modelo 3, direção/magnitude).
Episódios 'fora' = últimos n_val=int(N*0.2018) do dataset da fase 3 (split determinístico de factory.py), ASSUMINDO que o dataset da
fase 3 é a lista ordenada dos episódios anotados com end>0. 'dentro' = os demais (vistos no treino).
Uso: python phase0_offline.py [--frames]   (PYTHONPATH com o pacote da fase 3 + scripts/gr00t)
"""
import gc, json, os, sys
from types import SimpleNamespace

import av, numpy as np, pandas as pd
import huggingface_hub
huggingface_hub.model_info = lambda *a, **k: SimpleNamespace(tags=[])
import torch
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.deploy import gr00t_serve_simple as S
from psi.deploy.helpers import RequestMessage

ANN = os.path.expanduser("~/ann"); OUT = os.path.expanduser("~/Documents/Psi0/_scratch")
PKG2 = os.path.expanduser("~/gr00t_x/gr00t_subtask_pkg/weights"); PKG3 = os.path.expanduser("~/gr00t_x/phase3/gr00t_phase3_pkg/weights")
MODELS = {"approach": f"{PKG2}/approach", "grasp": f"{PKG2}/grasp", "phase3": f"{PKG3}/phase3"}
PROMPTS = {"approach": "walk to the shelf and stop centered in front of the blue tote with the arms raised.",
           "grasp": "grasp the blue tote.", "phase3": "carry the blue tote to the table and put it down."}
H = 40
GROUPS = {"mao_dir": slice(7, 14), "bracos": slice(14, 28), "altura": slice(31, 32), "vx": slice(32, 33), "vy": slice(33, 34), "vyaw": slice(34, 35)}

cuts = json.load(open(f"{ANN}/cuts.json"))["episodes"]
eps = sorted(int(k) for k, v in cuts.items() if v["end"] > 0)
n_val = int(len(eps) * 0.2018); held = eps[-n_val:]; seen = eps[:-n_val]
rng = np.random.RandomState(0)


def collect():
    """~200 quadros por segmento (100 fora do treino, 100 dentro) com estado e chunk gravado (H,36)."""
    cache = f"{OUT}/phase0_frames.npz"
    if os.path.exists(cache):
        return dict(np.load(cache, allow_pickle=True))
    rows = []
    for grp, pool, per in (("fora", held, 5), ("dentro", rng.choice(seen, 20, replace=False).tolist(), 5)):
        for e in pool:
            df = pd.read_parquet(f"{ANN}/data/chunk-000/episode_{e:06d}.parquet")
            St = np.stack(df["states"].values); A = np.stack(df["action"].values); n = len(df)
            end, ge = cuts[str(e)]["end"], cuts[str(e)]["grasp_end"]
            segs = {"approach": (0, end), "grasp": (end, ge), "carry": (ge, n - H)}
            want = {}
            for sname, (a, b) in segs.items():
                b = min(b, n - H)
                if b - a < 10: continue
                for i in rng.choice(np.arange(a, b), min(per, b - a), replace=False): want[int(i)] = sname
            c = av.open(f"{ANN}/videos/chunk-000/egocentric/episode_{e:06d}.mp4")
            for i, f in enumerate(c.decode(video=0)):
                if i in want:
                    rows.append((grp, e, want[i], i, f.to_ndarray(format="rgb24"), St[i], A[i:i + H]))
                if i >= max(want): break
            c.close(); print("ep", e, len(rows), flush=True)
    d = dict(grp=np.array([r[0] for r in rows]), ep=np.array([r[1] for r in rows]), seg=np.array([r[2] for r in rows]),
             idx=np.array([r[3] for r in rows]), img=np.stack([r[4] for r in rows]), st=np.stack([r[5] for r in rows]), gt=np.stack([r[6] for r in rows]))
    np.savez(cache, **d); return d


def load(name):
    cfg = S.ServerConfig(model_path=MODELS[name], embodiment_tag=EmbodimentTag.UNITREE_G1_N1D7, device="cuda:0")
    return S.Server(cfg)


def predict(srv, img, st, prompt, seed=None):
    if seed is not None: torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    req = RequestMessage(image={"rgb_head_stereo_left": img}, instruction=prompt, history={}, state={"states": st.reshape(1, 32)},
                         condition={}, gt_action=[], dataset_name="", timestamp="")
    obs = srv._build_observation(req)
    act, _ = srv.policy.get_action(obs)
    a = np.asarray(S.Server._action_to_psi_format(act)); return a.reshape(-1, a.shape[-1])[:H]


def err(pred, gt):
    return {g: float(np.abs(pred[:, s] - gt[:, s]).mean()) for g, s in GROUPS.items()}


def main():
    D = collect(); N = len(D["ep"]); print("frames", N, {k: int((D["seg"] == k).sum()) for k in ("approach", "grasp", "carry")}, flush=True)
    res = {"held_out_eps": held, "n_seen_pool": len(seen), "matrix": {}, "dirmag": {}}
    preds3 = None
    for name in MODELS:
        srv = load(name); P = np.zeros((N, H, 36), np.float32)
        for k in range(N): P[k] = predict(srv, D["img"][k], D["st"][k], PROMPTS[name], seed=k)
        for seg in ("approach", "grasp", "carry"):
            for grp in ("fora", "dentro"):
                m = (D["seg"] == seg) & (D["grp"] == grp)
                if m.sum() == 0: continue
                e = [err(P[k], D["gt"][k]) for k in np.nonzero(m)[0]]
                res["matrix"][f"{name}|{seg}|{grp}"] = {"n": int(m.sum()), **{g: float(np.mean([x[g] for x in e])) for g in GROUPS}}
        if name == "phase3":
            m = D["seg"] == "carry"
            res["dirmag"] = {grp: {g: {"pred_mean": float(P[(D["grp"] == grp) & m][:, :, GROUPS[g]].mean()), "gt_mean": float(D["gt"][(D["grp"] == grp) & m][:, :, GROUPS[g]].mean())}
                                    for g in ("mao_dir", "altura", "vx", "vy", "vyaw")} for grp in ("fora", "dentro")}
            # mão direita: fração de quadros com mão 'fechada' ; reportar min/max/med previstos vs gravados
            res["dirmag"]["mao_dir_quantis"] = {"pred": np.quantile(P[m][:, :, 7:14], [0.1, 0.5, 0.9]).tolist(), "gt": np.quantile(D["gt"][m][:, :, 7:14], [0.1, 0.5, 0.9]).tolist()}
            # Teste C: ablações em 60 quadros de carregar
            idx = rng.choice(np.nonzero(m)[0], 60, replace=False); abl = {}
            def run(fn, **kw): return np.stack([predict(srv, *fn(k), **kw) for k in idx])
            base = np.stack([P[k] for k in idx])
            def d(a, b): return float(np.abs(a - b).mean())
            noise = [d(base, run(lambda k: (D["img"][k], D["st"][k], PROMPTS["phase3"]), seed=1000 + s)) for s in range(3)]
            abl["ruido_semente"] = noise
            for pn in ("approach", "grasp"): abl[f"prompt_{pn}"] = d(base, run(lambda k: (D["img"][k], D["st"][k], PROMPTS[pn]), seed=7))
            abl["prompt_vazio"] = d(base, run(lambda k: (D["img"][k], D["st"][k], ""), seed=7))
            abl["imagem_preta"] = d(base, run(lambda k: (np.zeros_like(D["img"][k]), D["st"][k], PROMPTS["phase3"]), seed=7))
            perm = rng.permutation(len(idx))
            abl["imagem_de_outro_quadro"] = d(base, run(lambda k: (D["img"][idx[perm[list(idx).index(k)]]], D["st"][k], PROMPTS["phase3"]), seed=7))
            abl["estado_zero"] = d(base, run(lambda k: (D["img"][k], np.zeros(32, np.float32), PROMPTS["phase3"]), seed=7))
            # por grupo: vx prompt errado vs certo
            wrong = run(lambda k: (D["img"][k], D["st"][k], PROMPTS["approach"]), seed=7)
            abl["vx_medio_prompt_certo"] = float(base[:, :, 32].mean()); abl["vx_medio_prompt_approach"] = float(wrong[:, :, 32].mean())
            res["ablacoes"] = abl
        del srv; gc.collect(); torch.cuda.empty_cache()
        json.dump(res, open(f"{OUT}/phase0_results.json", "w"), indent=1); print("done", name, flush=True)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
