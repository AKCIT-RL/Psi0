"""Servidor multi-modelo (approach, grasp, phase3) em UM processo, uma porta por fase, com troca de GPU sob demanda.

Os três modelos juntos + o simulador não cabem na VRAM da 4090 (24,5 GB). Todos ficam na RAM (CPU); só o modelo da fase requisitada
fica na GPU: na primeira requisição de outra fase o ativo volta para a CPU e o novo sobe (alguns segundos; a simulação é síncrona).
Cada porta se comporta como o serve_subtask.py (prompt forçado, mediana de N amostras, dump de debug por requisição).

Uso: python serve_chain.py --paths approach=<dir>,grasp=<dir>,phase3=<dir> --ports 5556,5557,5558 --prompts-json '{"approach": "...", ...}'
     [--num-samples 5,5,1] [--debug-dir <dir>] [--seed N]
"""
import json, sys, threading, time
from dataclasses import dataclass
from typing import Any

import numpy as np
import tyro
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

import serve_subtask as SS   # aplica o patch de huggingface_hub e define SubtaskServer / _Ensemble


@dataclass
class Args:
    paths: str
    ports: str = "5556,5557,5558"
    prompts_json: str = "{}"
    num_samples: str = "5,5,1"
    embodiment_tag: str = "UNITREE_G1_N1D7"
    debug_dir: str = ""
    seed: int = -1
    host: str = "127.0.0.1"


def main():
    a = tyro.cli(Args, args=sys.argv[1:])
    paths = dict(kv.split("=", 1) for kv in a.paths.split(","))
    names = list(paths)
    ports = [int(p) for p in a.ports.split(",")]
    nsamp = [int(n) for n in a.num_samples.split(",")]
    prompts = json.loads(a.prompts_json)
    if a.seed >= 0:
        torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); np.random.seed(a.seed)
    from gr00t.data.embodiment_tags import EmbodimentTag

    servers = {}
    for name, ns in zip(names, nsamp):
        cfg = SS.Cfg(model_path=paths[name], embodiment_tag=EmbodimentTag[a.embodiment_tag], device="cpu", host=a.host,
                     force_prompt=prompts.get(name, ""), num_samples=ns, debug_dir=(f"{a.debug_dir}_{name}" if a.debug_dir else ""))
        servers[name] = SS.SubtaskServer(cfg)
        print(f"[chain-serve] {name} carregado na CPU (num_samples={ns})", flush=True)

    lock = threading.Lock(); state = {"active": None}

    def model_of(name):
        pol = servers[name].policy
        while hasattr(pol, "inner"): pol = pol.inner
        while hasattr(pol, "policy"): pol = pol.policy      # Gr00tSimPolicyWrapper -> Gr00tPolicy
        return pol.model

    def ensure(name):
        if state["active"] == name: return
        t0 = time.monotonic()
        if state["active"] is not None:
            model_of(state["active"]).to("cpu")
        torch.cuda.empty_cache()
        model_of(name).to("cuda:0")
        state["active"] = name
        print(f"[chain-serve] GPU: {name} ativo (troca em {time.monotonic() - t0:.1f}s, mem {torch.cuda.memory_allocated() / 1e9:.1f} GB)", flush=True)

    def make_app(name):
        app = FastAPI()
        srv = servers[name]

        def act(payload: dict[str, Any]):
            with lock:
                ensure(name)
                return srv.predict_action(payload)

        app.post("/act")(act)
        app.get("/health")(lambda: JSONResponse(content={"status": "ok", "phase": name, "active": state["active"]}))
        return app

    for name, port in zip(names, ports):
        cfg = uvicorn.Config(make_app(name), host=a.host, port=port, log_level="warning")
        s = uvicorn.Server(cfg); s.install_signal_handlers = lambda: None
        threading.Thread(target=s.run, daemon=True).start()
        print(f"[chain-serve] {name} em {a.host}:{port}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
