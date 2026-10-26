"""Servidor HTTP do GR00T sub-task (approach/grasp) com prompt forçado e dump de debug por requisição.

Envolve gr00t.deploy.gr00t_serve_simple.Server sem alterá-lo:
- força a instrução (o cliente do SIMPLE manda o texto gravado do episódio, que não é o prompt do sub-task);
- salva em --debug_dir um npz por requisição: imagem recebida, estado 32D, instrução recebida/usada, ação 36D (T,36).
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

import huggingface_hub
from types import SimpleNamespace

# transformers 4.57 consulta a API do HF (model_info) para detectar tokenizers Mistral, o que falha offline.
# O backbone é Cosmos-Reason2 (Qwen3-VL), não Mistral: devolver "sem tags" é o resultado correto.
huggingface_hub.model_info = lambda *a, **k: SimpleNamespace(tags=[])

from gr00t.deploy import gr00t_serve_simple as S


@dataclass
class Cfg(S.ServerConfig):
    force_prompt: str = ""
    """Se não vazio, substitui a instrução vinda do cliente."""
    debug_dir: str = ""
    """Se não vazio, grava um npz por requisição."""
    seed: int = -1
    """Se >= 0, semeia torch/numpy no início (amostragem de difusão reprodutível)."""
    num_samples: int = 1
    """N amostras de difusão por consulta; a ação devolvida é a MEDIANA elemento a elemento (consenso). MUDANÇA DE PROTOCOLO se > 1."""
    noise_scale: float = 1.0
    """Escala do ruído inicial da difusão no momento da inferência (1.0 = padrão do modelo; 0.0 = determinístico). MUDANÇA DE PROTOCOLO se != 1."""


class _Ensemble:
    """Envolve a política: roda N amostras e devolve a mediana por chave (modo majoritário da difusão)."""

    def __init__(self, inner, n):
        self.inner, self.n = inner, n

    def get_action(self, obs, *a, **k):
        outs = [self.inner.get_action(obs, *a, **k) for _ in range(self.n)]
        acts = [o[0] for o in outs]
        agg = {key: np.median(np.stack([np.asarray(x[key]) for x in acts]), axis=0) for key in acts[0]}
        return agg, outs[0][1]

    def __getattr__(self, name):
        return getattr(self.inner, name)


class SubtaskServer(S.Server):
    def __init__(self, cfg: Cfg):
        super().__init__(cfg)
        if cfg.num_samples > 1:
            self.policy = _Ensemble(self.policy, cfg.num_samples)
            print(f"[serve] ensemble median over {cfg.num_samples} samples", flush=True)
        self.force_prompt = cfg.force_prompt
        self.debug_dir = Path(cfg.debug_dir) if cfg.debug_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self._ctx = {}

    def _build_observation(self, request):
        recv = request.instruction
        if self.force_prompt:
            request.instruction = self.force_prompt
        obs = super()._build_observation(request)
        img = next(iter(request.image.values()))
        self._ctx = dict(
            recv_prompt=str(recv), used_prompt=str(request.instruction),
            image=np.asarray(img), state=np.asarray(request.state.get("states", np.zeros(0))),
        )
        return obs

    def _action_to_psi_format(self, action):
        out = S.Server._action_to_psi_format(action)
        self.n += 1
        if self.n <= 3 or self.n % 10 == 0:
            print(f"[req {self.n}] recv_prompt={self._ctx.get('recv_prompt')!r} used={self._ctx.get('used_prompt')!r}", flush=True)
        if self.debug_dir:
            a = np.asarray(out)
            np.savez(self.debug_dir / f"step_{self.n:05d}.npz", action=a.reshape(-1, a.shape[-1]), **self._ctx)
        return out


if __name__ == "__main__":
    cfg = tyro.cli(Cfg, args=sys.argv[1:])
    if cfg.seed >= 0:
        import torch
        torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed); np.random.seed(cfg.seed)
    if cfg.noise_scale != 1.0:
        import torch
        import gr00t.model.gr00t_n1d7.gr00t_n1d7 as _M

        class _TorchScaled:                    # o módulo do modelo chama torch.randn só para o ruído inicial da inferência
            def __getattr__(self, n): return getattr(torch, n)
            def randn(self, *a, **k): return cfg.noise_scale * torch.randn(*a, **k)

        _M.torch = _TorchScaled()
        print(f"[serve] noise_scale={cfg.noise_scale}", flush=True)
    srv = SubtaskServer(cfg)
    print(f"forced prompt: {cfg.force_prompt!r}", flush=True)
    srv.run(cfg.host, cfg.port)
