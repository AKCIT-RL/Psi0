"""[chain_assist2 = chain_assist + varredura por GIRO (H-B2): após ROTATE_AFTER passos de busca lateral sem achar a caixa, gira vyaw=ROT_VYAW (+ = esquerda) por ROT_T passos, depois volta à busca lateral; atributo self.vyaw lido pelo agente.]
Assistência de parada/busca para o approach (NÃO é do modelo; marcada nos logs). Módulo puro (numpy), testável fora do simulador.

Modos retornados por update(): 0 = nada, 1 = empurrão para frente (approach parou longe com a caixa visível e centralizada),
2 = busca lateral (approach parou e a caixa está fora de vista: varre vy=-V (direita) por SEARCH_T passos, depois +V (esquerda), alternando).
Convenção do robô: y>0 esquerda, então vy<0 vai para a direita.
"""
from collections import deque
import numpy as np


class AssistController:
    def __init__(self, forward=True, search=False, stall_steps=60, ymax_far=110.0, vx=0.2, search_vy=0.3,
                 search_after=60, search_period=200, min_vis_area=300, usable_area=1200, edge_dev=140, search_vx=0.2, creep_budget=150, rotate_after=400, rot_t=650, rot_vyaw=0.5):
        self.forward, self.search = forward, search
        self.stall_steps, self.ymax_far, self.vx, self.vy = stall_steps, ymax_far, vx, search_vy
        self.search_after, self.search_period, self.min_vis = search_after, search_period, min_vis_area
        self.usable_area, self.edge_dev = usable_area, edge_dev
        self.search_vx, self.creep_budget = search_vx, creep_budget
        self.rotate_after, self.rot_t, self.rot_vyaw = rotate_after, rot_t, rot_vyaw
        self.reset()

    def reset(self):
        self.spd = deque(maxlen=self.stall_steps)
        self.mode, self.fwd_steps, self.invis, self.vis_run, self.search_t, self.creep_used = 0, 0, 0, 0, 0, 0
        self.vyaw = None

    def update(self, L, cmd_v):
        """L = (area, cx, cy, xmin, xmax, ymin, ymax) da mancha azul ou None; cmd_v = |v| planar comandado pelo approach (nan se desconhecido).
        Retorna (mode, dvx_min, vy_override) — dvx_min: vx mínimo a impor (None se não); vy_override: vy a impor (None se não)."""
        self.spd.append(cmd_v if cmd_v == cmd_v else 1.0)
        stalled = len(self.spd) == self.spd.maxlen and float(np.mean(self.spd)) < 0.1
        visible = bool(L) and L[0] >= self.min_vis
        centered_vis = bool(L) and L[0] >= self.usable_area and abs(L[1] - 320) < 200   # caixa de fato utilizável: grande e centralizada (fatias que piscam no topo/borda não contam)
        self.invis = 0 if centered_vis else self.invis + 1           # "sem caixa utilizável": ausente ou só uma fatia na borda
        self.vis_run = self.vis_run + 1 if centered_vis else 0
        centered = bool(L) and abs(L[1] - 320) < 150
        ymax = L[6] if L else None

        if self.search and self.mode == 2:                       # em busca: termina quando a caixa fica centralizada e visível por 5 passos
            if self.vis_run >= 5:
                self.mode = 0
        elif self.search and self.mode == 0 and stalled and self.invis >= self.search_after:
            self.mode, self.search_t = 2, 0

        if self.forward and self.mode == 1:                      # empurrão: termina quando a mancha chega à posição de estacionamento
            if ymax is None or ymax >= self.ymax_far + 8 or self.fwd_steps > 250:
                self.mode = 0
        elif self.forward and self.mode == 0 and stalled and centered and ymax is not None and ymax < self.ymax_far:
            self.mode, self.fwd_steps = 1, 0

        self.vyaw = None
        if self.mode == 1:
            self.fwd_steps += 1
            return 1, self.vx, None
        if self.mode == 2:
            if L and L[0] >= self.min_vis and abs(L[1] - 320) >= self.edge_dev and L[0] < self.usable_area:   # mancha parcial na lateral: segue na direção dela (direita se cx>320)
                vy = -self.vy if L[1] > 320 else self.vy
            else:                                                         # nada visível: varre alternando (direita primeiro)
                vy = -self.vy if (self.search_t // self.search_period) % 2 == 0 else self.vy
            cyc = self.rotate_after + self.rot_t
            if self.rotate_after > 0 and (self.search_t % cyc) >= self.rotate_after and not (L and L[0] >= self.min_vis):
                self.vyaw = self.rot_vyaw; vy = 0.0                      # varredura por giro (nada visível)
            self.search_t += 1
            dvx = None
            if self.vyaw is not None: pass
            elif self.creep_used < self.creep_budget:     # avanço curto: a caixa só entra no campo de visão ao chegar mais perto
                dvx, self.creep_used = self.search_vx, self.creep_used + 1
            return 2, dvx, vy
        return 0, None, None
