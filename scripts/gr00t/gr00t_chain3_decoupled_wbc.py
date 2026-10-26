"""[PLANO_3_FASES, Fase 1] Cópia estendida do gr00t_chain_decoupled_wbc.py com 3ª fase (carry) e troca 2->3 por cabeça de imagem (CHAIN_HEAD2_*).
Variáveis novas: CHAIN_PORT_CARRY (5558), CHAIN_HEAD2_JSON, CHAIN_HEAD2_THR, CHAIN_HEAD2_MEANW, CHAIN_HEAD2_K, CHAIN_MIN_GRASP (passos mínimos na fase grasp antes de poder trocar), CHAIN_HEAD2_MODE (head | off).
Agente de cadeia approach -> grasp (GR00T N1.7 sub-tasks) para o SIMPLE. NÃO fica no submódulo: é montado no container
como simple/baselines/gr00t_chain_decoupled_wbc.py (ver docker_eval_chain.sh).

Dois servidores HTTP (um por peso). Fase 'approach' até a caixa-alvo estar centrada e ao alcance (info do simulador:
pose da caixa no referencial da base do robô), então troca (uma vez) para 'grasp'. A troca é "oracle" (usa estado do simulador).

Variáveis de ambiente:
  CHAIN_MODE        chain (padrão) | approach_only (só loga a distância, nunca troca; usado para calibrar)
  CHAIN_PORT_GRASP  porta do servidor grasp (padrão 5557); a do approach vem do --port do CLI
  CHAIN_X_MIN/MAX   faixa de distância à frente (m) para trocar (centro da caixa no referencial da base)
  CHAIN_Y_MAX       |y| máximo (m)       CHAIN_K  passos de controle consecutivos exigidos
  CHAIN_LOG         caminho do CSV por passo
  CHAIN_TRIGGER     oracle (padrão; usa pose do simulador) | stereo | mono | head (cabeça aprendida: imagem+estado+comando)  (stereo/mono só usam imagem + estado + comando; pose do simulador só é logada)
  CHAIN_CAL_A/B     x_est = A*feature + B (stereo: feature = 1/disparidade; mono: feature = área**-0.5), calibrados offline
  CHAIN_REQ_STOP    1 = exige também que o chunk atual do approach comande parar (|v| médio < CHAIN_STOP_V nos 20 primeiros passos)
  CHAIN_HEAD_JSON / CHAIN_HEAD_THR   pesos do logístico (switch_head_lin.json) e limiar de probabilidade (padrão 0.7)
  CHAIN_SEARCH      1 = assistência de busca (varre vy±CHAIN_SEARCH_VY quando o approach parou e a caixa não está visível), ver chain_assist.py
  CHAIN_ASSIST      1 = assistência de parada (NÃO é do modelo): se o approach parou (|v| médio<0.1 por CHAIN_ASSIST_STALL passos) com a caixa visível, centralizada e longe
                    (ymax da mancha < CHAIN_ASSIST_YMAX), soma vx=CHAIN_ASSIST_VX ao comando até a mancha chegar à posição de estacionamento (ymax>=CHAIN_ASSIST_YMAX+8)
  CHAIN_X_SMOOTH    janela (passos) da mediana de x_est (padrão 5)
"""
import json
import os
from collections import deque
import numpy as np

from simple.baselines.client import HttpActionClient
from simple.baselines.gr00t_n16_decoupled_wbc import STATE_SLICES, Gr00tN16DecoupledWbcAgent

PROMPTS = {
    "approach": "walk to the shelf and stop centered in front of the blue tote with the arms raised.",
    "grasp": "grasp the blue tote.",
    "carry": "carry the blue tote to the table and put it down.",
}
TABLE_XY = (-7.206, 0.625)          # centro da mesa branca (mundo)


def _blue(im):
    """Mancha azul (caixa-alvo): (area, cx, cy, xmin, xmax, ymin, ymax) ou None."""
    im = np.asarray(im)
    R, G, B = [im[..., i].astype(np.int16) for i in range(3)]
    m = (B > 170) & (B - R > 50) & (B - G > 25)
    n = int(m.sum())
    if n < 30:
        return None
    ys, xs = np.nonzero(m)
    return n, float(xs.mean()), float(ys.mean()), int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def _target_key_for_episode():
    """target_tote_key gravado no episódio (roots/<split>/meta/episodes.jsonl), via SIMPLE_EVAL_EPISODE_ID=<split>__episode_<i>."""
    ep = os.environ.get("SIMPLE_EVAL_EPISODE_ID", "")
    root = os.environ["CHAIN_ROOTS"]
    split, idx = ep.split("__episode_")
    with open(f"{root}/{split}/meta/episodes.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if int(d["episode_index"]) == int(idx):
                cfg = d["environment_config"]
                cfg = json.loads(cfg) if isinstance(cfg, str) else cfg
                return _find_key(cfg, "target_tote_key")
    raise KeyError(f"episódio {ep} não encontrado")


class Gr00tChain3DecoupledWbcAgent(Gr00tN16DecoupledWbcAgent):
    def __init__(self, robot, host, port, **kwargs):
        super().__init__(robot, host, port, **kwargs)
        self.clients = {
            "approach": self.client,
            "grasp": HttpActionClient(host, int(os.environ.get("CHAIN_PORT_GRASP", "5557"))),
            "carry": HttpActionClient(host, int(os.environ.get("CHAIN_PORT_CARRY", "5558"))),
        }
        self.mode = os.environ.get("CHAIN_MODE", "chain")
        self.x_min = float(os.environ.get("CHAIN_X_MIN", "0.0"))
        self.x_max = float(os.environ.get("CHAIN_X_MAX", "0.0"))
        self.y_max = float(os.environ.get("CHAIN_Y_MAX", "0.0"))
        self.k = int(os.environ.get("CHAIN_K", "10"))
        self.log_path = os.environ.get("CHAIN_LOG", "")
        self.trigger = os.environ.get("CHAIN_TRIGGER", "oracle")
        self.cal_a = float(os.environ.get("CHAIN_CAL_A", "0"))
        self.cal_b = float(os.environ.get("CHAIN_CAL_B", "0"))
        self.req_stop = os.environ.get("CHAIN_REQ_STOP", "0") == "1"
        self.stop_v = float(os.environ.get("CHAIN_STOP_V", "0.1"))
        self.smooth = int(os.environ.get("CHAIN_X_SMOOTH", "5"))
        self._xhist = []
        self._last_chunk = None
        self.head = None
        if self.trigger == "head":
            self.head = json.load(open(os.environ["CHAIN_HEAD_JSON"]))
            self.head_thr = float(os.environ.get("CHAIN_HEAD_THR", "0.7"))
            self.head_meanw = int(os.environ.get("CHAIN_HEAD_MEANW", "0"))     # >0: usa a média móvel (causal) de p nesta janela
            self.min_area = float(os.environ.get("CHAIN_MIN_AREA", "0"))
            self.cx_dev = float(os.environ.get("CHAIN_CX_DEV", "0"))                # >0: exige |cx-320| <= cx_dev       # área mínima da caixa azul (bloqueia congelamento inicial)
            self._phist = deque(maxlen=max(self.head_meanw, 1))
        self.head2 = None
        if os.environ.get("CHAIN_HEAD2_MODE", "head") == "head":
            self.head2 = json.load(open(os.environ["CHAIN_HEAD2_JSON"]))
            self.head2_thr = float(os.environ.get("CHAIN_HEAD2_THR", "0.5"))
            self.head2_meanw = int(os.environ.get("CHAIN_HEAD2_MEANW", "30"))
            self.head2_k = int(os.environ.get("CHAIN_HEAD2_K", "5"))
            self.min_grasp = int(os.environ.get("CHAIN_MIN_GRASP", "100"))
            self._p2hist = deque(maxlen=max(self.head2_meanw, 1)); self._streak2 = 0
        self._vhist = deque(maxlen=10)
        self.assist_ctl = None
        if os.environ.get("CHAIN_ASSIST", "0") == "1" or os.environ.get("CHAIN_SEARCH", "0") == "1":
            from simple.baselines.chain_assist import AssistController
            self.assist_ctl = AssistController(
                forward=os.environ.get("CHAIN_ASSIST", "0") == "1", search=os.environ.get("CHAIN_SEARCH", "0") == "1",
                ymax_far=float(os.environ.get("CHAIN_ASSIST_YMAX", "110")), vx=float(os.environ.get("CHAIN_ASSIST_VX", "0.2")),
                search_vy=float(os.environ.get("CHAIN_SEARCH_VY", "0.3")))
        self._assisting = 0
        _orig = self.clients["approach"].query_action

        def _rec(*a, **k):                       # guarda o último chunk do approach (comando de velocidade)
            out = _orig(*a, **k)
            self._last_chunk = np.asarray(out[0])
            return out

        self.clients["approach"].query_action = _rec
        self._phase = "approach"
        self._streak = 0
        self._step = 0
        self._target_key = None
        self._switch_step = None

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._phase, self._streak, self._step, self._switch_step = "approach", 0, 0, None
        self.client = self.clients["approach"]
        self._target_key = _target_key_for_episode()
        self._tote_z0 = None
        self._xhist, self._last_chunk, self._keys_printed = [], None, False
        self._vhist.clear(); self._assisting = 0
        if self.assist_ctl is not None: self.assist_ctl.reset()
        if self.head is not None: self._phist.clear()
        if self.head2 is not None: self._p2hist.clear(); self._streak2 = 0
        self._switch2_step = None
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            ep = os.environ.get("SIMPLE_EVAL_EPISODE_ID", "")
            self._log = open(self.log_path.replace(".csv", f"_{ep}.csv"), "w")
            self._log.write("step,phase,x,y,z_rel,tote_z,tote_z_gain,base_x,base_y,base_yaw,switch,areaL,cxL,cyL,yminL,ymaxL,areaR,cxR,disp,x_est,cmd_v,p_head,assist,p_head2,wx,wy,d_table\n")
        print(f"[chain] episode={os.environ.get('SIMPLE_EVAL_EPISODE_ID')} target_key={self._target_key} mode={self.mode}", flush=True)

    def _tote_in_base_frame(self, info):
        pose = np.asarray(info[self._target_key], dtype=np.float64)          # (7,) xyz + quat(w,x,y,z), mundo
        base = np.asarray(self.robot.prepare_obs()["floating_base_pose"], dtype=np.float64)
        q = base[3:7]
        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        d = pose[:3] - base[:3]
        c, s = np.cos(-yaw), np.sin(-yaw)
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]]), pose[2], base[0], base[1], yaw

    def _visual(self, observation):
        """Características só de imagem (deployáveis): mancha azul esquerda/direita e disparidade."""
        if not self._keys_printed:
            print(f"[chain] obs keys: {sorted(k for k in observation if isinstance(k, str))}", flush=True)
            self._keys_printed = True
        L = _blue(observation["head_stereo_left"])
        R = _blue(observation["head_stereo_right"]) if "head_stereo_right" in observation else None
        d = (L[1] - R[1]) if (L and R) else float("nan")
        return L, R, d

    def _head_p(self, observation, L, head=None):
        """Probabilidade de 'approach terminou' (logístico treinado nos cortes anotados). Só imagem + estado + comando."""
        proprio = np.asarray(observation["joint_qpos"])[None]
        st = np.concatenate([proprio[:, a:b] for _, a, b in STATE_SLICES] + [self._last_cmd_torso_rpyh[None]], axis=1)[0]
        nav = self._action_queue[0]["navigate_cmd"] if len(self._action_queue) else None
        pv = np.asarray(nav[:3], dtype=float) if nav is not None else (self._vhist[-1] if self._vhist else np.zeros(3))
        self._vhist.append(pv)
        mean_v = float(np.mean([np.abs(v[:2]).mean() for v in self._vhist]))
        blob = list(L[:1]) + list(L[1:3]) + list(L[3:7]) if L else [0] + [np.nan] * 6
        # ordem do treino: area, cx, cy, xmin, xmax, ymin, ymax
        feat = np.array([blob[0], blob[1], blob[2], blob[3], blob[4], blob[5], blob[6]] + list(st) + list(pv) + [mean_v], dtype=float)
        feat[np.isnan(feat)] = 0.0
        head = head or self.head
        feat = feat[head["cols_idx"]]                 # subconjunto de características do modelo (area é sempre a primeira)
        z = np.append(feat, 1.0 if feat[0] == 0 else 0.0)
        z = (z - np.array(head["mu"])) / np.array(head["sd"])
        return float(1.0 / (1.0 + np.exp(-(np.dot(z, head["w"]) + head["b"]))))

    def _x_est(self, L, d):
        if self.trigger == "stereo" and L and d == d and d > 1.0:
            return self.cal_a * (1.0 / d) + self.cal_b
        if self.trigger == "mono" and L:
            return self.cal_a * (L[0] ** -0.5) + self.cal_b
        return float("nan")

    def get_action(self, observation, instruction=None, info=None, conditions=None, **kwargs):
        sw = 0
        L, R, d = self._visual(observation)
        x_raw = self._x_est(L, d)
        if x_raw == x_raw:
            self._xhist.append(x_raw)
        x_est = float(np.median(self._xhist[-self.smooth:])) if (x_raw == x_raw and self._xhist) else float("nan")
        cmd_v = float(np.hypot(self._last_chunk[:20, 32], self._last_chunk[:20, 33]).mean()) if self._last_chunk is not None else float("nan")
        p_head = self._head_p(observation, L) if (self.head and self._phase == "approach") else float("nan")
        p2 = self._head_p(observation, L, self.head2) if (self.head2 is not None and self._phase == "grasp") else float("nan")
        rel = None
        if info is not None and self._target_key in info:
            rel, tz, bx, by, yaw = self._tote_in_base_frame(info)
            if self._tote_z0 is None:
                self._tote_z0 = tz
        if self._phase == "approach" and self.mode == "chain":
            if self.trigger == "oracle":
                ok = rel is not None and (self.x_min <= rel[0] <= self.x_max) and abs(rel[1]) <= self.y_max
            elif self.trigger == "head":
                self._phist.append(p_head)
                p_use = float(np.mean(self._phist)) if self.head_meanw > 0 else p_head
                ok = (p_use >= self.head_thr) and ((not L) or L[0] >= self.min_area) and (L is not None or self.min_area == 0)
                if ok and L is not None and self.cx_dev > 0 and abs(L[1] - 320) > self.cx_dev:
                    ok = False                              # caixa muito de lado na imagem
                if self.head_meanw > 0 and len(self._phist) < self.head_meanw:
                    ok = False
            else:
                ok = x_est == x_est and (self.x_min <= x_est <= self.x_max)
                if ok and self.req_stop:
                    ok = cmd_v == cmd_v and cmd_v < self.stop_v
            self._streak = self._streak + 1 if ok else 0
            if self._streak >= self.k:
                self._phase, self._switch_step, sw = "grasp", self._step, 1
                self.client = self.clients["grasp"]
                self._action_queue.clear()          # descarta ações do approach ainda na fila
                print(f"[chain] SWITCH approach->grasp at step {self._step} trigger={self.trigger} x_est={x_est:.3f} p_head={p_head:.2f} oracle_rel={None if rel is None else np.round(rel, 3)} cmd_v={cmd_v:.3f}", flush=True)
        if self._phase == "grasp" and self.mode == "chain" and self.head2 is not None:
            self._p2hist.append(p2)
            p2u = float(np.mean(self._p2hist)) if self.head2_meanw > 0 else p2
            ok2 = (p2u >= self.head2_thr) and len(self._p2hist) >= self.head2_meanw and (self._step - self._switch_step) >= self.min_grasp
            self._streak2 = self._streak2 + 1 if ok2 else 0
            if self._streak2 >= self.head2_k:
                self._phase, self._switch2_step, sw = "carry", self._step, 2
                self.client = self.clients["carry"]
                self._action_queue.clear()
                print(f"[chain] SWITCH grasp->carry at step {self._step} p2={p2u:.2f} oracle_rel={None if rel is None else np.round(rel, 3)}", flush=True)
        if self.log_path and rel is not None:
            wxy = np.asarray(info[self._target_key], dtype=float)[:2]; dtab = float(np.hypot(wxy[0] - TABLE_XY[0], wxy[1] - TABLE_XY[1]))
            f = lambda v: "nan" if v is None else f"{v:.2f}"
            Lz = L if L else (0, float("nan"), float("nan"), 0, 0, 0, 0)
            Rz = R if R else (0, float("nan"))
            self._log.write(f"{self._step},{self._phase},{rel[0]:.4f},{rel[1]:.4f},{rel[2]:.4f},{tz:.4f},{tz - self._tote_z0:.4f},{bx:.4f},{by:.4f},{yaw:.4f},{sw},"
                            f"{Lz[0]},{Lz[1]:.1f},{Lz[2]:.1f},{Lz[5]},{Lz[6]},{Rz[0]},{Rz[1]:.1f},{d:.2f},{x_est:.4f},{cmd_v:.3f},{p_head:.3f},{int(self._assisting)},{p2:.3f},{wxy[0]:.3f},{wxy[1]:.3f},{dtab:.3f}\n")
            if self._step % 50 == 0:
                self._log.flush()
        if self.assist_ctl is not None and self._phase == "approach":
            mode, dvx, vy = self.assist_ctl.update(L, cmd_v)
            if mode and mode != self._assisting:
                print(f"[chain] ASSIST mode {mode} at step {self._step}", flush=True)
            self._assisting = mode
            if mode and len(self._action_queue):
                nav = np.array(self._action_queue[0].parameters["navigate_cmd"], dtype=float, copy=True)   # o original é somente-leitura
                if dvx is not None:
                    nav[0] = max(float(nav[0]), dvx)
                if vy is not None:
                    nav[1] = vy
                self._action_queue[0].parameters["navigate_cmd"] = nav
        else:
            self._assisting = 0
        self._step += 1
        return super().get_action(observation, instruction=PROMPTS[self._phase], info=info, conditions=conditions, **kwargs)
