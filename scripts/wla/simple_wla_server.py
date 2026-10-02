#!/usr/bin/env python3
"""Servidor HTTP do UnifoLM-WLA para o cliente `HttpActionClient` do SIMPLE (stdlib, sem fastapi).

Contrato (SIMPLE: src/simple/baselines/client.py):
  GET  /health -> {"status": "ok"}
  POST /act    -> JSON com numpy como {"__numpy__": b64, "dtype", "shape"}
    request : image{"rgb_head_stereo_left": (H,W,3) RGB uint8}, instruction, history ({"reset": True} no 1º passo),
              state{"joint_qpos": (1,43) [perna esq 6, dir 6, cintura yaw/roll/pitch, braço esq 7, dir 7,
                                          mão esq 7, dir 7] (ordem SIMPLE G1Sonic),
                    "base_quat": (1,4) wxyz opc., "base_angvel": (1,3) opc., "yaw": (1,) opc.},
              gt_action (usado só com --oracle)
    response: {"action": (T,36) ação Ψ0 @30 FPS, "err": 0.0}  (T=30)
O servidor faz: estado SIMPLE -> estado WLA 60D (FK, fecho de mão) -> modelo -> desnormaliza ->
EE rel->abs -> IK -> ação Ψ0 36D (wla_adapter.to_simple). O cliente só executa via SONIC.

--oracle raw|adapter: sem modelo (CPU). `gt_action` = chunk (T,36) de ação Ψ0 @30 FPS vindo do cliente.
  raw     -> devolve gt_action intacto (controle do simulador, F4/S3).
  adapter -> gt_action -> chaves WLA (convert_action) -> EE abs -> IK (to_simple) -> ação Ψ0 (F4/S5).
Tratamento de SIGUSR1/SIGTERM: sai limpo (o job Slurm envia SIGUSR1@300).
"""

import argparse
import json
import logging
import signal
import sys
import threading
import time
from base64 import b64decode, b64encode
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from numpy.lib.format import descr_to_dtype, dtype_to_descr

PSI0 = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PSI0 / "src"), str(PSI0 / "third_party/unifolm-wla")]

from wla_adapter.convert_state import ee_pose_xyz_rpy  # noqa: E402
from wla_adapter.hands import joints_to_fig6d  # noqa: E402
from wla_adapter.pipeline import (  # noqa: E402
    PSI0_TOTE, S_BASE_ROT, S_EE, S_FIG, S_LEG, S_WAIST, WBT, NormStats,
    action_mask, invert_action, state_mask, state_unnorm,
)
from wla_adapter.to_simple import PSI0_ACTION_DIM, to_psi0_action  # noqa: E402

PROFILES = {"wbt": WBT, "psi0_tote": PSI0_TOTE}
IMG_KEY = "rgb_head_stereo_left"


def np_encode(o):
    if isinstance(o, (np.ndarray, np.generic)):
        o = np.asarray(o)
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {"__numpy__": b64encode(data).decode(), "dtype": dtype_to_descr(o.dtype), "shape": o.shape}
    raise TypeError(type(o))


def np_decode(d):
    if isinstance(d, dict):
        if "__numpy__" in d:
            a = np.frombuffer(b64decode(d["__numpy__"]), descr_to_dtype(d["dtype"]))
            return a.reshape(d["shape"]) if d["shape"] else a[0]
        return {k: np_decode(v) for k, v in d.items()}
    if isinstance(d, list):
        return [np_decode(v) for v in d]
    return d


def encode_tree(x):
    if isinstance(x, dict):
        return {k: encode_tree(v) for k, v in x.items()}
    if isinstance(x, (np.ndarray, np.generic)):
        return np_encode(x)
    return x


def gravity_from_quat_wxyz(q):
    """Gravidade projetada no frame do corpo: R^T @ [0,0,-1] (quaternion wxyz). Convenção de sinal do IMU do WBT
    NÃO verificada (só relevante ao perfil wbt; o padrão upright é [0,0,-1])."""
    w, x, y, z = q
    return np.array([-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))])


def simple_state_to_wla(state: dict, profile) -> tuple[np.ndarray, np.ndarray, float]:
    """state SIMPLE -> (state_unnorm (60,), q_arm (14,), yaw0)."""
    q = np.asarray(state["joint_qpos"], np.float64).reshape(-1)
    legs_l, legs_r, waist = q[0:6], q[6:12], q[12:15]
    arm = {"left": q[15:22], "right": q[22:29]}
    hand = {"left": q[29:36], "right": q[36:43]}
    row = {S_WAIST: waist}
    for side in ("left", "right"):
        row[S_EE[side]] = ee_pose_xyz_rpy(waist[None], arm[side][None], side)[0]
        row[S_FIG[side]] = joints_to_fig6d(hand[side], side)
    row[S_LEG["left"]], row[S_LEG["right"]] = legs_l, legs_r
    if profile.has_base_rot:
        g = np.array([0.0, 0.0, -1.0])
        if "base_quat" in state:
            g = gravity_from_quat_wxyz(np.asarray(state["base_quat"], np.float64).reshape(-1))
        om = np.asarray(state["base_angvel"], np.float64).reshape(-1) if "base_angvel" in state else np.zeros(3)
        row[S_BASE_ROT] = np.concatenate([g, om])
    yaw0 = float(np.asarray(state["yaw"]).reshape(-1)[0]) if "yaw" in state else 0.0
    return state_unnorm(row, profile), np.concatenate([arm["left"], arm["right"]]), yaw0


class Policy:
    def __init__(self, args):
        self.args = args
        self.profile = PROFILES[args.profile]
        self.lock = threading.Lock()
        self.steps = 0
        if args.oracle:
            return
        import torch
        from unifolm_wla.model.framework.base_framework import build_framework
        from unifolm_wla.model.framework.share_tools import dict_to_namespace, read_mode_config
        self.torch = torch
        self.device = torch.device(args.device if args.device != "cuda" else "cuda:0")
        # = eval_local_episode.load_model, com override opcional de atenção (CPU/dry-run: sdpa em vez de flash_attention_2)
        model_config, norm_stats = read_mode_config(Path(args.ckpt_path))
        if args.base_vlm:
            model_config["framework"]["qwenvl"]["base_vlm"] = args.base_vlm
        if args.device == "cpu":
            model_config["framework"]["qwenvl"]["attn_implementation"] = "sdpa"
        cfg = dict_to_namespace(model_config)
        cfg.trainer.pretrained_checkpoint = None
        self.model = build_framework(cfg=cfg)
        self.model.norm_stats = norm_stats
        if str(args.ckpt_path).endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(str(args.ckpt_path))
        else:
            sd = torch.load(args.ckpt_path, map_location="cpu")
        self.model.load_state_dict(sd, strict=True)
        if args.use_bf16 and self.device.type == "cuda":
            self.model = self.model.to(torch.bfloat16)
        elif self.device.type == "cpu":
            self.model = self.model.float()  # dry-run em CPU (o código do WLA usa autocast("cuda"): só valida o caminho, não a velocidade)
        self.model = self.model.to(self.device).eval()
        src = args.source_name if args.source_name in self.model.norm_stats else next(iter(self.model.norm_stats))
        logging.info("norm source: %s (available: %s)", src, list(self.model.norm_stats))
        st = self.model.norm_stats[src]
        f32 = lambda k1, k2: np.asarray(st[k1][k2], np.float32)  # noqa: E731
        self.stats = NormStats(f32("action", "offset"), f32("action", "scale"), f32("state", "offset"), f32("state", "scale"))
        self.amask, self.smask = action_mask(self.profile), state_mask(self.profile)
        self.role = "head_left"

    def _example(self, req, s_unnorm):
        import cv2
        from PIL import Image
        img = np.asarray(req["image"][IMG_KEY])
        if img.ndim == 4:
            img = img[-1]
        h, w = self.args.image_size
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        s_norm = (s_unnorm - self.stats.state_offset) / (self.stats.state_scale + 1e-8)
        return {
            "image": [Image.fromarray(img.astype(np.uint8))], "image_roles": [self.role],
            "lang": req.get("instruction") or self.args.instruction,
            "action_mask": self.amask, "state": s_norm.astype(np.float32), "state_mask": self.smask,
            "arm_type": "dual_with_legs", "robot_type": "unitree",
        }

    def act(self, req) -> np.ndarray:
        if self.args.oracle == "raw":
            return np.asarray(req["gt_action"], np.float32).reshape(-1, PSI0_ACTION_DIM)
        if self.args.oracle == "adapter":
            from wla_adapter.convert_action import action_keys
            from wla_adapter.geometry import xyz_rpy_to_se3
            gt = np.asarray(req["gt_action"], np.float64).reshape(-1, PSI0_ACTION_DIM)
            k = action_keys({"action": gt})
            inv = {"waist": k["action.waist_action_joint"].astype(np.float64),
                   "base_command": k["action.base_command"].astype(np.float64)}
            for side in ("left", "right"):
                inv[f"{side}_ee_T"] = xyz_rpy_to_se3(k[f"action.{side}_ee_pose_gripper_base"].astype(np.float64))
                inv[f"{side}_fig6d"] = k[f"action.{side}_fig6d"].astype(np.float64)
            _, q_arm, yaw0 = simple_state_to_wla(req["state"], PSI0_TOTE)
            A, info = to_psi0_action(inv, q_arm, yaw0=float(gt[0, 35]), dt=1 / 30)
            logging.info("oracle-adapter ik_pos_err_max=%.2e converged=%.2f", info["pos_err"].max(), info["converged"].mean())
            return A.astype(np.float32)
        t0 = time.perf_counter()
        s_un, q_arm, yaw0 = simple_state_to_wla(req["state"], self.profile)
        ex = self._example(req, s_un)
        with self.lock, self.torch.no_grad():
            out = self.model.predict_action([ex])
        pred = np.asarray(out["normalized_actions"][0], np.float64)  # (T,54) normalizado
        t1 = time.perf_counter()
        inv = invert_action(pred, s_un, self.stats, profile=self.profile)
        A, info = to_psi0_action(inv, q_arm, yaw0=yaw0, dt=1 / 30)
        t2 = time.perf_counter()
        self.steps += 1
        logging.info("step=%d model=%.2fs ik=%.2fs ik_pos_err_max=%.2e converged=%.2f", self.steps, t1 - t0, t2 - t1,
                     info["pos_err"].max(), info["converged"].mean())
        if self.args.debug_dir:
            np.savez(Path(self.args.debug_dir) / f"step_{self.steps:05d}.npz", state=s_un, pred_norm=pred, action=A,
                     pos_err=info["pos_err"], rot_err=info["rot_err"],
                     image=np.asarray(ex["image"][0], np.uint8), joint_qpos=np.asarray(req["state"]["joint_qpos"], np.float32).reshape(-1))
        return A.astype(np.float32)


def make_handler(policy: Policy):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, {"status": "ok"}) if self.path == "/health" else self._send(404, {"err": "not found"})

        def do_POST(self):
            if self.path != "/act":
                return self._send(404, {"err": "not found"})
            try:
                req = np_decode(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                action = policy.act(req)
                self._send(200, encode_tree({"action": action, "err": 0.0}))
            except Exception as e:  # o cliente imprime o texto do erro
                logging.exception("act failed")
                self._send(500, {"err": f"{type(e).__name__}: {e}"})

        def log_message(self, *a):
            pass

    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path")
    p.add_argument("--base_vlm")
    p.add_argument("--source_name", default="UnifoLM_WBT")
    p.add_argument("--profile", choices=list(PROFILES), default="wbt")
    p.add_argument("--image_size", type=int, nargs=2, default=[336, 448], metavar=("H", "W"))
    p.add_argument("--instruction", default="pick up the blue tote from the shelf and bring it to the table.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=22085)
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--oracle", choices=["raw", "adapter"])
    p.add_argument("--debug_dir")
    a = p.parse_args()
    if not a.oracle and not a.ckpt_path:
        p.error("--ckpt_path obrigatório sem --oracle")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.debug_dir:
        Path(a.debug_dir).mkdir(parents=True, exist_ok=True)
    policy = Policy(a)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(policy))
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, lambda *_: threading.Thread(target=srv.shutdown, daemon=True).start())
    logging.info("serving on %s:%d (profile=%s oracle=%s)", a.host, a.port, a.profile, a.oracle)
    srv.serve_forever()


if __name__ == "__main__":
    main()
