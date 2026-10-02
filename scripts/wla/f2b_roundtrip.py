"""Round-trip Ψ0→WLA→inversa em episódios crus. Não ajusta nada; só grava JSON.

TEST é recusado sem --allow-test (Fase 3). Não calcula estatística de normalização
(usa identidade) para não misturar splits. Compara pose do EE, não as juntas.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from wla_adapter.convert_action import action_keys  # noqa: E402
from wla_adapter.convert_state import state_keys  # noqa: E402
from wla_adapter.pipeline import H, PSI0_TOTE, NormStats, action_chunk, invert_action, state_unnorm  # noqa: E402
from wla_adapter.write_dataset import read_episode_raw, resample_episode  # noqa: E402

SPLIT = REPO / "docs/wla/split.json"


def episodes(split_name: str, limit: int | None, allow_test: bool) -> list[int]:
    spec = json.loads(SPLIT.read_text())
    if split_name == "test" and not allow_test:
        raise SystemExit("RECUSADO: split test exige --allow-test (Fase 3, só relatório)")
    ids = list(spec[split_name])
    return ids[:limit] if limit else ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=("train", "val", "test"))
    ap.add_argument("--limit", type=int, default=0, help="0 = todos os episódios do split")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--src", default=os.environ.get("PSI0_DATA", "/raid/user_marcospaulo/datasets/psi0") + "/G1ToteMix-psi0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ident = NormStats(np.zeros(54, np.float32), np.ones(54, np.float32), np.zeros(60, np.float32), np.ones(60, np.float32))
    pos_err, rot_err = [], []
    waist_err, base_err, leg_err, hand_err = [], [], [], []
    src = Path(args.src)
    for ep in episodes(args.split, args.limit, args.allow_test):
        raw = resample_episode(read_episode_raw(src, ep))
        sk, ak = state_keys(raw), action_keys(raw)
        waist_err.append(np.max(np.abs(ak["action.waist_action_joint"] - raw["action"][:, [30, 28, 29]])))
        base_err.append(np.max(np.abs(ak["action.base_command"] - raw["action"][:, [32, 33, 34, 31]])))
        leg_err.append(np.max(np.abs(sk["observation.state.left_leg"] - raw["leg_joints"][:, 0:6])))
        leg_err.append(np.max(np.abs(sk["observation.state.right_leg"] - raw["leg_joints"][:, 6:12])))
        m = len(raw["action"])
        for t in range(0, m, max(1, m // 4)):
            end = min(t + H, m)
            # janela com clamp no último frame, igual ao loader
            idx = np.clip(np.arange(t, t + H), 0, m - 1)
            row = {k: v[t] for k, v in sk.items()}
            win = {k: v[idx] for k, v in ak.items()}
            chunk = action_chunk(row, win, ident, PSI0_TOTE)
            inv = invert_action(chunk, state_unnorm(row, PSI0_TOTE), ident, profile=PSI0_TOTE)
            for side in ("left", "right"):
                got = inv[f"{side}_ee_xyz_rpy"][: end - t]
                exp = ak[f"action.{side}_ee_pose_gripper_base"][t:end]
                pos_err.append(np.max(np.abs(got[:, :3] - exp[:, :3])))
                rot_err.append(np.max(np.abs(got[:, 3:] - exp[:, 3:])))
                hand_err.append(np.max(np.abs(inv[f"{side}_fig6d"][: end - t] - ak[f"action.{side}_fig6d"][t:end])))
        print(f"ep {ep} frames={m}", flush=True)

    report = {
        "split": args.split,
        "allow_test": args.allow_test,
        "n_episodes": len(episodes(args.split, args.limit, args.allow_test)),
        "max_pos_m": float(np.max(pos_err)),
        "max_rot_rad": float(np.max(rot_err)),
        "max_waist": float(np.max(waist_err)),
        "max_base_cmd": float(np.max(base_err)),
        "max_leg": float(np.max(leg_err)),
        "max_fig6d": float(np.max(hand_err)),
        "pass": bool(
            np.max(pos_err) < 1e-5 and np.max(rot_err) < 1e-4 and np.max(waist_err) < 1e-6
            and np.max(base_err) < 1e-6 and np.max(leg_err) < 1e-6 and np.max(hand_err) < 1e-5
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
