"""Normalização (minmax_q/zscore) e ida-volta do pipeline com os stats reais do WLA."""

import numpy as np
from scipy.spatial.transform import Rotation
from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES, STATE_SLICES
from unifolm_wla.dataloader.multi_source_dataset.stats_utils import (
    get_normalizer,
    load_relative_stats,
    load_stats,
    normalize,
)

from wla_adapter import pipeline as P
from wla_adapter.geometry import rotation_angle, xyz_rpy_to_se3

rng = np.random.default_rng(0)
ns = P.load_norm_stats()
DATA = load_stats(P.WLA_STATS_DIR / "stats.json")
REL = load_relative_stats(P.WLA_STATS_DIR / "relative_stats.json")


def test_get_normalizer_roundtrip_random_and_real_stats():
    cases = [(DATA, k, "minmax_q") for k in P.ACTION_KEYS + P.STATE_KEYS if k in DATA]
    cases += [(REL, k, "zscore") for k in REL]
    assert len(cases) >= 20
    for stats, key, kind in cases:
        o, s = get_normalizer(stats, key, kind)
        x = rng.normal(size=(100, len(o))).astype(np.float64) * 5
        assert np.abs(normalize(x, o, s) * s + o - x).max() < 1e-6, key
        # nos próprios stats reais
        if kind == "minmax_q":
            lo, hi = (stats[key]["global_q01"], stats[key]["global_q99"]) if "global_q01" in stats[key] else (
                stats[key]["q01"], stats[key]["q99"])
            ok = (hi - lo) / 2 >= 1e-6
            assert np.allclose(normalize(lo, o, s)[ok], -1, atol=1e-5) and np.allclose(normalize(hi, o, s)[ok], 1, atol=1e-5)
        else:
            m = stats[key]["global_mean"]
            assert np.abs(normalize(m, o, s)).max() < 1e-5


def test_norm_vectors_slots():
    ao, asc = ns.action_offset, ns.action_scale
    o, s = get_normalizer(REL, "left_ee_pose_gripper_base", "zscore")
    o_r, s_r = get_normalizer(REL, "right_ee_pose_gripper_base", "zscore")
    assert np.array_equal(ao[SLICES["left_xyz_rotvec"]], o) and np.array_equal(asc[SLICES["right_xyz_rotvec"]], s_r)
    bo, bs = get_normalizer(DATA, P.A_BASE_CMD, "minmax_q")
    assert np.array_equal(ao[[32, 33, 34, 41]], bo) and np.array_equal(asc[[32, 33, 34, 41]], bs)
    # slots sem normalizador: identidade
    for sl in ("left_gripper", "right_gripper", "torso_joint"):
        assert (ao[SLICES[sl]] == 0).all() and (asc[SLICES[sl]] == 1).all()
    # estado: EE só xyz; rot6d intacto; gravidade intacta
    for side in ("left", "right"):
        sl = STATE_SLICES[f"{side}_xyz_rot6d"]
        assert (ns.state_offset[sl.start + 3:sl.stop] == 0).all() and (ns.state_scale[sl.start + 3:sl.stop] == 1).all()
        assert (ns.state_scale[sl.start:sl.start + 3] != 1).any()
    g = STATE_SLICES["base_rotvec"]
    assert (ns.state_offset[g.start:g.start + 3] == 0).all() and (ns.state_scale[g.start:g.start + 3] == 1).all()


def test_masks_exact():
    am, sm = P.action_mask(), P.state_mask()
    assert am.sum() == 49 and sm.sum() == 51
    for sl in ("left_gripper", "right_gripper", "torso_joint"):
        assert not am[SLICES[sl]].any()
    for sl in ("torso_joint", "left_gripper", "right_gripper", "base_vx_vy", "base_vw", "height"):
        assert not sm[STATE_SLICES[sl]].any()
    assert am[SLICES["left_leg_joint"]].all() and sm[STATE_SLICES["right_leg_joint"]].all()


def _synthetic():
    H = P.H

    def poses(n):
        return np.concatenate([rng.uniform(-0.5, 0.5, (n, 3)), rng.uniform(-1, 1, (n, 3))], axis=1).astype(np.float32)

    def quats(n):
        q = Rotation.random(n, random_state=int(rng.integers(1 << 30))).as_quat()
        return np.concatenate([rng.uniform(-1, 1, (n, 3)), q], axis=1).astype(np.float32)

    row = {
        P.S_EE["left"]: poses(1)[0], P.S_EE["right"]: poses(1)[0],
        P.S_FIG["left"]: rng.random(6).astype(np.float32), P.S_FIG["right"]: rng.random(6).astype(np.float32),
        P.S_LEG["left"]: rng.normal(size=6).astype(np.float32), P.S_LEG["right"]: rng.normal(size=6).astype(np.float32),
        P.S_WAIST: rng.normal(size=3).astype(np.float32),
        P.S_BASE_ROT: rng.normal(size=6).astype(np.float32),
        P.S_BASE_POSE: quats(1)[0],
    }
    win = {
        P.A_EE["left"]: poses(H), P.A_EE["right"]: poses(H),
        P.A_FIG["left"]: rng.random((H, 6)).astype(np.float32), P.A_FIG["right"]: rng.random((H, 6)).astype(np.float32),
        P.A_LEG["left"]: rng.normal(size=(H, 6)).astype(np.float32), P.A_LEG["right"]: rng.normal(size=(H, 6)).astype(np.float32),
        P.A_WAIST: rng.normal(size=(H, 3)).astype(np.float32),
        P.A_BASE_CMD: rng.normal(size=(H, 4)).astype(np.float32),
        P.A_BASE_POSE: quats(H),
    }
    return row, win


def test_action_chunk_invert_roundtrip_synthetic():
    for _ in range(20):
        row, win = _synthetic()
        a = P.action_chunk(row, win, ns)
        inv = P.invert_action(a, P.state_unnorm(row), ns, row[P.S_BASE_POSE])
        for side in ("left", "right"):
            T_raw = xyz_rpy_to_se3(win[P.A_EE[side]])
            assert np.abs(inv[f"{side}_ee_T"][:, :3, 3] - T_raw[:, :3, 3]).max() < 1e-5
            assert rotation_angle(inv[f"{side}_ee_T"][:, :3, :3], T_raw[:, :3, :3]).max() < 1e-5
            assert np.abs(inv[f"{side}_fig6d"] - win[P.A_FIG[side]]).max() < 1e-5
            assert np.abs(inv[f"{side}_leg"] - win[P.A_LEG[side]]).max() < 1e-5
        assert np.abs(inv["waist"] - win[P.A_WAIST]).max() < 1e-5
        assert np.abs(inv["base_command"] - win[P.A_BASE_CMD]).max() < 1e-5
        Tb = np.zeros((P.H, 4, 4))
        Tb[:, :3, :3] = Rotation.from_quat(win[P.A_BASE_POSE][:, 3:]).as_matrix()
        Tb[:, :3, 3], Tb[:, 3, 3] = win[P.A_BASE_POSE][:, :3], 1.0
        assert np.abs(inv["base_pose_T"][:, :3, 3] - Tb[:, :3, 3]).max() < 1e-5
        assert rotation_angle(inv["base_pose_T"][:, :3, :3], Tb[:, :3, :3]).max() < 1e-5
