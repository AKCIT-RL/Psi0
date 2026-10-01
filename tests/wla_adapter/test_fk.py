"""FK numpy (wla_adapter.geometry) vs pinocchio (fixtures gerados por scripts/wla/f2a_fk_fixtures.slurm)."""

from pathlib import Path

import numpy as np
import pytest

from wla_adapter.geometry import fk, fk_chain, joint_limits, rotation_angle

FIXTURE = Path(__file__).resolve().parents[2] / "_scratch/fixtures/fk_pinocchio.npz"
POS_TOL, ROT_TOL = 1e-6, 1e-6


@pytest.fixture(scope="module")
def fx():
    if not FIXTURE.exists():
        pytest.fail(f"fixtures ausentes: rode `sbatch scripts/wla/f2a_fk_fixtures.slurm` ({FIXTURE})")
    return np.load(FIXTURE)


def test_fixture_q_within_urdf_limits(fx):
    for side in ("left", "right"):
        lo, hi = joint_limits(side)
        q = np.concatenate([fx["q_waist_yrp"], fx[f"q_arm_{side}"]], axis=1)
        assert q.shape[0] == 1000
        assert (q >= lo).all() and (q <= hi).all()


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("tip", ["wrist_yaw", "hand_palm"])
@pytest.mark.parametrize("base", ["pelvis", "torso_link"])
def test_fk_matches_pinocchio(fx, side, tip, base):
    T = fk(fx["q_waist_yrp"], fx[f"q_arm_{side}"], side, base_link=base, tip_link=tip)
    ref = fx[f"T_{side}_{tip}_{base}"]
    assert np.abs(T[:, :3, 3] - ref[:, :3, 3]).max() < POS_TOL
    assert rotation_angle(T[:, :3, :3], ref[:, :3, :3]).max() < ROT_TOL


def test_torso_base_ignores_waist_and_chains(fx):
    side = "left"
    qw, qa = fx["q_waist_yrp"], fx[f"q_arm_{side}"]
    T_torso_tip = fk(qw, qa, side, base_link="torso_link")
    T_torso_tip2 = fk(np.zeros_like(qw), qa, side, base_link="torso_link")
    assert np.abs(T_torso_tip - T_torso_tip2).max() == 0
    names = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    T_pelvis_torso = fk_chain({n: qw[:, i] for i, n in enumerate(names)}, "pelvis", "torso_link")
    assert np.abs(T_pelvis_torso @ T_torso_tip - fk(qw, qa, side, base_link="pelvis")).max() < 1e-12
