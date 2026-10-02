"""Testes do mapeamento Dex3(7) <-> fig6d(6) (convenção WLA 1=aberto)."""

import numpy as np
import pytest

from wla_adapter.hands import (
    FIG6D_DIMS,
    JOINTS,
    action_hands_to_symmetric,
    closure,
    fig6d_to_joints,
    hand_limits,
    joints_to_fig6d,
    symmetric_to_action_hands,
)

SIDES = ("left", "right")


def test_limits_zero_is_open_and_close_matches_urdf():
    """q_open = 0; q_close = limite do URDF na direção de fechamento (sinais opostos L/R)."""
    ql, _ = hand_limits("left")
    qr, _ = hand_limits("right")
    _, qc_l = hand_limits("left")
    _, qc_r = hand_limits("right")
    assert np.all(ql == 0.0) and np.all(qr == 0.0)
    # index/middle fecham em sinais opostos entre as mãos (URDF)
    assert np.all(qc_l[3:7] < 0) and np.all(qc_r[3:7] > 0)
    # thumb_2: esquerda fecha para cima, direita para baixo
    assert qc_l[2] > 0 and qc_r[2] < 0
    # thumb_0 fecha positivo nos dois lados; thumb_1 negativo nos dois
    assert qc_l[0] > 0 and qc_r[0] > 0 and qc_l[1] < 0 and qc_r[1] < 0


@pytest.mark.parametrize("side", SIDES)
def test_open_is_one_closed_is_zero(side):
    q_open, q_close = hand_limits(side)
    f_open = joints_to_fig6d(q_open, side)
    f_closed = joints_to_fig6d(q_close, side)
    np.testing.assert_allclose(f_open, np.ones(6), atol=1e-12)
    np.testing.assert_allclose(f_closed, np.zeros(6), atol=1e-12)
    assert FIG6D_DIMS == ("thumb_oc", "thumb_lat", "index", "middle", "ring", "little")


@pytest.mark.parametrize("side", SIDES)
def test_roundtrip_on_manifold_exact(side):
    """fig6d -> joints -> fig6d é exato (<1e-6) no manifold de fecho (ring=little=middle)."""
    rng = np.random.default_rng(0)
    f = rng.random((500, 6))
    f[:, 4] = f[:, 5] = f[:, 3]  # manifold: ring/little duplicam middle (lossy documentado)
    q = fig6d_to_joints(f, side)
    f2 = joints_to_fig6d(q, side)
    assert np.abs(f2 - f).max() < 1e-6
    # e joints -> fig6d -> joints é exato quando as 2 juntas do dedo têm o mesmo fecho
    c = rng.random((500, 4))  # thumb_lat, thumb_oc, index, middle
    q_open, q_close = hand_limits(side)
    cj = np.stack([c[:, 0], c[:, 1], c[:, 1], c[:, 2], c[:, 2], c[:, 3], c[:, 3]], axis=1)
    q_on = q_open + cj * (q_close - q_open)
    q_rt = fig6d_to_joints(joints_to_fig6d(q_on, side), side)
    assert np.abs(q_rt - q_on).max() < 1e-6


@pytest.mark.parametrize("side", SIDES)
def test_offmanifold_inverse_stays_in_limits_and_monotonic(side):
    """Raw (fora do manifold) -> fig6d -> joints permanece nos limites; fig6d decresce ao fechar."""
    rng = np.random.default_rng(1)
    q_open, q_close = hand_limits(side)
    span = q_close - q_open
    q = q_open + rng.random((1000, 7)) * span
    q_rt = fig6d_to_joints(joints_to_fig6d(q, side), side)
    lo, hi = np.minimum(q_open, q_close), np.maximum(q_open, q_close)
    assert np.all(q_rt >= lo - 1e-12) and np.all(q_rt <= hi + 1e-12)
    # monotonicidade: fechar o index (juntas 3,4) reduz a dim index (2) do fig6d
    f0 = joints_to_fig6d(q_open, side)[2]
    f_half = joints_to_fig6d(0.5 * (q_open + q_close), side)[2]
    assert f_half < f0
    # ring/little espelham middle
    f = joints_to_fig6d(q, side)
    np.testing.assert_allclose(f[:, 3], f[:, 4])
    np.testing.assert_allclose(f[:, 3], f[:, 5])


def test_closure_clip():
    q_open, q_close = hand_limits("left")
    c = closure(q_close * 2.0, "left")  # além do limite -> clipa em 1
    np.testing.assert_allclose(c, np.ones(7))


def test_action_hand_reorder_involution_and_layout():
    rng = np.random.default_rng(2)
    a = rng.random((64, 14))
    sym = action_hands_to_symmetric(a)
    # esquerda: [t3, m2, i2] -> [t3, i2, m2]
    np.testing.assert_allclose(sym[:, 0:3], a[:, 0:3])   # thumb
    np.testing.assert_allclose(sym[:, 3:5], a[:, 5:7])   # index
    np.testing.assert_allclose(sym[:, 5:7], a[:, 3:5])   # middle
    # direita já é simétrica
    np.testing.assert_allclose(sym[:, 7:14], a[:, 7:14])
    # inversa exata
    np.testing.assert_allclose(symmetric_to_action_hands(sym), a, atol=0)
    assert len(JOINTS) == 7
