"""Round-trip ação Ψ0 -> chaves WLA (convert_action) -> adaptador (to_simple) -> ação Ψ0 (sintético, CPU)."""

import numpy as np
import pytest

from wla_adapter.convert_action import action_keys, A_EE, A_FIG, A_WAIST, A_BASE_CMD
from wla_adapter.geometry import fk_ee, joint_limits, rotation_angle, xyz_rpy_to_se3
from wla_adapter.hands import hand_limits, symmetric_to_action_hands
from wla_adapter.to_simple import integrate_target_yaw, to_psi0_action


def _smooth_action(H=12, seed=0):
    """Ação Ψ0 sintética: braços em trajetória suave dentro dos limites, mãos NO manifold de fecho."""
    rng = np.random.default_rng(seed)
    a = np.zeros((H, 36))
    sym = []
    for side in ("left", "right"):
        lo, hi = joint_limits(side)
        q0 = 0.3 * (lo[3:] + hi[3:]) / 2 + 0.2 * rng.standard_normal(7)
        q0 = np.clip(q0, lo[3:] + 0.2, hi[3:] - 0.2)
        c0 = 14 if side == "left" else 21
        a[:, c0:c0 + 7] = q0 + np.cumsum(0.01 * rng.standard_normal((H, 7)), axis=0)
        qo, qc = hand_limits(side)
        c = rng.random((H, 4))  # thumb_0, thumb_{1,2}, index_{0,1}, middle_{0,1}
        c = np.stack([c[:, 0], c[:, 1], c[:, 1], c[:, 2], c[:, 2], c[:, 3], c[:, 3]], axis=1)
        sym.append(qo + c * (qc - qo))
    a[:, 0:14] = symmetric_to_action_hands(np.concatenate(sym, axis=1))
    a[:, 28:31] = 0.05 * rng.standard_normal((H, 3))  # roll, pitch, yaw
    a[:, 31] = 0.74
    a[:, 32:35] = rng.uniform(-0.5, 0.5, (H, 3))
    a[:, 35] = 0.0
    return a


def _inv_from_keys(k):
    return {
        "left_ee_T": xyz_rpy_to_se3(k[A_EE.format(side="left")].astype(np.float64)),
        "right_ee_T": xyz_rpy_to_se3(k[A_EE.format(side="right")].astype(np.float64)),
        "left_fig6d": k[A_FIG.format(side="left")].astype(np.float64),
        "right_fig6d": k[A_FIG.format(side="right")].astype(np.float64),
        "waist": k[A_WAIST].astype(np.float64),
        "base_command": k[A_BASE_CMD].astype(np.float64),
    }


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_roundtrip_task_space_and_discretes(seed):
    a = _smooth_action(seed=seed)
    k = action_keys({"action": a})  # float32 (como no dataset convertido)
    inv = _inv_from_keys(k)
    seed_q = np.concatenate([a[0, 14:21], a[0, 21:28]])
    a2, info = to_psi0_action(inv, seed_q, yaw0=0.0, dt=1 / 30)

    # task-space: FK(IK) == EE alvo (alvo vindo de float32 => tolerância 1e-5)
    for i, side in enumerate(("left", "right")):
        sl = slice(14 + 7 * i, 21 + 7 * i)
        T = fk_ee(a2[:, [30, 28, 29]], a2[:, sl], side)
        assert np.abs(T[:, :3, 3] - inv[f"{side}_ee_T"][:, :3, 3]).max() < 1e-4
        assert rotation_angle(T[:, :3, :3], inv[f"{side}_ee_T"][:, :3, :3]).max() < 1e-4
    assert info["pos_err"].max() < 1e-4

    # discretos exatos (float32 do dataset): cintura, base, mãos no manifold
    np.testing.assert_allclose(a2[:, 28:31], a[:, 28:31], atol=1e-6)
    np.testing.assert_allclose(a2[:, 31:35], a[:, 31:35], atol=1e-6)
    np.testing.assert_allclose(a2[:, 0:14], a[:, 0:14], atol=1e-5)


def test_ik_close_to_dataset_joints_when_seeded():
    """Redundância 7-DoF: com seed no dataset a solução fica perto (não exata)."""
    a = _smooth_action(seed=3)
    inv = _inv_from_keys(action_keys({"action": a}))
    a2, _ = to_psi0_action(inv, np.concatenate([a[0, 14:21], a[0, 21:28]]))
    assert np.abs(a2[:, 14:28] - a[:, 14:28]).max() < 0.1  # rad (limite largo; medido de verdade no job de TRAIN)


def test_target_yaw_integration():
    vyaw = np.full(30, 0.5)
    y = integrate_target_yaw(vyaw, 0.0, 1 / 30)
    np.testing.assert_allclose(y[-1], 0.5, atol=1e-12)
    wrapped = integrate_target_yaw(np.full(30, 3.0), 3.0, 1 / 30)
    assert np.all(np.abs(wrapped) <= np.pi + 1e-12)


@pytest.mark.parametrize("side", ["left", "right"])
def test_ik_converges_from_perturbed_seed_and_respects_limits(side):
    """IK numérica: alvo = FK de q* (alcançável); seed = q* + 0.3 rad de ruído -> FK(IK)==alvo, q dentro dos limites."""
    from wla_adapter.geometry import ik
    rng = np.random.default_rng(7)
    lo, hi = joint_limits(side)
    n_ok = 0
    for _ in range(30):
        q_star = rng.uniform(lo[3:] * 0.5, hi[3:] * 0.5)
        qw = 0.05 * rng.standard_normal(3)
        T = fk_ee(qw, q_star, side)
        seed = np.clip(q_star + 0.3 * rng.standard_normal(7), lo[3:], hi[3:])
        _, q, info = ik(T, side, seed, q_waist0=qw, max_iter=200, tol_pos=1e-7, tol_rot=1e-7)
        assert np.all(q >= lo[3:] - 1e-12) and np.all(q <= hi[3:] + 1e-12)
        T2 = fk_ee(qw, q, side)
        n_ok += info["converged"] and np.abs(T2[:3, 3] - T[:3, 3]).max() < 1e-6 and rotation_angle(T2[:3, :3], T[:3, :3]) < 1e-6
    assert n_ok >= 27  # mínimos locais raros em seeds distantes; o deploy usa seed do passo anterior
