"""Convenções SE(3) do WLA: rot6d, xyz_rpy, relativo -> absoluto."""

import numpy as np
from scipy.spatial.transform import Rotation
from unifolm_wla.dataloader.multi_source_dataset.se3_utils import (
    compute_relative_actions,
    matrix_to_rot6d,
    se3_inverse,
)

from wla_adapter.geometry import (
    rel_to_abs,
    rot6d_to_matrix,
    rotation_angle,
    se3_to_xyz_rot6d,
    se3_to_xyz_rpy,
    xyz_rot6d_to_se3,
    xyz_rpy_to_se3,
)

rng = np.random.default_rng(0)


def rand_R(n):
    return Rotation.random(n, random_state=0).as_matrix()


def rand_T(n):
    T = np.zeros((n, 4, 4))
    T[:, :3, :3], T[:, :3, 3], T[:, 3, 3] = rand_R(n), rng.normal(size=(n, 3)), 1.0
    return T


def test_rot6d_order_and_roundtrip():
    R = rand_R(500)
    r6 = matrix_to_rot6d(R)
    assert np.array_equal(r6[:, :3], R[:, :, 0]) and np.array_equal(r6[:, 3:], R[:, :, 1])  # [R00,R10,R20,R01,R11,R21]
    assert np.abs(rot6d_to_matrix(r6) - R).max() < 1e-12
    # Gram-Schmidt tolera ruído float32 e devolve rotação própria
    R32 = rot6d_to_matrix(r6.astype(np.float32))
    assert np.abs(R32 @ np.swapaxes(R32, -1, -2) - np.eye(3)).max() < 1e-12
    assert np.allclose(np.linalg.det(R32), 1.0)
    assert rotation_angle(R32, R).max() < 1e-6


def test_xyz_rot6d_se3_roundtrip():
    T = rand_T(100)
    assert np.abs(xyz_rot6d_to_se3(se3_to_xyz_rot6d(T)) - T).max() < 1e-12


def test_xyz_rpy_convention_and_roundtrip():
    n = 1000
    pose = np.concatenate([
        rng.normal(size=(n, 3)),
        rng.uniform(-np.pi, np.pi, (n, 1)),
        rng.uniform(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, (n, 1)),
        rng.uniform(-np.pi, np.pi, (n, 1)),
    ], axis=1)
    T = xyz_rpy_to_se3(pose)
    r, p, y = pose[:, 3], pose[:, 4], pose[:, 5]
    Rx = Rotation.from_rotvec(np.stack([r, 0 * r, 0 * r], 1)).as_matrix()
    Ry = Rotation.from_rotvec(np.stack([0 * p, p, 0 * p], 1)).as_matrix()
    Rz = Rotation.from_rotvec(np.stack([0 * y, 0 * y, y], 1)).as_matrix()
    assert np.abs(T[:, :3, :3] - Rz @ Ry @ Rx).max() < 1e-12  # extrínseco xyz: R = Rz(yaw) Ry(pitch) Rx(roll)
    back = se3_to_xyz_rpy(T)
    assert np.abs(back - pose).max() < 1e-9


def test_se3_inverse():
    T = rand_T(50)
    assert np.abs(se3_inverse(T) @ T - np.eye(4)).max() < 1e-12


def test_relative_to_absolute_roundtrip():
    n = 200
    T_curr, T_fut = rand_T(n), rand_T(n)
    poses = lambda T: np.concatenate([T[:, :3, 3], Rotation.from_matrix(T[:, :3, :3]).as_euler("xyz")], axis=1)
    for i in range(n):
        rel = compute_relative_actions(poses(T_curr)[i], poses(T_fut)[i][None], "xyz_rpy")[0]
        T_abs = rel_to_abs(T_curr[i], rel)
        assert np.abs(T_abs[:3, 3] - T_fut[i, :3, 3]).max() < 1e-9
        assert rotation_angle(T_abs[:3, :3], T_fut[i, :3, :3]) < 1e-9


def test_relative_translation_is_in_current_ee_frame():
    T_curr = rand_T(1)[0]
    T_fut = T_curr.copy()
    T_fut[:3, 3] += np.array([0.1, -0.2, 0.3])  # deslocamento no mundo
    pose = lambda T: np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_euler("xyz")])
    rel = compute_relative_actions(pose(T_curr), pose(T_fut)[None], "xyz_rpy")[0]
    assert np.abs(rel[:3] - T_curr[:3, :3].T @ np.array([0.1, -0.2, 0.3])).max() < 1e-9
    assert np.abs(rel[3:]).max() < 1e-9
