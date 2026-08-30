"""SE(3) exponential/logarithmic map consistency."""

import numpy as np
import pytest

from monarc.common.se3 import exp_se3, exp_so3, invert_se3, log_se3, log_so3, transform_points


def test_so3_exp_log_roundtrip():
    rng = np.random.default_rng(1)
    omega = rng.normal(size=3) * 0.4
    R = exp_so3(omega)
    recovered = log_so3(R)
    R2 = exp_so3(recovered)
    assert np.allclose(R, R2, atol=1e-8)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)
    assert abs(np.linalg.det(R) - 1.0) < 1e-8


def test_se3_exp_log_roundtrip():
    rng = np.random.default_rng(2)
    xi = rng.normal(size=6) * 0.3
    T = exp_se3(xi)
    xi2 = log_se3(T)
    T2 = exp_se3(xi2)
    assert np.allclose(T, T2, atol=1e-7)


def test_se3_inverse_and_points():
    xi = np.array([1.0, -2.0, 0.5, 0.1, -0.2, 0.05])
    T = exp_se3(xi)
    points = np.array([[0.0, 0.0, 0.0], [3.0, -1.0, 2.0]])
    mapped = transform_points(T, points)
    back = transform_points(invert_se3(T), mapped)
    assert np.allclose(back, points, atol=1e-8)
