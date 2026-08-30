"""Camera ray / frustum projection on 2.5D points."""

import numpy as np

from monarc.common.frustum import Camera, camera_matrix, look_at_cw, points_in_frustum


def test_known_point_projects_to_principal_point():
    K = camera_matrix(100.0, 100.0, 50.0, 40.0)
    T_cw = np.eye(4)
    T_cw[2, 3] = 0.0
    cam = Camera(K=K, T_cw=T_cw, width=100, height=80)
    xyz = np.array([[0.0, 0.0, 10.0]])
    uv, z, valid = cam.project(xyz)
    assert valid[0]
    assert abs(uv[0, 0] - 50.0) < 1e-8
    assert abs(uv[0, 1] - 40.0) < 1e-8
    assert abs(z[0] - 10.0) < 1e-8


def test_look_at_places_target_in_frustum():
    K = camera_matrix(200.0, 200.0, 80.0, 60.0)
    eye = np.array([0.0, -40.0, 30.0])
    target = np.array([2.0, 0.0, 5.0])
    T_cw = look_at_cw(eye, target)
    assert abs(np.linalg.det(T_cw[:3, :3]) - 1.0) < 1e-8
    cam = Camera(K=K, T_cw=T_cw, width=160, height=120)
    xyz = np.vstack([target, np.array([200.0, 200.0, 0.0])])
    mask = points_in_frustum(cam, xyz)
    assert mask[0]
    assert not mask[1]
