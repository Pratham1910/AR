import numpy as np
import pytest

from app.services.pose.transforms import (
    compose_transforms,
    cv_pose_to_threejs,
    invert_transform,
    rvec_tvec_to_matrix,
)


def test_rvec_tvec_identity_rotation_matrix():
    rvec = np.zeros(3)
    tvec = np.array([1.0, 2.0, 3.0])
    matrix = rvec_tvec_to_matrix(rvec, tvec)
    assert np.allclose(matrix[:3, :3], np.eye(3))
    assert np.allclose(matrix[:3, 3], tvec)
    assert np.allclose(matrix[3], [0, 0, 0, 1])


def test_invert_transform_round_trip():
    rvec = np.array([0.1, 0.2, 0.3])
    tvec = np.array([0.5, -0.2, 1.0])
    matrix = rvec_tvec_to_matrix(rvec, tvec)
    inverted = invert_transform(matrix)
    identity = compose_transforms(matrix, inverted)
    assert np.allclose(identity, np.eye(4), atol=1e-9)


def test_compose_transforms_with_identity_is_noop():
    rvec = np.array([0.0, 0.4, 0.0])
    tvec = np.array([1.0, 0.0, 2.0])
    matrix = rvec_tvec_to_matrix(rvec, tvec)
    result = compose_transforms(np.eye(4), matrix)
    assert np.allclose(result, matrix)


def test_cv_pose_to_threejs_straight_ahead_no_rotation():
    """An object 2m directly in front of the camera, no rotation, should map
    to Three.js position (0, 0, -2) with an identity quaternion (Three.js
    cameras look down -Z, OpenCV cameras look down +Z)."""
    rvec = np.zeros(3)
    tvec = np.array([0.0, 0.0, 2.0])
    pose = cv_pose_to_threejs(rvec, tvec)
    assert pose.position == pytest.approx((0.0, 0.0, -2.0), abs=1e-9)
    assert pose.quaternion == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-9)


def test_cv_pose_to_threejs_flips_y():
    """OpenCV Y-down vs Three.js Y-up: an object above the camera axis in
    OpenCV space (negative Y) should be above it in Three.js space too
    (positive Y)."""
    rvec = np.zeros(3)
    tvec = np.array([0.0, -0.5, 1.0])
    pose = cv_pose_to_threejs(rvec, tvec)
    assert pose.position[1] > 0
