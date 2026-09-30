import math

import numpy as np
import pytest

from app.services.pose.calibration import CameraCalibration
from app.services.pose.transforms import (
    compose_transforms,
    cv_model_pose_to_threejs,
    cv_pose_to_threejs,
    euler_angles_deg,
    invert_transform,
    project_pose_axes,
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


def _quat_rotate(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=float)
    return 2 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2 * w * np.cross(u, v)


def test_cv_model_pose_identity_keeps_model_up_pointing_down_in_image():
    """OpenCV identity rotation means the model's +Y maps to camera +Y, which
    is DOWN in the image. In Three.js (Y up) the model's +Y must therefore map
    to world -Y, and +Z (toward the camera's viewing direction) to -Z."""
    t_co = np.eye(4)
    t_co[:3, 3] = [0.0, 0.0, 0.5]
    pose = cv_model_pose_to_threejs(t_co)
    assert pose.position == pytest.approx((0.0, 0.0, -0.5), abs=1e-9)
    assert _quat_rotate(pose.quaternion, [0, 1, 0]) == pytest.approx([0, -1, 0], abs=1e-9)
    assert _quat_rotate(pose.quaternion, [0, 0, 1]) == pytest.approx([0, 0, -1], abs=1e-9)
    assert _quat_rotate(pose.quaternion, [1, 0, 0]) == pytest.approx([1, 0, 0], abs=1e-9)


def test_cv_model_pose_upright_object_stays_upright():
    """A mesh standing upright in front of the camera (its +Y along camera
    -Y, i.e. up in the image) must come out with +Y pointing up in Three.js —
    the case that a C@R@C conversion would get upside down."""
    t_co = np.eye(4)
    t_co[:3, :3] = np.diag([1.0, -1.0, -1.0])  # model up = image up, model faces the camera
    t_co[:3, 3] = [0.0, 0.0, 0.5]
    pose = cv_model_pose_to_threejs(t_co)
    assert _quat_rotate(pose.quaternion, [0, 1, 0]) == pytest.approx([0, 1, 0], abs=1e-9)
    assert _quat_rotate(pose.quaternion, [0, 0, 1]) == pytest.approx([0, 0, 1], abs=1e-9)


def test_euler_angles_identity_is_zero():
    assert euler_angles_deg(np.zeros(3)) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)


def test_euler_angles_single_axis_rotations_are_unambiguous():
    """A rotation about exactly one axis must decompose back to that same
    single angle on that same axis, regardless of Euler-order convention —
    there's no ambiguity to worry about in this special case."""
    assert euler_angles_deg(np.array([math.pi / 2, 0.0, 0.0])) == pytest.approx((90.0, 0.0, 0.0), abs=1e-6)
    assert euler_angles_deg(np.array([0.0, math.pi / 4, 0.0])) == pytest.approx((0.0, 45.0, 0.0), abs=1e-6)
    assert euler_angles_deg(np.array([0.0, 0.0, math.pi / 6])) == pytest.approx((0.0, 0.0, 30.0), abs=1e-6)


def test_project_pose_axes_straight_ahead_no_rotation():
    """A pose with no rotation, 0.5m in front of a camera whose principal
    point is (320, 240): origin projects to the principal point, the X axis
    moves right by fx*axis_length/z, Y moves DOWN by the equivalent amount
    (OpenCV Y-down), and Z (pointing straight into the depth axis) doesn't
    move at all in image space — all confirmed against the exact pinhole
    formula, not just "some plausible-looking number"."""
    calibration = CameraCalibration.approximate(640, 480)
    fx = calibration.camera_matrix[0, 0]
    rvec = np.zeros(3)
    tvec = np.array([0.0, 0.0, 0.5])

    axes = project_pose_axes(rvec, tvec, calibration.camera_matrix, calibration.dist_coeffs, axis_length_m=0.1)

    expected_offset = fx * 0.1 / 0.5
    assert axes["origin"] == pytest.approx((320.0, 240.0), abs=1e-6)
    assert axes["x_axis"] == pytest.approx((320.0 + expected_offset, 240.0), abs=1e-6)
    assert axes["y_axis"] == pytest.approx((320.0, 240.0 + expected_offset), abs=1e-6)
    assert axes["z_axis"] == pytest.approx((320.0, 240.0), abs=1e-6)
