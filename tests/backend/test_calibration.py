import math

import pytest

import numpy as np

from app.services.pose.calibration import CameraCalibration, load_calibration


def test_approximate_calibration_aspect_ratio_matches_frame():
    cal = CameraCalibration.approximate(640, 480)
    assert cal.aspect_ratio() == pytest.approx(640 / 480)


def test_approximate_calibration_vertical_fov_is_consistent_with_horizontal():
    """
    approximate() fixes a 70deg *horizontal* FOV; for a non-square frame with
    fx == fy (square pixels), the vertical FOV must come out smaller than the
    horizontal one for a landscape (wider-than-tall) frame.
    """
    cal = CameraCalibration.approximate(640, 480, assumed_hfov_deg=70.0)
    vfov = cal.vertical_fov_deg()

    assert 0 < vfov < 70.0

    # Cross-check against the pinhole formula directly, independent of the
    # implementation: vfov = 2*atan(height / (2*fy)).
    fy = cal.camera_matrix[1, 1]
    expected = math.degrees(2 * math.atan(480 / (2 * fy)))
    assert vfov == pytest.approx(expected)


def test_square_frame_has_equal_horizontal_and_vertical_fov():
    cal = CameraCalibration.approximate(480, 480, assumed_hfov_deg=70.0)
    assert cal.vertical_fov_deg() == pytest.approx(70.0, abs=1e-6)


def _saved_calibration(tmp_path, width=1280, height=720):
    path = tmp_path / "camera.json"
    matrix = np.array([[1000.0, 0, 639.5], [0, 1000.0, 359.5], [0, 0, 1]])
    CameraCalibration(camera_matrix=matrix, dist_coeffs=np.zeros(5), image_width=width, image_height=height).save(path)
    return path


def test_calibration_is_used_as_is_at_its_own_resolution(tmp_path):
    cal = load_calibration(_saved_calibration(tmp_path), 1280, 720)
    assert not cal.is_approximate
    assert cal.camera_matrix[0, 0] == pytest.approx(1000.0)


def test_calibration_scales_to_another_resolution_with_the_same_aspect(tmp_path):
    """A 1280x720 calibration used on 1920x1080 frames: same lens, same field of view."""
    path = _saved_calibration(tmp_path)
    cal = load_calibration(path, 1920, 1080)
    assert not cal.is_approximate
    assert (cal.image_width, cal.image_height) == (1920, 1080)
    assert cal.camera_matrix[0, 0] == pytest.approx(1500.0)
    assert cal.camera_matrix[1, 1] == pytest.approx(1500.0)
    assert cal.camera_matrix[0, 2] == pytest.approx(959.5)  # still the image centre
    assert cal.camera_matrix[1, 2] == pytest.approx(539.5)
    assert cal.vertical_fov_deg() == pytest.approx(load_calibration(path, 1280, 720).vertical_fov_deg())


def test_calibration_at_another_aspect_falls_back_to_the_approximate_default(tmp_path):
    """640x480 is a different crop of the sensor than 1280x720; the file can't describe it."""
    cal = load_calibration(_saved_calibration(tmp_path), 640, 480)
    assert cal.is_approximate
