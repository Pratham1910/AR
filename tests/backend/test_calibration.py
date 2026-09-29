import math

import pytest

from app.services.pose.calibration import CameraCalibration


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
