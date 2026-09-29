"""
Renders a synthetic ArUco marker into a frame (rather than requiring a
physical printed marker + camera) so the detection -> solvePnP -> pose
pipeline can be verified in CI. See app/workers/generate_marker.py for the
printable version used in a real registration test.
"""

import cv2
import numpy as np

from app.services.pose.aruco_pose import ArucoPoseEstimator
from app.services.pose.calibration import CameraCalibration

WIDTH, HEIGHT = 640, 480
MARKER_LENGTH_M = 0.05


def _synthetic_frame_with_marker(marker_id: int = 3, marker_px: int = 200) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
    marker_bgr = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2BGR)

    frame = np.full((HEIGHT, WIDTH, 3), 200, dtype=np.uint8)  # light gray background
    x_off, y_off = (WIDTH - marker_px) // 2, (HEIGHT - marker_px) // 2
    frame[y_off : y_off + marker_px, x_off : x_off + marker_px] = marker_bgr
    return frame


def test_detects_and_estimates_pose_for_synthetic_marker():
    frame = _synthetic_frame_with_marker(marker_id=3)
    calibration = CameraCalibration.approximate(WIDTH, HEIGHT)
    estimator = ArucoPoseEstimator("DICT_4X4_50", MARKER_LENGTH_M)

    result = estimator.estimate(frame, calibration)

    assert result.found is True
    assert result.marker_id == 3
    assert result.rvec is not None
    assert result.tvec is not None
    # The marker is centered and facing the camera -> a plausible pose in front of it.
    assert result.tvec[2][0] > 0  # positive Z = in front of the camera
    assert result.reprojection_error_px is not None
    assert result.reprojection_error_px < 2.0  # should reproject very accurately for a synthetic, undistorted marker
    assert result.corners_px is not None
    assert len(result.corners_px) == 4


def test_target_marker_id_filters_out_non_matching_marker():
    frame = _synthetic_frame_with_marker(marker_id=3)
    calibration = CameraCalibration.approximate(WIDTH, HEIGHT)
    estimator = ArucoPoseEstimator("DICT_4X4_50", MARKER_LENGTH_M)

    result = estimator.estimate(frame, calibration, target_marker_id=7)

    assert result.found is False


def test_no_marker_in_frame_returns_not_found():
    blank_frame = np.full((HEIGHT, WIDTH, 3), 200, dtype=np.uint8)
    calibration = CameraCalibration.approximate(WIDTH, HEIGHT)
    estimator = ArucoPoseEstimator("DICT_4X4_50", MARKER_LENGTH_M)

    result = estimator.estimate(blank_frame, calibration)

    assert result.found is False
