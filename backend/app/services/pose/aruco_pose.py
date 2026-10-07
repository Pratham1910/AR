"""
Marker-based pose estimation (Project.md #24, #25): the initial registration
method, explicitly *not* the final product requirement — markerless/CAD-based
registration is a later swap-in behind the same PoseEstimator interface.

Uses solvePnP directly (rather than the deprecated
cv2.aruco.estimatePoseSingleMarkers) against the marker's known 3D corner
geometry, so it works across the opencv-contrib versions that still ship
cv2.aruco.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from app.services.markers.aruco_detector import ArucoDetector
from app.services.pose.calibration import CameraCalibration


@dataclass
class PoseEstimate:
    found: bool
    marker_id: int | None = None
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    reprojection_error_px: float | None = None
    # The marker's 4 detected image-space corners (pixel coords, in the
    # captured frame's own resolution), top-left/top-right/bottom-right/
    # bottom-left. Purely for drawing a visible "this is what was detected"
    # outline client-side (Project.md #57's debug-mode principle) — the QA/
    # pose math never uses this, only rvec/tvec.
    corners_px: list[tuple[float, float]] | None = None


class ArucoPoseEstimator:
    """Detects one ArUco marker per frame and solves its 6DoF pose."""

    def __init__(self, dictionary_name: str, marker_length_m: float):
        # Detection is shared with the marker -> product scanner; only the
        # pose solve below is specific to this class.
        self._detector = ArucoDetector(dictionary_name)
        self._marker_length_m = marker_length_m

        half = marker_length_m / 2.0
        # Marker corners in its own object space, matching the corner order
        # cv2.aruco.detectMarkers returns (top-left, top-right, bottom-right, bottom-left).
        self._object_points = np.array(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float64,
        )

    def estimate(
        self, frame: np.ndarray, calibration: CameraCalibration, target_marker_id: int | None = None
    ) -> PoseEstimate:
        detections = self._detector.detect(frame)
        if target_marker_id is not None:
            detections = [d for d in detections if d.marker_id == target_marker_id]
        if not detections:
            return PoseEstimate(found=False)
        detection = detections[0]  # first detected marker

        marker_corners = np.array(detection.corners_px, dtype=np.float32)
        found, rvec, tvec = cv2.solvePnP(
            self._object_points, marker_corners, calibration.camera_matrix, calibration.dist_coeffs
        )
        if not found:
            return PoseEstimate(found=False)

        reprojected, _ = cv2.projectPoints(
            self._object_points, rvec, tvec, calibration.camera_matrix, calibration.dist_coeffs
        )
        reprojection_error = float(np.mean(np.linalg.norm(reprojected.reshape(4, 2) - marker_corners, axis=1)))

        return PoseEstimate(
            found=True,
            marker_id=detection.marker_id,
            rvec=rvec,
            tvec=tvec,
            reprojection_error_px=reprojection_error,
            corners_px=[(float(x), float(y)) for x, y in marker_corners],
        )
