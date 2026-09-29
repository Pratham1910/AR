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

from app.services.pose.calibration import CameraCalibration

_DICTIONARY_NAMES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}


@dataclass
class PoseEstimate:
    found: bool
    marker_id: int | None = None
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    reprojection_error_px: float | None = None


class ArucoPoseEstimator:
    """Detects one ArUco marker per frame and solves its 6DoF pose."""

    def __init__(self, dictionary_name: str, marker_length_m: float):
        if dictionary_name not in _DICTIONARY_NAMES:
            raise ValueError(f"Unknown ArUco dictionary {dictionary_name!r}; supported: {list(_DICTIONARY_NAMES)}")
        self._dictionary = cv2.aruco.getPredefinedDictionary(_DICTIONARY_NAMES[dictionary_name])
        self._detector_params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._dictionary, self._detector_params)
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return PoseEstimate(found=False)

        ids_flat = ids.flatten()
        if target_marker_id is not None:
            if target_marker_id not in ids_flat:
                return PoseEstimate(found=False)
            index = int(np.where(ids_flat == target_marker_id)[0][0])
        else:
            index = 0  # first detected marker

        marker_corners = corners[index].reshape(4, 2)
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
            marker_id=int(ids_flat[index]),
            rvec=rvec,
            tvec=tvec,
            reprojection_error_px=reprojection_error,
        )
