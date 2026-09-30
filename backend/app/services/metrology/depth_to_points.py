"""
Depth image -> 3D point cloud back-projection (Project.md #26's dedicated
coordinate-transform discipline, extended to depth data): the upstream step
that turns a depth sensor's raw per-pixel depth into the point clouds
gap_measurement.py operates on.

Reuses the same pinhole camera model (OpenCV convention: X right, Y down,
Z forward) and CameraCalibration as the rest of the pose pipeline
(app/services/pose/calibration.py) — one camera-model implementation, not a
second one duplicated for depth.
"""

import numpy as np

from app.services.pose.calibration import CameraCalibration


def depth_to_points(
    depth_m: np.ndarray, calibration: CameraCalibration, mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Back-projects a per-pixel depth map (meters) into an (N, 3) array of 3D
    points in camera space. Pixels with depth <= 0 are treated as invalid
    (no return) and excluded, matching how real depth sensors report "no
    reading". `mask` (boolean, same shape as `depth_m`) further restricts
    the result to one object/region — e.g. a segmentation mask for "Part A".
    """
    height, width = depth_m.shape
    fx, fy = calibration.camera_matrix[0, 0], calibration.camera_matrix[1, 1]
    cx, cy = calibration.camera_matrix[0, 2], calibration.camera_matrix[1, 2]

    us, vs = np.meshgrid(np.arange(width), np.arange(height))

    valid = depth_m > 0
    if mask is not None:
        valid &= mask.astype(bool)

    z = depth_m[valid]
    x = (us[valid] - cx) * z / fx
    y = (vs[valid] - cy) * z / fy
    return np.stack([x, y, z], axis=-1)
