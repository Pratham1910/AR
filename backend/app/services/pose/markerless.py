"""
Markerless, approximate object placement (Project.md #24's stated future
upgrade over marker-based registration — built here as a first coarse
version, ahead of the original phase order, per explicit request).

IMPORTANT — this is NOT true 6DoF pose estimation:
  - Position (X, Y, Z) is estimated from the detected object's apparent pixel
    height vs. its assumed real-world height ("similar triangles" — the
    same principle a rangefinder camera uses), which requires the camera
    calibration to be reasonably accurate and the real-world height to be
    measured correctly. Get either wrong and the depth (Z) is wrong by the
    same ratio.
  - Orientation is NOT estimated at all — a single 2D bounding box carries no
    rotation information. The model is placed with an identity rotation
    (whatever orientation it was authored in).

This is deliberately labelled "approximate" everywhere it surfaces (Project.md
#25/#31/#53's "don't overclaim accuracy" principle) — it answers "roughly
where is the object", not "what is its precise 6DoF pose". True markerless
pose (feature matching against the CAD model, or a depth camera) is future
work; ArUco-marker-based pose (app/services/pose/aruco_pose.py) remains the
accurate option when a marker is available.
"""

import numpy as np

from app.schemas.vision import BoundingBox
from app.services.pose.calibration import CameraCalibration
from app.services.pose.transforms import Pose6DoF, cv_pose_to_threejs


def estimate_object_placement(
    bbox: BoundingBox, calibration: CameraCalibration, real_world_height_m: float
) -> Pose6DoF:
    """
    Depth from apparent size: Z = f_y * real_height / pixel_height. Then back
    -projects the box center to X/Y at that depth using the pinhole model.
    Reuses cv_pose_to_threejs for the OpenCV->Three.js conversion so there is
    still only one place that coordinate-system math happens (Project.md #26).
    """
    pixel_height = bbox.y2 - bbox.y1
    if pixel_height <= 0:
        raise ValueError("Bounding box has non-positive height; cannot estimate depth.")

    fx = calibration.camera_matrix[0, 0]
    fy = calibration.camera_matrix[1, 1]
    cx = calibration.camera_matrix[0, 2]
    cy = calibration.camera_matrix[1, 2]

    z = float(fy * real_world_height_m / pixel_height)
    x_center_px = (bbox.x1 + bbox.x2) / 2.0
    y_center_px = (bbox.y1 + bbox.y2) / 2.0
    x = float((x_center_px - cx) * z / fx)
    y = float((y_center_px - cy) * z / fy)

    tvec = np.array([x, y, z], dtype=np.float64)
    rvec = np.zeros(3, dtype=np.float64)  # no orientation signal available — identity rotation
    return cv_pose_to_threejs(rvec, tvec)
