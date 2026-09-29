"""Pose estimation API schemas (Project.md #24-#26, #63)."""

from pydantic import BaseModel


class PoseRequest(BaseModel):
    image_base64: str
    target_marker_id: int | None = None  # None = accept the first marker seen


class Vector3(BaseModel):
    x: float
    y: float
    z: float


class Quaternion(BaseModel):
    x: float
    y: float
    z: float
    w: float


class Vector2(BaseModel):
    x: float
    y: float


class PoseResponse(BaseModel):
    found: bool
    marker_id: int | None = None
    position: Vector3 | None = None  # Three.js-space (app/services/pose/transforms.py)
    quaternion: Quaternion | None = None
    reprojection_error_px: float | None = None
    # The marker's 4 detected corners in the captured frame's own pixel
    # space (top-left/top-right/bottom-right/bottom-left) — draw these as a
    # quadrilateral on the video to see exactly what was detected.
    corners_px: list[Vector2] | None = None
    calibration_is_approximate: bool
    calibration_source: str
