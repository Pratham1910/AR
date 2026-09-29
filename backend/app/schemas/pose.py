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


class ObjectRegistrationRequest(BaseModel):
    """Markerless registration (app/services/pose/markerless.py) — approximate, position-only."""

    image_base64: str
    target_class_label: str = "bottle"
    real_world_height_m: float
    confidence_threshold: float | None = None


class ObjectRegistrationResponse(BaseModel):
    found: bool
    class_label: str | None = None
    confidence: float | None = None
    bbox: list[float] | None = None  # [x1, y1, x2, y2] in the frame's own pixel space
    polygon: list[Vector2] | None = None  # outline for drawing, same pixel space
    position: Vector3 | None = None  # Three.js-space; None if not found
    quaternion: Quaternion | None = None  # always identity — see markerless.py docstring
    approximate: bool = True  # always true for this endpoint; distinguishes it from /pose's marker-based result
    calibration_is_approximate: bool
    calibration_source: str
    # The frontend overlay's Three.js camera MUST use this exact FOV/aspect
    # (not a guessed constant), or the 3D model visibly drifts off the real
    # object even when position/orientation are computed correctly — see
    # CameraCalibration.vertical_fov_deg()'s docstring.
    camera_vertical_fov_deg: float
    camera_aspect: float


class RegisterReferenceImageRequest(BaseModel):
    """Feature/keypoint tracking (app/services/pose/feature_tracker.py) — register a labeled surface."""

    asset_id: str
    image_base64: str
    label_width_m: float
    label_height_m: float


class RegisterReferenceImageResponse(BaseModel):
    feature_count: int
    quality: str  # "too_few" | "marginal" | "good" — see feature_tracker.RegistrationQuality


class FeaturePoseRequest(BaseModel):
    asset_id: str
    image_base64: str


class FeaturePoseResponse(BaseModel):
    found: bool
    position: Vector3 | None = None
    quaternion: Quaternion | None = None  # a REAL orientation estimate, unlike ObjectRegistrationResponse's
    num_matches: int = 0
    num_inliers: int = 0
    inlier_points_px: list[Vector2] | None = None  # for drawing what was actually matched
    calibration_is_approximate: bool
    calibration_source: str
    camera_vertical_fov_deg: float
    camera_aspect: float


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
    camera_vertical_fov_deg: float
    camera_aspect: float
