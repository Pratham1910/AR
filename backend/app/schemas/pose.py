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


class RotationDeg(BaseModel):
    """Tait-Bryan Euler angles in degrees — debug display only (Project.md's
    debug-mode requirement), never used for the actual placement math, which
    stays in matrix/quaternion form to avoid gimbal lock/order ambiguity."""

    rx: float
    ry: float
    rz: float


class PoseAxes(BaseModel):
    """2D-projected XYZ pose gizmo, in the frame's own pixel space, for drawing
    a debug "this is the pose I estimated" overlay (X=red, Y=green, Z=blue)."""

    origin: Vector2
    x_axis: Vector2
    y_axis: Vector2
    z_axis: Vector2


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
    rotation_deg: RotationDeg | None = None  # debug display — see RotationDeg
    axes: PoseAxes | None = None  # debug gizmo — see PoseAxes


class ModelPoseRequest(BaseModel):
    """Model-based (CAD) 6DoF pose: MegaPose matches the Model3D's own mesh."""

    model_config = {"protected_namespaces": ()}

    model_id: str
    image_base64: str
    session_id: str = "default"  # one tracking state per camera/session
    reset: bool = False  # drop the current track and do a full search this frame
    # YOLO class used to find the object for a full search. Defaults to the
    # model's linked Component.class_label; required if the model has none.
    class_label: str | None = None


class ModelPoseResponse(BaseModel):
    found: bool
    # "coarse+refine" = full search from a fresh YOLO box (slow, ~1s);
    # "refine" = tracking from the previous frame's pose (fast);
    # "no_detection" = no track and YOLO didn't find the object's class.
    mode: str
    score: float | None = None  # MegaPose pose score; low = poor match, track is dropped
    class_label: str | None = None
    bbox: list[float] | None = None  # YOLO box used for a full search, if one ran this frame
    # The pose of the MODEL ITSELF (its recentered mesh frame) — apply
    # directly, no anchor offset needed, unlike marker/feature modes.
    position: Vector3 | None = None
    quaternion: Quaternion | None = None
    rotation_deg: RotationDeg | None = None
    axes: PoseAxes | None = None
    elapsed_ms: float = 0.0
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
    rotation_deg: RotationDeg | None = None  # debug display — see RotationDeg
    axes: PoseAxes | None = None  # debug gizmo — see PoseAxes
