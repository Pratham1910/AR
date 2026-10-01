"""Pose estimation API schemas (Project.md #24-#26, #63)."""

from typing import Literal

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


class ARFrameRequest(BaseModel):
    """One camera frame for a live AR session (app/services/tracking/ar_session.py).
    The detector runs only while SEARCHING/LOST; while TRACKING only the tracker does."""

    model_config = {"protected_namespaces": ()}

    session_id: str  # one state machine per camera/session
    mode: Literal["markerless", "model"]  # optical-flow tracking vs MegaPose model-based tracking
    image_base64: str
    class_label: str  # what the detector looks for (must be a detectable class)
    # Required for "model"; optional for "markerless", where it only enables
    # the model's calibrated part checks (e.g. "is the cap still on?").
    model_id: str | None = None
    real_world_height_m: float | None = None  # required for "markerless" (depth from apparent size)
    # "model" only: match just this assembly part (glTF node index, e.g. a
    # bottle's body, which looks the same with or without its cap); the whole
    # assembly is still posed and rendered. None = match the whole model.
    track_part: int | None = None


class TrackedObjectOut(BaseModel):
    object_id: int
    class_label: str
    confidence: float
    first_seen_frame: int
    last_seen_frame: int
    bbox: list[float] | None = None  # [x1, y1, x2, y2] in the frame's own pixel space
    polygon: list[Vector2] | None = None
    velocity_px_s: Vector2 | None = None


class ARCounters(BaseModel):
    frame_index: int
    detection_runs: int  # detector calls, found or not (SEARCHING/RECOVERING only)
    detection_count: int  # successful detections: 1 at first lock, +1 per re-acquisition
    tracking_frames: int  # tracker calls
    frames_since_detection: int | None = None


class ARTimings(BaseModel):
    """Server-side milliseconds for this frame, by stage (0 when the stage didn't run)."""

    detection: float  # YOLO
    initialization: float  # initial 6DoF pose (+ tracker lock)
    tracking: float  # 2D tracker (optical flow) and overhead
    refinement: float  # MegaPose pose refinement (model-based)
    total: float  # whole request on the server


class DetectionOut(BaseModel):
    class_label: str
    confidence: float
    bbox: list[float]
    polygon: list[Vector2] | None = None


class CameraIntrinsics(BaseModel):
    """The exact camera model the pose was computed with — the renderer
    builds its projection from these instead of a field-of-view guess."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class PartCheckOut(BaseModel):
    """Is a calibrated part (e.g. the cap) still on the tracked object?"""

    node_index: int
    part_name: str
    state: Literal["present", "absent", "uncertain"]
    confidence: float  # 0 at the present/absent midpoint, 1 at a calibrated mean
    brightness: float  # measured mean HSV value of the part's region
    region_px: list[int]  # [x1, y1, x2, y2] that was measured, for drawing


class ARFrameResponse(BaseModel):
    state: Literal["SEARCHING", "INITIALIZING", "TRACKING", "LOST", "RECOVERING"]
    visible: bool  # draw the model: tracking, or holding the last valid pose while lost/re-acquiring
    monitoring: bool  # TRACKING but confidence below the "good" threshold (tracking with warning)
    object: TrackedObjectOut | None = None
    detection: DetectionOut | None = None  # what the detector found on this frame, if it ran
    # Filtered pose in renderer (Three.js) space — see docs/coordinates.md.
    # "model": the model's own pose; "markerless": approximate, no orientation.
    position: Vector3 | None = None
    quaternion: Quaternion | None = None
    rotation_deg: RotationDeg | None = None  # XYZ Euler of `quaternion`, for display
    axes: PoseAxes | None = None  # raw (unfiltered) measurement gizmo, model-based only
    approximate: bool  # True for markerless (no orientation)
    detector_ran: bool
    tracker_ran: bool
    timings_ms: ARTimings
    counters: ARCounters
    events: list[str]  # [SEARCHING]/[DETECTION]/[POSE]/[TRACKER]/[RECOVERY] lines from this frame
    # Calibrated parts' presence on the tracked object (only while TRACKING).
    part_checks: list[PartCheckOut] = []
    good_confidence: float
    lost_confidence: float
    intrinsics: CameraIntrinsics
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
