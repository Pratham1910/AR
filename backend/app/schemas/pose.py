"""Pose estimation API schemas (Project.md #24-#26, #63)."""

from typing import Literal

from pydantic import BaseModel

from app.schemas.component_inspection import ComponentInspectionOut


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
    # How the object is found (first lock, and after tracking is lost):
    # "model" (model-based mode only) from its 3D model alone — no class needed;
    # "class" with the YOLO detector, by `class_label` (always for markerless).
    # How the object is found (first lock, and after tracking is lost), in both modes:
    # "model" by its 3D model (needs model_id); "class" by the YOLO detector's
    # `class_label`; "box" from a box the user drew around it (`init_box`) — any
    # object, no model or class needed; re-found by its 3D model when one is given.
    # "guide": nothing is detected — the operator lines the object up with the
    # model's outline shown at the guide pose (guide_view.py), model mode only.
    detect_by: Literal["model", "class", "box", "guide"] = "model"
    init_box: list[float] | None = None  # [x1, y1, x2, y2] in the frame's pixels, for detect_by="box"
    class_label: str | None = None  # a detectable class; required when detecting by class
    # Required for "model"; optional for "markerless", where it only enables
    # the model's calibrated part checks (e.g. "is the cap still on?").
    model_id: str | None = None
    real_world_height_m: float | None = None  # required for "markerless" (depth from apparent size)
    # "model" only: match just this assembly part (glTF node index, e.g. a
    # bottle's body, which looks the same with or without its cap); the whole
    # assembly is still posed and rendered. None = match the whole model.
    track_part: int | None = None
    # Which side of the model faces where, for objects whose top and bottom
    # look alike (a flat box): "camera" = camera above, top facing it;
    # "upright" = camera at the side, top up in the image; "any" = no constraint.
    resting: Literal["any", "camera", "upright", "flat"] = "any"
    # The operator clicked the object (frame pixels): lock onto the segment
    # there instead of searching (model mode). Sent with one frame only.
    select_point: list[float] | None = None
    # The camera the frames come from (browser device label): a calibration
    # recorded for another camera isn't used for it.
    camera_label: str | None = None
    # Component inspection on this session's frames (needs model_id and a
    # 6DoF pose, i.e. model-based mode): "inspect" judges each component,
    # "learn" records the good state's appearance, "paused" keeps results.
    # A new `inspection_run` id starts the inspection over.
    inspection: Literal["off", "inspect", "learn", "paused"] = "off"
    inspection_run: str | None = None
    # Only these component IDs are inspected (the current procedure step's
    # parts); None = every required component.
    inspection_focus: list[str] | None = None
    # Fixed-station mode: use the model's locked pose instead of detecting and
    # tracking (POST /api/vision/station-lock saves one).
    station_lock: bool = False


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
    brightness: float  # the measured value (HSV brightness statistic `measure`) of the part's pixels
    region_px: list[int]  # [x1, y1, x2, y2] around what was measured, for drawing
    measure: str = "mean"  # which brightness statistic (calibration picks the most telling one)
    # "pose" checks: the outline of the part as drawn at the tracked pose — the exact pixels measured.
    region_polygon: list[Vector2] | None = None


class HandsOut(BaseModel):
    """Hands seen in this frame (the mask the tracker excluded), for the debug view."""

    available: bool  # a hand detector is configured and loaded
    sources: list[str]  # "mediapipe" / "person"
    hands: int  # hands MediaPipe found
    persons: int  # person regions from the segmentation model
    age_ms: float | None  # how old the mask used for this frame is
    outlines: list[list[Vector2]]  # the hand mask's outlines, frame pixels
    error: str | None = None


class StationOut(BaseModel):
    """Fixed-station mode: the frozen pose in use, and whether the part has moved off it."""

    locked: bool
    shift_px: float | None = None  # object region's shift since locking (frame pixels)
    moved: bool = False
    created_at: str | None = None
    max_shift_px: float | None = None


class TrackingQualityOut(BaseModel):
    """What the tracking confidence is made of (model-based mode)."""

    overlap: float | None = None  # posed model vs the tracked object region
    flow_support: float | None = None  # share of visible optical-flow points agreeing on the motion
    occlusion: float | None = None  # share of the object's outline covered by a hand
    held_ms: float | None = None  # how long the pose has been held because of occlusion
    refine_skipped_for_hand: bool = False
    # 3D edge fit of the shown pose: share of the model's visible edges on
    # image edges and of the object's image edges explained by the model
    # (0..1, both averaged); None = no edge model / too few edges in view.
    edge_fit: float | None = None
    edge_adjusted: bool = False  # the last edge alignment moved the pose
    # Orientation of the tracked pose (Phase 8): "unique" (model has no
    # symmetric twin), "confirmed" (features / edges pick this twin),
    # "ambiguous" (they don't — inspection pauses when features are configured).
    orientation_status: str | None = None
    orientation_reason: str | None = None
    twin_flips_suppressed: int = 0  # background corrections turned back from a symmetric twin
    corrections_rejected: int = 0  # large corrections the 3D edges didn't support
    orientation_switches: int = 0  # verified orientation changes adopted outright


class PoseFeatureOut(BaseModel):
    """A pose feature where the tracked pose puts it, and what the image shows there (diagnostics)."""

    feature_id: str
    status: str  # detected / not_detected / occluded / hidden / facing_away / outside
    evidence: float
    predicted_center_px: Vector2 | None = None
    predicted_contour_px: list[Vector2] = []
    observed_center_px: Vector2 | None = None  # the circle actually found near it
    residual_px: float | None = None


class GuideOut(BaseModel):
    """Guide view: where to draw the model's outline for the operator to line the object up with."""

    position: Vector3  # Three.js convention, like ARFrameResponse.position
    quaternion: Quaternion
    saved: bool  # the guide was saved for this model (else the default view)
    score: float | None = None  # the last check's edge fit at the guide, 0..1
    min_score: float  # what a lock needs


class ARFrameResponse(BaseModel):
    state: Literal["SEARCHING", "INITIALIZING", "TRACKING", "TRACKING_OCCLUDED", "LOST", "RECOVERING"]
    # The same, in the operator's terms: TRACKING / TRACKING_OCCLUDED /
    # REACQUIRING (lost, looking for it again) / TRACKING_LOST / SEARCHING / INITIALIZING.
    tracking_status: Literal[
        "SEARCHING", "INITIALIZING", "TRACKING", "TRACKING_OCCLUDED", "REACQUIRING", "TRACKING_LOST"
    ] = "SEARCHING"
    hands: HandsOut | None = None
    quality: TrackingQualityOut | None = None
    station: StationOut | None = None
    guide: GuideOut | None = None  # while searching in guide-view mode
    pose_features: list[PoseFeatureOut] = []  # at the last orientation decision (diagnostics)
    # Streamed frames: when the camera captured the frame this answers (sender's clock, ms),
    # so the client can predict the pose for the moment it draws.
    capture_ms: float | None = None
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
    # Names of this model's parts that have a presence calibration, sent in
    # every state, so the page can show "Cap: waiting for tracking".
    calibrated_parts: list[str] = []
    good_confidence: float
    lost_confidence: float
    intrinsics: CameraIntrinsics
    calibration_is_approximate: bool
    calibration_source: str
    calibration_note: str | None = None  # why this calibration is (or isn't) in use
    calibration_rms_px: float | None = None  # the calibration's own reprojection error
    # Lens distortion removed from the frame before pose estimation (worst
    # case, px). The video on screen still has it, so the overlay can be off
    # by up to this much near the image corners.
    distortion_removed_px: float = 0.0
    camera_vertical_fov_deg: float
    camera_aspect: float
    # Component inspection results, when requested (ARFrameRequest.inspection).
    inspection: ComponentInspectionOut | None = None


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
