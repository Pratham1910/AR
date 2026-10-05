"""
Vision API (Project.md #10, #63): perception + tracking only. This router
must never return a PASS/FAIL — that is the QA engine's job, reached only
through /api/inspection/*.
"""

import base64
import re
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.models.model3d import Model3D
from app.schemas.pose import (
    ARCounters,
    ARFrameRequest,
    ARFrameResponse,
    ARTimings,
    CameraIntrinsics,
    DetectionOut,
    PartCheckOut,
    FeaturePoseRequest,
    FeaturePoseResponse,
    ObjectRegistrationRequest,
    ObjectRegistrationResponse,
    PoseAxes,
    PoseRequest,
    PoseResponse,
    Quaternion,
    RegisterReferenceImageRequest,
    RegisterReferenceImageResponse,
    RotationDeg,
    TrackedObjectOut,
    Vector2,
    Vector3,
)
from app.schemas.vision import BoundingBox as VisionBoundingBox
from app.schemas.vision import Vector2 as VisionVector2
from app.schemas.vision import (
    DetectRequest,
    DetectResponse,
    SaveFrameRequest,
    SegmentedObject,
    SegmentRequest,
    SegmentResponse,
    StateRequest,
    StateResponse,
    TrackRequest,
    TrackResponse,
)
from app.services.pose.aruco_pose import ArucoPoseEstimator
from app.services.pose.calibration import load_calibration
from app.services.pose.feature_tracker import FeatureTracker, ReferencePlane, RegistrationQuality
from app.services.pose.markerless import estimate_object_placement
from app.services.model3d.glb_inspect import list_glb_parts
from app.services.pose.model_pose_client import ModelPoseClient, PoseServiceUnavailable
from app.services.pose.transforms import cv_pose_to_threejs, euler_angles_deg, project_pose_axes
from app.services.state_detection.state_engine import ComponentStateRule, StateEstimationError, StateEstimator
from app.services.tracking.ar_session import ARSession, ARTrackingConfig, TrackingState, euler_xyz_deg
from app.services.tracking.pose_filter import PoseFilterConfig
from app.services.tracking.tracker import ObjectTracker
from app.services.tracking.trackers import FlowPoseTracker, Frame, MegaPoseTracker
from app.services.vision.detector import build_detector, time_inference
from app.services.vision.part_presence import PresenceCalibration, region_brightness
from app.services.vision.segmentation import Segmenter, build_segmenter

router = APIRouter(prefix="/api/vision", tags=["vision"])

_settings = get_settings()
_detector = build_detector(_settings.model_path)
_pose_estimator = ArucoPoseEstimator(_settings.aruco_dictionary, _settings.aruco_marker_length_m)
_segmenter: Segmenter | None = None  # built lazily — first request pays the model download/load cost, not startup
_feature_tracker = FeatureTracker()
_reference_planes: dict[str, ReferencePlane] = {}  # in-memory cache, keyed by asset_id
_object_trackers: dict[str, ObjectTracker] = {}  # one ByteTrack instance per session_id (Project.md #17)
_model_pose_client = ModelPoseClient(_settings.pose_service_url, _settings.pose_service_timeout_s)


def _reference_plane_path(asset_id: str) -> Path:
    directory = Path(_settings.reference_images_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / asset_id


def _get_segmenter() -> Segmenter:
    global _segmenter
    if _segmenter is None:
        try:
            _segmenter = build_segmenter(_settings.segmentation_model_name)
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear 503, not a crash
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Could not load segmentation model {_settings.segmentation_model_name!r}: {exc}. "
                    "If this is a first-time download, check network access, or point "
                    "SEGMENTATION_MODEL_NAME at a local .pt file."
                ),
            ) from exc
    return _segmenter


def detectable_classes() -> list[str]:
    """Class labels the object-finding model (YOLO segmentation) can detect.
    Markerless and model-based registration find an object by one of these —
    a label outside this list (a typo, a custom name) is silently never found."""
    return _get_segmenter().class_names


def normalize_class_label(label: str) -> str:
    """Case-insensitive match to a detectable class, or a 400 listing the choices."""
    classes = detectable_classes()
    by_lower = {c.lower(): c for c in classes}
    match = by_lower.get(label.strip().lower())
    if match is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{label}' is not a class the detector knows, so this object would never be found. "
                f"Pick one of: {', '.join(sorted(classes))}"
            ),
        )
    return match


def forget_model_pose(model_id: str) -> None:
    """Drop a deleted model's tracking state and pose-service mesh (best effort)."""
    _model_pose_client.forget(model_id)


@router.get("/classes", response_model=list[str])
def list_detectable_classes() -> list[str]:
    return sorted(detectable_classes())


def decode_frame(image_base64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(image_base64)
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001 - surfaced as a 400 below
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}") from exc
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image_base64 as an image")
    return frame


def _debug_pose_gizmo(rvec, tvec, calibration, axis_length_m: float) -> tuple[PoseAxes, RotationDeg]:
    """Shared by every mode that has a real rvec/tvec (marker, feature tracking) — see schemas.pose's docstrings."""
    axes = project_pose_axes(rvec, tvec, calibration.camera_matrix, calibration.dist_coeffs, axis_length_m)
    rx, ry, rz = euler_angles_deg(rvec)
    return (
        PoseAxes(
            origin=Vector2(x=axes["origin"][0], y=axes["origin"][1]),
            x_axis=Vector2(x=axes["x_axis"][0], y=axes["x_axis"][1]),
            y_axis=Vector2(x=axes["y_axis"][0], y=axes["y_axis"][1]),
            z_axis=Vector2(x=axes["z_axis"][0], y=axes["z_axis"][1]),
        ),
        RotationDeg(rx=rx, ry=ry, rz=rz),
    )


@router.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest) -> DetectResponse:
    frame = decode_frame(request.image_base64)
    detections, inference_ms = time_inference(_detector, frame)
    return DetectResponse(detections=detections, model_version=_detector.model_version, inference_ms=inference_ms)


@router.post("/track", response_model=TrackResponse)
def track(request: TrackRequest) -> TrackResponse:
    """
    Detection + frame-to-frame identity (Project.md #17, Phase 2). Call this
    instead of /detect for anything that needs to know "is this the same
    physical object as last frame" — e.g. confirming a specific PCB (not just
    *a* PCB) was the one removed across several frames. `session_id` must be
    reused for every frame of the same camera stream; a new id starts a
    fresh ByteTrack instance with no memory of previous tracks.
    """
    frame = decode_frame(request.image_base64)
    detections, inference_ms = time_inference(_detector, frame)

    tracker = _object_trackers.setdefault(request.session_id, ObjectTracker())
    tracked_detections = tracker.update(detections)

    return TrackResponse(detections=tracked_detections, model_version=_detector.model_version, inference_ms=inference_ms)


@router.delete("/track/{session_id}")
def reset_track(session_id: str) -> dict:
    """Clears a tracking session's state — call when an inspection run ends or the camera stream restarts."""
    if session_id in _object_trackers:
        del _object_trackers[session_id]
    return {"reset": True, "session_id": session_id}


@router.post("/state", response_model=StateResponse)
def estimate_state(request: StateRequest) -> StateResponse:
    rule = ComponentStateRule(
        component_id=request.target_component_id,
        class_label=request.class_label,
        present_state_id=request.present_state_id,
        absent_state_id=request.absent_state_id,
    )
    estimator = StateEstimator([rule])
    try:
        state_id, confidence = estimator.estimate(request.detections, request.target_component_id)
    except StateEstimationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StateResponse(state_id=state_id, confidence=confidence, state_model_version=estimator.model_version)


@router.post("/segment", response_model=SegmentResponse)
def segment(request: SegmentRequest) -> SegmentResponse:
    """
    Live object outline (Project.md #16): draws the actual detected shape,
    not just a bounding box. Uses a stock COCO-pretrained model by default —
    see app/services/vision/segmentation.py for why that's acceptable here
    even though the procedure detector must be custom-trainable (#15).
    """
    frame = decode_frame(request.image_base64)
    segmenter = _get_segmenter()
    threshold = request.confidence_threshold or _settings.segmentation_confidence_threshold

    start = time.perf_counter()
    objects = segmenter.segment(frame, threshold)
    inference_ms = (time.perf_counter() - start) * 1000

    return SegmentResponse(objects=objects, model_version=segmenter.model_version, inference_ms=inference_ms)


@router.post("/object-registration", response_model=ObjectRegistrationResponse)
def object_registration(request: ObjectRegistrationRequest) -> ObjectRegistrationResponse:
    """
    Markerless registration (Project.md #24's future upgrade, built as a
    first approximate version now): finds `target_class_label` (e.g.
    "bottle") via segmentation, draws its outline, and estimates an
    approximate position from its apparent size vs. `real_world_height_m` —
    no printed marker needed, but see app/services/pose/markerless.py's
    docstring for exactly what this does and does not estimate (position
    only, no orientation; accuracy depends on calibration + the height
    measurement).
    """
    frame = decode_frame(request.image_base64)
    height_px, width_px = frame.shape[:2]
    calibration = load_calibration(_settings.camera_calibration_path, width_px, height_px)

    segmenter = _get_segmenter()
    threshold = request.confidence_threshold or _settings.segmentation_confidence_threshold
    objects = segmenter.segment(frame, threshold)

    target_class_label = normalize_class_label(request.target_class_label)
    matches = [o for o in objects if o.class_label == target_class_label]
    if not matches:
        return ObjectRegistrationResponse(
            found=False,
            approximate=True,
            calibration_is_approximate=calibration.is_approximate,
            calibration_source=calibration.source,
            camera_vertical_fov_deg=calibration.vertical_fov_deg(),
            camera_aspect=calibration.aspect_ratio(),
        )
    best = max(matches, key=lambda o: o.confidence)  # highest-confidence match if several

    try:
        pose = estimate_object_placement(best.bbox, calibration, request.real_world_height_m)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ObjectRegistrationResponse(
        found=True,
        class_label=best.class_label,
        confidence=best.confidence,
        bbox=[best.bbox.x1, best.bbox.y1, best.bbox.x2, best.bbox.y2],
        polygon=[Vector2(x=p.x, y=p.y) for p in best.polygon],
        position=Vector3(x=pose.position[0], y=pose.position[1], z=pose.position[2]),
        quaternion=Quaternion(x=pose.quaternion[0], y=pose.quaternion[1], z=pose.quaternion[2], w=pose.quaternion[3]),
        approximate=True,
        calibration_is_approximate=calibration.is_approximate,
        calibration_source=calibration.source,
        camera_vertical_fov_deg=calibration.vertical_fov_deg(),
        camera_aspect=calibration.aspect_ratio(),
    )


@router.post("/reference-image", response_model=RegisterReferenceImageResponse)
def register_reference_image(request: RegisterReferenceImageRequest) -> RegisterReferenceImageResponse:
    """
    Registers a reference photo for feature/keypoint ("image target")
    tracking (Project.md #24's markerless upgrade — app/services/pose/
    feature_tracker.py). Capture the object's labeled/textured surface
    filling the frame as closely as possible; a low feature_count here means
    live tracking will not work reliably (the surface needs real texture).
    """
    frame = decode_frame(request.image_base64)
    reference = _feature_tracker.register_reference(frame, request.label_width_m, request.label_height_m)

    reference.save(_reference_plane_path(request.asset_id))
    _reference_planes[request.asset_id] = reference

    feature_count = reference.descriptors.shape[0]
    return RegisterReferenceImageResponse(
        feature_count=feature_count, quality=RegistrationQuality.describe(feature_count)
    )


def _get_reference_plane(asset_id: str) -> ReferencePlane:
    if asset_id in _reference_planes:
        return _reference_planes[asset_id]
    path = _reference_plane_path(asset_id)
    if not ReferencePlane.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"No reference image registered for asset {asset_id!r}. POST /api/vision/reference-image first.",
        )
    reference = ReferencePlane.load(path)
    _reference_planes[asset_id] = reference
    return reference


@router.post("/feature-pose", response_model=FeaturePoseResponse)
def estimate_feature_pose(request: FeaturePoseRequest) -> FeaturePoseResponse:
    """
    Real 6DoF pose (position AND orientation) from matching live-camera
    features against a registered reference image, via RANSAC + solvePnP —
    see app/services/pose/feature_tracker.py's docstring for the principle
    and its honest limits (needs real texture; flat-plane approximation).
    """
    frame = decode_frame(request.image_base64)
    height_px, width_px = frame.shape[:2]
    calibration = load_calibration(_settings.camera_calibration_path, width_px, height_px)

    reference = _get_reference_plane(request.asset_id)
    estimate = _feature_tracker.estimate_pose(frame, reference, calibration)

    if not estimate.found:
        return FeaturePoseResponse(
            found=False,
            num_matches=estimate.num_matches,
            calibration_is_approximate=calibration.is_approximate,
            calibration_source=calibration.source,
            camera_vertical_fov_deg=calibration.vertical_fov_deg(),
            camera_aspect=calibration.aspect_ratio(),
        )

    pose = cv_pose_to_threejs(estimate.rvec, estimate.tvec)
    axes, rotation_deg = _debug_pose_gizmo(estimate.rvec, estimate.tvec, calibration, reference.label_width_m)
    return FeaturePoseResponse(
        found=True,
        position=Vector3(x=pose.position[0], y=pose.position[1], z=pose.position[2]),
        quaternion=Quaternion(x=pose.quaternion[0], y=pose.quaternion[1], z=pose.quaternion[2], w=pose.quaternion[3]),
        num_matches=estimate.num_matches,
        num_inliers=estimate.num_inliers,
        inlier_points_px=[Vector2(x=x, y=y) for x, y in estimate.inlier_points_px] if estimate.inlier_points_px else None,
        calibration_is_approximate=calibration.is_approximate,
        calibration_source=calibration.source,
        camera_vertical_fov_deg=calibration.vertical_fov_deg(),
        camera_aspect=calibration.aspect_ratio(),
        rotation_deg=rotation_deg,
        axes=axes,
    )


@router.post("/pose", response_model=PoseResponse)
def estimate_pose(request: PoseRequest) -> PoseResponse:
    """
    6DoF pose of a physical marker relative to the camera (Project.md #24-#26,
    Phase 5). Marker-based only for now — this is explicitly the initial
    registration method, not the final product requirement.
    """
    frame = decode_frame(request.image_base64)
    height, width = frame.shape[:2]
    calibration = load_calibration(_settings.camera_calibration_path, width, height)

    estimate = _pose_estimator.estimate(frame, calibration, request.target_marker_id)
    if not estimate.found:
        return PoseResponse(
            found=False,
            calibration_is_approximate=calibration.is_approximate,
            calibration_source=calibration.source,
            camera_vertical_fov_deg=calibration.vertical_fov_deg(),
            camera_aspect=calibration.aspect_ratio(),
        )

    pose = cv_pose_to_threejs(estimate.rvec, estimate.tvec)
    axes, rotation_deg = _debug_pose_gizmo(estimate.rvec, estimate.tvec, calibration, _settings.aruco_marker_length_m)
    return PoseResponse(
        found=True,
        marker_id=estimate.marker_id,
        position=Vector3(x=pose.position[0], y=pose.position[1], z=pose.position[2]),
        quaternion=Quaternion(x=pose.quaternion[0], y=pose.quaternion[1], z=pose.quaternion[2], w=pose.quaternion[3]),
        reprojection_error_px=estimate.reprojection_error_px,
        corners_px=[Vector2(x=x, y=y) for x, y in estimate.corners_px] if estimate.corners_px else None,
        calibration_is_approximate=calibration.is_approximate,
        calibration_source=calibration.source,
        camera_vertical_fov_deg=calibration.vertical_fov_deg(),
        camera_aspect=calibration.aspect_ratio(),
        rotation_deg=rotation_deg,
        axes=axes,
    )


# --- Live AR: detect once, then track (app/services/tracking/ar_session.py) ---

_AR_MAX_SESSIONS = 16
_ar_sessions: dict[str, tuple[tuple, ARSession]] = {}  # session_id -> (config key, session)


def find_object(frame_bgr: np.ndarray, class_label: str, threshold: float | None = None) -> SegmentedObject | None:
    """The most confident detection of `class_label` in the frame, if any."""
    objects = _get_segmenter().segment(frame_bgr, threshold or _settings.segmentation_confidence_threshold)
    matches = [o for o in objects if o.class_label == class_label]
    return max(matches, key=lambda o: o.confidence) if matches else None


def _ar_detector(class_label: str):
    """The expensive step, called by ARSession only while SEARCHING/RECOVERING."""

    def detect(frame: Frame) -> SegmentedObject | None:
        return find_object(frame.bgr, class_label)

    return detect


def find_object_by_model(frame_jpeg: bytes, label: str, name: str) -> SegmentedObject | None:
    """The registered mesh `label` found in the frame from its 3D model alone
    (pose service /detect) — no object class involved."""
    found = _model_pose_client.detect(label, frame_jpeg, _settings.cad_detect_min_score)
    if found is None:
        return None
    x1, y1, x2, y2 = found.bbox
    return SegmentedObject(
        class_label=name,
        confidence=found.score,
        bbox=VisionBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        polygon=[VisionVector2(x=x, y=y) for x, y in found.polygon],
    )


def register_model_mesh(model_id: str, read_glb, scale: float) -> None:
    """Makes sure the pose service has the whole model's mesh (needed to find it by its 3D model)."""
    _model_pose_client.ensure_registered(model_id, read_glb, scale)


def _cad_detector(label: str, name: str):
    """Model-based mode's detector: finds the whole assembly by its mesh."""

    def detect(frame: Frame) -> SegmentedObject | None:
        return find_object_by_model(frame.jpeg, label, name)

    return detect


_calibration_cache: dict[Path, tuple[float, PresenceCalibration]] = {}


def part_calibrations(model_id: str) -> list[PresenceCalibration]:
    """Presence calibrations saved for this model's parts (cached by file mtime)."""
    found = []
    for path in sorted(Path(_settings.part_calibration_dir).glob(f"{model_id}__node*.json")):
        mtime = path.stat().st_mtime
        cached = _calibration_cache.get(path)
        if cached is None or cached[0] != mtime:
            calibration = PresenceCalibration.load(path)
            if calibration is None:
                continue
            _calibration_cache[path] = cached = (mtime, calibration)
        found.append(cached[1])
    return found


def _ar_model(request: ARFrameRequest, db: Session) -> Model3D:
    try:
        model = db.get(Model3D, uuid.UUID(request.model_id or ""))
    except ValueError:
        model = None
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


def _build_ar_session(request: ARFrameRequest, class_label: str | None, model: Model3D | None) -> ARSession:
    """`class_label` None: find the object by its 3D model (model-based mode only)."""
    common = dict(
        detect_interval_ms=_settings.ar_detect_interval_ms,
        grace_frames=_settings.ar_grace_frames,
        lost_timeout_ms=_settings.ar_lost_timeout_ms,
        filter=PoseFilterConfig(
            enabled=_settings.ar_filter_enabled,
            translation_process_noise=_settings.ar_filter_translation_process_noise,
            translation_measurement_noise=_settings.ar_filter_translation_measurement_noise,
            rotation_process_noise_deg=_settings.ar_filter_rotation_process_noise_deg,
            rotation_measurement_noise_deg=_settings.ar_filter_rotation_measurement_noise_deg,
        ),
    )
    if model is None:
        config = ARTrackingConfig(
            good_confidence=_settings.ar_flow_good_confidence,
            lost_confidence=_settings.ar_flow_lost_confidence,
            **common,
        )
        tracker = FlowPoseTracker(request.real_world_height_m)
    else:
        glb_path = Path(_settings.models_3d_dir) / model.storage_key
        label, node_names = str(model.id), None
        try:
            if request.track_part is not None:
                part = next(
                    (p for p in list_glb_parts(glb_path.read_bytes()) if p.node_index == request.track_part), None
                )
                if part is None:
                    raise HTTPException(status_code=400, detail=f"This model has no part {request.track_part}")
                label, node_names = f"{model.id}__node{part.node_index}", [part.name]
            offset = _model_pose_client.ensure_registered(label, glb_path.read_bytes, model.scale, node_names)
            if class_label is None:
                # Found by the WHOLE assembly's shape even when tracking one
                # part: the box then covers the whole object, as the part
                # presence regions (fractions of that box) assume.
                _model_pose_client.ensure_registered(str(model.id), glb_path.read_bytes, model.scale)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"GLB file missing: {glb_path}") from exc
        except PoseServiceUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        config = ARTrackingConfig(
            good_confidence=_settings.ar_model_good_confidence,
            lost_confidence=_settings.ar_model_lost_confidence,
            **common,
        )
        tracker = MegaPoseTracker(_model_pose_client, label, _settings.model_pose_track_iterations, part_offset=offset)
        if class_label is None:
            return ARSession(model.name, _cad_detector(str(model.id), model.name), tracker, config)
    assert class_label is not None  # markerless always detects by class
    return ARSession(class_label, _ar_detector(class_label), tracker, config)


@router.post("/ar-session/frame", response_model=ARFrameResponse)
def ar_session_frame(request: ARFrameRequest, db: Session = Depends(get_db)) -> ARFrameResponse:
    """
    One camera frame through the session's SEARCHING / INITIALIZING /
    TRACKING / LOST / RECOVERING state machine (ar_session.py). The detector
    (YOLO) only runs while SEARCHING or RECOVERING, at most every
    ar_detect_interval_ms; while TRACKING only the tracker runs — optical
    flow ("markerless") or optical flow + the MegaPose refiner ("model").
    Changing mode/class/model/height starts a fresh session.
    """
    request_started = time.perf_counter()
    class_label: str | None = None
    if request.mode == "markerless" or request.detect_by == "class":
        if not request.class_label:
            raise HTTPException(status_code=400, detail="Pick an object class — it's what the detector looks for.")
        class_label = normalize_class_label(request.class_label)
    if request.mode == "model":
        if not request.model_id:
            raise HTTPException(status_code=400, detail="model_id is required for model-based tracking")
        model = _ar_model(request, db)
        key: tuple = ("model", class_label, str(model.id), model.scale, request.track_part)
    else:
        if not request.real_world_height_m or request.real_world_height_m <= 0:
            raise HTTPException(status_code=400, detail="real_world_height_m is required for markerless tracking")
        model = None
        key = ("markerless", class_label, request.real_world_height_m)

    # Keep the live session while the same thing is being tracked — its
    # tracker state is the whole point; rebuild only if that changed.
    existing = _ar_sessions.pop(request.session_id, None)
    session = existing[1] if existing is not None and existing[0] == key else _build_ar_session(request, class_label, model)
    _ar_sessions[request.session_id] = (key, session)
    while len(_ar_sessions) > _AR_MAX_SESSIONS:
        _ar_sessions.pop(next(iter(_ar_sessions)))  # oldest-used first

    frame_bgr = decode_frame(request.image_base64)
    height_px, width_px = frame_bgr.shape[:2]
    calibration = load_calibration(_settings.camera_calibration_path, width_px, height_px)
    try:
        result = session.step(Frame(frame_bgr, calibration))
    except PoseServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    obj = result.obj
    position = quaternion = rotation_deg = axes = None
    if result.visible and obj is not None and obj.position is not None:
        position = Vector3(x=obj.position[0], y=obj.position[1], z=obj.position[2])
        quaternion = Quaternion(x=obj.quaternion[0], y=obj.quaternion[1], z=obj.quaternion[2], w=obj.quaternion[3])
        rx, ry, rz = euler_xyz_deg(obj.quaternion)
        rotation_deg = RotationDeg(rx=rx, ry=ry, rz=rz)
        t_co = obj.extra.get("t_camera_object")
        if t_co is not None:
            rvec, _ = cv2.Rodrigues(t_co[:3, :3])
            axes, _ = _debug_pose_gizmo(rvec.reshape(3), t_co[:3, 3], calibration, axis_length_m=0.05)

    # Part presence (e.g. "is the cap still on?"): only while TRACKING, so the
    # object box is current; measured in the frame the backend received,
    # which never contains the browser's overlay.
    part_checks: list[PartCheckOut] = []
    calibrations = part_calibrations(request.model_id) if request.model_id else []
    if result.state == TrackingState.TRACKING and obj is not None and obj.bbox is not None:
        for cal in calibrations:
            brightness = region_brightness(frame_bgr, cal.region, obj.bbox)
            if brightness is None:
                continue
            state, confidence = cal.classify(brightness)
            part_checks.append(
                PartCheckOut(
                    node_index=cal.node_index,
                    part_name=cal.part_name,
                    state=state,
                    confidence=confidence,
                    brightness=brightness,
                    region_px=list(cal.region.pixels(obj.bbox)),
                )
            )

    k = calibration.camera_matrix
    detection = result.detection
    return ARFrameResponse(
        state=result.state.value,
        visible=result.visible,
        monitoring=result.monitoring,
        object=None
        if obj is None
        else TrackedObjectOut(
            object_id=obj.object_id,
            class_label=obj.class_label,
            confidence=obj.confidence,
            first_seen_frame=obj.first_seen_frame,
            last_seen_frame=obj.last_seen_frame,
            bbox=None if obj.bbox is None else [obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2],
            # Segmentation/tracking use the vision schemas' Vector2; responses use the pose schemas' one.
            polygon=None if obj.polygon is None else [Vector2(x=p.x, y=p.y) for p in obj.polygon],
            velocity_px_s=None if obj.velocity_px_s is None else Vector2(x=obj.velocity_px_s[0], y=obj.velocity_px_s[1]),
        ),
        detection=None
        if detection is None
        else DetectionOut(
            class_label=detection.class_label,
            confidence=detection.confidence,
            bbox=[detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2],
            polygon=[Vector2(x=p.x, y=p.y) for p in detection.polygon],
        ),
        position=position,
        quaternion=quaternion,
        rotation_deg=rotation_deg,
        axes=axes,
        approximate=request.mode == "markerless",
        detector_ran=result.detector_ran,
        tracker_ran=result.tracker_ran,
        timings_ms=ARTimings(**result.timings_ms, total=(time.perf_counter() - request_started) * 1000.0),
        counters=ARCounters(
            frame_index=session.frame_index,
            detection_runs=session.detection_runs,
            detection_count=session.detection_count,
            tracking_frames=session.tracking_frames,
            frames_since_detection=session.frames_since_detection,
        ),
        events=result.events,
        part_checks=part_checks,
        calibrated_parts=[cal.part_name for cal in calibrations],
        good_confidence=session.config.good_confidence,
        lost_confidence=session.config.lost_confidence,
        intrinsics=CameraIntrinsics(
            fx=float(k[0, 0]), fy=float(k[1, 1]), cx=float(k[0, 2]), cy=float(k[1, 2]), width=width_px, height=height_px
        ),
        calibration_is_approximate=calibration.is_approximate,
        calibration_source=calibration.source,
        camera_vertical_fov_deg=calibration.vertical_fov_deg(),
        camera_aspect=calibration.aspect_ratio(),
    )


@router.delete("/ar-session/{session_id}", status_code=204)
def end_ar_session(session_id: str) -> None:
    _ar_sessions.pop(session_id, None)


@router.post("/debug-frame")
def save_debug_frame(request: SaveFrameRequest) -> dict:
    """Stores the exact camera frame (full resolution, no overlay) under a
    label, for measuring/calibrating checks offline (e.g. cap on vs off)."""
    label = re.sub(r"[^A-Za-z0-9_-]+", "-", request.label.strip()).strip("-") or "frame"
    frame = decode_frame(request.image_base64)  # validates it's an image
    directory = Path(_settings.debug_frames_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{label}_{time.strftime('%Y%m%d-%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {"saved": path.name, "width": frame.shape[1], "height": frame.shape[0]}
