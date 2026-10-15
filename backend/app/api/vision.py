"""
Vision API (Project.md #10, #63): perception + tracking only. This router
must never return a PASS/FAIL — that is the QA engine's job, reached only
through /api/inspection/*.
"""

import asyncio
import logging
import base64
import json
import re
import struct
import time
import uuid
from typing import Literal
from pathlib import Path

import cv2
import numpy as np
from fastapi.concurrency import run_in_threadpool
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.component_inspection import end_inspection, run_inspection
from app.core.config import get_settings
from app.core.database import SessionLocal, get_db
from app.models.model3d import Model3D
from app.schemas.pose import (
    HandsOut,
    StationOut,
    TrackingQualityOut,
    ARCounters,
    ARFrameRequest,
    ARFrameResponse,
    GuideOut,
    PoseFeatureOut,
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
    ClipFrameRequest,
    ClipOut,
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
from app.services.pose.calibration import effective_calibration_path, load_calibration, undistort
from app.services.pose.edge_align import EdgeModel
from app.services.pose.outline_pose import outline_pose
from app.services.pose.pose_features import PoseFeature, load_pose_features, with_feature_edges
from app.services.pose.symmetry import find_symmetries
from app.services.pose.feature_tracker import FeatureTracker, ReferencePlane, RegistrationQuality
from app.services.pose.markerless import estimate_object_placement
from app.services.component_inspection.registry import assembly_center
from app.services.model3d.glb_inspect import GlbPartMesh, glb_part_meshes, list_glb_parts
from app.services.pose.model_pose_client import ModelPoseClient, PoseServiceUnavailable
from app.services.pose.transforms import cv_model_pose_to_threejs, cv_pose_to_threejs, euler_angles_deg, project_pose_axes
from app.services.state_detection.state_engine import ComponentStateRule, StateEstimationError, StateEstimator
from app.services.tracking.replay import clip_dir, load_clip
from app.services.tracking.ar_session import ARSession, ARTrackingConfig, TrackingState, euler_xyz_deg
from app.services.tracking.pose_filter import PoseFilterConfig
from app.services.tracking.tracker import ObjectTracker
from app.services.tracking.guide_view import GuideMatcher, default_guide_pose
from app.services.tracking.trackers import FlowPoseTracker, Frame, MegaPoseTracker, resting_side_ok
from app.services.tracking.station import StationLock
from app.services.tracking.ar_session import StepResult, TrackedObject
from app.services.vision.detector import build_detector, time_inference
from app.services.vision.part_presence import (
    PresenceCalibration,
    mask_features,
    mask_outline,
    project_part_mask,
    region_brightness,
)
from app.services.vision.hand_mask import HandMaskService, HandSegmenter, MediaPipeHands
from app.services.vision.segmentation import Segmenter, build_segmenter

router = APIRouter(prefix="/api/vision", tags=["vision"])

_settings = get_settings()
log = logging.getLogger("uvicorn.error")
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
    calibration = load_calibration(effective_calibration_path(_settings.camera_calibration_path), width_px, height_px, _settings.camera_assumed_hfov_deg)

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
    calibration = load_calibration(effective_calibration_path(_settings.camera_calibration_path), width_px, height_px, _settings.camera_assumed_hfov_deg)

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
    calibration = load_calibration(effective_calibration_path(_settings.camera_calibration_path), width, height, _settings.camera_assumed_hfov_deg)

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

# --- Fixed-station mode: a pose solved once and frozen (station.py) ---

_station_locks: dict[str, StationLock | None] = {}


def _station_dir() -> Path:
    return Path(_settings.station_locks_dir)


def station_lock(model_id: str) -> StationLock | None:
    if model_id not in _station_locks:
        _station_locks[model_id] = StationLock.load(_station_dir(), model_id)
    return _station_locks[model_id]


def _station_out(lock: StationLock | None, shift: float | None = None) -> StationOut:
    if lock is None:
        return StationOut(locked=False)
    return StationOut(
        locked=True,
        shift_px=shift,
        moved=shift is not None and shift > lock.max_shift_px,
        created_at=lock.created_at,
        max_shift_px=lock.max_shift_px,
    )


class StationLockRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    session_id: str  # the live session whose current pose to freeze
    model_id: str
    image_base64: str  # the frame now: the reference for detecting movement


@router.post("/station-lock", response_model=StationOut)
def create_station_lock(request: StationLockRequest) -> StationOut:
    """Freezes the session's current, healthy pose for this model (fixed camera + fixtured part)."""
    entry = _ar_sessions.get(request.session_id)
    session = entry[1] if entry else None
    obj = session.obj if session else None
    t_co = obj.extra.get("t_camera_object") if obj else None
    if session is None or obj is None or t_co is None or session.state != TrackingState.TRACKING:
        raise HTTPException(status_code=409, detail="Lock the pose while the object is being tracked (model-based, TRACKING).")
    if obj.confidence < session.config.good_confidence:
        raise HTTPException(
            status_code=409,
            detail=f"Tracking confidence {obj.confidence:.0%} is below {session.config.good_confidence:.0%}: align better before locking.",
        )
    frame = decode_frame(request.image_base64)
    if obj.bbox is None or obj.position is None or obj.quaternion is None:
        raise HTTPException(status_code=409, detail="No object region to lock yet — try again in a moment.")
    lock = StationLock.create(
        request.model_id, t_co, obj.position, obj.quaternion, frame,
        (obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2), _settings.station_max_shift_px,
    )
    lock.save(_station_dir())
    _station_locks[request.model_id] = lock
    return _station_out(lock, 0.0)


class StationSettings(BaseModel):
    # Which side of the model faces where (MegaPoseTracker resting): "camera"
    # = camera above, top facing it; "upright" = side view; "any" = no constraint.
    resting: Literal["any", "camera", "upright", "flat"] = "any"
    # Guide view (detect_by "guide"): the pose (T_camera_object, 4x4, metres)
    # the model's outline is shown at. None = default_guide_pose.
    guide_pose: list[list[float]] | None = None


def station_settings(model_id: str) -> StationSettings:
    path = Path(_settings.station_settings_dir) / f"{model_id}.json"
    try:
        return StationSettings(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return StationSettings()


@router.get("/station-settings/{model_id}", response_model=StationSettings)
def get_station_settings(model_id: str) -> StationSettings:
    return station_settings(model_id)


class GuideViewSave(BaseModel):
    session_id: str  # the live session whose current pose becomes the guide view


@router.post("/guide-view/{model_id}", response_model=StationSettings)
def save_guide_view(model_id: str, payload: GuideViewSave) -> StationSettings:
    """The pose being tracked now becomes the model's guide view: next time,
    its outline is drawn there for the operator to line the object up with."""
    entry = _ar_sessions.get(payload.session_id)
    session = entry[1] if entry else None
    t_co = session.obj.extra.get("t_camera_object") if session is not None and session.obj is not None else None
    if session is None or t_co is None or session.state != TrackingState.TRACKING:
        raise HTTPException(status_code=409, detail="Track the object first: the guide view is saved from the current pose.")
    settings = station_settings(model_id)
    settings.guide_pose = np.asarray(t_co, float).tolist()
    return _write_station_settings(model_id, settings)


@router.delete("/guide-view/{model_id}", response_model=StationSettings)
def reset_guide_view(model_id: str) -> StationSettings:
    """Back to the default guide view."""
    settings = station_settings(model_id)
    settings.guide_pose = None
    return _write_station_settings(model_id, settings)


def _write_station_settings(model_id: str, settings: StationSettings) -> StationSettings:
    directory = Path(_settings.station_settings_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{model_id}.json").write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    return settings


@router.put("/station-settings/{model_id}", response_model=StationSettings)
def put_station_settings(model_id: str, payload: StationSettings) -> StationSettings:
    """Updates the fields sent; the others (e.g. a saved guide view) are kept."""
    merged = station_settings(model_id).model_copy(update=payload.model_dump(exclude_unset=True))
    return _write_station_settings(model_id, merged)


@router.get("/station-lock/{model_id}", response_model=StationOut)
def get_station_lock(model_id: str) -> StationOut:
    return _station_out(station_lock(model_id))


@router.delete("/station-lock/{model_id}", response_model=StationOut)
def delete_station_lock(model_id: str) -> StationOut:
    StationLock.remove(_station_dir(), model_id)
    _station_locks[model_id] = None
    return StationOut(locked=False)


# --- Hands (Phase 7): a mask per frame, computed off the tracking thread ---

_hand_segmenter: HandSegmenter | None = None
_hand_segmenter_error: str | None = None
_hand_services: dict[str, HandMaskService] = {}
_guide_matchers: dict[str, GuideMatcher] = {}


def guide_pose_for(model: Model3D, edge_model: EdgeModel | None, camera_matrix: np.ndarray,
                   size: tuple[int, int], resting: str) -> tuple[np.ndarray, bool] | None:
    """(the model's guide pose, whether it was saved); None without the model's geometry."""
    saved = station_settings(str(model.id)).guide_pose
    if saved is not None:
        return np.asarray(saved, float), True
    if edge_model is None:
        return None
    return default_guide_pose(edge_model.vertices, camera_matrix, size, resting), False


def _pose_features_out(features: list[dict] | None) -> list[PoseFeatureOut]:
    def point(p):
        return Vector2(x=float(p[0]), y=float(p[1])) if p is not None else None

    return [
        PoseFeatureOut(
            feature_id=f["feature_id"], status=f["status"], evidence=float(f["evidence"]),
            predicted_center_px=point(f.get("predicted_center_px")),
            predicted_contour_px=[point(p) for p in f.get("predicted_contour_px") or []],
            observed_center_px=point(f.get("observed_center_px")),
            residual_px=f.get("residual_px"),
        )
        for f in features or []
    ]


def guide_lock_confidence(edge_score: float, good_confidence: float) -> float:
    """The lock confidence of a pose that passed the guide check. The edge-fit
    score is on its own scale (0.4-0.7 for a correct fit) and was already
    judged against the guide thresholds; used directly as a confidence, it
    fell below the session's "lost" level and every guide lock was thrown
    away. A verified lock counts as a good one; tracking confidence takes
    over from the next frame."""
    return max(float(edge_score), float(good_confidence))


def _guide_detection(edge_model: EdgeModel, t_co: np.ndarray, camera_matrix: np.ndarray, name: str, score: float) -> SegmentedObject:
    """The posed model's outline in the image, as the detection a lock starts from."""
    cam = edge_model.vertices @ t_co[:3, :3].T + t_co[:3, 3]
    px = cam @ camera_matrix.T
    px = px[:, :2] / px[:, 2:3]
    hull = cv2.convexHull(px.astype(np.float32)).reshape(-1, 2)
    (x1, y1), (x2, y2) = hull.min(axis=0), hull.max(axis=0)
    return SegmentedObject(
        class_label=name,
        confidence=score,
        bbox=VisionBoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
        polygon=[VisionVector2(x=float(x), y=float(y)) for x, y in hull],
    )


def _get_hand_segmenter() -> HandSegmenter | None:
    """Built once from settings (hand_mask_method). A source that fails to
    load is reported in every response (HandsOut.error), not skipped silently."""
    global _hand_segmenter, _hand_segmenter_error
    method = _settings.hand_mask_method
    if method == "off":
        return None
    if _hand_segmenter is None and _hand_segmenter_error is None:
        hands = persons = None
        problems = []
        if method in ("both", "mediapipe"):
            try:
                hands = MediaPipeHands(_settings.hand_landmarker_model)
            except Exception as exc:  # noqa: BLE001 - optional dependency / model file
                problems.append(f"MediaPipe hands unavailable: {exc}")
        if method in ("both", "person"):
            try:
                # Its own model instance: the hand worker runs beside the detector.
                person_model = build_segmenter(_settings.segmentation_model_name)
                threshold = _settings.hand_person_confidence

                def persons(frame: np.ndarray) -> list[np.ndarray]:
                    return [
                        np.array([[p.x, p.y] for p in o.polygon])
                        for o in person_model.segment(frame, threshold)
                        if o.class_label == "person" and len(o.polygon) >= 3
                    ]
            except Exception as exc:  # noqa: BLE001
                problems.append(f"person segmentation unavailable: {exc}")
        _hand_segmenter = HandSegmenter(hands, persons, _settings.hand_mask_margin_px)
        _hand_segmenter_error = "; ".join(problems) or None
        if problems:
            log.warning("[HANDS] %s", _hand_segmenter_error)
    return _hand_segmenter if _hand_segmenter is not None and _hand_segmenter.available else None


def _hand_service(session_id: str) -> HandMaskService | None:
    segmenter = _get_hand_segmenter()
    if segmenter is None:
        return None
    service = _hand_services.get(session_id)
    if service is None:
        service = HandMaskService(segmenter, _settings.hand_mask_interval_ms, _settings.hand_mask_max_age_ms)
        _hand_services[session_id] = service
        while len(_hand_services) > _AR_MAX_SESSIONS:
            _hand_services.pop(next(iter(_hand_services)))
    return service


def _tracking_status(result, session: ARSession) -> str:
    """The session state in the operator's terms (TRACKING / TRACKING_OCCLUDED /
    REACQUIRING / TRACKING_LOST / SEARCHING / INITIALIZING)."""
    state = result.state
    if state in (TrackingState.TRACKING, TrackingState.TRACKING_OCCLUDED):
        return state.value
    if state == TrackingState.LOST:
        return "TRACKING_LOST"
    if state == TrackingState.RECOVERING or (state == TrackingState.INITIALIZING and result.obj is not None):
        return "REACQUIRING"
    if state == TrackingState.SEARCHING and session.detection_count > 0:
        return "TRACKING_LOST"  # had it, gave up looking: model hidden
    return state.value


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
    found = _model_pose_client.detect(label, frame_jpeg, _settings.cad_detect_min_score, _settings.cad_detect_min_combined)
    if found is None:
        return None
    x1, y1, x2, y2 = found.bbox
    return SegmentedObject(
        class_label=name,
        confidence=found.score,
        bbox=VisionBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        polygon=[VisionVector2(x=x, y=y) for x, y in found.polygon],
    )


def select_object_at(
    frame_jpeg: bytes, x: float, y: float, model_id: str, name: str, camera_matrix: np.ndarray, events: list[str],
    vertices: np.ndarray | None = None, resting: str = "any",
) -> tuple[SegmentedObject, np.ndarray | None] | None:
    """The object the operator clicked: of the image segments under the click
    (usually nested — a panel, the whole object, a background region), the
    one the model's silhouette covers best, and the pose that does it
    (assembly frame), or None for the pose if only MegaPose's search could
    judge them.

    With the model's geometry (`vertices`, recentred metres) the pose comes
    from the outline itself (outline_pose: rotations x placement, resting
    side respected). Appearance plays no part: it's what fails on plain
    objects — on the user's black case MegaPose's full search posed the model
    edge-on (silhouette 55%) on the right region, the outline pose 98%.
    Without it, MegaPose's full search on each segment, judged by its
    silhouette IoU."""
    segments = _model_pose_client.segments_at(frame_jpeg, x, y, _settings.ar_select_max_area)
    segments = [s for s in segments if s.area_fraction >= _settings.ar_select_min_area][: _settings.ar_select_max_candidates]
    if not segments:
        events.append(f"[DETECTION] Nothing segmented at ({x:.0f}, {y:.0f})")
        return None
    best, best_iou, best_pose = None, -1.0, None
    for s in segments:
        if vertices is not None:
            found = outline_pose(
                vertices, np.asarray(s.polygon, float), camera_matrix, allowed=lambda t: resting_side_ok(t, resting)
            )
            if found is None:
                continue
            pose, iou = found.t_co, found.iou
        else:
            x1, y1, x2, y2 = s.bbox
            result = _model_pose_client.full_search(model_id, frame_jpeg, camera_matrix, [x1, y1, x2, y2], s.polygon)
            if result.t_camera_object is None:
                continue
            pose, iou = None, result.silhouette_iou or 0.0
            distance = float(result.t_camera_object[2, 3])
        distance = float(pose[2, 3]) if pose is not None else distance
        ok = _settings.ar_min_object_distance_m <= distance <= _settings.ar_max_object_distance_m
        log.info("[SELECT] segment %.1f%% of image at %s: silhouette IoU %.2f, %.2f m%s",
                 100 * s.area_fraction, [round(v) for v in s.bbox], iou, distance, "" if ok else " (outside working range)")
        if ok and iou > best_iou:
            best, best_iou, best_pose = s, iou, pose
    if best is None:
        events.append("[DETECTION] The model doesn't fit anything at the clicked point")
        return None
    events.append(f"[DETECTION] Picked the region the model fits best (silhouette {best_iou:.0%}) of {len(segments)} under the click")
    x1, y1, x2, y2 = best.bbox
    detection = SegmentedObject(
        class_label=name,
        confidence=best_iou,
        bbox=VisionBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        polygon=[VisionVector2(x=px, y=py) for px, py in best.polygon],
    )
    return detection, best_pose


def locate_object_and_pose(
    frame_jpeg: bytes, width: int, height: int, model_id: str, name: str
) -> tuple[SegmentedObject, np.ndarray, np.ndarray] | str:
    """Finds the model in one frame and solves its pose, as a fresh lock would
    (3D-model detection, then full search picked by silhouette). Returns
    (detection, T_camera_object, camera matrix), or why it couldn't."""
    obj = find_object_by_model(frame_jpeg, model_id, name)
    if obj is None:
        return "object not found"
    camera_matrix = load_calibration(effective_calibration_path(_settings.camera_calibration_path), width, height, _settings.camera_assumed_hfov_deg).camera_matrix
    b = obj.bbox
    result = _model_pose_client.full_search(
        model_id, frame_jpeg, camera_matrix, [b.x1, b.y1, b.x2, b.y2], [(p.x, p.y) for p in obj.polygon]
    )
    if result.t_camera_object is None or (result.silhouette_iou or 0.0) < _settings.ar_model_lost_confidence:
        return "pose rejected"
    distance = float(result.t_camera_object[2, 3])
    if not _settings.ar_min_object_distance_m <= distance <= _settings.ar_max_object_distance_m:
        return f"pose rejected (object would be {distance:.2f} m away)"
    return obj, result.t_camera_object, camera_matrix


def locate_model_pose(
    frame_jpeg: bytes, width: int, height: int, model_id: str, name: str
) -> tuple[np.ndarray, np.ndarray] | str:
    """(T_camera_object, camera matrix) of the model in one frame, or why not (see locate_object_and_pose)."""
    located = locate_object_and_pose(frame_jpeg, width, height, model_id, name)
    return located if isinstance(located, str) else (located[1], located[2])


_geometry_cache: dict[Path, tuple[float, dict[int, GlbPartMesh], np.ndarray]] = {}


def assembly_geometry(glb_path: Path) -> tuple[dict[int, GlbPartMesh], np.ndarray]:
    """The model's part triangles (by node index) and the assembly's bounding-box
    center (GLB units) — the center the pose service recenters on. Cached by mtime."""
    mtime = glb_path.stat().st_mtime
    cached = _geometry_cache.get(glb_path)
    if cached is None or cached[0] != mtime:
        meshes = glb_part_meshes(glb_path.read_bytes())
        points = np.concatenate([m.vertices for m in meshes.values()])
        _geometry_cache[glb_path] = cached = (mtime, meshes, (points.min(axis=0) + points.max(axis=0)) / 2)
    return cached[1], cached[2]


def measure_part(
    frame_bgr: np.ndarray,
    cal: PresenceCalibration,
    model: Model3D,
    t_camera_object: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[float, np.ndarray] | None:
    """A "pose" calibration's feature on the part's exact pixels at this pose, and those pixels."""
    meshes, center = assembly_geometry(Path(_settings.models_3d_dir) / model.storage_key)
    mesh = meshes.get(cal.node_index)
    if mesh is None:
        return None
    mask = project_part_mask(mesh, center, model.scale, t_camera_object, camera_matrix, frame_bgr.shape[:2])
    features = mask_features(frame_bgr, mask)
    return None if features is None else (features[cal.feature], mask)


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


def _box_detector(box: tuple[float, ...], name: str, fallback):
    """The user's drawn box, once (the first lock); after that, when tracking is
    lost, `fallback` (the 3D-model detector) if there is one, else nothing —
    the object needs a new box."""
    used = False

    def detect(frame: Frame) -> SegmentedObject | None:
        nonlocal used
        if used:
            return fallback(frame) if fallback is not None else None
        used = True
        x1, y1, x2, y2 = box
        return SegmentedObject(
            class_label=name,
            confidence=1.0,
            bbox=VisionBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            polygon=[VisionVector2(x=x, y=y) for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))],
        )

    return detect


def _finder(request: ARFrameRequest, class_label: str | None, model: Model3D | None):
    """(detector, label for the tracked object) for request.detect_by."""
    if request.detect_by == "class":
        assert class_label is not None
        return _ar_detector(class_label), class_label
    cad = None
    if model is not None:
        glb_path = Path(_settings.models_3d_dir) / model.storage_key
        try:
            # Found by the WHOLE assembly's shape even when tracking one part:
            # the box then covers the whole object, as part regions assume.
            _model_pose_client.ensure_registered(str(model.id), glb_path.read_bytes, model.scale)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"GLB file missing: {glb_path}") from exc
        except PoseServiceUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        cad = _cad_detector(str(model.id), model.name)
    name = model.name if model is not None else "object"
    if request.detect_by == "guide":
        # Nothing is detected: process_ar_frame locks once the object is lined
        # up with the guide (GuideMatcher).
        return (lambda frame: None), name
    if request.detect_by == "box":
        return _box_detector(tuple(request.init_box or ()), name, cad), name
    assert cad is not None  # checked in _ar_session_for
    return cad, name


_edge_models: dict[tuple, EdgeModel | None] = {}


def _edge_model_for(model: Model3D, glb: bytes) -> EdgeModel | None:
    """The model's 3D edges in the tracked frame (whole assembly, recentred
    on its bounding box like the pose service does, metres), cached per model
    and scale. None when disabled or too heavy."""
    if not _settings.ar_edge_align_enabled:
        return None
    key = (str(model.id), model.scale, _settings.ar_edge_crease_deg, _settings.ar_edge_max_points)
    if key not in _edge_models:
        meshes = glb_part_meshes(glb)
        edge_model = None
        if meshes and sum(len(m.faces) for m in meshes.values()) <= _settings.ar_edge_max_faces:
            center = assembly_center(meshes)
            vertex_sets, face_sets, base = [], [], 0
            for mesh in meshes.values():
                vertex_sets.append((mesh.vertices - center) * model.scale)
                face_sets.append(mesh.faces + base)
                base += len(mesh.vertices)
            edge_model = EdgeModel.from_mesh(
                np.concatenate(vertex_sets), np.concatenate(face_sets),
                crease_deg=_settings.ar_edge_crease_deg, max_points=_settings.ar_edge_max_points,
            )
        _edge_models[key] = edge_model
    return _edge_models[key]


_orientation_setups: dict[tuple, tuple[list[np.ndarray], list[PoseFeature]]] = {}


def _orientation_setup(model: Model3D, glb: bytes, edge_model: EdgeModel | None) -> tuple[list[np.ndarray], list[PoseFeature]]:
    """(the model's symmetries, its pose features) — cached per model, scale
    and mapping-file version. Symmetries come from the model's geometry; pose
    features from "poseFeatures" in its component mapping file."""
    if not _settings.ar_orientation_enabled or edge_model is None:
        return [np.eye(3)], []
    mapping = Path(_settings.component_registry_dir) / f"{model.id}.json"
    stamp = mapping.stat().st_mtime if mapping.exists() else None
    key = (str(model.id), model.scale, stamp, _settings.ar_symmetry_tolerance)
    if key not in _orientation_setups:
        symmetries = find_symmetries(edge_model.vertices, edge_model.faces, tolerance=_settings.ar_symmetry_tolerance)
        try:
            features, warnings = load_pose_features(mapping, glb, model.scale)
        except (ValueError, OSError) as exc:
            features, warnings = [], [f"pose features not loaded: {exc}"]
        for warning in warnings:
            log.warning("[ORIENTATION] %s: %s", model.name, warning)
        log.info("[ORIENTATION] %s: %d symmetric orientation(s), %d pose feature(s)", model.name, len(symmetries), len(features))
        _orientation_setups[key] = (symmetries, features)
    return _orientation_setups[key]


def _build_ar_session(request: ARFrameRequest, class_label: str | None, model: Model3D | None) -> ARSession:
    """Tracker by mode (optical flow for markerless, MegaPose + flow for model);
    how the object is found by request.detect_by (see _finder)."""
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
    if request.mode == "markerless":
        config = ARTrackingConfig(
            good_confidence=_settings.ar_flow_good_confidence,
            lost_confidence=_settings.ar_flow_lost_confidence,
            **common,
        )
        tracker = FlowPoseTracker(request.real_world_height_m)
    else:
        assert model is not None
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
            glb_bytes = glb_path.read_bytes()
            edge_model = _edge_model_for(model, glb_bytes)
            symmetries, pose_features = _orientation_setup(model, glb_bytes, edge_model)
            geometry_model = edge_model  # the CAD's own edges: orientation evidence
            if edge_model is not None and pose_features:
                edge_model = with_feature_edges(edge_model, pose_features)  # printed circles' rims: alignment only
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"GLB file missing: {glb_path}") from exc
        except PoseServiceUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        config = ARTrackingConfig(
            good_confidence=_settings.ar_model_good_confidence,
            lost_confidence=_settings.ar_model_lost_confidence,
            **common,
        )
        tracker = MegaPoseTracker(
            _model_pose_client,
            label,
            _settings.model_pose_track_iterations,
            part_offset=offset,
            still_motion_px=_settings.model_pose_still_motion_px,
            min_overlap=_settings.ar_model_lost_confidence,
            correction_gain_translation=_settings.model_pose_correction_gain_translation,
            correction_gain_rotation=_settings.model_pose_correction_gain_rotation,
            occlusion_min=_settings.tracking_occlusion_min,
            occlusion_hold_ms=_settings.tracking_occlusion_hold_ms,
            refine_max_occlusion=_settings.tracking_refine_max_occlusion,
            flow_support_full=_settings.tracking_flow_support_full,
            min_distance_m=_settings.ar_min_object_distance_m,
            max_distance_m=_settings.ar_max_object_distance_m,
            resting=request.resting,
            edge_model=edge_model,
            edge_tilt_starts_deg=_settings.ar_edge_tilt_starts_deg,
            edge_gain_translation=_settings.ar_edge_gain_translation,
            edge_gain_rotation=_settings.ar_edge_gain_rotation,
            symmetries=symmetries,
            pose_features=pose_features,
            orientation_min_evidence=_settings.ar_orientation_min_evidence,
            orientation_min_margin=_settings.ar_orientation_min_margin,
            correction_verify_deg=_settings.ar_correction_verify_deg,
            geometry_model=geometry_model,
        )
    detector, name = _finder(request, class_label, model)
    return ARSession(name, detector, tracker, config)


def _ar_session_for(request: ARFrameRequest, db: Session) -> tuple[ARSession, Model3D | None]:
    """The live session for this request's session_id, rebuilt only if what is tracked changed."""
    class_label: str | None = None
    if request.detect_by == "class":
        if not request.class_label:
            raise HTTPException(status_code=400, detail="Pick an object class — it's what the detector looks for.")
        class_label = normalize_class_label(request.class_label)
    box: tuple[float, ...] = ()
    if request.detect_by == "box":
        if not request.init_box or len(request.init_box) != 4:
            raise HTTPException(status_code=400, detail="Draw a box around the object first.")
        box = tuple(float(v) for v in request.init_box)
    model = _ar_model(request, db) if request.model_id else None
    if request.mode == "model":
        if model is None:
            raise HTTPException(status_code=400, detail="model_id is required for model-based tracking")
        # The model's saved resting side applies unless this request names one.
        if request.resting == "any":
            request.resting = station_settings(str(model.id)).resting
        key: tuple = ("model", request.detect_by, class_label, box, str(model.id), model.scale, request.track_part, request.resting)
    else:
        if not request.real_world_height_m or request.real_world_height_m <= 0:
            raise HTTPException(status_code=400, detail="real_world_height_m is required for markerless tracking")
        if request.detect_by == "guide":
            raise HTTPException(status_code=400, detail="Guide view needs model-based tracking (a 3D model).")
        if request.detect_by == "model" and model is None:
            raise HTTPException(
                status_code=400,
                detail="Finding the object by its 3D model needs a model — or choose Object class or Draw a box.",
            )
        key = ("markerless", request.detect_by, class_label, box, str(model.id) if model else None, request.real_world_height_m)

    # Keep the live session while the same thing is being tracked — its
    # tracker state is the whole point; rebuild only if that changed.
    existing = _ar_sessions.pop(request.session_id, None)
    session = existing[1] if existing is not None and existing[0] == key else _build_ar_session(request, class_label, model)
    _ar_sessions[request.session_id] = (key, session)
    while len(_ar_sessions) > _AR_MAX_SESSIONS:
        _ar_sessions.pop(next(iter(_ar_sessions)))  # oldest-used first
    return session, model


def decode_frame_bytes(raw: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode the frame as an image")
    return frame


def process_ar_frame(
    request: ARFrameRequest,
    session: ARSession,
    model: Model3D | None,
    encoded: bytes,
    started: float,
    capture_ms: float | None = None,
    db: Session | None = None,
) -> ARFrameResponse:
    """One encoded camera frame through the session's state machine, and what to draw."""
    frame_bgr = decode_frame_bytes(encoded)
    height_px, width_px = frame_bgr.shape[:2]
    calibration = load_calibration(
        effective_calibration_path(_settings.camera_calibration_path), width_px, height_px,
        _settings.camera_assumed_hfov_deg, camera_label=request.camera_label,
    )
    if calibration.has_distortion and request.mode == "model":
        # One distortion-free image for everything downstream (MegaPose, edges,
        # features, the overlay's pinhole camera) — never mixed with the raw one.
        frame_bgr = undistort(frame_bgr, calibration)
        encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
    # The newest hand mask (computed asynchronously; frames don't wait for it).
    hand = None
    service = _hand_service(request.session_id)
    if service is not None:
        now = time.monotonic()
        service.submit(frame_bgr, now)
        hand = service.latest(now)
    lock = station_lock(request.model_id) if request.station_lock and request.model_id else None
    station_out = None
    if request.station_lock and request.model_id:
        station_out = _station_out(lock)
    if lock is not None and lock.frame_size == (width_px, height_px):
        # Fixed station: the frozen pose, no detection or pose estimation.
        shift, _ = lock.shift(frame_bgr)
        station_out = _station_out(lock, shift)
        moved = station_out.moved
        if session.obj is None or session.obj.extra.get("station") is not True:
            session.obj = TrackedObject(
                object_id=1, class_label=model.name if model else "object", confidence=1.0,
                first_seen_frame=session.frame_index, last_seen_frame=session.frame_index, last_seen_at=time.monotonic(),
            )
        x1, y1, x2, y2 = lock.roi
        obj_now = session.obj
        obj_now.position, obj_now.quaternion = lock.position, lock.quaternion
        obj_now.bbox = VisionBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        obj_now.polygon = None
        obj_now.confidence = 0.0 if moved else 1.0
        obj_now.last_seen_frame = session.frame_index
        obj_now.extra = {"t_camera_object": lock.t_camera_object, "station": True, "confidence_source": "station lock"}
        session.frame_index += 1
        events = [f"[STATION] Object moved {shift:.1f} px since locking — re-align or unlock"] if moved else []
        result = StepResult(
            state=TrackingState.TRACKING if not moved else TrackingState.LOST,
            visible=True, monitoring=False, obj=obj_now, detection=None, detector_ran=False, tracker_ran=False,
            timings_ms={"detection": 0.0, "initialization": 0.0, "tracking": 0.0, "refinement": 0.0}, events=events,
        )
    else:
        result = None
    select_events: list[str] = []
    guide_out = None
    if (
        result is None and request.mode == "model" and request.detect_by == "guide" and model is not None
        and session.state in (TrackingState.SEARCHING, TrackingState.RECOVERING, TrackingState.LOST)
    ):
        edge_model = getattr(session.tracker, "edge_model", None)
        guide = guide_pose_for(model, edge_model, calibration.camera_matrix, (width_px, height_px), request.resting)
        if guide is not None and edge_model is not None:
            guide_pose, saved = guide
            matcher = _guide_matchers.setdefault(request.session_id, GuideMatcher(
                min_score=_settings.ar_guide_min_score, min_support=_settings.ar_guide_min_support,
                confirmations=_settings.ar_guide_confirmations,
            ))
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            locked = matcher.poll(edge_model, guide_pose, gray, hand.mask if hand else None, calibration.camera_matrix)
            if locked is not None:
                session.seed(
                    _guide_detection(
                        edge_model, locked.t_co, calibration.camera_matrix, model.name,
                        guide_lock_confidence(locked.score, session.config.good_confidence),
                    ),
                    select_events, initial_pose=locked.t_co,
                )
                select_events.insert(0, f"[GUIDE] Lined up with the guide view (edge fit {locked.score:.0%}) — locking")
                matcher.reset()
            shown = cv_model_pose_to_threejs(guide_pose)
            guide_out = GuideOut(
                position=Vector3(x=shown.position[0], y=shown.position[1], z=shown.position[2]),
                quaternion=Quaternion(x=shown.quaternion[0], y=shown.quaternion[1], z=shown.quaternion[2], w=shown.quaternion[3]),
                saved=saved,
                score=matcher.last.score if matcher.last else None,
                min_score=_settings.ar_guide_min_score,
            )
    try:
        if result is None and request.select_point and request.mode == "model" and model is not None:
            x, y = request.select_point[:2]
            edge_model = getattr(session.tracker, "edge_model", None)
            picked = select_object_at(
                encoded, x, y, str(model.id), model.name, calibration.camera_matrix, select_events,
                vertices=edge_model.vertices if edge_model is not None else None, resting=request.resting,
            )
            if picked is not None:
                session.seed(picked[0], select_events, initial_pose=picked[1])
        if result is None:
            result = session.step(Frame(frame_bgr, calibration, encoded=encoded, hand_mask=hand.mask if hand else None))
    except PoseServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if select_events:
        result.events[:0] = select_events
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
    t_co = obj.extra.get("t_camera_object") if obj is not None else None
    if result.state == TrackingState.TRACKING and obj is not None:
        for cal in calibrations:
            outline = None
            if cal.method == "pose":
                # The part's exact pixels at the tracked pose (model-based mode only).
                if t_co is None or model is None:
                    continue
                measured = measure_part(frame_bgr, cal, model, t_co, calibration.camera_matrix)
                if measured is None:
                    continue
                value, mask = measured
                rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
                region = [int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1]
                outline = [Vector2(x=x, y=y) for x, y in mask_outline(mask)]
            else:
                if obj.bbox is None:
                    continue
                value = region_brightness(frame_bgr, cal.region, obj.bbox)
                if value is None:
                    continue
                region = list(cal.region.pixels(obj.bbox))
            state, confidence = cal.classify(value)
            part_checks.append(
                PartCheckOut(
                    node_index=cal.node_index,
                    part_name=cal.part_name,
                    state=state,
                    confidence=confidence,
                    brightness=value,
                    region_px=region,
                    measure=cal.feature,
                    region_polygon=outline,
                )
            )

    # Component inspection: the same frame, with the pose tracked for it.
    # Only a confident TRACKING pose is a valid observation; anything else
    # is passed on as the reason the components can't be judged right now.
    inspection = None
    if request.inspection != "off" and model is not None and db is not None:
        pose = t_co if request.mode == "model" else None
        tracking_ok = result.state == TrackingState.TRACKING and not result.monitoring and pose is not None
        if request.mode != "model":
            tracking_reason = "needs a 6DoF pose — use Model-based (CAD) tracking"
        elif result.state != TrackingState.TRACKING:
            tracking_reason = f"tracking {result.state.value.lower()}"
        elif result.monitoring and obj is not None:
            tracking_reason = f"tracking confidence {obj.confidence:.2f} below {session.config.good_confidence:.2f}"
        else:
            tracking_reason = None
        if result.state == TrackingState.TRACKING_OCCLUDED:
            tracking_reason = "the object is covered (hand) — pose held"
        if station_out is not None and station_out.moved:
            tracking_reason = f"the part moved {station_out.shift_px:.1f} px off the station pose — re-align"
        orientation = (obj.extra.get("orientation") if obj is not None else None) or {}
        if tracking_ok and orientation.get("status") == "ambiguous" and "no pose features" not in orientation.get("reason", ""):
            # A symmetric part whose features don't settle which way round it
            # is: judging components on the wrong twin would report the wrong
            # ones missing. Pose uncertainty is not a defect.
            tracking_ok = False
            tracking_reason = f"orientation not confirmed — {orientation.get('reason')}"
        inspection = run_inspection(
            db, model, request.session_id, request.inspection, request.inspection_run, frame_bgr, pose,
            calibration.camera_matrix, tracking_ok, tracking_reason, hand.mask if hand else None,
            request.inspection_focus,
        )

    extra = obj.extra if obj is not None else {}
    quality = None
    if obj is not None and request.mode == "model":
        quality = TrackingQualityOut(
            overlap=extra.get("overlap"),
            flow_support=extra.get("flow_support"),
            occlusion=extra.get("occlusion"),
            held_ms=extra.get("occluded_for_ms"),
            refine_skipped_for_hand=bool(extra.get("refine_skipped_for_hand", False)),
            edge_fit=(extra.get("edge_fit") or {}).get("score"),
            edge_adjusted=bool((extra.get("edge_fit") or {}).get("adjusted", False)),
            orientation_status=(extra.get("orientation") or {}).get("status"),
            orientation_reason=(extra.get("orientation") or {}).get("reason"),
            twin_flips_suppressed=int((extra.get("correction_guard") or {}).get("twin_flips_suppressed", 0)),
            corrections_rejected=int((extra.get("correction_guard") or {}).get("corrections_rejected", 0)),
            orientation_switches=int((extra.get("correction_guard") or {}).get("orientation_switches", 0)),
        )
    hands_out = None
    if _settings.hand_mask_method != "off":
        hands_out = HandsOut(
            available=service is not None,
            sources=hand.sources if hand else [],
            hands=hand.hands if hand else 0,
            persons=hand.persons if hand else 0,
            age_ms=(time.monotonic() - hand.taken_at) * 1000 if hand else None,
            outlines=[[Vector2(x=x, y=y) for x, y in outline] for outline in hand.outlines()] if hand else [],
            error=_hand_segmenter_error or (service.error if service else None),
        )

    k = calibration.camera_matrix
    detection = result.detection
    return ARFrameResponse(
        inspection=inspection,
        tracking_status=_tracking_status(result, session),
        hands=hands_out,
        quality=quality,
        station=station_out,
        guide=guide_out,
        pose_features=_pose_features_out((extra.get("orientation") or {}).get("features") if obj is not None else None),
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
        timings_ms=ARTimings(**result.timings_ms, total=(time.perf_counter() - started) * 1000.0),
        capture_ms=capture_ms,
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
        calibration_note=calibration.note,
        calibration_rms_px=calibration.rms_px,
        distortion_removed_px=calibration.distortion_max_px() if request.mode == "model" else 0.0,
        camera_vertical_fov_deg=calibration.vertical_fov_deg(),
        camera_aspect=calibration.aspect_ratio(),
    )




@router.post("/ar-session/frame", response_model=ARFrameResponse)
def ar_session_frame(request: ARFrameRequest, db: Session = Depends(get_db)) -> ARFrameResponse:
    """
    One camera frame through the session's SEARCHING / INITIALIZING /
    TRACKING / LOST / RECOVERING state machine (ar_session.py). The detector
    (by default in model-based mode the 3D model itself, see
    find_object_by_model; otherwise YOLO by class) only runs while SEARCHING
    or RECOVERING, at most every ar_detect_interval_ms; while TRACKING only
    the tracker runs. Changing mode/class/model/height starts a fresh session.
    The live view streams over /ar-session/stream instead; this is for single
    frames ("Detect one frame") and the replay benchmark.
    """
    started = time.perf_counter()
    session, model = _ar_session_for(request, db)
    try:
        encoded = base64.b64decode(request.image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}") from exc
    return process_ar_frame(request, session, model, encoded, started, db=db)


@router.websocket("/ar-session/stream")
async def ar_session_stream(websocket: WebSocket) -> None:
    """
    Live tracking as a stream: the browser sends every new camera frame as it
    arrives and poses come back as they're computed, instead of one HTTP
    round trip per frame with nothing overlapping.

    Protocol: first a JSON text message with the ARFrameRequest fields
    (without image_base64); then binary messages = 8-byte little-endian
    float64 capture time (ms, the sender's clock) + the JPEG. Each processed
    frame is answered with an ARFrameResponse JSON text message echoing that
    capture time, so the client can predict the pose for "now". Frames that
    arrive while one is being processed are dropped except the newest: the
    tracker always works on the latest image and never falls behind the camera.
    A JSON text message later replaces the configuration.

    The frame source is separate from tracking on purpose: a native camera
    (e.g. a high-speed industrial one) can feed process_ar_frame the same way.
    """
    await websocket.accept()
    latest: dict = {}
    arrived = asyncio.Event()
    closed = False

    async def receive() -> None:
        nonlocal closed
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("text") is not None:
                    latest.pop("frame", None)
                    try:
                        latest["config"] = ARFrameRequest(**{**json.loads(message["text"]), "image_base64": ""})
                    except (ValueError, TypeError) as exc:  # pydantic's ValidationError is a ValueError
                        # Report it: a rejected configuration used to end the stream silently.
                        latest.pop("config", None)
                        await websocket.send_text(json.dumps({"error": f"Invalid tracking configuration: {exc}", "status": 422}))
                elif message.get("bytes"):
                    data = message["bytes"]
                    latest["frame"] = (struct.unpack("<d", data[:8])[0], data[8:])  # newest wins
                    arrived.set()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            closed = True
            arrived.set()

    receiver = asyncio.create_task(receive())
    try:
        while not closed:
            await arrived.wait()
            arrived.clear()
            frame = latest.pop("frame", None)
            config: ARFrameRequest | None = latest.get("config")
            if frame is None or config is None:
                continue
            capture_ms, encoded = frame
            started = time.perf_counter()
            if config.select_point is not None and latest.get("config") is config:
                # A click selects once, on the next frame — not on every frame after.
                latest["config"] = config.model_copy(update={"select_point": None})
            try:
                response = await run_in_threadpool(_stream_step, config, encoded, started, capture_ms)
                await websocket.send_text(response.model_dump_json())
            except HTTPException as exc:
                await websocket.send_text(json.dumps({"error": exc.detail, "status": exc.status_code, "capture_ms": capture_ms}))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        receiver.cancel()


def _stream_step(config: ARFrameRequest, encoded: bytes, started: float, capture_ms: float) -> ARFrameResponse:
    db = SessionLocal()
    try:
        session, model = _ar_session_for(config, db)
        return process_ar_frame(config, session, model, encoded, started, capture_ms, db=db)
    finally:
        db.close()


@router.delete("/ar-session/{session_id}", status_code=204)
def end_ar_session(session_id: str) -> None:
    _ar_sessions.pop(session_id, None)
    _hand_services.pop(session_id, None)
    _guide_matchers.pop(session_id, None)
    end_inspection(session_id)


def _clip_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip()).strip("-")
    if not safe:
        raise HTTPException(status_code=400, detail="Give the clip a name (letters, digits, - and _)")
    return safe


@router.post("/clips/{name}/frames")
def add_clip_frame(name: str, request: ClipFrameRequest) -> dict:
    """Appends one raw camera frame to a recorded clip (replayed by app/workers/replay_clip.py).
    Frame 0 starts the clip afresh."""
    directory = clip_dir(Path(_settings.clips_dir), _clip_name(name))
    frame = decode_frame(request.image_base64)  # validates it's an image
    if request.index == 0 and directory.exists():
        for old in directory.glob("*.jpg"):
            old.unlink()
        (directory / "index.jsonl").unlink(missing_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / f"{request.index:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with (directory / "index.jsonl").open("a") as index:
        index.write(json.dumps({"index": request.index, "t_ms": request.t_ms}) + "\n")
    return {"clip": directory.name, "frames": request.index + 1}


@router.get("/clips", response_model=list[ClipOut])
def list_clips() -> list[ClipOut]:
    out = []
    for directory in sorted(Path(_settings.clips_dir).glob("*/")):
        if (directory / "index.jsonl").exists():
            frames = load_clip(directory)
            out.append(ClipOut(name=directory.name, frames=len(frames), seconds=frames[-1].t_ms / 1000 if frames else 0.0))
    return out


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
