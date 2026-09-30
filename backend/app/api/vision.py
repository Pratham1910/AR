"""
Vision API (Project.md #10, #63): perception + tracking only. This router
must never return a PASS/FAIL — that is the QA engine's job, reached only
through /api/inspection/*.
"""

import base64
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.schemas.pose import (
    FeaturePoseRequest,
    FeaturePoseResponse,
    ObjectRegistrationRequest,
    ObjectRegistrationResponse,
    PoseRequest,
    PoseResponse,
    Quaternion,
    RegisterReferenceImageRequest,
    RegisterReferenceImageResponse,
    Vector2,
    Vector3,
)
from app.schemas.vision import (
    DetectRequest,
    DetectResponse,
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
from app.services.pose.transforms import cv_pose_to_threejs
from app.services.state_detection.state_engine import ComponentStateRule, StateEstimationError, StateEstimator
from app.services.tracking.tracker import ObjectTracker
from app.services.vision.detector import build_detector, time_inference
from app.services.vision.segmentation import Segmenter, build_segmenter

router = APIRouter(prefix="/api/vision", tags=["vision"])

_settings = get_settings()
_detector = build_detector(_settings.model_path)
_pose_estimator = ArucoPoseEstimator(_settings.aruco_dictionary, _settings.aruco_marker_length_m)
_segmenter: Segmenter | None = None  # built lazily — first request pays the model download/load cost, not startup
_feature_tracker = FeatureTracker()
_reference_planes: dict[str, ReferencePlane] = {}  # in-memory cache, keyed by asset_id
_object_trackers: dict[str, ObjectTracker] = {}  # one ByteTrack instance per session_id (Project.md #17)


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


def decode_frame(image_base64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(image_base64)
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001 - surfaced as a 400 below
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}") from exc
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image_base64 as an image")
    return frame


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

    matches = [o for o in objects if o.class_label == request.target_class_label]
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
    )
