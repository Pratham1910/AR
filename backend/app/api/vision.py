"""
Vision API (Project.md #10, #63): perception + tracking only. This router
must never return a PASS/FAIL — that is the QA engine's job, reached only
through /api/inspection/*.
"""

import base64
import time
from pathlib import Path

import uuid

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.models.asset import Component
from app.models.model3d import Model3D
from app.schemas.pose import (
    FeaturePoseRequest,
    FeaturePoseResponse,
    ModelPoseRequest,
    ModelPoseResponse,
    ObjectRegistrationRequest,
    ObjectRegistrationResponse,
    PoseAxes,
    PoseRequest,
    PoseResponse,
    Quaternion,
    RegisterReferenceImageRequest,
    RegisterReferenceImageResponse,
    RotationDeg,
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
from app.services.pose.model_pose_client import ModelPoseClient, PoseServiceUnavailable
from app.services.pose.transforms import (
    cv_model_pose_to_threejs,
    cv_pose_to_threejs,
    euler_angles_deg,
    project_pose_axes,
)
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


@router.post("/model-pose", response_model=ModelPoseResponse)
def estimate_model_pose(request: ModelPoseRequest, db: Session = Depends(get_db)) -> ModelPoseResponse:
    """
    Model-based (CAD) 6DoF pose: MegaPose (pose_service/, WSL2 + CUDA) matches
    the Model3D's own mesh against the frame, so the result is the pose of
    the model itself — not of a marker or a flat photo patch — and needs no
    anchor offset. The first frame (or after tracking is lost) runs YOLO to
    find the object's class, then a full coarse+refine search inside that box;
    later frames only refine from the previous pose.
    """
    try:
        model = db.get(Model3D, uuid.UUID(request.model_id))
    except ValueError:
        model = None
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    class_label = request.class_label
    if class_label is None and model.component_id is not None:
        component = db.get(Component, model.component_id)
        class_label = component.class_label if component else None
    if not class_label:
        raise HTTPException(
            status_code=400,
            detail="This model has no detection class (e.g. 'cup'); set one in the model settings",
        )
    class_label = normalize_class_label(class_label)

    frame = decode_frame(request.image_base64)
    height_px, width_px = frame.shape[:2]
    calibration = load_calibration(_settings.camera_calibration_path, width_px, height_px)

    def response(**kwargs) -> ModelPoseResponse:
        return ModelPoseResponse(
            class_label=class_label,
            calibration_is_approximate=calibration.is_approximate,
            calibration_source=calibration.source,
            camera_vertical_fov_deg=calibration.vertical_fov_deg(),
            camera_aspect=calibration.aspect_ratio(),
            **kwargs,
        )

    label = str(model.id)
    glb_path = Path(_settings.models_3d_dir) / model.storage_key
    try:
        _model_pose_client.ensure_registered(label, glb_path.read_bytes(), model.scale)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"GLB file missing: {glb_path}") from exc
    except PoseServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if request.reset:
        _model_pose_client.reset(label, request.session_id)

    bbox = None
    if not _model_pose_client.has_track(label, request.session_id):
        objects = _get_segmenter().segment(frame, _settings.segmentation_confidence_threshold)
        matches = [o for o in objects if o.class_label == class_label]
        if not matches:
            return response(found=False, mode="no_detection")
        best = max(matches, key=lambda o: o.confidence)
        bbox = [best.bbox.x1, best.bbox.y1, best.bbox.x2, best.bbox.y2]

    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not re-encode frame")
    try:
        result = _model_pose_client.estimate(
            label,
            request.session_id,
            jpeg.tobytes(),
            calibration.camera_matrix,
            bbox,
            _settings.model_pose_min_score,
            _settings.model_pose_track_iterations,
        )
    except PoseServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not result.found:
        return response(found=False, mode=result.mode, score=result.score, bbox=bbox, elapsed_ms=result.elapsed_ms)

    t_co = result.t_camera_object
    pose = cv_model_pose_to_threejs(t_co)
    rvec, _ = cv2.Rodrigues(t_co[:3, :3])
    axes, rotation_deg = _debug_pose_gizmo(rvec.reshape(3), t_co[:3, 3], calibration, axis_length_m=0.05)
    return response(
        found=True,
        mode=result.mode,
        score=result.score,
        bbox=bbox,
        position=Vector3(x=pose.position[0], y=pose.position[1], z=pose.position[2]),
        quaternion=Quaternion(x=pose.quaternion[0], y=pose.quaternion[1], z=pose.quaternion[2], w=pose.quaternion[3]),
        rotation_deg=rotation_deg,
        axes=axes,
        elapsed_ms=result.elapsed_ms,
    )
