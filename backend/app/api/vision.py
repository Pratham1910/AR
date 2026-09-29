"""
Vision API (Project.md #10, #63): perception + tracking only. This router
must never return a PASS/FAIL — that is the QA engine's job, reached only
through /api/inspection/*.
"""

import base64

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.schemas.pose import PoseRequest, PoseResponse, Quaternion, Vector3
from app.schemas.vision import DetectRequest, DetectResponse, StateRequest, StateResponse
from app.services.pose.aruco_pose import ArucoPoseEstimator
from app.services.pose.calibration import load_calibration
from app.services.pose.transforms import cv_pose_to_threejs
from app.services.state_detection.state_engine import ComponentStateRule, StateEstimationError, StateEstimator
from app.services.vision.detector import build_detector, time_inference

router = APIRouter(prefix="/api/vision", tags=["vision"])

_settings = get_settings()
_detector = build_detector(_settings.model_path)
_pose_estimator = ArucoPoseEstimator(_settings.aruco_dictionary, _settings.aruco_marker_length_m)


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
        )

    pose = cv_pose_to_threejs(estimate.rvec, estimate.tvec)
    return PoseResponse(
        found=True,
        marker_id=estimate.marker_id,
        position=Vector3(x=pose.position[0], y=pose.position[1], z=pose.position[2]),
        quaternion=Quaternion(x=pose.quaternion[0], y=pose.quaternion[1], z=pose.quaternion[2], w=pose.quaternion[3]),
        reprojection_error_px=estimate.reprojection_error_px,
        calibration_is_approximate=calibration.is_approximate,
        calibration_source=calibration.source,
    )
