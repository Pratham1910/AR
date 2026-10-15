"""
Camera calibration from the app (Phase 8): capture checkerboard frames with
the live camera, compute the intrinsics + lens distortion with OpenCV, and
use them from then on (data/calibration/camera.json, picked up by
load_calibration). The same computation as the command-line worker
(app/workers/calibrate_camera.py).

    1. Print a checkerboard (default 9x6 inner corners, 25 mm squares).
    2. POST /api/calibration/images 15-20 times, holding it at different
       angles, distances and positions across the image (each reply says
       whether the corners were found).
    3. POST /api/calibration/run with the board's geometry and the camera's name.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import get_settings
from app.services.pose.calibration import DEFAULT_CALIBRATION_PATH, CameraCalibration, effective_calibration_path
from app.workers.calibrate_camera import DEFAULT_IMAGES_DIR, calibrate

router = APIRouter(prefix="/api/calibration", tags=["calibration"])
_settings = get_settings()


class CalibrationStatus(BaseModel):
    calibrated: bool
    path: str | None = None
    camera_label: str | None = None
    created_at: str | None = None
    rms_px: float | None = None
    images_used: int | None = None
    image_width: int | None = None
    image_height: int | None = None
    distortion_max_px: float | None = None
    images_captured: int = 0


class CaptureRequest(BaseModel):
    image_base64: str
    pattern_cols: int = 9
    pattern_rows: int = 6


class CaptureResult(BaseModel):
    found: bool  # checkerboard corners detected in this frame (only those frames are kept)
    images_captured: int


class RunRequest(BaseModel):
    pattern_cols: int = 9
    pattern_rows: int = 6
    square_size_m: float = 0.025
    camera_label: str | None = None


def _images() -> list[Path]:
    return sorted(DEFAULT_IMAGES_DIR.glob("*.png")) + sorted(DEFAULT_IMAGES_DIR.glob("*.jpg"))


def status() -> CalibrationStatus:
    path = effective_calibration_path(_settings.camera_calibration_path)
    if path is None or not path.exists():
        return CalibrationStatus(calibrated=False, images_captured=len(_images()))
    c = CameraCalibration.from_file(path)
    return CalibrationStatus(
        calibrated=True, path=str(path), camera_label=c.camera_label, created_at=c.created_at, rms_px=c.rms_px,
        images_used=c.images_used, image_width=c.image_width, image_height=c.image_height,
        distortion_max_px=c.distortion_max_px(), images_captured=len(_images()),
    )


@router.get("", response_model=CalibrationStatus)
def get_calibration() -> CalibrationStatus:
    return status()


@router.post("/images", response_model=CaptureResult)
def capture_image(payload: CaptureRequest) -> CaptureResult:
    """Keep this frame for calibration if the checkerboard's corners are found in it."""
    raw = np.frombuffer(base64.b64decode(payload.image_base64), np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Not an image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, _ = cv2.findChessboardCorners(
        gray, (payload.pattern_cols, payload.pattern_rows), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if found:
        DEFAULT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(DEFAULT_IMAGES_DIR / f"calib_{int(time.time() * 1000)}.png"), image)
    return CaptureResult(found=bool(found), images_captured=len(_images()))


@router.delete("/images", response_model=CalibrationStatus)
def clear_images() -> CalibrationStatus:
    for path in _images():
        path.unlink(missing_ok=True)
    return status()


@router.post("/run", response_model=CalibrationStatus)
def run_calibration(payload: RunRequest) -> CalibrationStatus:
    """Compute the calibration from the captured frames and use it from now on."""
    if len(_images()) < 10:
        raise HTTPException(status_code=400, detail=f"Capture at least 10 checkerboard frames first ({len(_images())} so far).")
    try:
        result = calibrate(DEFAULT_IMAGES_DIR, payload.pattern_cols, payload.pattern_rows, payload.square_size_m,
                           payload.camera_label)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    target = Path(_settings.camera_calibration_path) if _settings.camera_calibration_path else DEFAULT_CALIBRATION_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    result.save(target)
    return status()


@router.delete("", response_model=CalibrationStatus)
def remove_calibration() -> CalibrationStatus:
    """Back to the approximate (field-of-view) intrinsics."""
    path = effective_calibration_path(_settings.camera_calibration_path)
    if path is not None:
        path.unlink(missing_ok=True)
    return status()
