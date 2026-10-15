"""
Fixed-station mode: the camera is mounted and the part sits in a fixture, so
its pose is solved once ("Lock pose here") and frozen. No detection and no
pose estimation run while locked: every frame reuses the saved pose, which
removes the frame-to-frame pose noise from component inspection and frees
the GPU.

The one thing that can break a frozen pose is the part (or the camera)
moving. Each frame the object's region is compared with the reference image
taken at lock time by phase correlation (a global shift estimate, ~1 ms on
the downscaled region): a missing screw changes a small patch and barely
moves the estimate, a moved part shifts everything. Above `max_shift_px` the
lock reports "moved" and inspection stops counting those frames until the
part is back or the station is re-aligned.

Locks are saved per model (data/station_locks/<model_id>.json + .png), so a
station keeps its alignment across restarts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REFERENCE_SIDE_PX = 256  # the reference region is compared at this size (longer side)


@dataclass
class StationLock:
    model_id: str
    t_camera_object: np.ndarray  # 4x4, OpenCV camera frame (as the tracker reports it)
    position: tuple[float, float, float]  # renderer space, as the overlay draws it
    quaternion: tuple[float, float, float, float]
    frame_size: tuple[int, int]  # width, height the pose was solved at
    roi: tuple[int, int, int, int]  # x1, y1, x2, y2 of the object region in the frame
    reference: np.ndarray  # grey, downscaled object region at lock time
    created_at: str
    max_shift_px: float = 4.0  # in full-frame pixels

    @staticmethod
    def create(
        model_id: str,
        t_camera_object: np.ndarray,
        position,
        quaternion,
        frame_bgr: np.ndarray,
        bbox: tuple[float, float, float, float],
        max_shift_px: float = 4.0,
    ) -> StationLock:
        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        # A little context around the object: its edges against the fixture
        # are what show a shift best.
        mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
        roi = (
            int(max(0, x1 - mx)), int(max(0, y1 - my)),
            int(min(width, x2 + mx)), int(min(height, y2 + my)),
        )
        return StationLock(
            model_id=model_id,
            t_camera_object=np.asarray(t_camera_object, float),
            position=tuple(float(v) for v in position),
            quaternion=tuple(float(v) for v in quaternion),
            frame_size=(width, height),
            roi=roi,
            reference=_region(frame_bgr, roi),
            created_at=datetime.now(timezone.utc).isoformat(),
            max_shift_px=max_shift_px,
        )

    def shift(self, frame_bgr: np.ndarray) -> tuple[float, float]:
        """(shift in full-frame px, correlation response 0..1) of the object region vs the reference."""
        current = _region(frame_bgr, self.roi)
        if current.shape != self.reference.shape:
            return float("inf"), 0.0
        window = cv2.createHanningWindow(current.shape[::-1], cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(self.reference, current, window)
        scale = max(self.roi[2] - self.roi[0], self.roi[3] - self.roi[1]) / max(self.reference.shape)
        return float(np.hypot(dx, dy) * scale), float(response)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(directory / f"{self.model_id}.png"), np.clip(self.reference, 0, 255).astype(np.uint8))
        (directory / f"{self.model_id}.json").write_text(json.dumps({
            "model_id": self.model_id,
            "t_camera_object": self.t_camera_object.tolist(),
            "position": list(self.position),
            "quaternion": list(self.quaternion),
            "frame_size": list(self.frame_size),
            "roi": list(self.roi),
            "created_at": self.created_at,
            "max_shift_px": self.max_shift_px,
        }, indent=2), encoding="utf-8")

    @staticmethod
    def load(directory: Path, model_id: str) -> StationLock | None:
        meta, image = directory / f"{model_id}.json", directory / f"{model_id}.png"
        if not meta.exists() or not image.exists():
            return None
        data = json.loads(meta.read_text(encoding="utf-8"))
        reference = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if reference is None:
            return None
        return StationLock(
            model_id=data["model_id"],
            t_camera_object=np.asarray(data["t_camera_object"], float),
            position=tuple(data["position"]),
            quaternion=tuple(data["quaternion"]),
            frame_size=tuple(data["frame_size"]),
            roi=tuple(data["roi"]),
            reference=reference.astype(np.float32),
            created_at=data["created_at"],
            max_shift_px=data.get("max_shift_px", 4.0),
        )

    @staticmethod
    def remove(directory: Path, model_id: str) -> None:
        for suffix in (".json", ".png"):
            (directory / f"{model_id}{suffix}").unlink(missing_ok=True)


def _region(frame_bgr: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = roi
    grey = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    scale = REFERENCE_SIDE_PX / max(grey.shape)
    grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return grey.astype(np.float32)
