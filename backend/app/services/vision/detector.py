"""
Object detection layer (Project.md #15, #3): answers "what object is present
and where", nothing else. Tracking, pose, state and QA are separate modules —
this file must never make a PASS/FAIL decision.

Two implementations are provided:
  - MockDetector: deterministic, config-driven fake detections. Lets the full
    procedure/QA/evidence pipeline be proven end-to-end (Project.md #78)
    before a custom YOLO model is trained on real assembly footage.
  - YoloDetector: thin wrapper around Ultralytics YOLO. Class list is
    config-driven, not hard-coded (Project.md #15) — it comes from the loaded
    model's own class names.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np

from app.schemas.vision import BoundingBox, Detection


class Detector(ABC):
    model_version: str = "unknown"

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single BGR frame and return raw detections."""


class MockDetector(Detector):
    """
    Returns a fixed, injectable set of detections. Used for local development
    and tests before a trained YOLO model exists, and to drive deterministic
    regression tests (Project.md #54) against recorded video.
    """

    model_version = "mock-detector-v0"

    def __init__(self, fixed_detections: list[Detection] | None = None):
        self._fixed_detections = fixed_detections or []

    def set_detections(self, detections: list[Detection]) -> None:
        self._fixed_detections = detections

    def detect(self, frame: np.ndarray) -> list[Detection]:  # noqa: ARG002 - frame ignored by design
        return list(self._fixed_detections)


class YoloDetector(Detector):
    """Wraps ultralytics.YOLO. Loaded lazily so importing this module never
    requires torch/ultralytics unless a real model path is configured."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.25):
        from ultralytics import YOLO  # local import: heavy, optional dependency

        self._model = YOLO(model_path)
        self._confidence_threshold = confidence_threshold
        self.model_version = model_path

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(frame, conf=self._confidence_threshold, verbose=False)
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls.item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                detections.append(
                    Detection(
                        class_label=names[cls_id],
                        confidence=float(box.conf.item()),
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )
        return detections


def build_detector(model_path: str | None) -> Detector:
    """
    Factory: returns a real YoloDetector when MODEL_PATH is configured,
    otherwise a MockDetector — the API/service layer never branches on this
    itself (Project.md #62, configuration-driven behavior).
    """
    if model_path:
        return YoloDetector(model_path)
    return MockDetector()


def time_inference(detector: Detector, frame: np.ndarray) -> tuple[list[Detection], float]:
    start = time.perf_counter()
    detections = detector.detect(frame)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return detections, elapsed_ms
