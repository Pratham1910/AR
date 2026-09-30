"""
Segmentation layer (Project.md #16): actual object *pixel outline*, not just
a bounding box — "Bounding box: PCB occupies rectangle. Segmentation: actual
PCB pixels." Kept as its own module rather than folded into detector.py
because it answers a different question (contour, not just presence/location)
and is used here purely for live visual feedback (Project.md #57's debug-mode
principle: show the operator exactly what the vision layer sees), not for any
QA decision — the QA engine never touches this module.

Uses a COCO-pretrained YOLOv8 segmentation model by default (not a custom one)
because COCO already includes a "bottle" class — no training needed to outline
the demo bottle. Project.md #15 requires the *procedure* detector to be
custom-trainable and config-driven; this module is a separate, generic
"show me what's out there" visualizer and is explicitly allowed to use a
stock pretrained model for that purpose.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.schemas.vision import BoundingBox, SegmentedObject, Vector2


class Segmenter(ABC):
    model_version: str = "unknown"

    @abstractmethod
    def segment(self, frame: np.ndarray, confidence_threshold: float) -> list[SegmentedObject]:
        """Run segmentation on a single BGR frame and return per-object outlines."""

    @property
    @abstractmethod
    def class_names(self) -> list[str]:
        """Every class label this model can ever return — anything else is never detected."""


class MockSegmenter(Segmenter):
    """Fixed/injectable outlines — mirrors detector.MockDetector's role for tests/offline dev."""

    model_version = "mock-segmenter-v0"

    def __init__(self, fixed_objects: list[SegmentedObject] | None = None):
        self._fixed_objects = fixed_objects or []

    def segment(self, frame: np.ndarray, confidence_threshold: float) -> list[SegmentedObject]:  # noqa: ARG002
        return list(self._fixed_objects)

    @property
    def class_names(self) -> list[str]:
        return sorted({o.class_label for o in self._fixed_objects})


class YoloSegmenter(Segmenter):
    """
    Wraps ultralytics YOLO's segmentation task. Loaded lazily so importing
    this module never requires torch/ultralytics or a network fetch unless
    this class is actually instantiated.
    """

    def __init__(self, model_name: str):
        from ultralytics import YOLO  # local import: heavy, optional dependency

        # A standard model name (e.g. "yolov8n-seg.pt") auto-downloads on
        # first use; a local path loads directly. Either way this is the one
        # place that decision is made (Project.md #62).
        self._model = YOLO(model_name)
        self.model_version = model_name

    @property
    def class_names(self) -> list[str]:
        return list(self._model.names.values())

    def segment(self, frame: np.ndarray, confidence_threshold: float) -> list[SegmentedObject]:
        results = self._model.predict(frame, conf=confidence_threshold, verbose=False)
        objects: list[SegmentedObject] = []
        for result in results:
            if result.masks is None:
                continue
            names = result.names
            polygons = result.masks.xy  # list of (N, 2) arrays, already in original-image pixel coords
            for box, polygon in zip(result.boxes, polygons):
                cls_id = int(box.cls.item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                objects.append(
                    SegmentedObject(
                        class_label=names[cls_id],
                        confidence=float(box.conf.item()),
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        polygon=[Vector2(x=float(px), y=float(py)) for px, py in polygon],
                    )
                )
        return objects


def build_segmenter(model_name: str) -> Segmenter:
    return YoloSegmenter(model_name)
