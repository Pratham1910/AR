"""
Vision schemas. Kept strictly to Perception + Tracking concerns
(Project.md #3) — no PASS/FAIL fields live here, that belongs to the QA
engine's schemas (app/schemas/inspection.py).
"""

from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Vector2(BaseModel):
    x: float
    y: float


class Detection(BaseModel):
    class_label: str
    confidence: float
    bbox: BoundingBox
    tracker_id: int | None = None


class SegmentedObject(BaseModel):
    """A detected object's pixel-space outline (Project.md #16), not just a box."""

    class_label: str
    confidence: float
    bbox: BoundingBox
    polygon: list[Vector2]  # in the captured frame's own pixel space


class SaveFrameRequest(BaseModel):
    """A raw camera frame to keep for offline analysis/calibration."""

    image_base64: str
    label: str  # e.g. "cap-on", "cap-off"


class DetectRequest(BaseModel):
    image_base64: str
    frame_ref: str | None = None


class DetectResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    detections: list[Detection]
    model_version: str
    inference_ms: float


class StateRequest(BaseModel):
    """
    State estimation takes detections, not raw pixels — separation of layers (#3).

    The presence/absence rule is passed explicitly here (class_label +
    present/absent state ids) rather than looked up from a procedure, so this
    endpoint is independently callable/testable; the inspection flow
    (app/api/inspection.py) derives the same rule from the step's own
    validation config and calls the state engine directly.
    """

    detections: list[Detection]
    target_component_id: str
    class_label: str
    present_state_id: str
    absent_state_id: str


class StateResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    state_id: str
    confidence: float
    state_model_version: str


class SegmentRequest(BaseModel):
    image_base64: str
    confidence_threshold: float | None = None  # falls back to Settings.segmentation_confidence_threshold


class SegmentResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    objects: list[SegmentedObject]
    model_version: str
    inference_ms: float


class TrackRequest(BaseModel):
    """
    Project.md #17 (Phase 2): frame-to-frame identity. `session_id` scopes a
    ByteTrack instance to one physical camera/inspection stream — pick any
    stable string for the duration of that stream (e.g. an inspection_run_id
    or a frontend-generated UUID), and reuse it for every frame.
    """

    session_id: str
    image_base64: str


class TrackResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    detections: list[Detection]  # tracker_id populated, stable per physical object across calls
    model_version: str
    inference_ms: float
