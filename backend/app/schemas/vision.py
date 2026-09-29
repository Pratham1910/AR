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


class Detection(BaseModel):
    class_label: str
    confidence: float
    bbox: BoundingBox
    tracker_id: int | None = None


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
