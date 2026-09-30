"""Inspection run API schemas — the operator-facing contract (Project.md #63)."""

import uuid

from pydantic import BaseModel

from app.models.enums import QAResult
from app.schemas.vision import Detection


class InspectionStartRequest(BaseModel):
    asset_id: uuid.UUID
    procedure_revision_id: uuid.UUID
    operator: str | None = None


class InspectionStartResponse(BaseModel):
    inspection_run_id: uuid.UUID
    first_step_id: uuid.UUID
    total_steps: int


class ObserveRequest(BaseModel):
    """Frame + detections submitted for one step (image_base64 XOR pre-computed detections)."""

    image_base64: str | None = None
    detections: list[Detection] | None = None


class ValidationDetail(BaseModel):
    objectDetected: bool
    trackingStable: bool | None = None
    poseValid: bool | None = None
    stateMatched: bool


class StepValidationResponse(BaseModel):
    """Mirrors the QA result shape in Project.md #8."""

    stepId: str
    expectedState: str
    observedState: str | None
    confidence: float
    result: QAResult
    validation: ValidationDetail
    evidence: list[str]
    reason: str | None = None


class InspectionStepStatus(BaseModel):
    step_id: uuid.UUID
    step_id_str: str
    title: str
    result: QAResult  # the original AI/rule decision — never overwritten by a manual override (Project.md #44)
    confidence: float | None
    manual_override_result: QAResult | None = None
    manual_override_by: str | None = None
    manual_override_reason: str | None = None


class ManualOverrideRequest(BaseModel):
    """Project.md #44: a qualified user can override an uncertain result — the original AI result is kept, not replaced."""

    result: QAResult
    reason: str


class ManualOverrideResponse(BaseModel):
    step_id: uuid.UUID
    original_result: QAResult
    manual_override_result: QAResult
    manual_override_by: str
    manual_override_reason: str
