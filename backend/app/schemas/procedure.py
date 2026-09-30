"""
Pydantic models for the platform-independent procedure JSON format
(Project.md #6). This is the authoring/interchange format — the procedure
engine loads it, and it maps onto the Procedure/ProcedureRevision/Step/State
DB tables (app/models) at publish time.
"""

from enum import Enum

from pydantic import BaseModel, Field

from app.models.enums import ActionType, ValidationMethod


class StateDefinition(BaseModel):
    id: str
    name: str


class ActionDefinition(BaseModel):
    type: ActionType


class TargetDefinition(BaseModel):
    componentId: str


class ValidationDefinition(BaseModel):
    methods: list[ValidationMethod]
    tolerance: dict | None = None


class StepDefinition(BaseModel):
    id: str
    title: str
    startingState: str
    action: ActionDefinition
    target: TargetDefinition
    expectedState: str
    validation: ValidationDefinition


class ProcedureDefinition(BaseModel):
    """Root of a procedure JSON file, e.g. data/procedures/pcb-removal-demo.json."""

    procedureId: str
    revision: str
    title: str
    assetId: str
    states: list[StateDefinition]
    steps: list[StepDefinition]

    def state_ids(self) -> set[str]:
        return {s.id for s in self.states}


class CandidateStepStatus(str, Enum):
    """Video-derived candidate steps require human review before becoming a
    production step (Project.md #28, #29) — this status tracks that."""

    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class CandidateStep(BaseModel):
    componentId: str  # which physical component this transition was observed on
    startState: str
    action: str
    endState: str
    status: CandidateStepStatus = CandidateStepStatus.PENDING_REVIEW
    source_video: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
