"""
Inspection runtime API (Project.md #41, #63): orchestrates procedure engine +
vision + state detection + QA engine + evidence, but implements none of their
logic itself (Project.md #74) — this module only wires them together and
persists the result.
"""

import base64
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import require_role
from app.core.config import get_settings
from app.core.database import get_db
from app.models.enums import InspectionRunStatus, QAResult, UserRole, ValidationMethod
from app.models.inspection import InspectionRun, InspectionStep, Observation
from app.models.procedure import ProcedureRevision, State
from app.models.step import Step
from app.models.user import User
from app.schemas.inspection import (
    InspectionStartRequest,
    InspectionStartResponse,
    InspectionStepStatus,
    ManualOverrideRequest,
    ManualOverrideResponse,
    ObserveRequest,
    StepValidationResponse,
    ValidationDetail,
)
from app.schemas.vision import Detection
from app.services.audit import audit_log
from app.services.evidence.storage import build_evidence_store
from app.services.qa_engine.engine import QAEngine, QAInput, QAThresholds
from app.services.state_detection.state_engine import ComponentStateRule, StateEstimator, action_expects_presence
from app.services.vision.detector import build_detector, time_inference
from app.api.vision import decode_frame  # reuse the same base64 decode path

_CAN_OVERRIDE = (UserRole.ADMIN, UserRole.QA_INSPECTOR)

router = APIRouter(prefix="/api/inspection", tags=["inspection"])

_settings = get_settings()
_detector = build_detector(_settings.model_path)
_qa_engine = QAEngine(
    QAThresholds(
        detection_confidence_threshold=_settings.qa_detection_confidence_threshold,
        state_confidence_threshold=_settings.qa_state_confidence_threshold,
    )
)
_evidence_store = None  # lazily built — avoids requiring MinIO for pure unit tests


def _get_evidence_store():
    global _evidence_store
    if _evidence_store is None:
        _evidence_store = build_evidence_store()
    return _evidence_store


@router.post("/start", response_model=InspectionStartResponse)
def start_inspection(payload: InspectionStartRequest, db: Session = Depends(get_db)) -> InspectionStartResponse:
    revision = db.get(ProcedureRevision, payload.procedure_revision_id)
    if revision is None:
        raise HTTPException(status_code=404, detail="Procedure revision not found")
    if not revision.steps:
        raise HTTPException(status_code=400, detail="Procedure revision has no steps")

    run = InspectionRun(
        asset_id=payload.asset_id,
        procedure_revision_id=revision.id,
        operator=payload.operator,
        status=InspectionRunStatus.IN_PROGRESS,
    )
    db.add(run)
    db.flush()

    for step in revision.steps:
        db.add(InspectionStep(inspection_run_id=run.id, step_id=step.id, result=QAResult.NOT_EVALUATED))

    db.commit()
    db.refresh(run)

    first_step = revision.steps[0]
    return InspectionStartResponse(inspection_run_id=run.id, first_step_id=first_step.id, total_steps=len(revision.steps))


class InspectionRunOut(BaseModel):
    id: uuid.UUID
    status: InspectionRunStatus
    steps: list[InspectionStepStatus]


@router.get("/{inspection_id}", response_model=InspectionRunOut)
def get_inspection(inspection_id: uuid.UUID, db: Session = Depends(get_db)) -> InspectionRunOut:
    run = db.get(InspectionRun, inspection_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Inspection run not found")
    return InspectionRunOut(
        id=run.id,
        status=run.status,
        steps=[
            InspectionStepStatus(
                step_id=s.step_id,
                step_id_str=s.step.step_id_str,
                title=s.step.title,
                result=s.result,
                confidence=s.confidence,
                manual_override_result=s.manual_override_result,
                manual_override_by=s.manual_override_by,
                manual_override_reason=s.manual_override_reason,
            )
            for s in run.steps
        ],
    )


def _get_inspection_step(db: Session, inspection_id: uuid.UUID, step_id: uuid.UUID) -> InspectionStep:
    inspection_step = (
        db.query(InspectionStep)
        .filter(InspectionStep.inspection_run_id == inspection_id, InspectionStep.step_id == step_id)
        .one_or_none()
    )
    if inspection_step is None:
        raise HTTPException(status_code=404, detail="Step not found in this inspection run")
    return inspection_step


@router.post("/{inspection_id}/step/{step_id}/observe")
def observe_step(
    inspection_id: uuid.UUID, step_id: uuid.UUID, payload: ObserveRequest, db: Session = Depends(get_db)
) -> dict:
    inspection_step = _get_inspection_step(db, inspection_id, step_id)
    step: Step = inspection_step.step

    if payload.detections is not None:
        detections = payload.detections
        frame_ref = None
        raw_bytes = None
    elif payload.image_base64 is not None:
        frame = decode_frame(payload.image_base64)
        detections, _ = time_inference(_detector, frame)
        raw_bytes = base64.b64decode(payload.image_base64)
        frame_ref = None
    else:
        raise HTTPException(status_code=400, detail="Provide either image_base64 or detections")

    # Presence of the target component's class => still in the starting
    # state; absence => the expected (post-action) state. A simplification
    # documented in app/services/state_detection/state_engine.py; swap for a
    # trained per-component state classifier without touching this endpoint.
    rule = ComponentStateRule(
        component_id=str(step.target_component_id),
        class_label=step.target_component.class_label if step.target_component else "",
        present_state_id=str(step.starting_state_id),
        absent_state_id=str(step.expected_state_id),
    )
    estimator = StateEstimator([rule])
    observed_state_id_internal, confidence = estimator.estimate(detections, str(step.target_component_id))
    observed_state = db.get(State, uuid.UUID(observed_state_id_internal))

    if raw_bytes is not None:
        frame_ref = _get_evidence_store().put_frame(inspection_step.id, raw_bytes)

    observation = Observation(
        inspection_step_id=inspection_step.id,
        detected_components={"detections": [d.model_dump() for d in detections]},
        raw_confidence=confidence,
        frame_ref=frame_ref,
    )
    db.add(observation)

    inspection_step.observed_state_id = observed_state.id if observed_state else None
    inspection_step.confidence = confidence
    db.commit()

    return {
        "observed_state": observed_state.state_id_str if observed_state else None,
        "confidence": confidence,
        "detections": [d.model_dump() for d in detections],
        "frame_ref": frame_ref,
    }


@router.post("/{inspection_id}/step/{step_id}/validate", response_model=StepValidationResponse)
def validate_step(inspection_id: uuid.UUID, step_id: uuid.UUID, db: Session = Depends(get_db)) -> StepValidationResponse:
    inspection_step = _get_inspection_step(db, inspection_id, step_id)
    step: Step = inspection_step.step

    latest_observation = (
        db.query(Observation)
        .filter(Observation.inspection_step_id == inspection_step.id)
        .order_by(Observation.id.desc())
        .first()
    )
    detections = (
        [Detection.model_validate(d) for d in latest_observation.detected_components.get("detections", [])]
        if latest_observation
        else []
    )
    required_methods = [r.method for r in step.validation_rules]

    qa_input = QAInput(
        expected_state_id=str(step.expected_state_id),
        observed_state_id=str(inspection_step.observed_state_id) if inspection_step.observed_state_id else None,
        state_confidence=inspection_step.confidence,
        detections=detections,
        required_component_class=step.target_component.class_label if step.target_component else None,
        required_methods=required_methods,
        tracking_stable=True if ValidationMethod.TRACKING in required_methods else None,
        pose_valid=True if ValidationMethod.POSE in required_methods else None,
        expects_presence=action_expects_presence(step.action_type),
    )
    decision = _qa_engine.evaluate(qa_input)

    inspection_step.result = decision.result
    inspection_step.evaluated_at = datetime.now(timezone.utc)
    db.commit()

    evidence_keys = [latest_observation.frame_ref] if latest_observation and latest_observation.frame_ref else []
    expected_state = db.get(State, step.expected_state_id)
    observed_state = db.get(State, inspection_step.observed_state_id) if inspection_step.observed_state_id else None

    return StepValidationResponse(
        stepId=step.step_id_str,
        expectedState=expected_state.state_id_str if expected_state else "",
        observedState=observed_state.state_id_str if observed_state else None,
        confidence=decision.confidence,
        result=decision.result,
        validation=decision.validation,
        evidence=evidence_keys,
        reason=decision.reason,
    )


@router.post("/{inspection_id}/step/{step_id}/override", response_model=ManualOverrideResponse)
def override_step_result(
    inspection_id: uuid.UUID,
    step_id: uuid.UUID,
    payload: ManualOverrideRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(*_CAN_OVERRIDE)),
) -> ManualOverrideResponse:
    """
    Project.md #44: a qualified user (QA Inspector/Admin) can override an
    uncertain result. The original AI/rule `result` is NEVER overwritten —
    it stays exactly as the QA engine produced it; the override is recorded
    alongside it in separate columns, with who did it, when, and why.
    """
    inspection_step = _get_inspection_step(db, inspection_id, step_id)
    original_result = inspection_step.result

    inspection_step.manual_override_result = payload.result
    inspection_step.manual_override_by = current_user.email
    inspection_step.manual_override_reason = payload.reason
    inspection_step.manual_override_at = datetime.now(timezone.utc)

    audit_log.record(
        db,
        user_id=current_user.id,
        action="manual_override",
        entity_type="InspectionStep",
        entity_id=str(inspection_step.id),
        details={
            "original_result": original_result.value,
            "override_result": payload.result.value,
            "reason": payload.reason,
        },
    )
    db.commit()

    return ManualOverrideResponse(
        step_id=step_id,
        original_result=original_result,
        manual_override_result=payload.result,
        manual_override_by=current_user.email,
        manual_override_reason=payload.reason,
    )


@router.post("/{inspection_id}/complete")
def complete_inspection(inspection_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    run = db.get(InspectionRun, inspection_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Inspection run not found")
    run.status = InspectionRunStatus.COMPLETED
    run.completed_at = datetime.now(timezone.utc)
    db.commit()
    return {"id": str(run.id), "status": run.status.value}
