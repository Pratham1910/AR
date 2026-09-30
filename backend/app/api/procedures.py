"""
Procedure authoring/publishing API (Project.md #39, #40, #63).

Accepts the platform-independent procedure JSON (Project.md #6) and persists
it as Procedure -> ProcedureRevision -> States/Steps/ValidationRules. Publishing
a revision (moving draft -> published) is the only way an InspectionRun can
reference it, per Project.md #48/#67 (never silently promote a model/procedure).
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import require_role
from app.core.database import get_db
from app.models.asset import Asset, Component
from app.models.enums import RevisionStatus, UserRole
from app.models.procedure import Procedure, ProcedureRevision, State
from app.models.step import Step, ValidationRule
from app.models.user import User
from app.schemas.procedure import ProcedureDefinition
from app.services.audit import audit_log

router = APIRouter(prefix="/api/procedures", tags=["procedures"])

_CAN_PUBLISH = (UserRole.ADMIN, UserRole.ENGINEER, UserRole.TECHNICAL_AUTHOR)


class ProcedureRevisionOut(BaseModel):
    id: uuid.UUID
    revision_label: str
    status: RevisionStatus
    step_count: int

    model_config = {"from_attributes": True}


class ProcedureOut(BaseModel):
    id: uuid.UUID
    procedure_id_str: str
    title: str
    asset_id: uuid.UUID
    revisions: list[ProcedureRevisionOut]

    model_config = {"from_attributes": True}


def _get_or_create_state(db: Session, state_id_str: str, name: str) -> State:
    state = db.query(State).filter(State.state_id_str == state_id_str).one_or_none()
    if state is None:
        state = State(state_id_str=state_id_str, name=name)
        db.add(state)
        db.flush()
    return state


def _get_or_create_component(db: Session, asset_id: uuid.UUID, component_id_str: str) -> Component:
    component = db.query(Component).filter(Component.component_id_str == component_id_str).one_or_none()
    if component is None:
        # Authoring convenience: a step may reference a component that hasn't
        # been explicitly created yet. class_label defaults to the id itself
        # and should be corrected via PUT /api/assets/{id}/components later.
        component = Component(
            asset_id=asset_id,
            component_id_str=component_id_str,
            name=component_id_str,
            class_label=component_id_str.split("-")[0].lower(),
        )
        db.add(component)
        db.flush()
    return component


@router.get("", response_model=list[ProcedureOut])
def list_procedures(db: Session = Depends(get_db)) -> list[Procedure]:
    return db.query(Procedure).all()


@router.post("", response_model=ProcedureOut, status_code=201)
def create_procedure(definition: ProcedureDefinition, db: Session = Depends(get_db)) -> Procedure:
    asset = db.query(Asset).filter(Asset.id == definition.assetId).one_or_none()
    if asset is None:
        # Also accept a human-friendly asset name/string id for authoring convenience.
        asset = db.query(Asset).filter(Asset.name == definition.assetId).one_or_none()
    if asset is None:
        raise HTTPException(status_code=404, detail=f"Asset {definition.assetId!r} not found")

    procedure = (
        db.query(Procedure).filter(Procedure.procedure_id_str == definition.procedureId).one_or_none()
    )
    if procedure is None:
        procedure = Procedure(procedure_id_str=definition.procedureId, asset_id=asset.id, title=definition.title)
        db.add(procedure)
        db.flush()

    revision = ProcedureRevision(
        procedure_id=procedure.id, revision_label=definition.revision, status=RevisionStatus.DRAFT
    )
    db.add(revision)
    db.flush()

    state_lookup: dict[str, State] = {
        s.id: _get_or_create_state(db, s.id, s.name) for s in definition.states
    }

    for index, step_def in enumerate(definition.steps):
        target_component = _get_or_create_component(db, asset.id, step_def.target.componentId)
        step = Step(
            revision_id=revision.id,
            step_id_str=step_def.id,
            title=step_def.title,
            starting_state_id=state_lookup[step_def.startingState].id,
            expected_state_id=state_lookup[step_def.expectedState].id,
            action_type=step_def.action.type,
            target_component_id=target_component.id,
            order_index=index,
        )
        db.add(step)
        db.flush()
        for method in step_def.validation.methods:
            db.add(ValidationRule(step_id=step.id, method=method, config=step_def.validation.tolerance or {}))

    db.commit()
    db.refresh(procedure)
    return procedure


@router.get("/{procedure_id}", response_model=ProcedureOut)
def get_procedure(procedure_id: uuid.UUID, db: Session = Depends(get_db)) -> Procedure:
    procedure = db.get(Procedure, procedure_id)
    if procedure is None:
        raise HTTPException(status_code=404, detail="Procedure not found")
    return procedure


@router.post("/{procedure_id}/publish", response_model=ProcedureRevisionOut)
def publish_latest_revision(
    procedure_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(*_CAN_PUBLISH)),
) -> ProcedureRevision:
    """Publishing is gated (Project.md #45) — only Engineer/Technical Author/Admin roles may promote a draft."""
    procedure = db.get(Procedure, procedure_id)
    if procedure is None:
        raise HTTPException(status_code=404, detail="Procedure not found")
    draft = (
        db.query(ProcedureRevision)
        .filter(ProcedureRevision.procedure_id == procedure_id, ProcedureRevision.status == RevisionStatus.DRAFT)
        .order_by(ProcedureRevision.created_at.desc())
        .first()
    )
    if draft is None:
        raise HTTPException(status_code=400, detail="No draft revision to publish")
    draft.status = RevisionStatus.PUBLISHED
    audit_log.record(
        db,
        user_id=current_user.id,
        action="publish_procedure_revision",
        entity_type="ProcedureRevision",
        entity_id=str(draft.id),
        details={"procedure_id": str(procedure_id), "revision_label": draft.revision_label},
    )
    db.commit()
    db.refresh(draft)
    return draft
