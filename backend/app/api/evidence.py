"""Standalone evidence retrieval (Project.md #33, #63) — ad-hoc, outside a step's own observe/validate flow."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.evidence import Evidence
from app.services.evidence.storage import build_evidence_store

router = APIRouter(prefix="/api/evidence", tags=["evidence"])


class EvidenceOut(BaseModel):
    id: uuid.UUID
    inspection_step_id: uuid.UUID
    storage_key: str
    type: str
    url: str

    model_config = {"from_attributes": True}


@router.get("/{evidence_id}", response_model=EvidenceOut)
def get_evidence(evidence_id: uuid.UUID, db: Session = Depends(get_db)) -> EvidenceOut:
    evidence = db.get(Evidence, evidence_id)
    if evidence is None:
        raise HTTPException(status_code=404, detail="Evidence not found")
    url = build_evidence_store().get_frame_url(evidence.storage_key)
    return EvidenceOut(
        id=evidence.id,
        inspection_step_id=evidence.inspection_step_id,
        storage_key=evidence.storage_key,
        type=evidence.type.value,
        url=url,
    )
