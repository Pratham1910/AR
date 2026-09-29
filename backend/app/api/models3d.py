"""
Model3D API (Project.md #20, #63): registers a GLB/glTF file (already placed
under the models_3d_dir served at /static/models — see app/main.py) against
an Asset/Component, so the frontend viewer can discover it by asset instead
of hard-coding a filename.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.asset import Asset, Component
from app.models.model3d import Model3D

router = APIRouter(prefix="/api/models3d", tags=["3d"])


class Model3DCreate(BaseModel):
    asset_id: uuid.UUID
    component_id: uuid.UUID | None = None
    name: str
    storage_key: str  # filename under data/models/, e.g. "bottle.glb"
    format: str = "glb"
    # See app/models/model3d.py — corrects for GLBs not authored at
    # 1 unit = 1 meter. Defaults to 1.0 (assume correctly scaled); if the AR
    # overlay renders comically large/tiny/invisible, this is very likely why.
    scale: float = 1.0


class Model3DOut(BaseModel):
    id: uuid.UUID
    asset_id: uuid.UUID
    component_id: uuid.UUID | None
    name: str
    format: str
    storage_key: str
    scale: float
    url: str

    model_config = {"from_attributes": True}


def _to_out(model: Model3D) -> Model3DOut:
    return Model3DOut(
        id=model.id,
        asset_id=model.asset_id,
        component_id=model.component_id,
        name=model.name,
        format=model.format,
        storage_key=model.storage_key,
        scale=model.scale,
        url=f"/static/models/{model.storage_key}",
    )


@router.get("", response_model=list[Model3DOut])
def list_models(asset_id: uuid.UUID | None = None, db: Session = Depends(get_db)) -> list[Model3DOut]:
    query = db.query(Model3D)
    if asset_id is not None:
        query = query.filter(Model3D.asset_id == asset_id)
    return [_to_out(m) for m in query.all()]


@router.post("", response_model=Model3DOut, status_code=201)
def register_model(payload: Model3DCreate, db: Session = Depends(get_db)) -> Model3DOut:
    if db.get(Asset, payload.asset_id) is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    if payload.component_id is not None and db.get(Component, payload.component_id) is None:
        raise HTTPException(status_code=404, detail="Component not found")

    model = Model3D(**payload.model_dump())
    db.add(model)
    db.commit()
    db.refresh(model)
    return _to_out(model)
