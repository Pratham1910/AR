"""
Model3D API (Project.md #20, #63): registers a GLB/glTF file (already placed
under the models_3d_dir served at /static/models — see app/main.py) against
an Asset/Component, so the frontend viewer can discover it by asset instead
of hard-coding a filename.
"""

import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.models.asset import Asset, Component
from app.models.model3d import Model3D
from app.services.model3d.glb_inspect import GlbParseError, compute_scale_for_real_height

router = APIRouter(prefix="/api/models3d", tags=["3d"])
_settings = get_settings()

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


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


def _safe_storage_filename(directory: Path, original_filename: str) -> str:
    """
    Sanitizes an uploaded filename (no path traversal, no odd characters) and
    de-duplicates against whatever's already in models_3d_dir, so two
    uploads named the same thing don't silently overwrite each other.
    """
    stem = Path(original_filename).stem or "model"
    suffix = Path(original_filename).suffix or ".glb"
    safe_stem = _SAFE_FILENAME_RE.sub("-", stem).strip("-") or "model"

    candidate = f"{safe_stem}{suffix}"
    counter = 1
    while (directory / candidate).exists():
        candidate = f"{safe_stem}-{counter}{suffix}"
        counter += 1
    return candidate


@router.post("/upload", response_model=Model3DOut, status_code=201)
async def upload_model(
    asset_id: uuid.UUID = Form(...),
    name: str = Form(...),
    file: UploadFile = File(...),
    component_id: uuid.UUID | None = Form(None),
    real_world_height_m: float | None = Form(None),
    db: Session = Depends(get_db),
) -> Model3DOut:
    """
    Uploads a .glb file, saves it under models_3d_dir (served at
    /static/models/*), and registers it against an existing Asset.

    If `real_world_height_m` is given, the model's `scale` (Project.md #20's
    "unit correction", the exact fix bottle.glb needed by hand) is computed
    automatically from the GLB's own mesh bounds — see
    app/services/model3d/glb_inspect.py. Without it, scale defaults to 1.0,
    which is very likely wrong for any GLB not specifically authored at
    1 unit = 1 meter.
    """
    if db.get(Asset, asset_id) is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    if component_id is not None and db.get(Component, component_id) is None:
        raise HTTPException(status_code=404, detail="Component not found")
    if not file.filename or not file.filename.lower().endswith(".glb"):
        raise HTTPException(status_code=400, detail="Only .glb files are supported")

    contents = await file.read()

    scale = 1.0
    if real_world_height_m is not None:
        try:
            scale = compute_scale_for_real_height(contents, real_world_height_m)
        except GlbParseError as exc:
            raise HTTPException(status_code=400, detail=f"Could not read this .glb file: {exc}") from exc

    models_dir = Path(_settings.models_3d_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    storage_key = _safe_storage_filename(models_dir, file.filename)
    (models_dir / storage_key).write_bytes(contents)

    model = Model3D(
        asset_id=asset_id,
        component_id=component_id,
        name=name,
        format="glb",
        storage_key=storage_key,
        scale=scale,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return _to_out(model)
