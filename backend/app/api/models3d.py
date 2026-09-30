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
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.models.asset import Asset, Component
from app.models.model3d import Model3D
from app.api.vision import forget_model_pose, normalize_class_label
from app.services.model3d.fbx_convert import BlenderNotFound, FbxConversionError, convert_fbx_to_glb
from app.services.model3d.glb_inspect import GlbParseError, compute_glb_bounds, compute_scale_for_real_height

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
    # If no component_id is given but this is set, a Component is created
    # with this class_label and linked — see _get_or_create_detection_component.
    detection_class_label: str | None = None


class Model3DOut(BaseModel):
    id: uuid.UUID
    asset_id: uuid.UUID
    component_id: uuid.UUID | None
    name: str
    format: str
    storage_key: str
    scale: float
    url: str
    # The linked Component's class_label, if any — what the vision layer
    # should call this object (e.g. "cup", "bottle"). Lets the frontend
    # auto-fill "Object class" for markerless registration instead of
    # leaving a free-typed field that can silently disagree with whichever
    # asset/model is actually selected (a real bug this fixes: switching
    # assets used to leave the previous asset's object class behind).
    component_class_label: str | None = None
    # How tall the model renders in the AR overlay (mesh height x scale), so a
    # wrong scale is visible at a glance (e.g. 20m instead of 0.24m).
    real_height_m: float | None = None
    # See app/models/model3d.py — the model's own local transform relative to
    # the tracked reference plane (marker or feature-tracking target).
    # Defaults to zero offset/identity rotation (model planted directly at
    # the tracked pose), which is only correct by coincidence.
    anchor_offset_x: float = 0.0
    anchor_offset_y: float = 0.0
    anchor_offset_z: float = 0.0
    anchor_rotation_x: float = 0.0
    anchor_rotation_y: float = 0.0
    anchor_rotation_z: float = 0.0
    anchor_rotation_w: float = 1.0

    model_config = {"from_attributes": True}


class Model3DAnchorUpdate(BaseModel):
    anchor_offset_x: float
    anchor_offset_y: float
    anchor_offset_z: float
    anchor_rotation_x: float
    anchor_rotation_y: float
    anchor_rotation_z: float
    anchor_rotation_w: float


def _glb_path(model: Model3D) -> Path:
    return Path(_settings.models_3d_dir) / model.storage_key


def _to_out(model: Model3D, db: Session) -> Model3DOut:
    component_class_label = None
    if model.component_id is not None:
        component = db.get(Component, model.component_id)
        component_class_label = component.class_label if component else None
    try:
        real_height_m = compute_glb_bounds(_glb_path(model).read_bytes()).height * model.scale
    except (OSError, GlbParseError):
        real_height_m = None
    return Model3DOut(
        real_height_m=real_height_m,
        id=model.id,
        asset_id=model.asset_id,
        component_id=model.component_id,
        name=model.name,
        format=model.format,
        storage_key=model.storage_key,
        scale=model.scale,
        url=f"/static/models/{model.storage_key}",
        component_class_label=component_class_label,
        anchor_offset_x=model.anchor_offset_x,
        anchor_offset_y=model.anchor_offset_y,
        anchor_offset_z=model.anchor_offset_z,
        anchor_rotation_x=model.anchor_rotation_x,
        anchor_rotation_y=model.anchor_rotation_y,
        anchor_rotation_z=model.anchor_rotation_z,
        anchor_rotation_w=model.anchor_rotation_w,
    )


def _get_or_create_detection_component(db: Session, asset_id: uuid.UUID, class_label: str, name: str) -> Component:
    """
    One Component per (asset, class_label) — reuses an existing one so
    uploading a second model for the same object doesn't create duplicates.
    """
    existing = (
        db.query(Component)
        .filter(Component.asset_id == asset_id, Component.class_label == class_label)
        .one_or_none()
    )
    if existing is not None:
        return existing

    component = Component(
        asset_id=asset_id,
        component_id_str=f"{class_label.upper()}-{uuid.uuid4().hex[:8]}",
        name=name,
        class_label=class_label,
    )
    db.add(component)
    db.flush()
    return component


@router.get("", response_model=list[Model3DOut])
def list_models(asset_id: uuid.UUID | None = None, db: Session = Depends(get_db)) -> list[Model3DOut]:
    query = db.query(Model3D)
    if asset_id is not None:
        query = query.filter(Model3D.asset_id == asset_id)
    # Newest first: the frontend picks models[0] when a user hasn't chosen one
    # explicitly, and defaulting to the oldest upload was a recurring source
    # of "which model is actually showing" confusion.
    query = query.order_by(Model3D.created_at.desc())
    return [_to_out(m, db) for m in query.all()]


@router.post("", response_model=Model3DOut, status_code=201)
def register_model(payload: Model3DCreate, db: Session = Depends(get_db)) -> Model3DOut:
    if db.get(Asset, payload.asset_id) is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    if payload.component_id is not None and db.get(Component, payload.component_id) is None:
        raise HTTPException(status_code=404, detail="Component not found")

    component_id = payload.component_id
    if component_id is None and payload.detection_class_label:
        component_id = _get_or_create_detection_component(
            db, payload.asset_id, normalize_class_label(payload.detection_class_label), payload.name
        ).id

    model = Model3D(
        asset_id=payload.asset_id,
        component_id=component_id,
        name=payload.name,
        storage_key=payload.storage_key,
        format=payload.format,
        scale=payload.scale,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return _to_out(model, db)


@router.delete("/{model_id}", status_code=204)
def delete_model(model_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    """
    Removes a model. Its .glb (and original .fbx, if any) is MOVED to
    models_3d_dir/.deleted/ rather than erased, so a mistaken delete can be
    undone by moving the file back and re-uploading/registering it. A file
    still used by another model row is left in place.
    """
    model = db.get(Model3D, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    storage_key = model.storage_key
    db.delete(model)
    db.commit()

    still_used = db.query(Model3D).filter(Model3D.storage_key == storage_key).count() > 0
    if not still_used:
        models_dir = Path(_settings.models_3d_dir)
        trash = models_dir / ".deleted"
        for path in (models_dir / storage_key, (models_dir / storage_key).with_suffix(".fbx")):
            if path.exists():
                trash.mkdir(exist_ok=True)
                target = trash / path.name
                counter = 1
                while target.exists():
                    target = trash / f"{path.stem}-{counter}{path.suffix}"
                    counter += 1
                path.replace(target)
    forget_model_pose(str(model_id))


class Model3DSettingsUpdate(BaseModel):
    real_world_height_m: float | None = None  # recomputes scale from the GLB's own bounds
    detection_class_label: str | None = None  # must be a class the detector knows


@router.patch("/{model_id}", response_model=Model3DOut)
def update_model_settings(model_id: uuid.UUID, payload: Model3DSettingsUpdate, db: Session = Depends(get_db)) -> Model3DOut:
    """Fix an uploaded model's real-world size or detection class in place —
    the two settings that make a model silently fail to overlay if wrong."""
    model = db.get(Model3D, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    if payload.real_world_height_m is not None:
        if payload.real_world_height_m <= 0:
            raise HTTPException(status_code=400, detail="Real height must be positive")
        try:
            model.scale = compute_scale_for_real_height(_glb_path(model).read_bytes(), payload.real_world_height_m)
        except (OSError, GlbParseError) as exc:
            raise HTTPException(status_code=400, detail=f"Could not read this model's .glb: {exc}") from exc
    if payload.detection_class_label is not None:
        label = normalize_class_label(payload.detection_class_label)
        model.component_id = _get_or_create_detection_component(db, model.asset_id, label, model.name).id
    db.commit()
    db.refresh(model)
    return _to_out(model, db)


@router.patch("/{model_id}/anchor", response_model=Model3DOut)
def update_anchor(model_id: uuid.UUID, payload: Model3DAnchorUpdate, db: Session = Depends(get_db)) -> Model3DOut:
    """
    Saves the model's calibrated anchor offset (see app/models/model3d.py) —
    called once the user has nudged the AR overlay into visual alignment
    with the physical object, so the same offset is reused on every future
    tracking session instead of re-calibrating from scratch each time.
    """
    model = db.get(Model3D, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    model.anchor_offset_x = payload.anchor_offset_x
    model.anchor_offset_y = payload.anchor_offset_y
    model.anchor_offset_z = payload.anchor_offset_z
    model.anchor_rotation_x = payload.anchor_rotation_x
    model.anchor_rotation_y = payload.anchor_rotation_y
    model.anchor_rotation_z = payload.anchor_rotation_z
    model.anchor_rotation_w = payload.anchor_rotation_w
    db.commit()
    db.refresh(model)
    return _to_out(model, db)


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
    detection_class_label: str | None = Form(
        None, description="e.g. 'cup', 'bottle' — a COCO class the markerless-detection endpoint can look for"
    ),
    db: Session = Depends(get_db),
) -> Model3DOut:
    """
    Uploads a .glb or .fbx file, saves it under models_3d_dir (served at
    /static/models/*), and registers it against an existing Asset.

    An .fbx is converted to .glb with headless Blender first (see
    app/services/model3d/fbx_convert.py) — everything downstream only speaks
    GLB. The original .fbx is kept next to it with the same name stem.
    Blender applies the FBX's own unit scale, so an FBX authored in real
    units comes out in meters even without `real_world_height_m`.

    If `real_world_height_m` is given, the model's `scale` (Project.md #20's
    "unit correction", the exact fix bottle.glb needed by hand) is computed
    automatically from the GLB's own mesh bounds — see
    app/services/model3d/glb_inspect.py. Without it, scale defaults to 1.0,
    which is very likely wrong for any GLB not specifically authored at
    1 unit = 1 meter.

    If `detection_class_label` is given (and no explicit `component_id`), a
    Component carrying that class_label is created/reused and linked, so the
    frontend can auto-fill markerless registration's "Object class" from
    whichever asset is selected instead of a free-typed field that can
    silently disagree with it.
    """
    if db.get(Asset, asset_id) is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    if component_id is not None and db.get(Component, component_id) is None:
        raise HTTPException(status_code=404, detail="Component not found")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".glb", ".fbx"):
        raise HTTPException(status_code=400, detail="Only .glb and .fbx files are supported")
    # A GLB's units are whatever the exporter used (often not meters), so
    # without a real height the overlay renders absurdly sized — the model is
    # "detected" but never visibly overlaps. FBX carries its own units.
    if suffix == ".glb" and real_world_height_m is None:
        raise HTTPException(
            status_code=400,
            detail="Give the object's real height (m) — a .glb's own units are rarely meters",
        )
    if detection_class_label:
        detection_class_label = normalize_class_label(detection_class_label)

    contents = await file.read()
    source_fbx: bytes | None = None
    if suffix == ".fbx":
        source_fbx = contents
        try:
            contents = await run_in_threadpool(convert_fbx_to_glb, source_fbx, _settings.blender_path)
        except BlenderNotFound as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except FbxConversionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    scale = 1.0
    if real_world_height_m is not None:
        try:
            scale = compute_scale_for_real_height(contents, real_world_height_m)
        except GlbParseError as exc:
            raise HTTPException(status_code=400, detail=f"Could not read this .glb file: {exc}") from exc

    if component_id is None and detection_class_label:
        component_id = _get_or_create_detection_component(db, asset_id, detection_class_label, name).id

    models_dir = Path(_settings.models_3d_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    storage_key = _safe_storage_filename(models_dir, str(Path(file.filename).with_suffix(".glb")))
    (models_dir / storage_key).write_bytes(contents)
    if source_fbx is not None:
        (models_dir / Path(storage_key).with_suffix(".fbx")).write_bytes(source_fbx)

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
    return _to_out(model, db)
