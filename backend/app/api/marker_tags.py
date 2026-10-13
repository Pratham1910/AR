"""
ArUco product tags API (docs/marker-tags.md): four printed tags per product,
one per corner, and the two ways a camera uses them.

    mapping        /sets, /suggest, /config         tag id -> product corner (marker_tags table)
    printing       /markers/{id}.png, /sets/{id}/sheet.png
    product scan   /identify   any ONE mapped tag -> product -> its 3D model / procedure
    tracking       /track      all FOUR tags -> region they enclose -> YOLO finds the
                               object inside that region only -> tracked over frames

Like every router here this only wires services together: tag logic in
app/services/markers, the ArUco detector and the YOLO segmenter from
app/api/vision (the same ones /pose and the AR session use). It reports an
identity and an object, never a QA PASS/FAIL, and it does not change any
/api/vision endpoint.
"""

import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from app.api.vision import decode_frame, detect_aruco_markers, segment_objects
from app.core.config import get_settings
from app.core.database import get_db
from app.models.asset import Asset, Component
from app.models.enums import MarkerCorner, RevisionStatus
from app.models.marker_tag import MarkerTag
from app.models.model3d import Model3D
from app.models.procedure import Procedure
from app.schemas.marker_tag import (
    CornerStatusOut,
    DetectedObjectOut,
    FrameRequest,
    IdentifyResponse,
    ProductModelOut,
    ProductOut,
    ProductProcedureOut,
    SuggestedIdsOut,
    TagConfigOut,
    TagOut,
    TagSetIn,
    TagSetOut,
    TrackResponse,
)
from app.schemas.vision import Vector2
from app.services.audit import audit_log
from app.services.markers.geometry import QUAD_ORDER
from app.services.markers.identification import LookupStatus, Registry, SetStatus, TagAssignment, lookup_by_any_tag
from app.services.markers.mapping import MappingError, suggest_free_ids, validate_tag_set
from app.services.markers.region_detection import RegionDetection, RegionObject, detect_in_region
from app.services.markers.tags import CORNER_LABELS, TagSighting, dictionary_size, read_tags, render_sheet_png, render_tag_png
from app.services.markers.tracking import FourTagTracker, Located, ProductLookupSession, TrackOutcome, TrackState
from app.services.pose.aruco_pose import ARUCO_DICTIONARIES

router = APIRouter(prefix="/api/marker-tags", tags=["marker-tags"])
log = logging.getLogger("uvicorn.error").getChild("marker_tags")  # shown in the server log, like ar_session.py

_settings = get_settings()
_MAX_SESSIONS = 16  # per kind; oldest-used dropped first (in memory, like the AR sessions)
_trackers: dict[str, FourTagTracker] = {}
_lookups: dict[str, ProductLookupSession] = {}


def _dictionary() -> str:
    return _settings.aruco_dictionary


def _all_tags(db: Session) -> list[MarkerTag]:
    """Every mapped tag of the dictionary in use, with its product loaded."""
    return (
        db.query(MarkerTag)
        .options(joinedload(MarkerTag.asset))
        .filter(MarkerTag.dictionary == _dictionary())
        .order_by(MarkerTag.marker_id)
        .all()
    )


def _registry(tags: list[MarkerTag]) -> Registry:
    return {t.marker_id: TagAssignment(str(t.asset_id), MarkerCorner(t.corner)) for t in tags}


# --- What a product offers ----------------------------------------------------


def _procedure_file(model_id: uuid.UUID) -> Path:
    # Same location app/api/models3d.py stores attached procedures in.
    return Path(_settings.procedures_dir) / f"{model_id}.procedure.json"


def _procedure_title(model_id: uuid.UUID) -> str | None:
    try:
        package = json.loads(_procedure_file(model_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    procedure = package.get("procedure", {}) if isinstance(package, dict) else {}
    return procedure.get("title") or procedure.get("name") or "Procedure"


def _model_out(model: Model3D) -> ProductModelOut:
    return ProductModelOut(id=model.id, name=model.name, url=f"/static/models/{model.storage_key}", scale=model.scale)


def _models(db: Session, asset_id: uuid.UUID) -> list[Model3D]:
    return db.query(Model3D).filter(Model3D.asset_id == asset_id).order_by(Model3D.created_at.desc()).all()


def _detection_class(db: Session, models: list[Model3D]) -> str | None:
    """The detection class set on the product's newest model that has one
    (upload / Model settings store it as the linked Component's class_label)."""
    for model in models:
        if model.component_id is not None:
            component = db.get(Component, model.component_id)
            if component is not None and component.class_label:
                return component.class_label
    return None


def _product(db: Session, asset: Asset) -> ProductOut:
    models = _models(db, asset.id)
    procedure_model = next((m for m in models if _procedure_file(m.id).is_file()), None)
    procedures = db.query(Procedure).filter(Procedure.asset_id == asset.id).order_by(Procedure.procedure_id_str).all()
    procedures_out = []
    for p in procedures:
        published = [r for r in sorted(p.revisions, key=lambda r: r.created_at, reverse=True) if r.status == RevisionStatus.PUBLISHED]
        procedures_out.append(
            ProductProcedureOut(
                id=p.id,
                procedure_id_str=p.procedure_id_str,
                title=p.title,
                published_revision_id=published[0].id if published else None,
                step_count=published[0].step_count if published else 0,
            )
        )
    return ProductOut(
        asset_id=asset.id,
        asset_name=asset.name,
        description=asset.description,
        detection_class=_detection_class(db, models),
        model=_model_out(models[0]) if models else None,
        procedure_model=_model_out(procedure_model) if procedure_model else None,
        procedure_title=_procedure_title(procedure_model.id) if procedure_model else None,
        procedures=procedures_out,
    )


# --- Mapping --------------------------------------------------------------------


def _set_out(db: Session, asset: Asset, ids: dict[str, int]) -> TagSetOut:
    models = _models(db, asset.id)
    return TagSetOut(
        asset_id=asset.id,
        asset_name=asset.name,
        dictionary=_dictionary(),
        detection_class=_detection_class(db, models),
        model_name=models[0].name if models else None,
        has_model_procedure=any(_procedure_file(m.id).is_file() for m in models),
        procedure_count=db.query(Procedure).filter(Procedure.asset_id == asset.id).count(),
        **ids,
    )


@router.get("/config", response_model=TagConfigOut)
def get_config(db: Session = Depends(get_db)) -> TagConfigOut:
    size = dictionary_size(_dictionary())
    used = [t.marker_id for t in _all_tags(db)]
    return TagConfigOut(
        dictionary=_dictionary(),
        dictionary_size=size,
        supported_dictionaries=list(ARUCO_DICTIONARIES),
        used_ids=used,
        free_ids=size - len(used),
        identify_frames=_settings.marker_tag_identify_frames,
        confirm_frames=_settings.marker_tag_confirm_frames,
        hold_ms=_settings.marker_tag_hold_ms,
    )


@router.get("/sets", response_model=list[TagSetOut])
def list_sets(db: Session = Depends(get_db)) -> list[TagSetOut]:
    by_asset: dict[uuid.UUID, tuple[Asset, dict[str, int]]] = {}
    for tag in _all_tags(db):
        by_asset.setdefault(tag.asset_id, (tag.asset, {}))[1][tag.corner] = tag.marker_id
    # Sets are written whole; rows edited by hand into a partial set are not
    # a usable mapping and are left out rather than half-shown.
    complete = [(a, ids) for a, ids in by_asset.values() if len(ids) == len(MarkerCorner)]
    return [_set_out(db, a, ids) for a, ids in sorted(complete, key=lambda pair: pair[0].name.lower())]


@router.get("/suggest", response_model=SuggestedIdsOut)
def suggest(asset_id: uuid.UUID | None = None, db: Session = Depends(get_db)) -> SuggestedIdsOut:
    """Four unused ids. A product's own current ids count as free for it, so
    re-suggesting for it does not use up four more."""
    taken = {t.marker_id for t in _all_tags(db) if t.asset_id != asset_id}
    ids = suggest_free_ids(taken, dictionary_size(_dictionary()))
    if len(ids) < len(MarkerCorner):
        raise HTTPException(
            status_code=409,
            detail=f"{_dictionary()} has only {len(ids)} unused id(s) left; a product needs four. "
            "Remove a mapping, or set ARUCO_DICTIONARY to a larger dictionary (and re-map).",
        )
    return SuggestedIdsOut(dictionary=_dictionary(), ids=ids)


@router.put("/sets/{asset_id}", response_model=TagSetOut)
def put_set(asset_id: uuid.UUID, payload: TagSetIn, db: Session = Depends(get_db)) -> TagSetOut:
    """Assigns or replaces a product's four tags — all four or nothing."""
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Product (asset) not found")
    requested = {corner: getattr(payload, corner.value) for corner in MarkerCorner}
    existing = _all_tags(db)
    try:
        validate_tag_set(
            requested,
            str(asset_id),
            {t.marker_id: (str(t.asset_id), t.asset.name) for t in existing},
            dictionary_size(_dictionary()),
        )
    except MappingError as exc:
        raise HTTPException(status_code=409 if exc.conflict else 400, detail=str(exc)) from exc

    previous = {t.corner: t.marker_id for t in existing if t.asset_id == asset_id}
    for tag in existing:
        if tag.asset_id == asset_id:
            db.delete(tag)
    db.flush()  # free this product's old ids first: it may be swapping them between corners
    for corner, marker_id in requested.items():
        db.add(MarkerTag(asset_id=asset_id, dictionary=_dictionary(), marker_id=marker_id, corner=corner.value))
    audit_log.record(
        db,
        user_id=None,  # unauthenticated, like the asset and model endpoints beside it
        action="set_marker_tags",
        entity_type="Asset",
        entity_id=str(asset_id),
        details={"dictionary": _dictionary(), "previous": previous, "tags": {c.value: i for c, i in requested.items()}},
    )
    try:
        db.commit()
    except IntegrityError as exc:  # two saves racing for one id: the unique constraints decide
        db.rollback()
        raise HTTPException(status_code=409, detail="One of these tag ids was just assigned to another product.") from exc
    log.info("marker tags for %s set to %s", asset.name, {c.value: i for c, i in requested.items()})
    return _set_out(db, asset, {c.value: i for c, i in requested.items()})


@router.delete("/sets/{asset_id}", status_code=204)
def delete_set(asset_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    tags = [t for t in _all_tags(db) if t.asset_id == asset_id]
    if not tags:
        raise HTTPException(status_code=404, detail="This product has no tags assigned")
    for tag in tags:
        db.delete(tag)
    audit_log.record(
        db,
        user_id=None,
        action="delete_marker_tags",
        entity_type="Asset",
        entity_id=str(asset_id),
        details={"dictionary": _dictionary(), "tags": {t.corner: t.marker_id for t in tags}},
    )
    db.commit()


# --- Printing ---------------------------------------------------------------------


def _png(content: bytes, filename: str, download: bool) -> Response:
    disposition = "attachment" if download else "inline"
    return Response(content=content, media_type="image/png", headers={"Content-Disposition": f'{disposition}; filename="{filename}"'})


@router.get("/markers/{marker_id}.png")
def tag_png(
    marker_id: int,
    size_px: int = Query(400, ge=100, le=2000),
    download: bool = False,
    db: Session = Depends(get_db),
) -> Response:
    """One printable tag. If it is mapped, the caption names its product and corner."""
    if not 0 <= marker_id < dictionary_size(_dictionary()):
        raise HTTPException(status_code=404, detail=f"{_dictionary()} has no tag id {marker_id}")
    tag = next((t for t in _all_tags(db) if t.marker_id == marker_id), None)
    caption = f"{_dictionary()}  id {marker_id}"
    if tag is not None:
        caption = f"{tag.asset.name}  {CORNER_LABELS[MarkerCorner(tag.corner)]}  id {marker_id}"
    return _png(render_tag_png(_dictionary(), marker_id, size_px, caption), f"aruco_{_dictionary()}_id{marker_id}.png", download)


@router.get("/sets/{asset_id}/sheet.png")
def sheet_png(
    asset_id: uuid.UUID,
    size_px: int = Query(400, ge=100, le=1500),
    download: bool = False,
    db: Session = Depends(get_db),
) -> Response:
    """A product's four tags on one page, each in its corner position."""
    tags = {MarkerCorner(t.corner): t for t in _all_tags(db) if t.asset_id == asset_id}
    if len(tags) != len(MarkerCorner):
        raise HTTPException(status_code=404, detail="This product has no tags assigned")
    name = tags[MarkerCorner.TOP_LEFT].asset.name
    image = render_sheet_png(_dictionary(), {c: t.marker_id for c, t in tags.items()}, size_px, name)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name)
    return _png(image, f"aruco_tags_{safe}.png", download)


# --- Shared by both scans -----------------------------------------------------------


def _session(store: dict, session_id: str, factory):
    session = store.pop(session_id, None) or factory()
    store[session_id] = session
    while len(store) > _MAX_SESSIONS:
        store.pop(next(iter(store)))
    return session


def _tag_out(s: TagSighting, by_id: dict[int, MarkerTag]) -> TagOut:
    tag = by_id.get(s.marker_id)
    return TagOut(
        marker_id=s.marker_id,
        corners_px=[Vector2(x=float(x), y=float(y)) for x, y in s.corners],
        asset_id=tag.asset_id if tag else None,
        asset_name=tag.asset.name if tag else None,
        corner=tag.corner if tag else None,
        estimated=s.estimated,
    )


def _object_out(r: RegionObject, held: bool = False) -> DetectedObjectOut:
    b = r.obj.bbox
    return DetectedObjectOut(
        class_label=r.obj.class_label,
        confidence=r.obj.confidence,
        bbox=[b.x1, b.y1, b.x2, b.y2],
        polygon=r.obj.polygon,
        inside=r.inside,
        held=held,
    )


# --- Product scan: any one tag ---------------------------------------------------------


@router.post("/identify", response_model=IdentifyResponse)
def identify(request: FrameRequest, db: Session = Depends(get_db)) -> IdentifyResponse:
    """
    Single-tag product scan: names the product that any one mapped tag in
    view belongs to, so its 3D model and procedure can be opened. A lookup —
    no layout check, no object detection — so it is not evidence the product
    is physically there; that is /track's job, which shares no state with it.
    Unmapped tags name nothing, and tags of two products are reported as
    ambiguous rather than picked between.
    """
    frame = decode_frame(request.image_base64)
    height, width = frame.shape[:2]
    sightings, _ = read_tags(detect_aruco_markers, frame)
    mapped = _all_tags(db)
    by_id = {t.marker_id: t for t in mapped}
    names = {str(t.asset_id): t.asset.name for t in mapped}
    lookup = lookup_by_any_tag(sightings, _registry(mapped))

    session: ProductLookupSession = _session(
        _lookups, request.session_id, lambda: ProductLookupSession(_settings.marker_tag_identify_frames)
    )
    streak = session.record(lookup.asset_id if lookup.status is LookupStatus.FOUND else None)
    product = None
    name = names.get(lookup.asset_id) if lookup.asset_id else None

    if lookup.status is LookupStatus.NO_MARKERS:
        state, message = "NO_MARKERS", "No tag in view — show any one of the product's tags to the camera."
    elif lookup.status is LookupStatus.UNKNOWN_MARKERS:
        state = "UNKNOWN_MARKERS"
        message = f"Tag id(s) {lookup.unknown_ids} are not assigned to any product (ArUco Tags tab)."
    elif lookup.status is LookupStatus.AMBIGUOUS:
        state = "AMBIGUOUS"
        message = f"Tags of different products in view ({', '.join(names[a] for a in lookup.asset_ids)}) — show one product's tags."
    elif not session.identified:
        state = "IDENTIFYING"
        message = f"Tag {lookup.matched_ids[0]} belongs to {name} — confirming ({streak}/{session.required})."
    else:
        state = "IDENTIFIED"
        product = _product(db, db.get(Asset, uuid.UUID(lookup.asset_id)))
        message = f"{name} — identified by tag {', '.join(map(str, lookup.matched_ids))}."

    return IdentifyResponse(
        state=state,
        message=message,
        identified=product is not None,
        tags=[_tag_out(s, by_id) for s in sightings],
        candidate_asset_id=uuid.UUID(lookup.asset_id) if lookup.asset_id else None,
        candidate_asset_name=name,
        stable_frames=streak,
        required_frames=session.required,
        product=product,
        frame_width=width,
        frame_height=height,
    )


@router.delete("/identify/{session_id}", status_code=204)
def end_identify(session_id: str) -> None:
    _lookups.pop(session_id, None)


# --- Four-tag tracking -------------------------------------------------------------------


def _corner_list(corners) -> str:
    return ", ".join(CORNER_LABELS[c].lower() for c in corners)


def _track_message(located: Located, outcome: TrackOutcome, names: dict[str, str], expected: str | None) -> str:
    found = located.found
    name = names.get(found.asset_id or "", "the product")
    state = outcome.state
    if state is TrackState.NO_MARKERS:
        return "No tags in view — show the product with all four corner tags."
    if state is TrackState.UNKNOWN_MARKERS:
        return f"Tag id(s) {found.unknown_ids} are not assigned to any product (ArUco Tags tab)."
    if state is TrackState.DUPLICATE_MARKERS:
        return f"Tag id(s) {found.duplicate_ids} appear more than once — remove the extra copy."
    if state is TrackState.INCOMPLETE:
        return f"{name}: {4 - len(found.missing)} of 4 tags in view — missing {_corner_list(found.missing)}."
    if state is TrackState.MULTIPLE_PRODUCTS:
        return f"More than one complete tag set in view ({', '.join(names.get(a, a) for a in found.other_asset_ids)}) — show one product at a time."
    if state is TrackState.MIXED_PRODUCTS:
        if found.problem:
            return f"{name}: {found.problem}."
        others = ", ".join(names.get(a, a) for a in found.other_asset_ids)
        return f"Tags of different products in view ({name}, {others}); {name} is missing {_corner_list(found.missing)}."
    if state is TrackState.LAYOUT_INVALID:
        return f"{name}: all four tags in view, but {found.problem}."
    covered = f" ({_corner_list(located.estimated)} tag covered — position estimated)" if located.estimated else ""
    looking_for = f"a {expected}" if expected else "an object"
    if state is TrackState.OBJECT_NOT_FOUND:
        return f"{name}: tags OK{covered}, but no object detected inside them."
    if state is TrackState.WRONG_OBJECT:
        seen = outcome.obj.obj.class_label if outcome.obj else "something else"
        return f"{name}: expected {looking_for} inside the tags, but the detector sees a {seen}."
    label = outcome.obj.obj.class_label if outcome.obj else "object"
    if state is TrackState.VERIFYING:
        return f"{name}: {label} found inside the tags{covered} — hold still."
    held = " (object momentarily not detected — holding)" if outcome.object_held else ""
    return f"Tracking {name}: {label} inside the four tags{covered}{held}."


def _corner_status(located: Located, mapped: list[MarkerTag]) -> list[CornerStatusOut]:
    """The candidate product's four registered tags: read, estimated, or missing."""
    found = located.found
    if not found.asset_id:
        return []
    ids = {MarkerCorner(t.corner): t.marker_id for t in mapped if str(t.asset_id) == found.asset_id}
    out = []
    for corner in QUAD_ORDER:
        tag = found.tags.get(corner)
        status = "missing" if tag is None else ("estimated" if tag.estimated else "seen")
        centre = None if tag is None else Vector2(x=float(tag.center[0]), y=float(tag.center[1]))
        out.append(CornerStatusOut(corner=corner.value, marker_id=ids.get(corner, -1), status=status, center_px=centre))
    return out


def _arrangement(located: Located) -> tuple[str | None, str | None]:
    found = located.found
    if found.status is SetStatus.COMPLETE:
        return "valid", None
    if found.status is SetStatus.LAYOUT_INVALID or (found.status is SetStatus.MIXED_PRODUCTS and found.problem):
        return "invalid", found.problem
    if found.status in (SetStatus.INCOMPLETE, SetStatus.MIXED_PRODUCTS):
        return "incomplete", None
    if found.status in (SetStatus.DUPLICATE_MARKERS, SetStatus.MULTIPLE_PRODUCTS):
        return "invalid", found.status.value.replace("_", " ").lower()
    return None, None


@router.post("/track", response_model=TrackResponse)
def track(request: FrameRequest, db: Session = Depends(get_db)) -> TrackResponse:
    """
    One camera frame of four-tag tracking; call it with every frame.

    The four tags of one product define a quadrilateral from their detected
    corners (`region`). The YOLO segmentation model the other AR modes use is
    run on that region plus a context band (`search_area`; everything beyond
    it blanked, so objects elsewhere in view are never seen), and an object
    counts only if most of its mask is inside the quadrilateral itself;
    anything else it sees is listed in `rejected`. After
    marker_tag_confirm_frames agreeing frames the state is TRACKING and
    `product` is set. This identifies and segments the object; it does not
    estimate the object's 6DoF pose.
    """
    frame = decode_frame(request.image_base64)
    height, width = frame.shape[:2]
    tracker: FourTagTracker = _session(
        _trackers,
        request.session_id,
        lambda: FourTagTracker(
            _settings.marker_tag_hold_ms,
            _settings.marker_tag_confirm_frames,
            _settings.marker_tag_min_visible_for_hold,
            _settings.marker_tag_min_region_in_tags,
        ),
    )
    now_ms = time.monotonic() * 1000.0

    started = time.perf_counter()
    sightings, duplicates = read_tags(detect_aruco_markers, frame)
    markers_ms = (time.perf_counter() - started) * 1000.0

    mapped = _all_tags(db)
    by_id = {t.marker_id: t for t in mapped}
    names = {str(t.asset_id): t.asset.name for t in mapped}
    located = tracker.locate(sightings, _registry(mapped), duplicates, now_ms)
    found = located.found

    expected: str | None = None
    detection: RegionDetection | None = None
    yolo_ms: float | None = None
    if found.status is SetStatus.COMPLETE and found.region is not None:
        expected = _detection_class(db, _models(db, uuid.UUID(found.asset_id)))
        yolo_started = time.perf_counter()
        detection = detect_in_region(
            frame,
            found.region,
            segment_objects,
            expected,
            _settings.marker_tag_min_mask_inside,
            _settings.marker_tag_context,
        )
        yolo_ms = (time.perf_counter() - yolo_started) * 1000.0

    outcome = tracker.finish(located, detection, now_ms)
    message = _track_message(located, outcome, names, expected)
    # One log line per state change (not per frame): with no camera at hand,
    # `docker compose logs backend | grep four-tag` shows what a scan saw.
    if outcome.state.value != getattr(tracker, "last_state", None):
        tracker.last_state = outcome.state.value
        log.info(
            "four-tag %s: %s | tags read %s | %s",
            request.session_id,
            outcome.state.value,
            [s.marker_id for s in sightings],
            message,
        )

    product = None
    if outcome.state is TrackState.TRACKING and found.asset_id:
        product = _product(db, db.get(Asset, uuid.UUID(found.asset_id)))

    # Tags to draw: everything read, plus the re-projected (covered) ones.
    shown = list(sightings) + [found.tags[c] for c in located.estimated if c in found.tags]
    return TrackResponse(
        state=outcome.state.value,
        message=message,
        tracking=outcome.state is TrackState.TRACKING,
        asset_id=uuid.UUID(found.asset_id) if found.asset_id else None,
        asset_name=names.get(found.asset_id) if found.asset_id else None,
        tags=[_tag_out(s, by_id) for s in sorted(shown, key=lambda s: s.marker_id)],
        corners=_corner_status(located, mapped),
        arrangement=_arrangement(located)[0],
        arrangement_problem=_arrangement(located)[1],
        missing_corners=[c.value for c in found.missing],
        estimated_corners=[c.value for c in located.estimated],
        region=None if found.region is None else [Vector2(x=float(x), y=float(y)) for x, y in found.region],
        search_rect=list(detection.search_rect) if detection else None,
        search_area=None
        if detection is None or detection.search_polygon is None
        else [Vector2(x=float(x), y=float(y)) for x, y in detection.search_polygon],
        expected_class=expected,
        object=_object_out(outcome.obj, outcome.object_held) if outcome.obj else None,
        rejected=[_object_out(r) for r in detection.rejected] if detection else [],
        stable_frames=outcome.streak,
        required_frames=tracker.confirm_frames,
        product=product,
        frame_width=width,
        frame_height=height,
        markers_ms=markers_ms,
        yolo_ms=yolo_ms,
    )


@router.delete("/track/{session_id}", status_code=204)
def end_track(session_id: str) -> None:
    """Forgets a tracking session; the next frame starts from scratch."""
    _trackers.pop(session_id, None)

