"""
ArUco product tags API: four tags per product, and the scan that turns them
into a verified product identity.

    tags in the frame -> which product's four corners? -> layout plausible?
        -> YOLO: is that product's class actually there, at the tags?
            -> same answer for a few frames -> CONFIRMED -> its 3D model / procedures

Like every router here this only wires services together
(app/services/markers for the tag logic, the existing ArUco detector and
object-finding model from app/api/vision). It reports an identity, never a
QA PASS/FAIL. Mapping tags is management of reference data, kept apart from
/api/vision's pose and tracking endpoints, which it does not change.
"""

import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from app.api.vision import decode_frame, detect_aruco_markers, detectable_classes, segment_objects
from app.core.config import get_settings
from app.core.database import get_db
from app.models.asset import Asset, Component
from app.models.enums import MarkerCorner, RevisionStatus
from app.models.marker_tag import MarkerTag
from app.models.model3d import Model3D
from app.models.procedure import Procedure
from app.schemas.marker_tag import (
    IdentifyResponse,
    ScannedTagOut,
    ScanProcedureOut,
    ScanProductOut,
    ScanRequest,
    ScanResponse,
    ScanVerificationOut,
    SuggestedIdsOut,
    TagConfigOut,
    TagSetIn,
    TagSetOut,
)
from app.schemas.pose import Vector2
from app.services.audit import audit_log
from app.services.markers.identification import (
    Identification,
    IdentificationStatus,
    LayoutLimits,
    TagAssignment,
    TagLookupStatus,
    identify_product,
    lookup_product_by_tag,
)
from app.services.markers.registry import TagSetError, suggest_free_ids, validate_tag_set
from app.services.markers.scan_session import ScanSession
from app.services.markers.tags import CORNER_LABELS, detect_tags, dictionary_size, render_sheet_png, render_tag_png
from app.services.markers.verification import (
    Verification,
    VerificationStatus,
    crop_to_region,
    to_frame_coordinates,
    verify_product,
)

router = APIRouter(prefix="/api/marker-tags", tags=["marker-tags"])

_settings = get_settings()
_MAX_SESSIONS = 16
_scan_sessions: dict[str, ScanSession] = {}  # session_id -> held tags + confirmation streak (in memory, like AR sessions)
_identify_sessions: dict[str, ScanSession] = {}  # product scan: its own streaks, never shared with the above


def _dictionary() -> str:
    return _settings.aruco_dictionary


def _all_tags(db: Session) -> list[MarkerTag]:
    """Every mapped tag of the dictionary in use, with its asset loaded."""
    return (
        db.query(MarkerTag)
        .options(joinedload(MarkerTag.asset))
        .filter(MarkerTag.dictionary == _dictionary())
        .order_by(MarkerTag.marker_id)
        .all()
    )


def _product_model(db: Session, asset_id: uuid.UUID) -> tuple[Model3D | None, str | None]:
    """
    (model to load, YOLO class to verify against) for a product.

    The model is the asset's newest — what the 3D workspace opens by default.
    The class is the detection class set on a model at upload / in Model
    settings (its linked Component's class_label): the newest model's if it
    has one, else the newest model that does.
    """
    models = db.query(Model3D).filter(Model3D.asset_id == asset_id).order_by(Model3D.created_at.desc()).all()
    for model in models:
        if model.component_id is not None:
            component = db.get(Component, model.component_id)
            if component is not None and component.class_label:
                return models[0], component.class_label
    return (models[0] if models else None), None


def _set_out(db: Session, asset: Asset, corners: dict[str, int]) -> TagSetOut:
    model, detection_class = _product_model(db, asset.id)
    return TagSetOut(
        asset_id=asset.id,
        asset_name=asset.name,
        dictionary=_dictionary(),
        detection_class=detection_class,
        model_id=model.id if model else None,
        model_name=model.name if model else None,
        **corners,
    )


def _sets(db: Session) -> list[TagSetOut]:
    by_asset: dict[uuid.UUID, tuple[Asset, dict[str, int]]] = {}
    for tag in _all_tags(db):
        by_asset.setdefault(tag.asset_id, (tag.asset, {}))[1][tag.corner] = tag.marker_id
    # A set is always written whole; anything else (rows edited by hand) is
    # not a usable mapping and is left out rather than half-shown.
    complete = [(asset, corners) for asset, corners in by_asset.values() if len(corners) == len(MarkerCorner)]
    return [_set_out(db, asset, corners) for asset, corners in sorted(complete, key=lambda pair: pair[0].name)]


# --- Mapping ---------------------------------------------------------------


@router.get("/config", response_model=TagConfigOut)
def get_config(db: Session = Depends(get_db)) -> TagConfigOut:
    size = dictionary_size(_dictionary())
    used = [tag.marker_id for tag in _all_tags(db)]
    return TagConfigOut(
        dictionary=_dictionary(),
        dictionary_size=size,
        used_ids=used,
        free_ids=size - len(used),
        confirm_frames=_settings.marker_tag_confirm_frames,
    )


@router.get("/suggest", response_model=SuggestedIdsOut)
def suggest_ids(asset_id: uuid.UUID | None = None, db: Session = Depends(get_db)) -> SuggestedIdsOut:
    """Four unused tag ids for a new product. A product's own current ids
    count as free for it, so re-suggesting for it doesn't burn four more."""
    taken = {tag.marker_id for tag in _all_tags(db) if tag.asset_id != asset_id}
    ids = suggest_free_ids(taken, dictionary_size(_dictionary()))
    if len(ids) < len(MarkerCorner):
        raise HTTPException(
            status_code=409,
            detail=(
                f"{_dictionary()} has only {len(ids)} unused tag id(s) left; a product needs four. "
                "Remove an unused mapping, or switch ARUCO_DICTIONARY to a larger dictionary."
            ),
        )
    return SuggestedIdsOut(dictionary=_dictionary(), ids=ids)


@router.get("/sets", response_model=list[TagSetOut])
def list_sets(db: Session = Depends(get_db)) -> list[TagSetOut]:
    return _sets(db)


@router.put("/sets/{asset_id}", response_model=TagSetOut)
def put_set(asset_id: uuid.UUID, payload: TagSetIn, db: Session = Depends(get_db)) -> TagSetOut:
    """Assigns (or replaces) a product's four tags, all or nothing."""
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found")

    requested = {corner: getattr(payload, corner.value) for corner in MarkerCorner}
    existing = _all_tags(db)
    try:
        validate_tag_set(
            requested,
            str(asset_id),
            {tag.marker_id: (str(tag.asset_id), tag.asset.name) for tag in existing},
            dictionary_size(_dictionary()),
        )
    except TagSetError as exc:
        raise HTTPException(status_code=409 if exc.conflict else 400, detail=str(exc)) from exc

    previous = {tag.corner: tag.marker_id for tag in existing if tag.asset_id == asset_id}
    for tag in existing:
        if tag.asset_id == asset_id:
            db.delete(tag)
    db.flush()  # free this product's old ids before re-inserting (it may be rearranging them)
    for corner, marker_id in requested.items():
        db.add(MarkerTag(asset_id=asset_id, dictionary=_dictionary(), marker_id=marker_id, corner=corner.value))
    audit_log.record(
        db,
        user_id=None,  # the mapping API is unauthenticated, like the asset/model APIs it sits beside
        action="set_marker_tags",
        entity_type="Asset",
        entity_id=str(asset_id),
        details={"dictionary": _dictionary(), "previous": previous, "tags": {c.value: i for c, i in requested.items()}},
    )
    try:
        db.commit()
    except IntegrityError as exc:
        # Two saves racing for the same id: the unique constraints decide.
        db.rollback()
        raise HTTPException(status_code=409, detail="One of these tag ids was just assigned to another product.") from exc
    return _set_out(db, asset, {c.value: i for c, i in requested.items()})


@router.delete("/sets/{asset_id}", status_code=204)
def delete_set(asset_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    tags = [tag for tag in _all_tags(db) if tag.asset_id == asset_id]
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
        details={"dictionary": _dictionary(), "tags": {tag.corner: tag.marker_id for tag in tags}},
    )
    db.commit()


# --- Printable tags --------------------------------------------------------


def _png(content: bytes, filename: str) -> Response:
    return Response(
        content=content, media_type="image/png", headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@router.get("/markers/{marker_id}.png")
def download_tag(
    marker_id: int, size_px: int = Query(600, ge=100, le=2000), db: Session = Depends(get_db)
) -> Response:
    """One tag as a PNG. If it is mapped, the caption names its product and corner."""
    if not 0 <= marker_id < dictionary_size(_dictionary()):
        raise HTTPException(status_code=404, detail=f"{_dictionary()} has no tag id {marker_id}")
    tag = next((t for t in _all_tags(db) if t.marker_id == marker_id), None)
    caption = f"id {marker_id}"
    if tag is not None:
        caption = f"{tag.asset.name}  {CORNER_LABELS[MarkerCorner(tag.corner)]}  id {marker_id}"
    return _png(render_tag_png(_dictionary(), marker_id, size_px, caption), f"aruco_{_dictionary()}_id{marker_id}.png")


@router.get("/sets/{asset_id}/sheet.png")
def download_sheet(
    asset_id: uuid.UUID, size_px: int = Query(500, ge=100, le=1500), db: Session = Depends(get_db)
) -> Response:
    """A product's four tags on one printable page, in their corner positions."""
    tags = {MarkerCorner(tag.corner): tag for tag in _all_tags(db) if tag.asset_id == asset_id}
    if len(tags) != len(MarkerCorner):
        raise HTTPException(status_code=404, detail="This product has no tags assigned")
    name = tags[MarkerCorner.TOP_LEFT].asset.name
    sheet = render_sheet_png(_dictionary(), {corner: tag.marker_id for corner, tag in tags.items()}, size_px, name)
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name)
    return _png(sheet, f"aruco_tags_{safe_name}.png")


# --- Scan ------------------------------------------------------------------


def _session(session_id: str) -> ScanSession:
    session = _scan_sessions.pop(session_id, None) or ScanSession(
        _settings.marker_tag_hold_ms, _settings.marker_tag_confirm_frames
    )
    _scan_sessions[session_id] = session
    while len(_scan_sessions) > _MAX_SESSIONS:
        _scan_sessions.pop(next(iter(_scan_sessions)))  # oldest-used first
    return session


def _corner_names(corners: list[MarkerCorner]) -> str:
    return ", ".join(CORNER_LABELS[c] for c in corners)


def _identification_message(result: Identification, names: dict[str, str]) -> tuple[str, str]:
    """(state, message) for every outcome that stops before verification."""
    name = names.get(result.asset_id or "", "a product")
    status = result.status
    if status is IdentificationStatus.NO_MARKERS:
        return status.value, "No tags in view — show the product with its four corner tags."
    if status is IdentificationStatus.UNKNOWN_MARKERS:
        return status.value, f"Tag id(s) {result.unknown_ids} are not assigned to any product."
    if status is IdentificationStatus.DUPLICATE_MARKERS:
        return status.value, f"Tag id(s) {result.duplicate_ids} appear more than once — remove the extra copy."
    if status is IdentificationStatus.INCOMPLETE:
        seen = 4 - len(result.missing_corners)
        return status.value, f"{name}: {seen} of 4 tags found — missing {_corner_names(result.missing_corners)}."
    if status is IdentificationStatus.MULTIPLE_PRODUCTS:
        listed = ", ".join(names.get(a, a) for a in result.other_asset_ids)
        return status.value, f"More than one tagged product in view ({listed}) — show one at a time."
    if status is IdentificationStatus.MIXED_PRODUCTS:
        others = ", ".join(names.get(a, a) for a in result.other_asset_ids)
        if result.quad is not None:
            return status.value, f"A tag of {others} is inside {name}'s four tags — not confirming either."
        return (
            status.value,
            f"Tags of different products in view ({name}, {others}) and no complete set — "
            f"{name} is missing {_corner_names(result.missing_corners)}.",
        )
    return status.value, f"{name}: all four tags found, but not in a valid layout. {result.layout_problem}"


def _verification_message(result: Verification, name: str, streak: int, required: int) -> tuple[str, str]:
    expected = result.expected_class
    status = result.status
    if status is VerificationStatus.VERIFIED:
        if streak >= required:
            return "CONFIRMED", f"{name} identified — four tags match and a {expected} is inside them."
        return "VERIFYING", f"{name}: tags match and a {expected} is inside them — hold still ({streak}/{required})."
    if status is VerificationStatus.NO_CLASS:
        return (
            "CANNOT_VERIFY",
            f"{name}: tags match, but the product has no detection class, so it cannot be verified. "
            "Set one on its 3D model (Model settings → detection class).",
        )
    if status is VerificationStatus.UNKNOWN_CLASS:
        return (
            "CANNOT_VERIFY",
            f"{name}: tags match, but its detection class '{expected}' is not one the detector knows.",
        )
    if status is VerificationStatus.WRONG_OBJECT:
        return (
            "WRONG_OBJECT",
            f"Tags say {name} (expected a {expected}), but the camera sees a {result.obj.class_label} inside them.",
        )
    return "OBJECT_NOT_FOUND", f"{name}: tags match, but no {expected} is detected inside the four tags."


def _product_out(db: Session, asset: Asset, model: Model3D | None, detection_class: str | None, pose_marker_id: int) -> ScanProductOut:
    procedures = db.query(Procedure).filter(Procedure.asset_id == asset.id).order_by(Procedure.procedure_id_str).all()
    return ScanProductOut(
        asset_id=asset.id,
        asset_name=asset.name,
        detection_class=detection_class,
        model_id=model.id if model else None,
        model_name=model.name if model else None,
        model_url=f"/static/models/{model.storage_key}" if model else None,
        has_model_procedure=model is not None
        and (Path(_settings.procedures_dir) / f"{model.id}.procedure.json").is_file(),
        procedures=[
            ScanProcedureOut(
                procedure_id=p.id,
                procedure_id_str=p.procedure_id_str,
                title=p.title,
                published_revision_id=next(
                    (
                        r.id
                        for r in sorted(p.revisions, key=lambda r: r.created_at, reverse=True)
                        if r.status == RevisionStatus.PUBLISHED
                    ),
                    None,
                ),
            )
            for p in procedures
        ],
        pose_marker_id=pose_marker_id,
    )


@router.post("/scan", response_model=ScanResponse)
def scan(request: ScanRequest, db: Session = Depends(get_db)) -> ScanResponse:
    """
    One camera frame of a product scan; call it with every frame.

    YOLO runs only once the tags have named a candidate with a valid layout,
    and then only on the area those four tags enclose (`region` in the
    response) — the rest of the frame is blanked before detection, so an
    object elsewhere in view can neither verify nor contradict the product.
    `confirmed` turns true after marker_tag_confirm_frames consecutive frames
    that identified and verified the same product; `product` is set only then.
    """
    frame = decode_frame(request.image_base64)
    height, width = frame.shape[:2]
    session = _session(request.session_id)

    started = time.perf_counter()
    seen, duplicate_ids = detect_tags(detect_aruco_markers, frame)
    markers_ms = (time.perf_counter() - started) * 1000.0
    tags = session.with_held(seen, time.monotonic() * 1000.0)

    mapped = _all_tags(db)
    by_id = {tag.marker_id: tag for tag in mapped}
    names = {str(tag.asset_id): tag.asset.name for tag in mapped}
    registry = {tag.marker_id: TagAssignment(str(tag.asset_id), MarkerCorner(tag.corner)) for tag in mapped}
    limits = LayoutLimits(
        max_size_ratio=_settings.marker_tag_max_size_ratio,
        max_opposite_side_ratio=_settings.marker_tag_max_opposite_side_ratio,
        min_side_in_tags=_settings.marker_tag_min_side_in_tags,
    )
    identification = identify_product(tags, registry, limits, duplicate_ids)

    verification: Verification | None = None
    product: ScanProductOut | None = None
    yolo_ms: float | None = None
    if identification.status is IdentificationStatus.CANDIDATE:
        asset = db.get(Asset, uuid.UUID(identification.asset_id))
        model, detection_class = _product_model(db, asset.id)
        objects = []
        known_classes = detectable_classes() if detection_class else []
        if detection_class in known_classes:
            # The detector gets the area inside the four tags and nothing
            # else — the rest of the camera view is blanked out first.
            yolo_started = time.perf_counter()
            crop, origin = crop_to_region(frame, identification.region)
            objects = to_frame_coordinates(segment_objects(crop), origin)
            yolo_ms = (time.perf_counter() - yolo_started) * 1000.0
        verification = verify_product(
            detection_class, known_classes, objects, identification.region, _settings.marker_tag_min_inside
        )
        verified = verification.status is VerificationStatus.VERIFIED
        streak = session.record(identification.asset_id if verified else None)
        state, message = _verification_message(verification, asset.name, streak, session.confirm_frames)
        if session.is_confirmed():
            top_left = next(i for i, a in registry.items() if a == TagAssignment(str(asset.id), MarkerCorner.TOP_LEFT))
            product = _product_out(db, asset, model, detection_class, top_left)
    else:
        streak = session.record(None)
        state, message = _identification_message(identification, names)

    candidate = identification.asset_id
    obj = verification.obj if verification else None
    return ScanResponse(
        state=state,
        message=message,
        confirmed=product is not None,
        tags=[
            ScannedTagOut(
                marker_id=d.marker_id,
                corners_px=[Vector2(x=float(x), y=float(y)) for x, y in d.corners],
                held=d.held,
                asset_id=by_id[d.marker_id].asset_id if d.marker_id in by_id else None,
                asset_name=by_id[d.marker_id].asset.name if d.marker_id in by_id else None,
                corner=by_id[d.marker_id].corner if d.marker_id in by_id else None,
            )
            for d in sorted(tags, key=lambda d: d.marker_id)
        ],
        candidate_asset_id=uuid.UUID(candidate) if candidate else None,
        candidate_asset_name=names.get(candidate) if candidate else None,
        missing_corners=[c.value for c in identification.missing_corners],
        quad=None
        if identification.quad is None
        else [Vector2(x=float(x), y=float(y)) for x, y in identification.quad],
        region=None
        if identification.region is None
        else [Vector2(x=float(x), y=float(y)) for x, y in identification.region],
        verification=None
        if verification is None
        else ScanVerificationOut(
            status=verification.status.value,
            expected_class=verification.expected_class,
            detected_class=obj.class_label if obj else None,
            confidence=obj.confidence if obj else None,
            inside=verification.inside,
            bbox=[obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2] if obj else None,
            polygon=[Vector2(x=p.x, y=p.y) for p in obj.polygon] if obj else None,
        ),
        stable_frames=streak,
        required_frames=session.confirm_frames,
        product=product,
        dictionary=_dictionary(),
        frame_width=width,
        frame_height=height,
        markers_ms=markers_ms,
        yolo_ms=yolo_ms,
    )


# --- Product scan: look a product up by any one tag --------------------------


def _identify_session(session_id: str) -> ScanSession:
    # hold_ms=0: a product is only ever named from tags in the current frame.
    session = _identify_sessions.pop(session_id, None) or ScanSession(0.0, _settings.marker_tag_identify_frames)
    _identify_sessions[session_id] = session
    while len(_identify_sessions) > _MAX_SESSIONS:
        _identify_sessions.pop(next(iter(_identify_sessions)))
    return session


@router.post("/identify", response_model=IdentifyResponse)
def identify(request: ScanRequest, db: Session = Depends(get_db)) -> IdentifyResponse:
    """
    Product scan: names the product that ANY ONE registered tag in view
    belongs to, so its 3D model and procedure can be opened. Deliberately
    lenient — no four-tag set, no layout check, no YOLO — and therefore not
    evidence that the product is physically there: confirming a product for
    tracking is /scan's job, which this endpoint shares no state or
    thresholds with.

    It still refuses to guess: an unmapped tag names no product, and tags of
    two products in view are reported as ambiguous rather than resolved.
    """
    frame = decode_frame(request.image_base64)
    height, width = frame.shape[:2]
    seen, _ = detect_tags(detect_aruco_markers, frame)

    mapped = _all_tags(db)
    by_id = {tag.marker_id: tag for tag in mapped}
    names = {str(tag.asset_id): tag.asset.name for tag in mapped}
    registry = {tag.marker_id: TagAssignment(str(tag.asset_id), MarkerCorner(tag.corner)) for tag in mapped}
    lookup = lookup_product_by_tag(seen, registry)

    session = _identify_session(request.session_id)
    streak = session.record(lookup.asset_id if lookup.status is TagLookupStatus.FOUND else None)
    product: ScanProductOut | None = None
    candidate_name = names.get(lookup.asset_id) if lookup.asset_id else None

    if lookup.status is TagLookupStatus.NO_MARKERS:
        state, message = "NO_MARKERS", "No tag in view — show any one of the product's tags to the camera."
    elif lookup.status is TagLookupStatus.UNKNOWN_MARKERS:
        state = "UNKNOWN_MARKERS"
        message = f"Tag id(s) {lookup.unknown_ids} are not mapped to any product — no product selected."
    elif lookup.status is TagLookupStatus.AMBIGUOUS:
        state = "AMBIGUOUS"
        listed = ", ".join(names[a] for a in lookup.asset_ids)
        message = f"Tags of different products in view ({listed}) — show the tags of one product only."
    elif not session.is_confirmed():
        state = "IDENTIFYING"
        message = f"Tag {lookup.matched_ids[0]} belongs to {candidate_name} — confirming ({streak}/{session.confirm_frames})."
    else:
        state = "IDENTIFIED"
        asset = db.get(Asset, uuid.UUID(lookup.asset_id))
        model, detection_class = _product_model(db, asset.id)
        top_left = next(i for i, a in registry.items() if a == TagAssignment(lookup.asset_id, MarkerCorner.TOP_LEFT))
        product = _product_out(db, asset, model, detection_class, top_left)
        message = f"{asset.name} — identified by tag {', '.join(str(i) for i in lookup.matched_ids)}."

    return IdentifyResponse(
        state=state,
        message=message,
        identified=product is not None,
        tags=[
            ScannedTagOut(
                marker_id=d.marker_id,
                corners_px=[Vector2(x=float(x), y=float(y)) for x, y in d.corners],
                held=False,
                asset_id=by_id[d.marker_id].asset_id if d.marker_id in by_id else None,
                asset_name=by_id[d.marker_id].asset.name if d.marker_id in by_id else None,
                corner=by_id[d.marker_id].corner if d.marker_id in by_id else None,
            )
            for d in sorted(seen, key=lambda d: d.marker_id)
        ],
        candidate_asset_id=uuid.UUID(lookup.asset_id) if lookup.asset_id else None,
        candidate_asset_name=candidate_name,
        stable_frames=streak,
        required_frames=session.confirm_frames,
        product=product,
        dictionary=_dictionary(),
        frame_width=width,
        frame_height=height,
    )


@router.delete("/identify/{session_id}", status_code=204)
def end_identify(session_id: str) -> None:
    _identify_sessions.pop(session_id, None)


@router.delete("/scan/{session_id}", status_code=204)
def end_scan(session_id: str) -> None:
    """Forgets a scan's held tags and confirmation streak; the next frame starts clean."""
    _scan_sessions.pop(session_id, None)
