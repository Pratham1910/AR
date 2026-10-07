"""
Marker -> product API: identifies which products are in front of the camera
from printed fiducial markers, and leads from a product to its existing 3D
models (/api/models3d) and procedures.

Detection (app/services/markers) only yields ids and pixel corners; the
marker_bindings table (app/models/marker.py) maps an id to an Asset. Neither
knows about the other, so a new marker or product is a binding row, and a new
marker family is a new detector.
"""

import uuid
from dataclasses import dataclass

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.vision import decode_frame, detectable_classes, segment_objects  # same decode path and YOLO model
from app.core.config import get_settings
from app.core.database import get_db
from app.models.asset import Asset
from app.models.enums import RevisionStatus
from app.models.marker import MarkerBinding
from app.models.model3d import Model3D
from app.models.procedure import Procedure, ProcedureRevision
from app.schemas.marker import (
    DetectedMarkerOut,
    MarkerBindingIn,
    MarkerBindingOut,
    MarkerConfigOut,
    MarkerDetectRequest,
    MarkerDetectResponse,
    MarkerProductRef,
    MarkerScanRequest,
    MarkerScanResponse,
    ProductOut,
    ProductProcedureOut,
    ProductStepOut,
    ScanGeometryOut,
    ScanOut,
)
from app.schemas.pose import Vector2
from app.services.markers.factory import build_marker_detector
from app.services.markers.presence import MarkerPresenceTracker, TrackedMarker
from app.services.scan.product_verifier import ProductDetection, Verification, verify_product
from app.services.scan.positioning import ScanGateConfig, scan_area
from app.services.scan.scan_gate import ScanGate, ScanObservation, ScanResult, ScanState
from app.services.vision.detector import YoloDetector

router = APIRouter(prefix="/api/markers", tags=["markers"])

_settings = get_settings()
_detector = build_marker_detector(_settings.marker_family, _settings.aruco_dictionary)

_gate_config = ScanGateConfig(
    area_width=_settings.scan_area_width,
    area_height=_settings.scan_area_height,
    center_tolerance=_settings.scan_center_tolerance,
    min_marker_size=_settings.scan_min_marker_size,
    max_marker_size=_settings.scan_max_marker_size,
    max_product_fill=_settings.scan_max_product_fill,
    marker_size_waived_fill=_settings.scan_marker_size_waived_fill,
    min_squareness=_settings.scan_min_squareness,
    max_roll_deg=_settings.scan_max_roll_deg,
    frame_edge_margin=_settings.scan_frame_edge_margin,
    hold_ms=_settings.scan_hold_ms,
    max_speed=_settings.scan_max_speed,
    verification_grace_ms=_settings.scan_verification_grace_ms,
    rearm_ms=_settings.scan_rearm_ms,
)
_ignored_classes = frozenset(c.strip().lower() for c in _settings.scan_ignored_classes.split(",") if c.strip())
_product_detector: YoloDetector | None = None  # built lazily, only if PRODUCT_MODEL_PATH is set


@dataclass
class _Session:
    presence: MarkerPresenceTracker
    gate: ScanGate


_MAX_SESSIONS = 16
_sessions: dict[str, _Session] = {}  # one per camera stream (session_id)


def _session(session_id: str) -> _Session:
    session = _sessions.pop(session_id, None) or _Session(
        MarkerPresenceTracker(_settings.marker_hold_ms), ScanGate(_gate_config)
    )
    _sessions[session_id] = session
    while len(_sessions) > _MAX_SESSIONS:
        _sessions.pop(next(iter(_sessions)))  # oldest-used first
    return session


def _active_bindings(db: Session):
    return db.query(MarkerBinding).filter(
        MarkerBinding.family == _detector.family, MarkerBinding.dictionary == _detector.dictionary
    )


def _binding_out(binding: MarkerBinding) -> MarkerBindingOut:
    return MarkerBindingOut(
        id=binding.id,
        family=binding.family,
        dictionary=binding.dictionary,
        marker_id=binding.marker_id,
        asset_id=binding.asset_id,
        asset_name=binding.asset.name,
    )


@router.get("/config", response_model=MarkerConfigOut)
def marker_config() -> MarkerConfigOut:
    return MarkerConfigOut(
        family=_detector.family,
        dictionary=_detector.dictionary,
        marker_count=_detector.marker_count,
        hold_ms=_settings.marker_hold_ms,
    )


@router.get("/bindings", response_model=list[MarkerBindingOut])
def list_bindings(db: Session = Depends(get_db)) -> list[MarkerBindingOut]:
    """Bindings for the active family/dictionary — the only ones the scanner can resolve."""
    return [_binding_out(b) for b in _active_bindings(db).order_by(MarkerBinding.marker_id)]


@router.put("/bindings", response_model=MarkerBindingOut)
def set_binding(payload: MarkerBindingIn, db: Session = Depends(get_db)) -> MarkerBindingOut:
    """Binds a marker id to a product, replacing whatever that id pointed at."""
    count = _detector.marker_count
    if count is not None and payload.marker_id >= count:
        raise HTTPException(
            status_code=400,
            detail=f"{_detector.dictionary} only has ids 0-{count - 1}, so marker {payload.marker_id} would never be detected.",
        )
    if db.get(Asset, payload.asset_id) is None:
        raise HTTPException(status_code=404, detail="Asset not found")

    binding = _active_bindings(db).filter(MarkerBinding.marker_id == payload.marker_id).one_or_none()
    if binding is None:
        binding = MarkerBinding(
            family=_detector.family, dictionary=_detector.dictionary, marker_id=payload.marker_id
        )
        db.add(binding)
    binding.asset_id = payload.asset_id
    db.commit()
    db.refresh(binding)
    return _binding_out(binding)


@router.delete("/bindings/{binding_id}", status_code=204)
def delete_binding(binding_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    binding = db.get(MarkerBinding, binding_id)
    if binding is None:
        raise HTTPException(status_code=404, detail="Binding not found")
    db.delete(binding)
    db.commit()


def _track(session_id: str | None, detections) -> list[TrackedMarker]:
    if session_id is None:
        return [TrackedMarker(d, visible=True, ms_since_seen=0.0) for d in detections]
    return _session(session_id).presence.update(detections)


def _markers_out(db: Session, tracked: list[TrackedMarker]) -> list[DetectedMarkerOut]:
    """Each marker with the product bound to its id, ordered by id."""
    marker_ids = {t.detection.marker_id for t in tracked}
    bindings = (
        {b.marker_id: b for b in _active_bindings(db).filter(MarkerBinding.marker_id.in_(marker_ids))}
        if marker_ids
        else {}
    )
    markers = []
    for item in sorted(tracked, key=lambda t: t.detection.marker_id):
        detection = item.detection
        binding = bindings.get(detection.marker_id)
        center_x, center_y = detection.center_px
        markers.append(
            DetectedMarkerOut(
                marker_id=detection.marker_id,
                corners_px=[Vector2(x=x, y=y) for x, y in detection.corners_px],
                center_px=Vector2(x=center_x, y=center_y),
                visible=item.visible,
                ms_since_seen=item.ms_since_seen,
                status="known" if binding else "unknown",
                product=MarkerProductRef(asset_id=binding.asset_id, name=binding.asset.name) if binding else None,
            )
        )
    return markers


@router.post("/detect", response_model=MarkerDetectResponse)
def detect_markers(request: MarkerDetectRequest, db: Session = Depends(get_db)) -> MarkerDetectResponse:
    """
    Every marker in the frame, each with its id, four pixel corners and the
    product bound to it (status "unknown" if none). With a session_id, a
    marker that just left the frame is still returned for MARKER_HOLD_MS with
    visible=false.
    """
    frame = decode_frame(request.image_base64)
    height_px, width_px = frame.shape[:2]
    tracked = _track(request.session_id, _detector.detect(frame))
    return MarkerDetectResponse(
        family=_detector.family,
        dictionary=_detector.dictionary,
        frame_width=width_px,
        frame_height=height_px,
        markers=_markers_out(db, tracked),
    )


def _expected_class(db: Session, asset_id: uuid.UUID) -> str | None:
    """The detector class a product should show up as: its newest 3D model's detection class."""
    model = (
        db.query(Model3D)
        .filter(Model3D.asset_id == asset_id, Model3D.component_id.isnot(None))
        .order_by(Model3D.created_at.desc())
        .first()
    )
    return model.component.class_label if model is not None and model.component is not None else None


def _product_classes() -> list[str]:
    """Every class the product-verification model can report."""
    global _product_detector
    if not _settings.product_model_path:
        return detectable_classes()
    if _product_detector is None:
        try:
            _product_detector = YoloDetector(_settings.product_model_path, _settings.scan_candidate_confidence)
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear 503, not a crash
            raise HTTPException(
                status_code=503, detail=f"Could not load PRODUCT_MODEL_PATH {_settings.product_model_path!r}: {exc}"
            ) from exc
    return _product_detector.class_names


def _detect_products(frame) -> list[ProductDetection]:
    if _product_detector is not None:
        found = _product_detector.detect(frame)
    else:
        found = segment_objects(frame, _settings.scan_candidate_confidence)
    return [
        ProductDetection(d.class_label, d.confidence, (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2)) for d in found
    ]


_HINT_TEXT = {
    "show_complete_product": "Show the complete product",
    "move_closer": "Move CLOSER",
    "move_farther": "Move FARTHER",
    "move_left": "Move LEFT",
    "move_right": "Move RIGHT",
    "move_up": "Move UP",
    "move_down": "Move DOWN",
    "center": "Center the product",
    "straighten": "Straighten the product",
    "hold_steady": "Hold STEADY",
}


def _scan_message(result: ScanResult, product_name: str | None) -> str:
    verification = result.verification
    expected = verification.expected_class if verification else None
    if result.state is ScanState.CONFIRMED:
        return f"✓ Product confirmed: {product_name}"
    if result.state is ScanState.SCANNING:
        return "SCANNING…"
    if result.problem == "no_marker":
        return "Show the marker"
    if result.problem == "multiple_markers":
        return "Multiple markers in view — show only one"
    if result.problem == "unknown_marker":
        return f"Marker {result.marker_id} is not registered to a product"
    if result.problem == "unverifiable":
        if expected is None:
            return f"Cannot verify {product_name}: it has no detection class (set one on its 3D model)"
        return f"Cannot verify {product_name}: the detection model has no '{expected}' class"
    if result.problem == "no_product" and not result.hints:
        return f"Show the complete product ({expected})"
    if result.problem == "uncertain" and not result.hints:
        return "Verifying product…"
    if result.problem == "mismatch":
        return f"Product mismatch — expected {product_name} ({expected}), detected {verification.detection.class_label}"
    return " · ".join(_HINT_TEXT[h] for h in result.hints)


@router.post("/scan", response_model=MarkerScanResponse)
def scan_product(request: MarkerScanRequest, db: Session = Depends(get_db)) -> MarkerScanResponse:
    """
    One frame of the automatic scan: marker -> product it names -> is that
    product really at the marker (object detector) -> is it well placed and
    still. Call it with every frame; `scan.state` reaches CONFIRMED by itself
    after SCAN_HOLD_MS of a good, steady view, and stays there (same
    `scan.confirmation`) until that product has left the scan box. Anything
    else in `scan` says what is missing and which way to move.
    """
    frame = decode_frame(request.image_base64)
    height_px, width_px = frame.shape[:2]
    session = _session(request.session_id)
    detections = _detector.detect(frame)
    markers = _markers_out(db, session.presence.update(detections))

    marker = product = None
    verification: Verification | None = None
    if len(detections) == 1:
        marker = detections[0]
        product = next(m.product for m in markers if m.visible and m.marker_id == marker.marker_id)
        if product is not None:
            expected = _expected_class(db, product.asset_id)
            classes = _product_classes()
            # Only pay for inference when the model could recognise this product at all.
            can_verify = expected is not None and expected.lower() in {c.lower() for c in classes}
            verification = verify_product(
                marker,
                expected,
                classes,
                _detect_products(frame) if can_verify else [],
                _settings.scan_min_product_confidence,
                _ignored_classes,
                _settings.scan_association_reach,
            )

    result = session.gate.update(
        ScanObservation(
            frame_width=width_px,
            frame_height=height_px,
            marker_count=len(detections),
            marker=marker,
            known=product is not None,
            verification=verification,
            mirrored=request.mirrored,
        )
    )
    if result.state is ScanState.CONFIRMED and (marker is None or marker.marker_id != result.marker_id):
        # Latched from an earlier frame: report the product that was confirmed, not what is in view now.
        binding = _active_bindings(db).filter(MarkerBinding.marker_id == result.marker_id).one_or_none()
        product = MarkerProductRef(asset_id=binding.asset_id, name=binding.asset.name) if binding else None

    verified = result.verification
    seen = verified.detection if verified else None
    position = result.positioning
    geometry = None
    if position is not None:
        geometry = ScanGeometryOut(
            target_center=Vector2(x=position.target_center[0], y=position.target_center[1]),
            marker_center=Vector2(x=position.marker_center[0], y=position.marker_center[1]),
            product_center=Vector2(x=position.product_center[0], y=position.product_center[1])
            if position.product_center
            else None,
            offset_x=position.offset_x,
            offset_y=position.offset_y,
            marker_size=position.marker_size,
            product_fill=position.product_fill,
            distance=position.distance,
            squareness=position.squareness,
            roll_deg=position.roll_deg,
            inside_area=position.inside_area,
            complete=position.complete,
            speed=result.speed,
        )
    return MarkerScanResponse(
        family=_detector.family,
        dictionary=_detector.dictionary,
        frame_width=width_px,
        frame_height=height_px,
        markers=markers,
        scan=ScanOut(
            state=result.state.value,
            message=_scan_message(result, product.name if product else None),
            problem=result.problem,
            hints=list(result.hints),
            progress=result.progress,
            stages=result.stages,
            confirmation=result.confirmation,
            marker_id=result.marker_id,
            product=product,
            expected_class=verified.expected_class if verified else None,
            detected_class=seen.class_label if seen else None,
            detected_confidence=seen.confidence if seen else None,
            product_bbox=list(seen.bbox) if seen else None,
            scan_area=list(scan_area(width_px, height_px, _gate_config)),
            geometry=geometry,
        ),
    )


@router.delete("/sessions/{session_id}", status_code=204)
def end_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


@router.get("/products/{asset_id}", response_model=ProductOut)
def get_product(asset_id: uuid.UUID, db: Session = Depends(get_db)) -> ProductOut:
    """A product's instructions: the steps of each procedure's latest published revision."""
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found")

    procedures = []
    for procedure in db.query(Procedure).filter(Procedure.asset_id == asset_id).order_by(Procedure.title):
        revision = (
            db.query(ProcedureRevision)
            .filter(
                ProcedureRevision.procedure_id == procedure.id,
                ProcedureRevision.status == RevisionStatus.PUBLISHED,
            )
            .order_by(ProcedureRevision.created_at.desc())
            .first()
        )
        if revision is None:
            continue
        procedures.append(
            ProductProcedureOut(
                procedure_id=procedure.id,
                procedure_id_str=procedure.procedure_id_str,
                title=procedure.title,
                revision_id=revision.id,
                revision_label=revision.revision_label,
                steps=[
                    ProductStepOut(
                        step_id_str=step.step_id_str,
                        title=step.title,
                        action_type=step.action_type.value,
                        component=step.target_component.name if step.target_component else None,
                        starting_state=step.starting_state.name,
                        expected_state=step.expected_state.name,
                    )
                    for step in revision.steps
                ],
            )
        )
    return ProductOut(asset_id=asset.id, name=asset.name, description=asset.description, procedures=procedures)
