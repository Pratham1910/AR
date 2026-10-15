"""
Component inspection API: the component registry of a 3D model, its learned
good-state reference, and the per-session inspection the live AR stream
runs (vision.process_ar_frame calls run_inspection with the frame's own
tracked pose).
"""

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.models.asset import Component
from app.models.model3d import Model3D
from app.schemas.component_inspection import (
    ComponentInspectionOut,
    ComponentOut,
    ComponentRegistryOut,
    ComponentResultOut,
    InspectionSummaryOut,
    NodeOut,
    VerdictOut,
)
from app.services.component_inspection.decision import DecisionConfig
from app.services.component_inspection.registry import ComponentRegistry, registry_from_glb
from app.services.component_inspection.session import InspectionConfig, InspectionSession
from app.services.component_inspection.verifiers import (
    CalibratedPresenceVerifier,
    ComponentVerifier,
    DetectorVerifier,
    ReferenceAppearanceVerifier,
    ReferenceSet,
    UltralyticsComponentDetector,
)
from app.services.component_inspection.visibility import VisibilityConfig
from app.services.model3d.glb_inspect import GlbParseError
from app.services.vision.part_presence import PresenceCalibration

router = APIRouter(prefix="/api/models3d", tags=["3d"])
_settings = get_settings()
log = logging.getLogger("app.inspection")


def _glb_path(model: Model3D) -> Path:
    return Path(_settings.models_3d_dir) / model.storage_key


def mapping_path(model_id: str) -> Path:
    return Path(_settings.component_registry_dir) / f"{model_id}.json"


def reference_path(model_id: str) -> Path:
    return Path(_settings.component_reference_dir) / f"{model_id}.json"


def named_parts(db: Session, model: Model3D) -> dict[str, str]:
    """Parts named in the Parts panel: GLB node name -> the user's name (Component.cad_node_id)."""
    query = db.query(Component).filter(Component.asset_id == model.asset_id, Component.cad_node_id.isnot(None))
    if model.component_id is not None:
        query = query.filter(Component.parent_id == model.component_id)
    return {c.cad_node_id: c.name for c in query.all()}


def _calibrations(model_id: str) -> list[PresenceCalibration]:
    found = []
    for path in sorted(Path(_settings.part_calibration_dir).glob(f"{model_id}__node*.json")):
        calibration = PresenceCalibration.load(path)
        if calibration is not None:
            found.append(calibration)
    return found


# ── registry ────────────────────────────────────────────────────────────

_registry_cache: dict[str, tuple[tuple, ComponentRegistry]] = {}


def registry_for(db: Session, model: Model3D) -> ComponentRegistry:
    """The model's component registry, rebuilt only when the GLB, its scale,
    the mapping file or the part names change."""
    glb, mapping = _glb_path(model), mapping_path(str(model.id))
    names = named_parts(db, model)
    try:
        key = (glb.stat().st_mtime, model.scale, mapping.stat().st_mtime if mapping.exists() else None, tuple(sorted(names.items())))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"GLB file missing: {glb}") from exc
    cached = _registry_cache.get(str(model.id))
    if cached is not None and cached[0] == key:
        return cached[1]
    try:
        registry = registry_from_glb(model.name, glb.read_bytes(), model.scale, names, mapping)
    except GlbParseError as exc:
        raise HTTPException(status_code=400, detail=f"Could not read this model's .glb: {exc}") from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Component mapping {mapping}: {exc}") from exc
    _registry_cache[str(model.id)] = (key, registry)
    return registry


def _registry_out(model: Model3D, registry: ComponentRegistry) -> ComponentRegistryOut:
    references = ReferenceSet.load(reference_path(str(model.id)))
    calibrated = {c.node_index for c in _calibrations(str(model.id)) if c.method == "pose"}
    _, detector_name = _detector()
    return ComponentRegistryOut(
        model_id=str(model.id),
        object_id=registry.object_id,
        source=registry.source,
        mapping_path=registry.mapping_path,
        components=[
            ComponentOut(
                id=c.id, type=c.type, display_name=c.display_name, required=c.required,
                inspection_mode=c.inspection_mode, source=c.source, node_index=c.node_index,
                node_name=c.node_name, piece=c.piece, mesh_status=c.mesh_status, problem=c.problem,
                center_m=c.center_m, size_m=c.size_m, color=c.color,
                node_matrix=c.node_matrix.tolist() if c.node_matrix is not None else None,
                reference_samples=references.components[c.id].samples if references and c.id in references.components else 0,
                calibrated=c.piece is None and c.node_index in calibrated,
            )
            for c in registry.components
        ],
        nodes=[NodeOut(**vars(n)) for n in registry.nodes],
        warnings=registry.warnings,
        detector=detector_name,
    )


def _model(db: Session, model_id: uuid.UUID) -> Model3D:
    model = db.get(Model3D, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.get("/{model_id}/components", response_model=ComponentRegistryOut)
def get_components(model_id: uuid.UUID, db: Session = Depends(get_db)) -> ComponentRegistryOut:
    """The model's inspectable components (and every mesh node, for mapping more)."""
    model = _model(db, model_id)
    return _registry_out(model, registry_for(db, model))


@router.put("/{model_id}/components", response_model=ComponentRegistryOut)
def put_components(model_id: uuid.UUID, mapping: dict, db: Session = Depends(get_db)) -> ComponentRegistryOut:
    """Saves an explicit component mapping ({"objectId", "components": [...]},
    see app/services/component_inspection/registry.py). Rejected, and nothing
    saved, if it doesn't parse; entries that don't resolve are kept and reported."""
    model = _model(db, model_id)
    if not isinstance(mapping.get("components"), list):
        raise HTTPException(status_code=400, detail="Expected {'objectId': ..., 'components': [...]}")
    path = mapping_path(str(model.id))
    previous = path.read_bytes() if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    try:
        registry = registry_for(db, model)
    except HTTPException:
        if previous is None:
            path.unlink()
        else:
            path.write_bytes(previous)
        raise
    return _registry_out(model, registry)


@router.delete("/{model_id}/components", response_model=ComponentRegistryOut)
def delete_components(model_id: uuid.UUID, db: Session = Depends(get_db)) -> ComponentRegistryOut:
    """Removes the mapping file: back to named parts / name-matched nodes."""
    model = _model(db, model_id)
    mapping_path(str(model.id)).unlink(missing_ok=True)
    return _registry_out(model, registry_for(db, model))


@router.delete("/{model_id}/components/reference", status_code=204)
def delete_reference(model_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    """Forgets the learned good state (learn it again from the AR page)."""
    model = _model(db, model_id)
    reference_path(str(model.id)).unlink(missing_ok=True)
    for _, live in _sessions.values():
        if live.session.model_id == str(model.id):
            for verifier in live.session.verifiers:
                if isinstance(verifier, ReferenceAppearanceVerifier):
                    verifier.references = None


# ── live inspection (called from vision.process_ar_frame) ───────────────

_detector_state: dict[str, object] = {}


def _detector() -> tuple[DetectorVerifier | None, str | None]:
    """The configured component detector, loaded once. A configured detector
    that fails to load is reported, not silently skipped."""
    if "verifier" not in _detector_state:
        weights, classes = _settings.component_detector_weights, _settings.component_detector_classes
        verifier, name = None, None
        if weights:
            try:
                class_map = json.loads(classes) if classes else {}
                if not class_map:
                    raise ValueError("COMPONENT_DETECTOR_CLASSES must map the model's class names to component types")
                verifier = DetectorVerifier(UltralyticsComponentDetector(weights, class_map))
                name = f"{Path(weights).name} ({', '.join(sorted(set(class_map.values())))})"
            except Exception as exc:  # noqa: BLE001 - any load failure is reported the same way
                log.error("[INSPECTION] component detector %s failed to load: %s", weights, exc)
                name = f"{Path(weights).name} FAILED TO LOAD: {exc}"
        _detector_state["verifier"], _detector_state["name"] = verifier, name
    return _detector_state["verifier"], _detector_state["name"]  # type: ignore[return-value]


def _verifiers(model_id: str) -> list[ComponentVerifier]:
    verifiers: list[ComponentVerifier] = [
        CalibratedPresenceVerifier(_calibrations(model_id)),
        ReferenceAppearanceVerifier(
            ReferenceSet.load(reference_path(model_id)),
            present_z=_settings.inspection_reference_present_z,
            absent_z=_settings.inspection_reference_absent_z,
            max_view_angle_deg=_settings.inspection_reference_view_deg,
        ),
    ]
    detector, _ = _detector()
    if detector is not None:
        verifiers.append(detector)
    return verifiers


def _config() -> InspectionConfig:
    s = _settings
    return InspectionConfig(
        interval_s=s.inspection_interval_s,
        decision=DecisionConfig(
            confirm_missing_after=s.inspection_confirm_missing_after,
            confirm_present_after=s.inspection_confirm_present_after,
            clear_missing_after=s.inspection_clear_missing_after,
            stale_after_s=s.inspection_stale_after_s,
            viewpoint_reset_deg=s.inspection_viewpoint_reset_deg,
        ),
        visibility=VisibilityConfig(
            min_tracking_confidence=s.ar_model_good_confidence,
            min_area_px=s.inspection_min_area_px,
            min_sharpness=s.inspection_min_sharpness,
            min_cad_visible=s.inspection_min_cad_visible,
            max_hand_cover=s.inspection_max_hand_cover,
        ),
    )


class _LiveInspection:
    """An InspectionSession plus what the AR loop needs to run it off its own
    thread: the latest finished results, and whether a pass is in flight."""

    MESSAGE_SHOWN_S = 6.0  # keep a one-off message (e.g. "Learned ...") in the results this long

    def __init__(self, session: InspectionSession):
        self.session = session
        self.lock = threading.Lock()  # one pass at a time
        self.busy = False
        self.last_frame: np.ndarray | None = None  # the frame of the latest finished pass (report evidence)
        # Built before any pass runs, so a frame never waits for the worker.
        self.latest: ComponentInspectionOut = _inspection_out(session, False, None)
        self.message: tuple[float, str] | None = None

    def run(
        self, mode: str, frame_bgr, t_co, camera_matrix, tracking_ok: bool, tracking_reason: str | None, hand_mask=None,
        focus=None,
    ) -> None:
        """One pass on one frame and the pose tracked for that same frame."""
        try:
            with self.lock:
                message = self.session.set_mode(mode)  # type: ignore[arg-type]
                if message:
                    self.message = (time.monotonic(), message)
                result = self.session.process(
                    frame_bgr, t_co, camera_matrix, tracking_ok, tracking_reason, hand_mask=hand_mask, focus=focus
                )
                if result.ran:
                    self.last_frame = frame_bgr
                shown = self.message[1] if self.message and time.monotonic() - self.message[0] < self.MESSAGE_SHOWN_S else None
                self.latest = _inspection_out(self.session, result.ran, shown)
        except Exception:  # noqa: BLE001 - a failed pass must not kill the worker or the AR stream
            log.exception("[INSPECTION] pass failed")
        finally:
            self.busy = False


_MAX_SESSIONS = 16
_sessions: dict[str, tuple[tuple, _LiveInspection]] = {}  # AR session id -> (key, inspection)
_sessions_lock = threading.Lock()
# Passes run here, not on the AR stream's thread: a pass (~200 ms on the
# bottle) would otherwise stall tracking. One worker: passes are CPU-bound.
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inspection")


def run_inspection(
    db: Session,
    model: Model3D,
    session_id: str,
    mode: str,
    run_id: str | None,
    frame_bgr: np.ndarray,
    t_co: np.ndarray | None,
    camera_matrix: np.ndarray,
    tracking_ok: bool,
    tracking_reason: str | None,
    hand_mask: np.ndarray | None = None,
    focus: list[str] | None = None,
) -> ComponentInspectionOut:
    """Inspection for one AR frame, with the pose tracked for this frame.

    With inspection_async (the default) the pass runs on a worker thread when
    none is in flight and the latest finished results are returned at once,
    so tracking never waits for inspection; a frame arriving while a pass
    runs is simply not inspected. Each pass uses one frame and that frame's
    own pose together, so results always match the image they came from."""
    registry = registry_for(db, model)
    key = (str(model.id), run_id, id(registry))
    with _sessions_lock:
        existing = _sessions.pop(session_id, None)
        if existing is not None and existing[0] == key:
            live = existing[1]
        else:
            run = run_id or uuid.uuid4().hex[:8]
            log_path = Path(_settings.inspection_log_dir) / str(model.id) / f"{session_id}-{run}.jsonl"
            live = _LiveInspection(InspectionSession(str(model.id), registry, _verifiers(str(model.id)), _config(),
                                                     reference_path(str(model.id)), log_path, run_id=run))
        _sessions[session_id] = (key, live)
        while len(_sessions) > _MAX_SESSIONS:
            _sessions.pop(next(iter(_sessions)))
        start = not live.busy
        if start:
            live.busy = True
    if start:
        args = (mode, frame_bgr, t_co, camera_matrix, tracking_ok, tracking_reason, hand_mask, focus)
        if _settings.inspection_async:
            _worker.submit(live.run, *args)
        else:
            live.run(*args)
    return live.latest


def live_inspection(session_id: str) -> "_LiveInspection | None":
    entry = _sessions.get(session_id)
    return entry[1] if entry else None


def end_inspection(session_id: str) -> None:
    with _sessions_lock:
        _sessions.pop(session_id, None)


def _inspection_out(session: InspectionSession, ran: bool, message: str | None) -> ComponentInspectionOut:
    learn_counts = session.learner.counts() if session.learner is not None else None
    components = []
    for c in session.registry.components:
        state = session.decisions[c.id].state
        frame = session.frames[c.id]
        projection, visibility, evidence = frame.projection, frame.visibility, frame.evidence
        roi = projection.roi if projection is not None else None
        components.append(
            ComponentResultOut(
                id=c.id,
                status=state.status.value,
                reason=state.reason,
                required=c.required,
                score=state.p_present,
                confidence=evidence.confidence if evidence else None,
                observable=state.observable,
                visibility=visibility.status if visibility else None,
                visibility_reasons=visibility.reasons if visibility else [],
                last_valid=session.wall_time(state.last_valid_s),
                valid_observations=state.valid_observations,
                center_px=list(projection.center_px) if projection is not None and projection.center_px else None,
                bbox_px=list(roi) if roi else None,
                outline_px=[list(p) for p in projection.outline] if projection is not None and roi else None,
                verdicts=[VerdictOut(verifier=v.verifier, decision=v.decision, confidence=v.confidence, reason=v.reason)
                          for v in (evidence.verdicts if evidence else [])],
                history=state.history[-5:],
                learn_samples=learn_counts.get(c.id, 0) if learn_counts is not None else None,
            )
        )
    notes = list(session.registry.warnings)
    if session.focus:
        notes.insert(0, f"Step focus: {', '.join(sorted(session.focus))}")
    if session.occlusion_note:
        notes.append(session.occlusion_note)
    summary = session.summary()
    return ComponentInspectionOut(
        run_id=session.run_id,
        mode=session.mode,
        passes=session.passes,
        inspected_this_frame=ran,
        summary=InspectionSummaryOut(**vars(summary)),
        components=components,
        message=message,
        notes=notes,
    )
