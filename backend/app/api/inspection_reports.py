"""
Inspection reports (saved from a live component inspection) and the
workstation's system status.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api import component_inspection
from app.core.config import get_settings
from app.core.database import get_db
from app.models.model3d import Model3D
from app.services.component_inspection.reports import ReportStore, evidence_image, new_report_id
from app.services.pose.calibration import load_calibration

router = APIRouter(tags=["reports"])
_settings = get_settings()


def _store() -> ReportStore:
    return ReportStore(Path(_settings.inspection_reports_dir))


class SaveReportRequest(BaseModel):
    session_id: str  # the AR session whose inspection to record
    station: str | None = None
    operator: str | None = None
    note: str | None = None


class ReviewRequest(BaseModel):
    note: str | None = None
    disposition: Literal["approved", "rework"] | None = None
    reviewer: str | None = None


@router.post("/api/inspection-reports", status_code=201)
def save_report(payload: SaveReportRequest, db: Session = Depends(get_db)) -> dict:
    """Records the session's inspection as it stands now (latest finished pass)."""
    live = component_inspection.live_inspection(payload.session_id)
    if live is None or live.latest.passes == 0:
        raise HTTPException(status_code=404, detail="No inspection has run in this session yet — start one first.")
    session = live.session
    model = db.get(Model3D, _as_uuid(session.model_id))
    out = live.latest
    registry = {c.id: c for c in session.registry.components}
    components = []
    for r in out.components:
        c = registry.get(r.id)
        components.append({
            "id": r.id,
            "type": c.type if c else None,
            "display_name": c.display_name if c else r.id,
            "required": r.required,
            "status": r.status,
            "reason": r.reason,
            "score": r.score,
            "valid_observations": r.valid_observations,
            "last_valid": r.last_valid,
            "visibility": r.visibility,
            "verdicts": [v.model_dump() for v in r.verdicts],
            "outline_px": r.outline_px,
        })
    frame = live.last_frame
    evidence = evidence_image(frame, components) if frame is not None else None
    report = {
        "id": new_report_id(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": session.model_id,
        "model_name": model.name if model else session.registry.object_id,
        "object_id": session.registry.object_id,
        "station": payload.station,
        "operator": payload.operator,
        "run_id": out.run_id,
        "passes": out.passes,
        "verdict": out.summary.verdict,
        "summary": out.summary.model_dump(),
        "components": [{k: v for k, v in c.items() if k != "outline_px"} for c in components],
        "notes": out.notes,
        "note": payload.note or "",
        "disposition": None,
    }
    return _store().save(report, evidence)


@router.get("/api/inspection-reports")
def list_reports() -> list[dict]:
    return _store().list()


@router.get("/api/inspection-reports/{report_id}")
def get_report(report_id: str) -> dict:
    report = _store().get(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.get("/api/inspection-reports/{report_id}/evidence.jpg")
def get_evidence(report_id: str) -> FileResponse:
    store = _store()
    report = store.get(report_id)
    if report is None or not store.evidence_path(report_id).exists():
        raise HTTPException(status_code=404, detail="No evidence image for this report")
    return FileResponse(store.evidence_path(report_id), media_type="image/jpeg")


@router.patch("/api/inspection-reports/{report_id}")
def review_report(report_id: str, payload: ReviewRequest) -> dict:
    """The inspector's note and disposition. The findings themselves never change."""
    report = _store().review(report_id, payload.note, payload.disposition, payload.reviewer)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.get("/api/system/status")
def system_status() -> dict:
    """What the workstation is running with: lens model, pose service, hand and component detectors."""
    from app.api import vision  # imported here: vision builds detectors at import

    calibration = load_calibration(_settings.camera_calibration_path, 1280, 720, _settings.camera_assumed_hfov_deg)
    pose: dict = {"url": _settings.pose_service_url, "ok": False}
    try:
        response = httpx.get(f"{_settings.pose_service_url}/health", timeout=2.0)
        body = response.json()
        pose.update(ok=bool(body.get("ok")), device=body.get("device"), cuda=body.get("cuda"),
                    model=body.get("model"), objects=len(body.get("objects") or []), error=body.get("gpu_error"))
    except (httpx.HTTPError, ValueError) as exc:
        pose["error"] = f"not reachable ({type(exc).__name__}) — start it in WSL (pose_service/README.md)"
    hands = None
    if _settings.hand_mask_method != "off":
        segmenter = vision._get_hand_segmenter()
        hands = {
            "method": _settings.hand_mask_method,
            "available": segmenter is not None,
            "mediapipe": bool(segmenter and segmenter.hands is not None),
            "person": bool(segmenter and segmenter.persons is not None),
            "error": vision._hand_segmenter_error,
        }
    _, detector = component_inspection._detector()
    return {
        "camera": {
            "assumed_hfov_deg": _settings.camera_assumed_hfov_deg,
            "vertical_fov_deg_720p": round(calibration.vertical_fov_deg(), 1),
            "calibration_source": calibration.source,
            "approximate": calibration.is_approximate,
        },
        "pose_service": pose,
        "hands": hands,
        "component_detector": detector,
        "inspection": {
            "confirm_missing_after": _settings.inspection_confirm_missing_after,
            "confirm_present_after": _settings.inspection_confirm_present_after,
            "interval_s": _settings.inspection_interval_s,
        },
    }


def _as_uuid(value: str):
    import uuid

    try:
        return uuid.UUID(value)
    except ValueError:
        return None
