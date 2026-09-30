"""
Video-to-procedure candidate extraction API (Project.md #28/#29, Phase 3).

Returns candidates only — it never writes a Procedure/Step row. A human
reviews the candidates (approve/reject each) and an authoring workflow turns
approved ones into a real ProcedureDefinition via the existing
POST /api/procedures endpoint (app/api/procedures.py). Skipping that human
step is explicitly disallowed by Project.md #28.
"""

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.config import get_settings
from app.schemas.procedure import CandidateStep
from app.services.state_detection.state_engine import ComponentStateRule
from app.services.video.candidate_extraction import DebounceConfig, extract_candidates_from_video
from app.services.video.extraction import VideoExtractionError
from app.services.vision.detector import build_detector

router = APIRouter(prefix="/api/video", tags=["video"])

_settings = get_settings()
_detector = build_detector(_settings.model_path)


class StateRuleInput(BaseModel):
    component_id: str
    class_label: str
    present_state_id: str
    absent_state_id: str


class ExtractCandidatesResponse(BaseModel):
    candidates: list[CandidateStep]
    rules_applied: int


@router.post("/extract-candidates", response_model=ExtractCandidatesResponse)
async def extract_candidates(
    file: UploadFile = File(...),
    rules: str = Form(..., description="JSON array of {component_id, class_label, present_state_id, absent_state_id}"),
    sample_every_n_frames: int = Form(15),
    min_consecutive_frames: int = Form(2),
) -> ExtractCandidatesResponse:
    """
    Uploads a video and one presence/absence rule per component of interest
    (the same rule shape the live inspection flow uses — see
    app/services/state_detection/state_engine.py), and returns every
    debounced state transition found, as PENDING_REVIEW candidates.
    """
    try:
        rule_inputs = [StateRuleInput.model_validate(r) for r in json.loads(rules)]
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid rules JSON: {exc}") from exc

    if not rule_inputs:
        raise HTTPException(status_code=400, detail="At least one rule is required")

    component_rules = [
        ComponentStateRule(
            component_id=r.component_id,
            class_label=r.class_label,
            present_state_id=r.present_state_id,
            absent_state_id=r.absent_state_id,
        )
        for r in rule_inputs
    ]

    contents = await file.read()
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        candidates = extract_candidates_from_video(
            tmp_path,
            _detector,
            component_rules,
            sample_every_n_frames=sample_every_n_frames,
            debounce=DebounceConfig(min_consecutive_frames=min_consecutive_frames),
        )
    except VideoExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return ExtractCandidatesResponse(candidates=candidates, rules_applied=len(component_rules))
