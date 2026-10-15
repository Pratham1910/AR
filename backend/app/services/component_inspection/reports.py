"""
Inspection reports: a saved snapshot of one live component inspection —
verdict, every component's status and the reason for it, who and where —
plus an evidence frame with each component's outline drawn in its status
colour. Stored as files (data/inspection_reports/<id>.json + .jpg), like
the rest of the inspection data, so no database migration is needed.

A report is a record: its findings never change after saving. Only the
inspector's note and the disposition (approved / sent to rework) can be
set afterwards, and when they were is kept.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# Outline colours (BGR) by status, matching the UI's.
_COLORS = {
    "PRESENT": (154, 227, 95),
    "MISSING_CONFIRMED": (92, 92, 255),
    "MISSING_CANDIDATE": (32, 176, 255),
    "UNCERTAIN": (180, 168, 154),
    "OCCLUDED": (180, 168, 154),
    "NOT_INSPECTED": (180, 168, 154),
}

DISPOSITIONS = ("approved", "rework")
_ID = re.compile(r"^[0-9a-f]{12}$")


def new_report_id() -> str:
    return uuid.uuid4().hex[:12]


def valid_id(report_id: str) -> bool:
    return bool(_ID.match(report_id))


def evidence_image(frame_bgr: np.ndarray, components: list[dict]) -> bytes:
    """The frame with each component's projected outline in its status colour,
    labelled with its ID — what the inspection looked at."""
    image = frame_bgr.copy()
    for c in components:
        outline = c.get("outline_px")
        if not outline or len(outline) < 3:
            continue
        color = _COLORS.get(c["status"], (180, 168, 154))
        points = np.round(np.asarray(outline)).astype(np.int32)
        thickness = 3 if c["status"] in ("MISSING_CONFIRMED", "MISSING_CANDIDATE") else 2
        cv2.polylines(image, [points], True, color, thickness, cv2.LINE_AA)
        x, y = points[:, 0].max() + 8, points[:, 1].min() + 14
        label = f"{c['id']} {c['status'].replace('_CONFIRMED', '').replace('_', ' ')}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (int(x) - 3, int(y) - h - 5), (int(x) + w + 3, int(y) + 4), (20, 15, 11), -1)
        cv2.putText(image, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("Could not encode the evidence image")
    return data.tobytes()


class ReportStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def _json(self, report_id: str) -> Path:
        return self.directory / f"{report_id}.json"

    def evidence_path(self, report_id: str) -> Path:
        return self.directory / f"{report_id}.jpg"

    def save(self, report: dict, evidence_jpeg: bytes | None) -> dict:
        self.directory.mkdir(parents=True, exist_ok=True)
        report = {**report, "evidence": evidence_jpeg is not None}
        if evidence_jpeg is not None:
            self.evidence_path(report["id"]).write_bytes(evidence_jpeg)
        self._json(report["id"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def get(self, report_id: str) -> dict | None:
        if not valid_id(report_id) or not self._json(report_id).exists():
            return None
        return json.loads(self._json(report_id).read_text(encoding="utf-8"))

    def list(self, limit: int = 200) -> list[dict]:
        """Newest first; summaries only (no per-component detail)."""
        if not self.directory.exists():
            return []
        reports = []
        for path in self.directory.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            reports.append({k: v for k, v in data.items() if k != "components"})
        reports.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return reports[:limit]

    def review(self, report_id: str, note: str | None, disposition: str | None, reviewer: str | None) -> dict | None:
        """Sets the inspector's note and/or disposition; the findings stay as saved."""
        report = self.get(report_id)
        if report is None:
            return None
        now = datetime.now(timezone.utc).isoformat()
        if note is not None:
            report["note"] = note
        if disposition is not None:
            if disposition not in DISPOSITIONS:
                raise ValueError(f"disposition must be one of {DISPOSITIONS}")
            report["disposition"] = disposition
            report["disposition_at"] = now
            report["disposition_by"] = reviewer
        report["updated_at"] = now
        self._json(report_id).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
