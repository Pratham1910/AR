"""
Per-model tracking shells (see tracking_shell.py): built on request, stored
next to the models (`<models_3d_dir>/tracking/<model id>.glb` + `.json`), and
used for tracking only while they were built from the model's current GLB
(a re-uploaded GLB makes the shell stale; it's then ignored, never mixed).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.services.model3d.tracking_shell import build_tracking_glb


def _paths(model_id: str) -> tuple[Path, Path]:
    folder = Path(get_settings().models_3d_dir) / "tracking"
    return folder / f"{model_id}.glb", folder / f"{model_id}.json"


def build(model_id: str, glb: bytes) -> dict:
    """Build and store the shell; returns its report."""
    shell, report = build_tracking_glb(glb)
    glb_path, json_path = _paths(model_id)
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    glb_path.write_bytes(shell)
    info = asdict(report) | {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shell_sha256": hashlib.sha256(shell).hexdigest(),
    }
    json_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def status(model_id: str, glb: bytes | None) -> dict | None:
    """The stored report (+ "stale": built from another version of the GLB), or None."""
    glb_path, json_path = _paths(model_id)
    if not (glb_path.exists() and json_path.exists()):
        return None
    info = json.loads(json_path.read_text(encoding="utf-8"))
    info["stale"] = glb is not None and info.get("source_sha256") != hashlib.sha256(glb).hexdigest()
    return info


def load(model_id: str, glb: bytes) -> tuple[bytes, list[float], str] | None:
    """(shell GLB, the original's frame centre in GLB units, shell hash) when a
    shell built from exactly this GLB exists, else None (track the original)."""
    info = status(model_id, glb)
    if info is None or info["stale"]:
        return None
    return _paths(model_id)[0].read_bytes(), info["frame_center"], info["shell_sha256"]


def remove(model_id: str) -> None:
    for path in _paths(model_id):
        path.unlink(missing_ok=True)
