"""
Recorded camera clips and their replay metrics — Phae-4.md's movement tests
(still, move left/right, closer/further, rotate, move the camera) as a
repeatable benchmark instead of eyeballing a live view.

A clip is raw camera frames with their capture times (data/clips/<name>/).
Replay feeds them through the live AR endpoint at the recorded pace, always
taking the newest frame that has "arrived" — like the browser loop, so a slow
tracker skips frames exactly as it would live — and measures what the user
would have seen.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_CV_TO_GL = np.diag([1.0, -1.0, -1.0])


@dataclass
class ClipFrame:
    index: int
    t_ms: float  # capture time since the clip started
    path: Path


def clip_dir(root: Path, name: str) -> Path:
    return Path(root) / name


def load_clip(directory: Path) -> list[ClipFrame]:
    """The clip's frames in capture order (index.jsonl: one {"index", "t_ms"} per line)."""
    frames = []
    for line in (directory / "index.jsonl").read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            frames.append(ClipFrame(entry["index"], float(entry["t_ms"]), directory / f"{entry['index']:05d}.jpg"))
    frames.sort(key=lambda f: f.index)
    return frames


def next_frame(frames: list[ClipFrame], elapsed_ms: float, after: int) -> int | None:
    """
    Position (in `frames`) of the frame a live loop would grab now: the newest
    one captured by `elapsed_ms`, later than position `after`. If none has
    arrived yet, the next one (the loop would wait for it). None at the end.
    """
    if after + 1 >= len(frames):
        return None
    best = after + 1
    for i in range(after + 1, len(frames)):
        if frames[i].t_ms <= elapsed_ms:
            best = i
        else:
            break
    return best


def threejs_pose_to_cv(position, quaternion) -> np.ndarray:
    """The 4x4 OpenCV-camera pose of a model shown at this Three.js position/quaternion
    (inverse of transforms.cv_model_pose_to_threejs)."""
    x, y, z, w = quaternion
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    pose = np.eye(4)
    pose[:3, :3] = _CV_TO_GL @ rotation
    pose[:3, 3] = _CV_TO_GL @ np.asarray(position, float)
    return pose


@dataclass
class ReplayRecord:
    clip_index: int
    t_ms: float  # capture time of the frame processed
    latency_ms: float  # request round trip
    state: str
    detector_ran: bool
    position: tuple[float, float, float] | None  # shown pose, renderer frame (m)
    quaternion: tuple[float, float, float, float] | None
    overlap: float | None = None  # silhouette IoU of the shown pose vs the object's outline (sampled)


def _angle_deg(q1, q2) -> float:
    dot = abs(float(np.dot(q1, q2)))
    return math.degrees(2 * math.acos(min(1.0, dot)))


def summarize(records: list[ReplayRecord], clip_frames: int, clip_seconds: float) -> dict:
    """Numbers for comparing runs. Steps are between consecutive processed frames while tracking."""
    tracking = [r for r in records if r.state == "TRACKING"]
    latencies = np.array([r.latency_ms for r in records]) if records else np.zeros(1)
    steps_mm, turns_deg = [], []
    for a, b in zip(records, records[1:]):
        if a.state == b.state == "TRACKING" and a.position and b.position:
            steps_mm.append(float(np.linalg.norm(np.subtract(b.position, a.position))) * 1000)
            turns_deg.append(_angle_deg(a.quaternion, b.quaternion))
    steps, turns = np.array(steps_mm or [0.0]), np.array(turns_deg or [0.0])
    # A jump: a step far bigger than the clip's typical motion — what reads as "the model twitched".
    jump_mm = max(5.0, 4 * float(np.median(steps)))
    jump_deg = max(5.0, 4 * float(np.median(turns)))
    lost_events = sum(1 for a, b in zip(records, records[1:]) if a.state == "TRACKING" and b.state != "TRACKING")
    overlaps = np.array([r.overlap for r in records if r.overlap is not None])

    def pct(values: np.ndarray, q: float) -> float:
        return round(float(np.percentile(values, q)), 2)

    return {
        "clip_frames": clip_frames,
        "clip_seconds": round(clip_seconds, 1),
        "frames_processed": len(records),
        "updates_per_second": round(len(records) / clip_seconds, 1) if clip_seconds > 0 else 0.0,
        "tracking_share": round(len(tracking) / len(records), 3) if records else 0.0,
        "detector_runs": sum(r.detector_ran for r in records),
        "lost_events": lost_events,
        "latency_ms_p50": pct(latencies, 50),
        "latency_ms_p95": pct(latencies, 95),
        "step_mm_p50": pct(steps, 50),
        "step_mm_p95": pct(steps, 95),
        "turn_deg_p50": pct(turns, 50),
        "turn_deg_p95": pct(turns, 95),
        "jumps": int(((steps > jump_mm) | (turns > jump_deg)).sum()),
        "overlap_mean": round(float(overlaps.mean()), 3) if len(overlaps) else None,
        "overlap_min": round(float(overlaps.min()), 3) if len(overlaps) else None,
        "overlap_samples": int(len(overlaps)),
    }


# Which direction is better, for the comparison with the previous run.
HIGHER_IS_BETTER = {"updates_per_second", "tracking_share", "overlap_mean", "overlap_min"}
LOWER_IS_BETTER = {"detector_runs", "lost_events", "latency_ms_p50", "latency_ms_p95", "step_mm_p95", "turn_deg_p95", "jumps"}
