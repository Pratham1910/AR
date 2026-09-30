"""
Multi-object tracking (Project.md #17, Phase 2): frame-to-frame identity for
detected components — a separate architectural layer from detection itself
(Project.md #3: detection answers "what", tracking answers "where over
time"; neither one decides PASS/FAIL).

Uses ByteTrack via the `supervision` library: track association from
detections only, no extra re-identification model to download, matching
Project.md #17's explicit tolerance requirements (movement, rotation,
temporary occlusion, partial visibility) via ByteTrack's own two-stage
high/low-confidence matching and short-term track buffering.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

from app.schemas.vision import BoundingBox, Detection


class ObjectTracker:
    """
    One instance per physical camera / inspection session — ByteTrack is
    stateful and needs continuous frame-to-frame updates to maintain
    identity, so instances are cached per session_id by the API layer
    (app/api/vision.py), not created fresh per request.
    """

    def __init__(self, lost_track_buffer_frames: int = 30):
        self._tracker = sv.ByteTrack(lost_track_buffer=lost_track_buffer_frames)
        self._class_names: dict[int, str] = {}
        self._name_to_class_id: dict[str, int] = {}

    def _class_id_for(self, label: str) -> int:
        if label in self._name_to_class_id:
            return self._name_to_class_id[label]
        class_id = len(self._name_to_class_id)
        self._name_to_class_id[label] = class_id
        self._class_names[class_id] = label
        return class_id

    def update(self, detections: list[Detection]) -> list[Detection]:
        """
        Feeds one frame's detections through ByteTrack and returns the same
        detections with `tracker_id` populated (stable across calls for the
        same physical object) — or an empty list if none were detected this
        frame (still advances ByteTrack's internal frame counter so lost
        tracks age out correctly per `lost_track_buffer_frames`).

        Note: a brand-new track (one that doesn't match anything already
        being tracked) is NOT returned on the frame it first appears —
        ByteTrack requires it to be matched again on the next frame before
        "activating" it (verified directly against the underlying
        `supervision` library). This suppresses one-off spurious detections
        rather than tracking every fleeting false positive; callers should
        expect a ~1 frame delay before a genuinely new object gets a stable
        `tracker_id`.
        """
        if not detections:
            self._tracker.update_with_detections(sv.Detections.empty())
            return []

        xyxy = np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([self._class_id_for(d.class_label) for d in detections], dtype=int)

        sv_detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        tracked = self._tracker.update_with_detections(sv_detections)

        results: list[Detection] = []
        for i in range(len(tracked)):
            x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
            class_id_value = int(tracked.class_id[i]) if tracked.class_id is not None else -1
            tracker_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None
            confidence_value = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            results.append(
                Detection(
                    class_label=self._class_names.get(class_id_value, "unknown"),
                    confidence=confidence_value,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    tracker_id=tracker_id,
                )
            )
        return results

    def reset(self) -> None:
        """Clears all track state — e.g. when an inspection run ends or the camera session restarts."""
        self._tracker.reset()
