"""
Keeps a marker "present" for a short while after it stops being detected, so
a hand passing over it, motion blur or a brief exit from the frame doesn't
make its product flicker away. A held marker is reported with visible=False
and its last known corners; once it has been gone longer than hold_ms it is
dropped.
"""

import time
from dataclasses import dataclass

from app.services.markers.detector import MarkerDetection


@dataclass(frozen=True)
class TrackedMarker:
    detection: MarkerDetection
    visible: bool  # detected on this frame (False = held from an earlier one)
    ms_since_seen: float


class MarkerPresenceTracker:
    def __init__(self, hold_ms: float):
        self._hold_ms = hold_ms
        self._last_seen: dict[int, tuple[float, MarkerDetection]] = {}

    def update(self, detections: list[MarkerDetection], now_ms: float | None = None) -> list[TrackedMarker]:
        now = time.monotonic() * 1000.0 if now_ms is None else now_ms
        tracked = [TrackedMarker(d, visible=True, ms_since_seen=0.0) for d in detections]
        for detection in detections:
            self._last_seen[detection.marker_id] = (now, detection)

        visible_ids = {d.marker_id for d in detections}
        for marker_id, (seen_at, detection) in list(self._last_seen.items()):
            if marker_id in visible_ids:
                continue
            age = now - seen_at
            if age > self._hold_ms:
                del self._last_seen[marker_id]
            else:
                tracked.append(TrackedMarker(detection, visible=False, ms_since_seen=age))
        return tracked
