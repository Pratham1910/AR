"""
What a product scan remembers between frames (one per camera session).

Two things, both to keep a live scan from flickering without making it
gullible:

- Tags seen a moment ago are held for `hold_ms`, so a tag lost for a frame or
  two (glare, a passing hand) doesn't drop the product back to "3 of 4 tags".
  A held tag is flagged as such and expires on its own.
- A product is confirmed only after `confirm_frames` consecutive frames that
  identified AND verified it. Any other outcome resets the count.
"""

from dataclasses import replace

from app.services.markers.tags import MarkerDetection


class ScanSession:
    def __init__(self, hold_ms: float = 600.0, confirm_frames: int = 3):
        self._hold_ms = hold_ms
        self._confirm_frames = max(1, confirm_frames)
        self._last_seen: dict[int, tuple[float, MarkerDetection]] = {}
        self._streak_asset: str | None = None
        self._streak = 0

    @property
    def confirm_frames(self) -> int:
        return self._confirm_frames

    def with_held(self, detections: list[MarkerDetection], now_ms: float) -> list[MarkerDetection]:
        """This frame's tags plus those seen within the last `hold_ms`."""
        for detection in detections:
            self._last_seen[detection.marker_id] = (now_ms, detection)
        self._last_seen = {i: seen for i, seen in self._last_seen.items() if now_ms - seen[0] <= self._hold_ms}
        current = {d.marker_id for d in detections}
        held = [replace(d, held=True) for i, (_, d) in self._last_seen.items() if i not in current]
        return detections + held

    def record(self, verified_asset_id: str | None) -> int:
        """Counts consecutive verified frames for one product; returns the count."""
        if verified_asset_id is None or verified_asset_id != self._streak_asset:
            self._streak = 0
        self._streak_asset = verified_asset_id
        if verified_asset_id is not None:
            self._streak += 1
        return self._streak

    def is_confirmed(self) -> bool:
        return self._streak >= self._confirm_frames
