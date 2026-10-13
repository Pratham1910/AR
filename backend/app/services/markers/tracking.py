"""
Four-tag tracking over time: one session per camera (docs/marker-tags.md,
"Tracking states").

Per frame the API asks two things of a session:

  locate()  which product's four tags are in view and the region they enclose
            — bridging a covered tag of the product already being tracked
  finish()  given what the detector found inside that region, the state

Tracking is declared only after `confirm_frames` consecutive frames in which
all four tags AND the object inside them agree. Once tracking:

- A covered tag (a hand over a corner) is re-projected from the visible tags
  of the same product: the homography that maps those tags' corners from the
  last frame in which all four were read to where they are now is applied to
  the covered tag's last corners. Allowed for `hold_ms` after the last full
  read and only with `min_visible` tags still visible; estimated tags are
  flagged in the response. A product is never *acquired* from estimated tags.
- A frame in which the detector misses the object keeps the last object for
  `hold_ms`, flagged as held.

Pure state over tag positions and verdicts; the detector is run by the API.
"""

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from app.models.enums import MarkerCorner
from app.services.markers.geometry import QUAD_ORDER
from app.services.markers.identification import FourTagSet, Registry, SetStatus, check_set, find_four_tag_set
from app.services.markers.region_detection import RegionDetection, RegionObject, RegionStatus
from app.services.markers.tags import TagSighting


class TrackState(str, Enum):
    # Stopped before the region (FourTagSet statuses other than COMPLETE).
    NO_MARKERS = "NO_MARKERS"
    UNKNOWN_MARKERS = "UNKNOWN_MARKERS"
    DUPLICATE_MARKERS = "DUPLICATE_MARKERS"
    INCOMPLETE = "INCOMPLETE"
    MIXED_PRODUCTS = "MIXED_PRODUCTS"
    MULTIPLE_PRODUCTS = "MULTIPLE_PRODUCTS"
    LAYOUT_INVALID = "LAYOUT_INVALID"
    # Region found; what is inside it.
    OBJECT_NOT_FOUND = "OBJECT_NOT_FOUND"
    WRONG_OBJECT = "WRONG_OBJECT"
    VERIFYING = "VERIFYING"  # object found, counting consecutive frames
    TRACKING = "TRACKING"


@dataclass
class Located:
    found: FourTagSet
    # Tags re-projected because they were covered; empty when all four were read.
    estimated: list[MarkerCorner] = field(default_factory=list)


@dataclass
class TrackOutcome:
    state: TrackState
    streak: int
    obj: RegionObject | None = None
    object_held: bool = False  # the detector missed this frame; last object kept
    lost: bool = False  # tracking just ended (hold expired / product gone)


class FourTagTracker:
    def __init__(self, hold_ms: float, confirm_frames: int, min_visible: int, min_region_in_tags: float):
        self.hold_ms = hold_ms
        self.confirm_frames = confirm_frames
        self.min_visible = min_visible
        self.min_region_in_tags = min_region_in_tags
        self._reset()

    def _reset(self) -> None:
        self.asset_id: str | None = None
        self._reference: dict[MarkerCorner, np.ndarray] = {}  # corners at the last full read
        self._reference_ms = 0.0
        self._streak = 0
        self._tracking = False
        self._last_object: RegionObject | None = None
        self._last_object_ms = 0.0

    @property
    def tracking(self) -> bool:
        return self._tracking

    def locate(self, sightings: list[TagSighting], registry: Registry, duplicate_ids: list[int], now_ms: float) -> Located:
        found = find_four_tag_set(sightings, registry, duplicate_ids, self.min_region_in_tags)
        if found.status is SetStatus.COMPLETE:
            if found.asset_id != self.asset_id:
                self._reset()
                self.asset_id = found.asset_id
            self._reference = {corner: s.corners.copy() for corner, s in found.tags.items()}
            self._reference_ms = now_ms
            return Located(found)

        if self._tracking and now_ms - self._reference_ms <= self.hold_ms and found.status is not SetStatus.DUPLICATE_MARKERS:
            bridged = self._bridge(sightings, registry)
            if bridged is not None:
                return bridged
        return Located(found)

    def _bridge(self, sightings: list[TagSighting], registry: Registry) -> Located | None:
        """Re-project the locked product's covered tags from its visible ones."""
        ids = {a.corner: marker_id for marker_id, a in registry.items() if a.asset_id == self.asset_id}
        visible = {
            registry[s.marker_id].corner: s
            for s in sightings
            if s.marker_id in registry and registry[s.marker_id].asset_id == self.asset_id
        }
        if not self.min_visible <= len(visible) < len(QUAD_ORDER) or len(ids) != len(QUAD_ORDER):
            return None
        src = np.vstack([self._reference[c] for c in visible]).astype(np.float32)
        dst = np.vstack([visible[c].corners for c in visible]).astype(np.float32)
        homography, _ = cv2.findHomography(src, dst, 0)
        if homography is None:
            return None
        tags = dict(visible)
        missing = [c for c in QUAD_ORDER if c not in visible]
        for corner in missing:
            projected = cv2.perspectiveTransform(self._reference[corner].reshape(-1, 1, 2).astype(np.float32), homography)
            tags[corner] = TagSighting(ids[corner], projected.reshape(4, 2).astype(np.float64), estimated=True)
        others = [s for s in sightings if s.marker_id in registry and registry[s.marker_id].asset_id != self.asset_id]
        result = check_set(self.asset_id, tags, others, self.min_region_in_tags)
        if result.status is not SetStatus.COMPLETE:
            return None
        return Located(result, estimated=missing)

    def finish(self, located: Located, detection: RegionDetection | None, now_ms: float) -> TrackOutcome:
        """The frame's state, given the detector's verdict on the region
        (None when there was no usable region)."""
        found = located.found
        if found.status is not SetStatus.COMPLETE or detection is None:
            was_tracking = self._tracking
            self._streak = 0
            self._tracking = False
            if now_ms - self._reference_ms > self.hold_ms:
                self._last_object = None
            return TrackOutcome(TrackState(found.status.value), 0, lost=was_tracking)

        if detection.status is RegionStatus.FOUND:
            self._streak += 1
            self._last_object, self._last_object_ms = detection.best, now_ms
            if self._streak >= self.confirm_frames:
                self._tracking = True
            state = TrackState.TRACKING if self._tracking else TrackState.VERIFYING
            return TrackOutcome(state, self._streak, detection.best)

        if self._tracking and self._last_object is not None and now_ms - self._last_object_ms <= self.hold_ms:
            return TrackOutcome(TrackState.TRACKING, self._streak, self._last_object, object_held=True)

        was_tracking = self._tracking
        self._streak = 0
        self._tracking = False
        state = TrackState.WRONG_OBJECT if detection.status is RegionStatus.WRONG_OBJECT else TrackState.OBJECT_NOT_FOUND
        return TrackOutcome(state, 0, detection.best, lost=was_tracking)


class ProductLookupSession:
    """Single-tag product scan: the same product named in `required`
    consecutive frames before it is shown, so one misread frame does not
    flash a product up."""

    def __init__(self, required: int):
        self.required = required
        self._asset_id: str | None = None
        self._streak = 0

    def record(self, asset_id: str | None) -> int:
        if asset_id is None or asset_id != self._asset_id:
            self._asset_id = asset_id
            self._streak = 1 if asset_id is not None else 0
        else:
            self._streak += 1
        return self._streak

    @property
    def identified(self) -> bool:
        return self._asset_id is not None and self._streak >= self.required
