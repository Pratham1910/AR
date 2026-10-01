"""
Detect once, then track: the live AR state machine (one per camera session).

    SEARCHING    --detector finds the object-->                     INITIALIZING
    INITIALIZING --initial 6DoF pose + tracker lock succeed-->      TRACKING
    INITIALIZING --lock fails-->                                    SEARCHING / RECOVERING
    TRACKING     --tracker confidence below lost threshold-->       LOST
    LOST         --tracker recovers within the grace frames-->      TRACKING
    LOST         --grace frames used up-->                          RECOVERING
    RECOVERING   --detector re-finds the object-->                  INITIALIZING (same object id)
    RECOVERING   --not re-found within the timeout-->               SEARCHING (model hidden)

The detector runs only in SEARCHING and RECOVERING, at most once per
`detect_interval_ms`. Detection and pose initialization are separate steps
(SEARCHING finds the object; the next frame, INITIALIZING, computes its
initial pose), so the slow first pose search shows up as its own state. In
TRACKING only the tracker runs. Every accepted pose goes through PoseFilter.

The tracker is pluggable (PoseTracker): optical flow for markerless mode,
optical flow + the MegaPose refiner for model-based mode. This module never
touches images or models itself, so it's unit-testable with fakes.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np

from app.schemas.vision import BoundingBox, SegmentedObject, Vector2
from app.services.tracking.pose_filter import PoseFilter, PoseFilterConfig

log = logging.getLogger("uvicorn.error").getChild("ar")


class TrackingState(str, Enum):
    SEARCHING = "SEARCHING"
    INITIALIZING = "INITIALIZING"
    TRACKING = "TRACKING"
    LOST = "LOST"
    RECOVERING = "RECOVERING"


@dataclass
class Measurement:
    """One tracker output. `confidence` is on the tracker's own scale; the
    thresholds in ARTrackingConfig are chosen per tracker."""

    confidence: float
    position: tuple[float, float, float] | None = None  # renderer (Three.js) space, meters
    quaternion: tuple[float, float, float, float] | None = None
    bbox: BoundingBox | None = None
    polygon: list[Vector2] | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # e.g. t_camera_object, flow_ms, pose_service_ms


class PoseTracker(Protocol):
    # Whether update() can still recover after a miss (MegaPose refining from
    # the last good pose can; optical flow whose points were lost can't).
    can_recover: bool

    def initialize(self, frame: Any, detection: SegmentedObject) -> Measurement: ...

    def update(self, frame: Any) -> Measurement: ...

    def commit(self, measurement: Measurement) -> None:
        """The session accepted this measurement as the object's pose."""


@dataclass
class ARTrackingConfig:
    detect_interval_ms: float = 150.0  # detector rate cap while SEARCHING/RECOVERING
    good_confidence: float = 0.7  # at/above: healthy; below: tracking with warning
    lost_confidence: float = 0.4  # below: tracking lost
    grace_frames: int = 2  # LOST frames that hold the last pose before RECOVERING
    lost_timeout_ms: float = 1500.0  # not re-acquired this long after loss -> hide, SEARCHING
    filter: PoseFilterConfig = field(default_factory=PoseFilterConfig)


@dataclass
class TrackedObject:
    object_id: int
    class_label: str
    confidence: float
    first_seen_frame: int
    last_seen_frame: int
    last_seen_at: float
    position: tuple[float, float, float] | None = None  # filtered
    quaternion: tuple[float, float, float, float] | None = None  # filtered
    bbox: BoundingBox | None = None
    polygon: list[Vector2] | None = None
    velocity_px_s: tuple[float, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    state: TrackingState
    visible: bool  # draw the model (tracking, or holding the last valid pose while lost/re-acquiring)
    monitoring: bool  # TRACKING with confidence below good_confidence ("tracking with warning")
    obj: TrackedObject | None
    detection: SegmentedObject | None  # what the detector found this frame, if it ran and found it
    detector_ran: bool
    tracker_ran: bool
    timings_ms: dict[str, float]  # detection / initialization / tracking / refinement
    events: list[str]


def _center(bbox: BoundingBox | None) -> tuple[float, float] | None:
    return None if bbox is None else ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)


def euler_xyz_deg(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Display only: XYZ Euler angles (degrees) of a quaternion, as Three.js's Euler('XYZ')."""
    x, y, z, w = q
    m11, m12, m13 = 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)
    m22, m23 = 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)
    m32, m33 = 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)
    ry = math.asin(max(-1.0, min(1.0, m13)))
    if abs(m13) < 0.9999999:
        rx, rz = math.atan2(-m23, m33), math.atan2(-m12, m11)
    else:
        rx, rz = math.atan2(m32, m22), 0.0
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


class ARSession:
    def __init__(
        self,
        class_label: str,
        detector: Callable[[Any], SegmentedObject | None],
        tracker: PoseTracker,
        config: ARTrackingConfig,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.class_label = class_label
        self.detector = detector
        self.tracker = tracker
        self.config = config
        self.clock = clock
        self.filter = PoseFilter(config.filter)

        self.state = TrackingState.SEARCHING
        self.obj: TrackedObject | None = None
        self.frame_index = 0
        self.detection_runs = 0  # detector calls, found or not
        self.detection_count = 0  # successful detections: 1 at first lock, +1 per re-acquisition
        self.tracking_frames = 0
        self.last_detection_frame: int | None = None
        self._next_object_id = 1
        self._pending: SegmentedObject | None = None  # found by the detector, pose not yet initialized
        self._last_detection_at: float | None = None
        self._lost_since: float | None = None
        self._lost_frames = 0
        self._announced_search = False
        self._last_status_log = 0.0
        self._last_moved_log = 0.0

    @property
    def frames_since_detection(self) -> int | None:
        return None if self.last_detection_frame is None else self.frame_index - self.last_detection_frame

    # --- helpers -----------------------------------------------------------

    def _emit(self, events: list[str], message: str) -> None:
        events.append(message)
        log.info(message)

    def _detector_due(self, now: float) -> bool:
        return self._last_detection_at is None or (now - self._last_detection_at) * 1000.0 >= self.config.detect_interval_ms

    def _accept(self, m: Measurement, now: float, events: list[str], fresh_lock: bool) -> None:
        obj = self.obj
        assert obj is not None
        old_center, new_center = _center(obj.bbox), _center(m.bbox)
        dt = now - obj.last_seen_at
        if not fresh_lock and old_center and new_center and dt > 0:
            obj.velocity_px_s = ((new_center[0] - old_center[0]) / dt, (new_center[1] - old_center[1]) / dt)
            speed = float(np.hypot(*obj.velocity_px_s))
            if speed > 40.0 and now - self._last_moved_log >= 1.0:
                self._last_moved_log = now
                log.info("[TRACKER] Object moved (ID=%d, %.0f px/s)", obj.object_id, speed)
        elif fresh_lock:
            obj.velocity_px_s = None  # a fresh lock carries no motion history
        obj.confidence = m.confidence
        obj.last_seen_frame = self.frame_index
        obj.last_seen_at = now
        if m.position is not None and m.quaternion is not None:
            if fresh_lock:
                self.filter.reset(m.position, m.quaternion, now)
                obj.position, obj.quaternion = m.position, m.quaternion
            else:
                obj.position, obj.quaternion = self.filter.update(m.position, m.quaternion, now)
        if m.bbox is not None:
            obj.bbox, obj.polygon = m.bbox, m.polygon
        obj.extra = m.extra
        self.tracker.commit(m)

    def _run_detector(self, frame: Any, now: float, events: list[str], recovering: bool) -> tuple[SegmentedObject | None, float]:
        started = time.perf_counter()
        self._last_detection_at = now
        self.detection_runs += 1
        if recovering:
            self._emit(events, "[RECOVERY] Running detector...")
        detection = self.detector(frame)
        elapsed = (time.perf_counter() - started) * 1000.0
        if detection is not None:
            self.detection_count += 1
            self.last_detection_frame = self.frame_index
            self._pending = detection
            self.state = TrackingState.INITIALIZING
            verb = "reacquired" if recovering else "detected"
            self._emit(
                events,
                f"[DETECTION] {detection.class_label} {verb} (confidence={detection.confidence:.2f}) — "
                f"Detection count = {self.detection_count}",
            )
        return detection, elapsed

    def _initialize(self, frame: Any, now: float, events: list[str]) -> float:
        """INITIALIZING: initial 6DoF pose + tracker lock from the pending detection."""
        detection, self._pending = self._pending, None
        started = time.perf_counter()
        m = self.tracker.initialize(frame, detection)
        elapsed = (time.perf_counter() - started) * 1000.0
        reacquiring = self.obj is not None
        if m.confidence < self.config.lost_confidence:
            self._emit(events, f"[POSE] Could not initialize pose (confidence={m.confidence:.2f})")
            self.state = TrackingState.RECOVERING if reacquiring else TrackingState.SEARCHING
            return elapsed

        if not reacquiring:
            self.obj = TrackedObject(
                object_id=self._next_object_id,
                class_label=detection.class_label,
                confidence=m.confidence,
                first_seen_frame=self.frame_index,
                last_seen_frame=self.frame_index,
                last_seen_at=now,
            )
            self._next_object_id += 1
        self._emit(events, "[POSE] Initial pose calculated")
        self._emit(
            events,
            f"[TRACKER] Tracker {'re-initialized' if reacquiring else 'initialized'} — Object ID = {self.obj.object_id}",
        )
        self._accept(m, now, events, fresh_lock=True)
        self.state = TrackingState.TRACKING
        self._lost_since = None
        self._lost_frames = 0
        self._announced_search = False
        return elapsed

    def _log_tracking(self, now: float) -> None:
        if now - self._last_status_log < 1.0 or self.obj is None:
            return
        self._last_status_log = now
        obj = self.obj
        log.info("[TRACKER] Tracking — confidence = %.2f (object ID = %d)", obj.confidence, obj.object_id)
        if obj.position is not None:
            rx, ry, rz = euler_xyz_deg(obj.quaternion)
            log.info(
                "[POSE] Updated pose: X=%.3f Y=%.3f Z=%.3f Rx=%.1f Ry=%.1f Rz=%.1f",
                *obj.position, rx, ry, rz,
            )

    # --- the state machine ----------------------------------------------------

    def step(self, frame: Any) -> StepResult:
        now = self.clock()
        self.frame_index += 1
        events: list[str] = []
        timings = {"detection": 0.0, "initialization": 0.0, "tracking": 0.0, "refinement": 0.0}
        detector_ran = tracker_ran = False
        found: SegmentedObject | None = None
        cfg = self.config

        def run_tracker() -> Measurement:
            nonlocal tracker_ran
            started = time.perf_counter()
            m = self.tracker.update(frame)
            total = (time.perf_counter() - started) * 1000.0
            refinement = float(m.extra.get("pose_service_ms", 0.0))
            timings["refinement"] += refinement
            timings["tracking"] += max(0.0, total - refinement)
            tracker_ran = True
            self.tracking_frames += 1
            return m

        if self.state == TrackingState.INITIALIZING:
            timings["initialization"] = self._initialize(frame, now, events)

        elif self.state == TrackingState.TRACKING:
            m = run_tracker()
            if m.confidence >= cfg.lost_confidence:
                self._accept(m, now, events, fresh_lock=False)
                self._log_tracking(now)
            else:
                self.obj.confidence = m.confidence  # real, current confidence; pose stays the last valid one
                self.state = TrackingState.LOST
                self._lost_since = now
                self._lost_frames = 1
                self._emit(events, f"[TRACKER] Tracking lost (confidence = {m.confidence:.2f})")

        elif self.state == TrackingState.LOST:
            recovered = False
            if self.tracker.can_recover:
                m = run_tracker()
                self.obj.confidence = m.confidence
                if m.confidence >= cfg.lost_confidence:
                    recovered = True
                    self.state = TrackingState.TRACKING
                    self._lost_since = None
                    self._accept(m, now, events, fresh_lock=False)
                    self._emit(events, f"[TRACKER] Tracking recovered (confidence = {m.confidence:.2f})")
            if not recovered:
                self._lost_frames += 1
                if self._lost_frames > cfg.grace_frames:
                    self.state = TrackingState.RECOVERING
                    self._last_detection_at = None  # detector may run right away

        if self.state == TrackingState.RECOVERING:
            if (now - self._lost_since) * 1000.0 >= cfg.lost_timeout_ms:
                self._emit(events, f"[TRACKER] Object ID = {self.obj.object_id} not reacquired — model hidden")
                self.state = TrackingState.SEARCHING
                self.obj = None
                self._lost_since = None
            elif self._detector_due(now):
                found, timings["detection"] = self._run_detector(frame, now, events, recovering=True)
                detector_ran = True

        if self.state == TrackingState.SEARCHING and not detector_ran:
            if not self._announced_search:
                self._announced_search = True
                self._emit(events, "[SEARCHING] Running detector...")
            if self._detector_due(now):
                found, timings["detection"] = self._run_detector(frame, now, events, recovering=False)
                detector_ran = True

        conf = self.obj.confidence if self.obj else 0.0
        return StepResult(
            state=self.state,
            visible=self.obj is not None and self.obj.position is not None and self.state != TrackingState.SEARCHING,
            monitoring=self.state == TrackingState.TRACKING and conf < cfg.good_confidence,
            obj=self.obj,
            detection=found,
            detector_ran=detector_ran,
            tracker_ran=tracker_ran,
            timings_ms=timings,
            events=events,
        )
