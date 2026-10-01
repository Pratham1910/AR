"""
Detect-once, then track: the live AR state machine (one per camera session).

    SEARCHING --detector finds object, tracker locks on--> TRACKING
    TRACKING  --several low-confidence frames in a row-->  LOST
    LOST      --tracker recovers, or detector reacquires--> TRACKING
    LOST      --no recovery within the timeout-->           SEARCHING

The detector (YOLO) runs only in SEARCHING and LOST, and then at most once
per `detect_interval_ms`. In TRACKING only the tracker runs; it updates the
pose of the object it already has, it never asks "where is the cup?" again.
The object keeps one identity (object_id) across tracking and re-acquisition.

The tracker is pluggable (PoseTracker): optical flow for markerless mode,
the MegaPose refiner for model-based mode. This module never touches images
or models itself, so its behaviour is unit-testable with fakes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np

from app.schemas.vision import BoundingBox, SegmentedObject, Vector2

log = logging.getLogger("uvicorn.error").getChild("ar")


class TrackingState(str, Enum):
    SEARCHING = "SEARCHING"
    TRACKING = "TRACKING"
    LOST = "LOST"


@dataclass
class Measurement:
    """One tracker output. `confidence` is on the tracker's own scale; the
    thresholds in ARTrackingConfig are chosen per tracker."""

    confidence: float
    position: tuple[float, float, float] | None = None  # Three.js space, meters
    quaternion: tuple[float, float, float, float] | None = None
    bbox: BoundingBox | None = None
    polygon: list[Vector2] | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # tracker-specific, e.g. t_camera_object


class PoseTracker(Protocol):
    # Whether update() can still recover after the session declared the
    # object LOST (MegaPose refining from the last good pose can; optical
    # flow whose points were lost can't, so it needs the detector).
    can_recover: bool

    def initialize(self, frame: Any, detection: SegmentedObject) -> Measurement: ...

    def update(self, frame: Any) -> Measurement: ...

    def commit(self, measurement: Measurement) -> None:
        """The session accepted this measurement as the object's pose."""


@dataclass
class ARTrackingConfig:
    detect_interval_ms: float = 150.0  # detector rate cap while SEARCHING/LOST
    good_confidence: float = 0.7  # at/above: solid tracking; below: shown, but "monitoring"
    lost_confidence: float = 0.4  # below: a miss
    grace_frames: int = 2  # misses in a row still TRACKING (last pose held) before LOST
    lost_timeout_ms: float = 1500.0  # LOST this long without recovery -> hide, SEARCHING


@dataclass
class TrackedObject:
    object_id: int
    class_label: str
    confidence: float
    first_seen_frame: int
    last_seen_frame: int
    last_seen_at: float
    position: tuple[float, float, float] | None = None
    quaternion: tuple[float, float, float, float] | None = None
    bbox: BoundingBox | None = None
    polygon: list[Vector2] | None = None
    velocity_px_s: tuple[float, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    state: TrackingState
    visible: bool  # draw the model (TRACKING, or LOST within the grace period at the last valid pose)
    monitoring: bool  # tracking, but confidence below good_confidence
    obj: TrackedObject | None
    detector_ran: bool
    tracker_ran: bool
    detect_ms: float
    track_ms: float
    events: list[str]


def _center(bbox: BoundingBox | None) -> tuple[float, float] | None:
    return None if bbox is None else ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)


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

        self.state = TrackingState.SEARCHING
        self.obj: TrackedObject | None = None
        self.frame_index = 0
        self.detection_runs = 0
        self.detections_found = 0
        self.tracking_frames = 0
        self._next_object_id = 1
        self._last_detection_at: float | None = None
        self._lost_since: float | None = None
        self._misses = 0
        self._announced_search = False
        self._last_status_log = 0.0
        self._last_moved_log = 0.0

    # --- helpers -----------------------------------------------------------

    def _emit(self, events: list[str], message: str) -> None:
        events.append(message)
        log.info(message)

    def _detector_due(self, now: float) -> bool:
        return self._last_detection_at is None or (now - self._last_detection_at) * 1000.0 >= self.config.detect_interval_ms

    def _accept(self, m: Measurement, now: float, events: list[str]) -> None:
        obj = self.obj
        assert obj is not None
        old_center, new_center = _center(obj.bbox), _center(m.bbox)
        dt = now - obj.last_seen_at
        if old_center and new_center and dt > 0:
            obj.velocity_px_s = ((new_center[0] - old_center[0]) / dt, (new_center[1] - old_center[1]) / dt)
            speed = float(np.hypot(*obj.velocity_px_s))
            if speed > 40.0 and now - self._last_moved_log >= 1.0:
                self._last_moved_log = now
                self._emit(events, f"[TRACKER] Object moved (ID={obj.object_id}, {speed:.0f} px/s)")
        obj.confidence = m.confidence
        obj.last_seen_frame = self.frame_index
        obj.last_seen_at = now
        if m.position is not None:
            obj.position, obj.quaternion = m.position, m.quaternion
        if m.bbox is not None:
            obj.bbox, obj.polygon = m.bbox, m.polygon
        obj.extra = m.extra
        self.tracker.commit(m)

    def _detect_and_lock(self, frame: Any, now: float, events: list[str], recovering: bool) -> float:
        """Runs the detector once; on success initializes the tracker and
        enters TRACKING. Returns the time spent (ms)."""
        started = time.perf_counter()
        self._last_detection_at = now
        self.detection_runs += 1
        if recovering:
            self._emit(events, "[RECOVERY] Running detector")
        detection = self.detector(frame)
        if detection is None:
            return (time.perf_counter() - started) * 1000.0
        self.detections_found += 1
        m = self.tracker.initialize(frame, detection)
        elapsed = (time.perf_counter() - started) * 1000.0
        if m.confidence < self.config.lost_confidence:
            self._emit(
                events,
                f"[TRACKER] Found {detection.class_label} but could not lock on (confidence={m.confidence:.2f})",
            )
            return elapsed

        if recovering and self.obj is not None:
            self._emit(events, f"[DETECTION] {detection.class_label} reacquired")
            self._emit(events, f"[TRACKER] Reinitialized object ID={self.obj.object_id}")
        else:
            self.obj = TrackedObject(
                object_id=self._next_object_id,
                class_label=detection.class_label,
                confidence=m.confidence,
                first_seen_frame=self.frame_index,
                last_seen_frame=self.frame_index,
                last_seen_at=now,
            )
            self._next_object_id += 1
            self._emit(events, f"[DETECTION] {detection.class_label} found (confidence={detection.confidence:.2f})")
            self._emit(events, "[POSE] Initial pose calculated")
            self._emit(events, f"[TRACKER] Initialized object ID={self.obj.object_id}")
        self._accept(m, now, events)
        self.obj.velocity_px_s = None  # a fresh lock carries no motion history
        self.state = TrackingState.TRACKING
        self._misses = 0
        self._lost_since = None
        self._announced_search = False
        return elapsed

    # --- the state machine ----------------------------------------------------

    def step(self, frame: Any) -> StepResult:
        now = self.clock()
        self.frame_index += 1
        events: list[str] = []
        detector_ran = tracker_ran = False
        detect_ms = track_ms = 0.0
        cfg = self.config

        if self.state == TrackingState.TRACKING:
            started = time.perf_counter()
            m = self.tracker.update(frame)
            track_ms = (time.perf_counter() - started) * 1000.0
            tracker_ran = True
            self.tracking_frames += 1
            if m.confidence >= cfg.lost_confidence:
                self._misses = 0
                self._accept(m, now, events)
                if now - self._last_status_log >= 1.0:
                    self._last_status_log = now
                    log.info(
                        "[TRACKER] Tracking object ID=%d confidence=%.2f (%d tracking frames, %d detector runs)",
                        self.obj.object_id, m.confidence, self.tracking_frames, self.detection_runs,
                    )
            else:
                self._misses += 1  # last valid pose is kept and still shown...
                self.obj.confidence = m.confidence  # ...but the confidence shown is the real, current one
                if self._misses > cfg.grace_frames:
                    self.state = TrackingState.LOST
                    self._lost_since = now
                    # The detector may run straight away on the next LOST frame.
                    self._last_detection_at = None
                    self._emit(
                        events,
                        f"[TRACKER] Object lost (ID={self.obj.object_id}, confidence={m.confidence:.2f} "
                        f"for {self._misses} frames)",
                    )

        elif self.state == TrackingState.LOST:
            recovered = False
            if self.tracker.can_recover:
                started = time.perf_counter()
                m = self.tracker.update(frame)
                track_ms = (time.perf_counter() - started) * 1000.0
                tracker_ran = True
                self.tracking_frames += 1
                self.obj.confidence = m.confidence
                if m.confidence >= cfg.lost_confidence:
                    recovered = True
                    self.state = TrackingState.TRACKING
                    self._misses = 0
                    self._lost_since = None
                    self._accept(m, now, events)
                    self._emit(events, f"[TRACKER] Recovered object ID={self.obj.object_id} (confidence={m.confidence:.2f})")
            if not recovered:
                if (now - self._lost_since) * 1000.0 >= cfg.lost_timeout_ms:
                    self._emit(events, f"[TRACKER] Object ID={self.obj.object_id} not recovered — hiding model")
                    self.state = TrackingState.SEARCHING
                    self.obj = None
                    self._lost_since = None
                elif self._detector_due(now):
                    detect_ms = self._detect_and_lock(frame, now, events, recovering=True)
                    detector_ran = True

        if self.state == TrackingState.SEARCHING and not detector_ran:
            if not self._announced_search:
                self._announced_search = True
                self._emit(events, f"[DETECTION] Searching for {self.class_label}...")
            if self._detector_due(now):
                detect_ms = self._detect_and_lock(frame, now, events, recovering=False)
                detector_ran = True

        conf = self.obj.confidence if self.obj else 0.0
        return StepResult(
            state=self.state,
            visible=self.obj is not None and self.state in (TrackingState.TRACKING, TrackingState.LOST),
            monitoring=self.state == TrackingState.TRACKING and conf < cfg.good_confidence,
            obj=self.obj,
            detector_ran=detector_ran,
            tracker_ran=tracker_ran,
            detect_ms=detect_ms,
            track_ms=track_ms,
            events=events,
        )
