"""
The detect-once/track state machine, walked through Phase-3.md's expected
test sequence with a fake detector, fake tracker and a controllable clock.
"""

import pytest

from app.schemas.vision import BoundingBox, SegmentedObject
from app.services.tracking.ar_session import ARSession, ARTrackingConfig, Measurement, TrackingState


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance_ms(self, ms: float) -> None:
        self.t += ms / 1000.0


class FakeDetector:
    def __init__(self):
        self.calls = 0
        self.cup_visible = False

    def __call__(self, frame):
        self.calls += 1
        if not self.cup_visible:
            return None
        return SegmentedObject(class_label="cup", confidence=0.9, bbox=BoundingBox(x1=100, y1=100, x2=200, y2=250), polygon=[])


class FakeTracker:
    def __init__(self, can_recover: bool = False):
        self.can_recover = can_recover
        self.confidence = 0.95
        self.x = 0.0
        self.init_calls = 0
        self.update_calls = 0
        self.committed: list[Measurement] = []

    def _m(self) -> Measurement:
        return Measurement(
            confidence=self.confidence,
            position=(self.x, 0.0, -0.4),
            quaternion=(0.0, 0.0, 0.0, 1.0),
            bbox=BoundingBox(x1=100 + self.x * 1000, y1=100, x2=200 + self.x * 1000, y2=250),
        )

    def initialize(self, frame, detection):
        self.init_calls += 1
        return self._m()

    def update(self, frame):
        self.update_calls += 1
        return self._m()

    def commit(self, measurement):
        self.committed.append(measurement)


CONFIG = ARTrackingConfig(detect_interval_ms=150, good_confidence=0.7, lost_confidence=0.4, grace_frames=2, lost_timeout_ms=1500)


def make(can_recover=False):
    clock, detector, tracker = Clock(), FakeDetector(), FakeTracker(can_recover)
    return ARSession("cup", detector, tracker, CONFIG, clock=clock), clock, detector, tracker


def run(session, clock, frames, ms_per_frame=33):
    results = []
    for _ in range(frames):
        clock.advance_ms(ms_per_frame)
        results.append(session.step(frame=None))
    return results


def all_events(results):
    return [e for r in results for e in r.events]


def test_full_phase3_sequence_detects_once_then_tracks():
    session, clock, detector, tracker = make()

    # 1. Camera starts, no cup: searching, detector rate-capped to every 150ms.
    first = session.step(frame=None)
    assert first.state == TrackingState.SEARCHING and not first.visible
    assert "[DETECTION] Searching for cup..." in first.events
    results = run(session, clock, 30)  # ~1s at 30fps
    assert all(r.state == TrackingState.SEARCHING for r in results)
    assert 6 <= detector.calls <= 8  # ~1000ms / 150ms, not 31
    assert sum(r.detector_ran for r in results) == detector.calls - 1

    # 2-3. Cup appears: detected, initial pose, tracker initialized, model shown.
    detector.cup_visible = True
    results = run(session, clock, 6)
    locked = next(r for r in results if r.state == TrackingState.TRACKING)
    assert locked.visible and locked.obj.object_id == 1
    events = all_events(results)
    assert "[DETECTION] cup found (confidence=0.90)" in events
    assert "[POSE] Initial pose calculated" in events
    assert "[TRACKER] Initialized object ID=1" in events
    calls_at_lock = detector.calls
    tracking_frames_at_lock = session.tracking_frames

    # 4-9. Stationary, then moving: ONLY the tracker runs, pose follows it.
    results = run(session, clock, 100)
    tracker.x = 0.05
    results += run(session, clock, 50)
    assert detector.calls == calls_at_lock  # the detector never ran again
    assert not any(r.detector_ran for r in results)
    assert all(r.tracker_ran and r.state == TrackingState.TRACKING for r in results)
    assert not any(e.startswith("[DETECTION]") for e in all_events(results))
    assert results[-1].obj.position == (0.05, 0.0, -0.4)
    assert session.tracking_frames - tracking_frames_at_lock == 150
    assert session.detection_runs == calls_at_lock

    # 10. Cup hidden briefly: model held at the last valid pose, no flicker.
    tracker.confidence = 0.1
    held = run(session, clock, 2)  # within grace_frames
    assert all(r.state == TrackingState.TRACKING and r.visible for r in held)
    assert held[-1].obj.position == (0.05, 0.0, -0.4)
    assert held[-1].obj.confidence == 0.1  # shown confidence is the real one, pose is the held one
    lost = run(session, clock, 1)[0]
    assert lost.state == TrackingState.LOST and lost.visible
    assert any(e.startswith("[TRACKER] Object lost") for e in lost.events)

    # Recovery while LOST: detector runs, rate-capped, model still at last pose.
    detector.cup_visible = False
    calls_before = detector.calls
    results = run(session, clock, 20)  # 660ms
    assert all(r.state == TrackingState.LOST and r.visible for r in results)
    assert 4 <= detector.calls - calls_before <= 6
    assert "[RECOVERY] Running detector" in all_events(results)

    # 11. Hidden long enough: model hides, back to SEARCHING.
    results = run(session, clock, 40)
    assert results[-1].state == TrackingState.SEARCHING and not results[-1].visible
    assert any("not recovered" in e for e in all_events(results))

    # 12. Cup returns: detector runs, tracker re-initializes, detector stops again.
    detector.cup_visible = True
    tracker.confidence = 0.95
    results = run(session, clock, 10)
    assert results[-1].state == TrackingState.TRACKING and results[-1].visible
    calls_after_relock = detector.calls
    run(session, clock, 30)
    assert detector.calls == calls_after_relock


def test_recovery_during_lost_keeps_object_identity():
    session, clock, detector, tracker = make()
    detector.cup_visible = True
    run(session, clock, 2)
    tracker.confidence = 0.1
    run(session, clock, 3)  # -> LOST
    assert session.state == TrackingState.LOST
    tracker.confidence = 0.95
    results = run(session, clock, 10)  # detector reacquires
    events = all_events(results)
    assert "[DETECTION] cup reacquired" in events
    assert "[TRACKER] Reinitialized object ID=1" in events
    assert results[-1].obj.object_id == 1
    assert results[-1].state == TrackingState.TRACKING


def test_tracker_that_can_recover_relocks_without_the_detector():
    session, clock, detector, tracker = make(can_recover=True)
    detector.cup_visible = True
    run(session, clock, 2)
    tracker.confidence = 0.1
    run(session, clock, 3)
    assert session.state == TrackingState.LOST
    calls = detector.calls
    tracker.confidence = 0.9
    result = run(session, clock, 1)[0]
    assert result.state == TrackingState.TRACKING
    assert detector.calls == calls
    assert any(e.startswith("[TRACKER] Recovered object ID=1") for e in result.events)


def test_detected_but_unlockable_object_stays_searching():
    session, clock, detector, tracker = make()
    detector.cup_visible = True
    tracker.confidence = 0.1  # e.g. a plain surface with nothing to follow
    results = run(session, clock, 10)
    assert all(r.state == TrackingState.SEARCHING and not r.visible for r in results)
    assert any("could not lock on" in e for e in all_events(results))


@pytest.mark.parametrize("confidence, monitoring", [(0.95, False), (0.55, True)])
def test_monitoring_band_between_lost_and_good_thresholds(confidence, monitoring):
    session, clock, detector, tracker = make()
    detector.cup_visible = True
    run(session, clock, 2)
    tracker.confidence = confidence
    result = run(session, clock, 1)[0]
    assert result.state == TrackingState.TRACKING
    assert result.monitoring is monitoring


def test_only_accepted_measurements_are_committed_to_the_tracker():
    session, clock, detector, tracker = make(can_recover=True)
    detector.cup_visible = True
    run(session, clock, 2)
    committed = len(tracker.committed)
    tracker.confidence = 0.1
    run(session, clock, 2)
    assert len(tracker.committed) == committed  # a bad frame never becomes the next reference
