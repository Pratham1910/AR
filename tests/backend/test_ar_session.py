"""
The detect-once/track state machine, walked through Phase-3.md's / Phae-4.md's
expected sequences with a fake detector, fake tracker and a controllable clock.
"""

import pytest

from app.schemas.vision import BoundingBox, SegmentedObject
from app.services.tracking.ar_session import ARSession, ARTrackingConfig, Measurement, TrackingState, euler_xyz_deg
from app.services.tracking.pose_filter import PoseFilterConfig

S = TrackingState


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
            extra={"pose_service_ms": 0.0},
        )

    def initialize(self, frame, detection):
        self.init_calls += 1
        return self._m()

    def update(self, frame):
        self.update_calls += 1
        return self._m()

    def commit(self, measurement):
        self.committed.append(measurement)


def config(**overrides):
    base = dict(
        detect_interval_ms=150,
        good_confidence=0.7,
        lost_confidence=0.4,
        grace_frames=2,
        lost_timeout_ms=1500,
        filter=PoseFilterConfig(enabled=False),  # exact positions in these tests; the filter has its own tests
    )
    return ARTrackingConfig(**{**base, **overrides})


def make(can_recover=False, **overrides):
    clock, detector, tracker = Clock(), FakeDetector(), FakeTracker(can_recover)
    return ARSession("cup", detector, tracker, config(**overrides), clock=clock), clock, detector, tracker


def run(session, clock, frames, ms_per_frame=33):
    results = []
    for _ in range(frames):
        clock.advance_ms(ms_per_frame)
        results.append(session.step(frame=None))
    return results


def all_events(results):
    return [e for r in results for e in r.events]


def test_phase4_success_criteria_sequence():
    session, clock, detector, tracker = make()

    # 1-2. Camera starts, no cup: SEARCHING, detector rate-capped (~every 150ms, not every frame).
    first = session.step(frame=None)
    assert first.state == S.SEARCHING and not first.visible
    assert "[SEARCHING] Running detector..." in first.events
    results = run(session, clock, 30)
    assert all(r.state == S.SEARCHING for r in results)
    assert 6 <= detector.calls <= 8
    assert session.detection_count == 0

    # 3. Cup appears: detected -> INITIALIZING (detection and pose init are separate steps).
    detector.cup_visible = True
    results = run(session, clock, 6)
    found = next(r for r in results if r.detection is not None)
    assert found.state == S.INITIALIZING and not found.visible
    assert session.detection_count == 1
    # 4-7. Next frame: initial pose, tracker initialized, model shown, detector stops.
    locked = results[results.index(found) + 1]
    assert locked.state == S.TRACKING and locked.visible and locked.obj.object_id == 1
    assert locked.timings_ms["initialization"] >= 0.0 and not locked.detector_ran
    events = all_events(results)
    assert "[DETECTION] cup detected (confidence=0.90) — Detection count = 1" in events
    assert "[POSE] Initial pose calculated" in events
    assert "[TRACKER] Tracker initialized — Object ID = 1" in events
    calls_at_lock, frames_at_lock = detector.calls, session.tracking_frames

    # 8-13. Moving: only the tracker runs, detection_count stays 1, model follows.
    results = run(session, clock, 100)
    tracker.x = 0.05
    results += run(session, clock, 50)
    assert detector.calls == calls_at_lock and session.detection_count == 1
    assert not any(r.detector_ran for r in results)
    assert all(r.tracker_ran and r.state == S.TRACKING for r in results)
    assert not any(e.startswith(("[DETECTION]", "[SEARCHING]")) for e in all_events(results))
    assert results[-1].obj.position == (0.05, 0.0, -0.4)
    assert session.tracking_frames - frames_at_lock == 150
    assert session.frames_since_detection > 150

    # Tracking lost (cup hidden): model held at the last valid pose; frames 1-2
    # hold, frame 3 attempts recovery with the detector.
    tracker.confidence = 0.1
    detector.cup_visible = False
    lost = run(session, clock, 3)
    assert lost[0].state == S.LOST and any(e.startswith("[TRACKER] Tracking lost") for e in lost[0].events)
    assert all(r.visible and r.obj.position == (0.05, 0.0, -0.4) for r in lost)
    assert lost[0].obj.confidence == 0.1  # shown confidence is the real one
    assert [r.detector_ran for r in lost] == [False, False, True]
    assert lost[-1].state == S.RECOVERING

    # 15. RECOVERING: detector at a controlled interval, model still held.
    calls_before = detector.calls
    results = run(session, clock, 20)
    assert all(r.state == S.RECOVERING and r.visible for r in results)
    assert 4 <= detector.calls - calls_before <= 6
    assert "[RECOVERY] Running detector..." in all_events(results)

    # 16-19. Cup reacquired: detection_count = 2, same object, tracker re-initialized, detector stops.
    detector.cup_visible = True
    tracker.confidence = 0.95
    results = run(session, clock, 8)
    events = all_events(results)
    assert any("cup reacquired" in e and "Detection count = 2" in e for e in events)
    assert "[TRACKER] Tracker re-initialized — Object ID = 1" in events
    assert results[-1].state == S.TRACKING and results[-1].obj.object_id == 1
    calls = detector.calls
    run(session, clock, 30)
    assert detector.calls == calls and session.detection_count == 2


def test_not_reacquired_within_timeout_hides_model_and_searches_again():
    session, clock, detector, tracker = make()
    detector.cup_visible = True
    run(session, clock, 3)
    tracker.confidence = 0.1
    detector.cup_visible = False
    results = run(session, clock, 60)  # ~2s > 1.5s timeout
    assert results[-1].state == S.SEARCHING and not results[-1].visible and results[-1].obj is None
    assert any("not reacquired — model hidden" in e for e in all_events(results))
    # A later detection is a NEW object identity.
    detector.cup_visible = True
    tracker.confidence = 0.95
    results = run(session, clock, 8)
    assert results[-1].state == S.TRACKING and results[-1].obj.object_id == 2


def test_recoverable_tracker_relocks_during_grace_without_the_detector():
    session, clock, detector, tracker = make(can_recover=True)
    detector.cup_visible = True
    run(session, clock, 3)
    tracker.confidence = 0.1
    assert run(session, clock, 1)[0].state == S.LOST
    calls = detector.calls
    tracker.confidence = 0.9
    result = run(session, clock, 1)[0]
    assert result.state == S.TRACKING and detector.calls == calls
    assert any(e.startswith("[TRACKER] Tracking recovered") for e in result.events)


def test_detected_but_unlockable_object_returns_to_searching():
    session, clock, detector, tracker = make()
    detector.cup_visible = True
    tracker.confidence = 0.1  # e.g. a wrong-shaped model, or nothing to follow
    results = run(session, clock, 10)
    assert all(not r.visible for r in results)
    assert {r.state for r in results} <= {S.SEARCHING, S.INITIALIZING}
    assert any("Could not initialize pose" in e for e in all_events(results))


@pytest.mark.parametrize("confidence, monitoring", [(0.95, False), (0.55, True)])
def test_tracking_with_warning_band(confidence, monitoring):
    session, clock, detector, tracker = make()
    detector.cup_visible = True
    run(session, clock, 3)
    tracker.confidence = confidence
    result = run(session, clock, 1)[0]
    assert result.state == S.TRACKING and result.monitoring is monitoring


def test_rejected_measurements_never_become_the_trackers_reference():
    session, clock, detector, tracker = make(can_recover=True)
    detector.cup_visible = True
    run(session, clock, 3)
    committed = len(tracker.committed)
    tracker.confidence = 0.1
    run(session, clock, 2)
    assert len(tracker.committed) == committed


def test_accepted_poses_are_filtered_and_a_fresh_lock_resets_the_filter():
    session, clock, detector, tracker = make(filter=PoseFilterConfig())
    detector.cup_visible = True
    run(session, clock, 3)
    assert session.obj.position == (0.0, 0.0, -0.4)  # fresh lock: exact, no blending
    tracker.x = 0.05  # sudden 5 cm jump in one frame
    filtered = run(session, clock, 1)[0].obj.position[0]
    assert 0.0 < filtered < 0.05  # smoothed, not applied raw
    converged = run(session, clock, 30)[-1].obj.position[0]
    assert converged == pytest.approx(0.05, abs=0.002)


def test_euler_display_matches_threejs_xyz_order():
    import math

    half = math.radians(30) / 2
    assert euler_xyz_deg((math.sin(half), 0.0, 0.0, math.cos(half))) == pytest.approx((30.0, 0.0, 0.0))
    assert euler_xyz_deg((0.0, math.sin(half), 0.0, math.cos(half))) == pytest.approx((0.0, 30.0, 0.0))
    assert euler_xyz_deg((0.0, 0.0, math.sin(half), math.cos(half))) == pytest.approx((0.0, 0.0, 30.0))
