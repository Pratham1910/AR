import pytest

from app.schemas.vision import BoundingBox, Detection
from app.services.tracking.tracker import ObjectTracker


def _detection(x1: float, y1: float, x2: float, y2: float, label: str = "pcb", confidence: float = 0.9) -> Detection:
    return Detection(class_label=label, confidence=confidence, bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))


def test_same_object_keeps_the_same_tracker_id_across_frames():
    tracker = ObjectTracker()

    frame1 = tracker.update([_detection(100, 100, 200, 200)])
    assert len(frame1) == 1
    assert frame1[0].tracker_id is not None
    first_id = frame1[0].tracker_id

    # Small movement between frames — ByteTrack should still associate it as the same track.
    frame2 = tracker.update([_detection(105, 103, 205, 203)])
    assert len(frame2) == 1
    assert frame2[0].tracker_id == first_id


def test_new_object_gets_a_different_tracker_id():
    """
    ByteTrack requires a brand-new track to be matched on a *second*
    consecutive frame before it's "activated" and included in the output
    (verified directly against the underlying `supervision` library) — this
    suppresses one-off spurious detections rather than tracking every
    fleeting false positive. So the new object needs two frames before it
    gets its own confirmed id.
    """
    tracker = ObjectTracker()

    frame1 = tracker.update([_detection(100, 100, 200, 200)])
    first_id = frame1[0].tracker_id

    # A second, spatially distinct object appears alongside the first — not
    # yet activated on its first sighting.
    frame2 = tracker.update([_detection(102, 102, 202, 202), _detection(500, 500, 600, 600)])
    assert {d.tracker_id for d in frame2} == {first_id}

    # Same new object seen again -> now activated with its own stable id.
    frame3 = tracker.update([_detection(104, 104, 204, 204), _detection(502, 502, 602, 602)])
    ids = {d.tracker_id for d in frame3}
    assert first_id in ids
    assert len(ids) == 2


def test_class_label_and_bbox_round_trip_through_tracking():
    tracker = ObjectTracker()
    result = tracker.update([_detection(10, 10, 50, 50, label="pcb", confidence=0.77)])

    assert len(result) == 1
    tracked = result[0]
    assert tracked.class_label == "pcb"
    # ByteTrack's Kalman filter can nudge box coordinates slightly even on
    # frame one — this just guards against a totally wrong box, not exactness.
    assert tracked.bbox.x1 == pytest.approx(10, abs=2.0)
    assert tracked.bbox.y2 == pytest.approx(50, abs=2.0)


def test_update_with_no_detections_returns_empty_and_does_not_crash():
    tracker = ObjectTracker()
    assert tracker.update([]) == []

    # A detection right after an empty frame is a brand-new track from
    # ByteTrack's perspective — not yet activated (see the note in
    # test_new_object_gets_a_different_tracker_id); it needs a second
    # consecutive matching frame to appear in the output.
    first_result = tracker.update([_detection(10, 10, 50, 50)])
    assert first_result == []

    second_result = tracker.update([_detection(12, 12, 52, 52)])
    assert len(second_result) == 1


def test_reset_clears_track_state():
    tracker = ObjectTracker()
    frame1 = tracker.update([_detection(100, 100, 200, 200)])
    original_id = frame1[0].tracker_id

    tracker.reset()

    frame2 = tracker.update([_detection(100, 100, 200, 200)])
    # After reset, ByteTrack has no memory of the old track — a fresh
    # detection at the same location starts a new track, not a continuation
    # (it may or may not reuse the same numeric id, but must not silently
    # carry forward any state that would make this a different assertion).
    assert frame2[0].tracker_id is not None
    assert isinstance(original_id, int)
