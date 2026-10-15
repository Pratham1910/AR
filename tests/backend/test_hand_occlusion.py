"""
Phase 7: hands in view. The hand mask (hand_mask.py), and tracking and
inspection that keep hand pixels out (flow_tracker.py, trackers.py,
ar_session.py, component_inspection).
"""

import cv2
import numpy as np
import pytest

from app.schemas.vision import BoundingBox, SegmentedObject, Vector2
from app.services.component_inspection.decision import ComponentStatus
from app.services.tracking.ar_session import ARSession, ARTrackingConfig, Measurement, TrackingState
from app.services.tracking.flow_tracker import FlowBoxTracker
from app.services.tracking.trackers import without_hand
from app.services.vision.hand_mask import HandMaskService, HandSegmenter, landmark_mask

from test_flow_tracker import BACKGROUND, H, OBJECT, W, frame as object_frame
from test_pose_fusion import SlowPoseService, _locked, _pose, _textured_frame

# ── the hand mask ───────────────────────────────────────────────────────

# An open right hand (MediaPipe's 21 landmarks), palm ~80 px wide, wrist at the bottom.
HAND = np.array([
    [300, 400],                                      # 0 wrist
    [255, 375], [235, 345], [222, 318], [212, 295],  # thumb
    [265, 300], [262, 255], [260, 228], [258, 205],  # index
    [300, 295], [300, 245], [300, 215], [300, 190],  # middle
    [335, 300], [338, 255], [340, 228], [342, 207],  # ring
    [362, 312], [372, 280], [378, 260], [383, 242],  # little
], float)


def test_landmark_mask_covers_palm_and_fingers_only():
    mask = landmark_mask(HAND, (H, W))
    assert mask[350, 300] and mask[205, 258] and mask[190, 300]  # palm, index tip, middle tip
    assert not mask[100, 300] and not mask[400, 500]  # above the fingers, beside the hand
    assert 0.03 < mask.mean() < 0.12


class FakeHands:
    def __init__(self, hands):
        self.hands = hands

    def landmarks(self, frame):
        return self.hands


def test_segmenter_unions_mediapipe_and_person_regions():
    person = np.array([[500, 300], [630, 300], [630, 470], [500, 470]], float)
    seg = HandSegmenter(FakeHands([HAND]), lambda f: [person], margin_px=0)
    result = seg.segment(np.zeros((H, W, 3), np.uint8), taken_at=1.0)
    assert result.hands == 1 and result.persons == 1 and result.sources == ["mediapipe", "person"]
    assert result.mask[350, 300] and result.mask[400, 560] and not result.mask[50, 50]
    assert len(result.outlines()) == 2


def test_mask_service_rate_limits_and_ages_out():
    calls = []
    seg = HandSegmenter(FakeHands([HAND]), None, margin_px=0)
    original = seg.segment
    seg.segment = lambda f, t: calls.append(t) or original(f, t)
    service = HandMaskService(seg, interval_ms=100, max_age_ms=400, motion_margin_px=8, run_async=False)
    img = np.zeros((H, W, 3), np.uint8)
    for t in (0.0, 0.05, 0.12, 0.15):
        service.submit(img, t)
    assert calls == [0.0, 0.12]
    fresh, older = service.latest(0.12), service.latest(0.42)
    assert fresh.mask.sum() < older.mask.sum()  # an older mask is grown for the hand's motion since
    assert service.latest(0.6) is None  # too old to trust


def test_failing_hand_source_is_reported_not_raised():
    def broken(frame):
        raise RuntimeError("model file missing")

    service = HandMaskService(HandSegmenter(None, broken), run_async=False)
    service.submit(np.zeros((H, W, 3), np.uint8), 0.0)
    assert service.latest(0.0) is None and "model file missing" in service.error


# ── the object's outline without the hand ───────────────────────────────


def _square(x1, y1, x2, y2):
    return SegmentedObject(class_label="cup", confidence=0.9, bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                           polygon=[Vector2(x=x, y=y) for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))])


def test_hand_is_cut_out_of_the_detected_outline():
    """Detector merged cup (x 100-200) and the hand holding it (x 200-300) into one region."""
    hand = np.zeros((H, W), bool)
    hand[95:306, 200:306] = True  # the hand reaches past the region's edge, as real (dilated) masks do
    cleaned, share = without_hand(_square(100, 100, 300, 300), hand)
    assert share == pytest.approx(0.5, abs=0.02)
    assert cleaned.bbox.x2 <= 201 and cleaned.bbox.x1 == pytest.approx(100, abs=1)


def test_an_outline_that_is_mostly_hand_is_kept_for_the_pose_check_to_judge():
    hand = np.zeros((H, W), bool)
    hand[100:300, 120:300] = True
    cleaned, share = without_hand(_square(100, 100, 300, 300), hand)
    assert share > 0.75 and cleaned.bbox.x2 == 300


# ── optical flow ignores the hand ───────────────────────────────────────

HAND_PATCH = np.kron(np.random.default_rng(7).integers(0, 255, (8, 8)), np.ones((10, 10))).astype(np.uint8)  # 80x80


def with_hand(img: np.ndarray, x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
    """A strongly textured "hand" pasted at (x, y), and its mask."""
    img = img.copy()
    img[y : y + 80, x : x + 80] = HAND_PATCH
    mask = np.zeros(img.shape, bool)
    mask[y : y + 80, x : x + 80] = True
    return img, mask


def test_hand_moving_over_a_still_object_does_not_move_it():
    """Acceptance 2: a hand passes across the cup — no pose change."""
    tracker = FlowBoxTracker()
    first, _ = with_hand(object_frame(300, 240), 230, 200)
    _, hand0 = with_hand(object_frame(300, 240), 230, 200)
    assert tracker.initialize(first, BoundingBox(x1=240, y1=160, x2=360, y2=320), None, exclude=hand0)
    for step in range(1, 8):
        img, hand = with_hand(object_frame(300, 240), 230 + 8 * step, 200 + 3 * step)
        result = tracker.update(img, exclude=hand)
        assert result.ok
    cx, cy = (result.bbox.x1 + result.bbox.x2) / 2, (result.bbox.y1 + result.bbox.y2) / 2
    assert (cx, cy) == pytest.approx((300, 240), abs=1.0)
    assert result.occlusion > 0.1


def test_without_the_mask_the_same_hand_would_be_tracked_as_object():
    """Why the mask matters: seeded on the hand, flow follows the hand."""
    tracker = FlowBoxTracker()
    img, _ = with_hand(np.full((H, W), 120, np.uint8), 260, 200)  # a plain object: only the hand has texture
    assert tracker.initialize(img, BoundingBox(x1=240, y1=160, x2=360, y2=320), None)
    for step in range(1, 6):
        img, _ = with_hand(np.full((H, W), 120, np.uint8), 260 + 8 * step, 200)
        result = tracker.update(img)
    assert (result.bbox.x1 + result.bbox.x2) / 2 > 330  # dragged ~40 px by the hand


def test_points_hidden_by_the_hand_are_occlusion_not_failure():
    tracker = FlowBoxTracker()
    assert tracker.initialize(object_frame(300, 240), BoundingBox(x1=240, y1=160, x2=360, y2=320), None)
    hand = np.zeros((H, W), bool)
    hand[160:240, 240:360] = True  # covers the top half
    result = tracker.update(object_frame(300, 240), exclude=hand)
    assert result.ok and result.confidence > 0.9 and result.hand_points > 0
    assert result.occlusion == pytest.approx(0.5, abs=0.05)


# ── model-based tracker: hold, resume, give up ──────────────────────────


def _covered(frame, top=0, bottom=480):
    """The test scene's object (x 240-360, y 160-320) under a hand: grey pixels + mask."""
    bgr = frame.bgr.copy()
    bgr[top:bottom, 230:370] = 128
    mask = np.zeros(bgr.shape[:2], bool)
    mask[top:bottom, 230:370] = True
    out = type(frame)(bgr, frame.calibration, hand_mask=mask)
    return out


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_covered_object_holds_its_last_pose_then_is_lost():
    """Acceptance 4: a hand covers the cup — occluded, plausible (held) pose briefly, then lost."""
    service = SlowPoseService(refine_pose=_pose())
    service.release.set()
    tracker = _locked(service)
    tracker.clock = clock = Clock()
    tracker.occlusion_hold_ms = 1000
    before = tracker.update(_textured_frame()).extra["t_camera_mesh"].copy()
    refines = service.refines
    m = tracker.update(_covered(_textured_frame()))
    assert m.extra["occluded"] and m.extra["occlusion"] > 0.9
    assert m.extra["t_camera_mesh"] == pytest.approx(before)  # held, not moved by anything in the hand
    assert service.refines == refines  # no MegaPose on a hand-covered crop
    clock.t = 0.9
    assert tracker.update(_covered(_textured_frame())).extra.get("occluded")
    clock.t = 1.2
    gone = tracker.update(_covered(_textured_frame()))
    assert not gone.extra.get("occluded") and gone.confidence == 0.0


def test_tracking_resumes_from_visible_pixels_when_the_hand_moves_away():
    """Acceptance 7-ish: object visible again — flow picks up without a new detection."""
    service = SlowPoseService(refine_pose=_pose())
    service.release.set()
    tracker = _locked(service)
    tracker.clock = Clock()
    tracker.update(_textured_frame())
    assert tracker.update(_covered(_textured_frame())).extra.get("occluded")
    tracker.update(_textured_frame())  # hand gone: flow re-seeded on the object
    m = tracker.update(_textured_frame(dx=10))
    assert not m.extra.get("occluded") and m.confidence > 0.5
    assert m.extra["t_camera_mesh"][0, 3] > 0  # following the object's motion again


def test_partly_covered_object_is_tracked_from_its_visible_part_and_not_refined():
    service = SlowPoseService(refine_pose=_pose())
    service.release.set()
    tracker = _locked(service)
    refines = service.refines
    m = tracker.update(_covered(_textured_frame(dx=10), top=160, bottom=230))  # top ~45% under a hand
    assert not m.extra.get("occluded") and m.extra["occlusion"] > 0.3
    assert m.extra["refine_skipped_for_hand"] and service.refines == refines
    assert m.extra["t_camera_mesh"][0, 3] > 0  # still followed the motion, from the uncovered part


# ── session states ──────────────────────────────────────────────────────


class ScriptedTracker:
    can_recover = True

    def __init__(self, script):
        self.script = list(script)

    def initialize(self, frame, detection):
        return Measurement(confidence=0.9, position=(0, 0, 0.5), quaternion=(0, 0, 0, 1), bbox=detection.bbox,
                           extra={"t_camera_object": np.eye(4), "lock": True})

    def update(self, frame):
        kind = self.script.pop(0)
        if kind == "ok":
            return Measurement(confidence=0.9, position=(0.01, 0, 0.5), quaternion=(0, 0, 0, 1),
                               bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10), extra={"t_camera_object": np.eye(4)})
        if kind == "occluded":
            return Measurement(confidence=0.3, position=(0.5, 0.5, 0.5), quaternion=(0, 0, 0, 1),
                               extra={"occluded": True, "occlusion": 0.8, "t_camera_object": np.eye(4) * 2})
        return Measurement(confidence=0.0)

    def commit(self, m):
        pass


def _session(script):
    detection = SegmentedObject(class_label="cup", confidence=0.9, bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10), polygon=[])
    session = ARSession("cup", lambda f: detection, ScriptedTracker(script), ARTrackingConfig(lost_confidence=0.5))
    session.step(None)  # SEARCHING -> detected
    session.step(None)  # INITIALIZING -> TRACKING
    return session


def test_occlusion_is_its_own_state_and_never_moves_the_pose():
    session = _session(["ok", "occluded", "occluded", "ok"])
    session.step(None)
    position = session.obj.position
    r = session.step(None)
    assert r.state == TrackingState.TRACKING_OCCLUDED and r.visible and not r.monitoring
    assert any("occluded" in e for e in r.events)
    session.step(None)
    assert session.obj.position == position  # the tracker's measurement position (0.5, ...) was ignored
    assert session.obj.extra["t_camera_object"][0, 0] == 1.0  # still the last accepted pose
    r = session.step(None)
    assert r.state == TrackingState.TRACKING and any("visible again" in e for e in r.events)


def test_occlusion_that_lasts_too_long_becomes_lost():
    session = _session(["ok", "occluded", "lost", "lost", "lost", "lost"])
    session.step(None)
    assert session.step(None).state == TrackingState.TRACKING_OCCLUDED
    assert session.step(None).state == TrackingState.LOST


def test_tracking_status_names():
    from types import SimpleNamespace

    from app.api.vision import _tracking_status

    s = SimpleNamespace(detection_count=1)
    r = lambda state, obj=None: SimpleNamespace(state=state, obj=obj)  # noqa: E731
    assert _tracking_status(r(TrackingState.TRACKING_OCCLUDED), s) == "TRACKING_OCCLUDED"
    assert _tracking_status(r(TrackingState.LOST), s) == "TRACKING_LOST"
    assert _tracking_status(r(TrackingState.RECOVERING), s) == "REACQUIRING"
    assert _tracking_status(r(TrackingState.INITIALIZING, obj=object()), s) == "REACQUIRING"
    assert _tracking_status(r(TrackingState.SEARCHING), s) == "TRACKING_LOST"
    assert _tracking_status(r(TrackingState.SEARCHING), SimpleNamespace(detection_count=0)) == "SEARCHING"


# ── inspection under a hand ─────────────────────────────────────────────


def test_component_under_a_hand_is_occluded_not_missing(tmp_path):
    """Acceptance 8: a hand over a (present, but now hidden) screw never makes it missing."""
    from glb_builder import plate_with_screws
    from test_component_inspection import jittered, learn, make_session, render
    from test_component_projection import K, SHAPE
    from app.services.component_inspection.projection import project_component
    from app.services.component_inspection.registry import registry_from_glb

    registry = registry_from_glb("m1", plate_with_screws(), scale=1.0)
    session = make_session(registry, tmp_path, confirm_missing_after=3)
    learn(session, registry)
    target = registry.get("SCREW_004")
    for k in range(10):
        t_co = jittered(k)
        frame = render(registry, t_co, seed=500 + k)
        p = project_component(target.geometry, t_co, K, SHAPE)
        x1, y1, x2, y2 = p.roi
        hand = np.zeros(SHAPE, bool)
        hand[y1 - 20 : y2 + 20, x1 - 20 : x2 + 20] = True
        frame[hand] = (90, 140, 200)  # skin-coloured: looks nothing like a screw head
        session.process(frame, t_co, K, True, None, now=200 + k * 0.3, hand_mask=hand)
    state = session.decisions["SCREW_004"].state
    assert state.status == ComponentStatus.OCCLUDED and "covered by a hand" in state.reason
    assert session.decisions["SCREW_001"].state.status == ComponentStatus.PRESENT  # others unaffected
