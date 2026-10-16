"""The tracker with orientation logic (Phase 8): locks choose the right twin, background corrections can't flip it."""

import numpy as np

from app.eval.synthetic_scene import K, SHAPE, box_mesh, corner_circle, look_at, project, render, turned
from app.schemas.vision import BoundingBox, SegmentedObject
from app.services.pose.calibration import CameraCalibration
from app.services.pose.edge_align import EdgeModel
from app.services.pose.model_pose_client import PoseServiceResult
from app.services.pose.pose_features import PoseFeature, with_feature_edges
from app.services.pose.symmetry import find_symmetries, rotation_angle_deg
from app.services.tracking.trackers import Frame, MegaPoseTracker

V, F = box_mesh()
MARK = corner_circle()
FEATURE = PoseFeature("UR_CIRCLE", "circle", MARK.center, MARK.normal, MARK.radius)
TRUTH = look_at((0.03, 0.30, 0.20))
FLIP = np.diag([-1.0, 1.0, -1.0])
SYM = find_symmetries(V, F)


def _twin(t):
    out = np.array(t, float)
    out[:3, :3] = out[:3, :3] @ FLIP
    return out


def _frame(t=TRUTH, hand=None):
    gray, mask = render(V, F, t, [MARK], hand=hand)
    calibration = CameraCalibration.approximate(SHAPE[1], SHAPE[0])
    calibration.camera_matrix = K.copy()
    return Frame(np.dstack([gray] * 3), calibration, hand_mask=mask)


def _detection(t=TRUTH):
    px = project(V, t)
    (x1, y1), (x2, y2) = px.min(axis=0), px.max(axis=0)
    return SegmentedObject(class_label="box", confidence=0.9, bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2), polygon=[])


class FakeMegaPose:
    """Full search and refines answer `pose` (MegaPose returning whichever twin)."""

    def __init__(self, pose, refine_pose=None):
        self.pose, self.refine_pose = pose, refine_pose if refine_pose is not None else pose

    def full_search(self, label, jpeg, k, bbox, outline=None):
        return PoseServiceResult(self.pose, 0.9, "coarse+refine", 900.0, tuple(bbox), 0.95)

    def refine(self, label, jpeg, k, prev_pose, iterations):
        box = tuple(float(v) for v in (*project(V, self.refine_pose).min(axis=0), *project(V, self.refine_pose).max(axis=0)))
        return PoseServiceResult(self.refine_pose, 0.9, "refine", 250.0, box)


def _tracker(service, features=(FEATURE,), symmetries=SYM, **kw):
    geometry = EdgeModel.from_mesh(V, F)
    model = with_feature_edges(geometry, list(features))
    return MegaPoseTracker(service, "box", refine_iterations=2, edge_model=model, geometry_model=geometry, async_refine=False,
                           symmetries=symmetries, pose_features=list(features), resting="camera",
                           edge_gain_translation=1.0, edge_gain_rotation=1.0, **kw)


def _correct_once(tracker, frame):
    tracker._needs_settle = True  # ask for one correction on the next (still) frame
    tracker.update(frame)
    return tracker.update(frame)


def test_a_lock_on_the_twin_is_turned_the_right_way_round():
    m = _tracker(FakeMegaPose(_twin(TRUTH))).initialize(_frame(), _detection())
    assert m.extra["orientation"]["status"] == "confirmed" and m.extra["orientation"]["flipped"]
    assert rotation_angle_deg(m.extra["t_camera_object"], TRUTH) < 1.5
    feature = m.extra["orientation"]["features"][0]
    assert feature["status"] == "detected" and feature["residual_px"] < 2.5


def test_without_orientation_logic_the_twin_is_what_you_get():
    """Before Phase 8: the lock keeps MegaPose's twin (the reported flip)."""
    m = _tracker(FakeMegaPose(_twin(TRUTH)), features=(), symmetries=None).initialize(_frame(), _detection())
    assert m.extra["orientation"] is None
    assert rotation_angle_deg(m.extra["t_camera_object"], TRUTH) > 170


def test_a_required_feature_under_a_hand_refuses_the_lock():
    c = project(MARK.center[None], TRUTH)[0].astype(int)
    required = PoseFeature("UR_CIRCLE", "circle", MARK.center, MARK.normal, MARK.radius, required_for_initialization=True)
    tracker = _tracker(FakeMegaPose(TRUTH), features=(required,))
    m = tracker.initialize(_frame(hand=(c[0] - 45, c[1] - 45, c[0] + 45, c[1] + 45)), _detection())
    assert m.confidence == 0.0 and "UR_CIRCLE" in m.extra["reject_reason"]


def test_a_correction_that_flips_to_the_twin_is_turned_back():
    service = FakeMegaPose(TRUTH)
    tracker = _tracker(service)
    tracker.commit(tracker.initialize(_frame(), _detection()))
    service.refine_pose = _twin(TRUTH)  # MegaPose "corrects" to the twin
    m = _correct_once(tracker, _frame())
    assert rotation_angle_deg(m.extra["t_camera_object"], TRUTH) < 2.0
    assert tracker.guard_counts["twin_flips_suppressed"] == 1


def test_a_real_turnaround_shown_by_the_features_switches_outright():
    """The tracker believed the twin (say, locked while the circle was covered);
    a correction to the truth, with the circle in view, is adopted whole — not blended."""
    service = FakeMegaPose(_twin(TRUTH))
    tracker = _tracker(service, features=(), symmetries=SYM)  # no features at the lock: it can't tell
    tracker.commit(tracker.initialize(_frame(), _detection()))
    assert rotation_angle_deg(tracker._output, TRUTH) > 170
    tracker.pose_features = [FEATURE]  # the circle is now known and visible
    service.refine_pose = TRUTH
    m = _correct_once(tracker, _frame())
    angle = rotation_angle_deg(m.extra["t_camera_object"], TRUTH)
    assert angle < 2.0, angle  # all the way, not 90 deg in between
    assert tracker.guard_counts["orientation_switches"] == 1


def test_a_large_correction_the_edges_dont_support_is_rejected():
    service = FakeMegaPose(TRUTH)
    tracker = _tracker(service, features=(), symmetries=None)
    tracker.commit(tracker.initialize(_frame(), _detection()))
    service.refine_pose = turned(TRUTH, (0, 1, 0), 35)  # a 35 deg "correction" of a box that didn't move
    m = _correct_once(tracker, _frame())
    assert rotation_angle_deg(m.extra["t_camera_object"], TRUTH) < 2.0
    assert tracker.guard_counts["corrections_rejected"] == 1
    assert "not supported" in m.extra["correction_guard"]["last"]["reason"]


def test_a_covered_circle_does_not_settle_the_lock_by_its_absence_elsewhere():
    """The circle is under a hand: the empty opposite corner is no proof (a missing print could be the defect)."""
    c = project(MARK.center[None], TRUTH)[0].astype(int)
    tracker = _tracker(FakeMegaPose(_twin(TRUTH)))
    m = tracker.initialize(_frame(hand=(c[0] - 45, c[1] - 45, c[0] + 45, c[1] + 45)), _detection())
    assert m.extra["orientation"]["status"] == "ambiguous"


def test_a_guide_lock_counts_as_operator_confirmed_orientation():
    """No features configured: the outline the operator lined up with settles which way round it is."""
    tracker = _tracker(FakeMegaPose(TRUTH), features=())
    m = tracker.initialize(_frame(), _detection(), initial_pose=TRUTH, pose_source="guide")
    assert m.extra["orientation"]["status"] == "confirmed" and "guide" in m.extra["orientation"]["reason"]
    assert rotation_angle_deg(m.extra["t_camera_object"], TRUTH) < 1.5


def test_visible_features_overrule_a_guide_lined_up_the_wrong_way_round():
    """The operator lined the box up turned round; the circle in view says otherwise."""
    tracker = _tracker(FakeMegaPose(TRUTH))
    m = tracker.initialize(_frame(), _detection(), initial_pose=_twin(TRUTH), pose_source="guide")
    assert m.extra["orientation"]["status"] == "confirmed" and m.extra["orientation"]["flipped"]
    assert rotation_angle_deg(m.extra["t_camera_object"], TRUTH) < 1.5


# ── steadiness once locked ───────────────────────────────────────────────


def test_a_tiny_correction_is_not_applied():
    """MegaPose a couple of millimetres off on a still object: the model must not creep."""
    service = FakeMegaPose(TRUTH)
    tracker = _tracker(service, features=(), symmetries=None)
    tracker.commit(tracker.initialize(_frame(), _detection()))
    shown = tracker._output.copy()
    nudged = TRUTH.copy()
    nudged[:3, 3] += [0.0015, -0.001, 0.0]  # 1.8 mm
    service.refine_pose = nudged
    m = _correct_once(tracker, _frame())
    assert np.allclose(m.extra["t_camera_object"], shown)
    assert tracker.guard_counts["corrections_kept"] == 1 and "dead-band" in tracker._last_guard["reason"]


def test_a_correction_that_fits_the_edges_worse_is_not_applied():
    service = FakeMegaPose(TRUTH)
    tracker = _tracker(service, features=(), symmetries=None)
    tracker.commit(tracker.initialize(_frame(), _detection()))
    shown = tracker._output.copy()
    service.refine_pose = turned(TRUTH, (0, 1, 0), 6)  # a few degrees off — worse than what's shown
    m = _correct_once(tracker, _frame())
    assert rotation_angle_deg(m.extra["t_camera_object"], shown) < 0.5
    assert tracker.guard_counts["corrections_kept"] == 1


def test_a_correction_that_fits_better_is_still_applied():
    """The shown pose drifted (6 deg); MegaPose has the truth: that one goes in."""
    service = FakeMegaPose(TRUTH)
    tracker = _tracker(service, features=(), symmetries=None)
    tracker.commit(tracker.initialize(_frame(), _detection()))
    drifted = turned(TRUTH, (0, 1, 0), 6)
    tracker._base, tracker._output = drifted.copy(), drifted.copy()
    service.refine_pose = TRUTH
    m = _correct_once(tracker, _frame())
    assert rotation_angle_deg(m.extra["t_camera_object"], TRUTH) < rotation_angle_deg(drifted, TRUTH) - 1.0  # pulled towards it
    assert tracker.guard_counts["corrections_kept"] == 0
