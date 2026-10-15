"""Click-to-select: lock onto the segment under the operator's click that the model fits best (app/api/vision.py)."""

import numpy as np

import app.api.vision as vision
from app.services.pose.model_pose_client import ImageSegment, PoseServiceResult
from app.services.tracking.ar_session import ARSession, ARTrackingConfig, TrackingState
from app.schemas.vision import BoundingBox, SegmentedObject

K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
PANEL = ImageSegment((477, 91, 775, 220), 0.039, [(477, 91), (775, 91), (775, 220), (477, 220)])
CASE = ImageSegment((474, 92, 791, 609), 0.174, [(474, 92), (791, 92), (791, 609), (474, 609)])
SPECK = ImageSegment((500, 100, 505, 104), 0.0001, [(500, 100), (505, 100), (505, 104)])


def _pose(z):
    t = np.eye(4)
    t[2, 3] = z
    return t


class FakeService:
    """The model squeezed into the panel strip fits it badly; the whole case well."""

    def __init__(self, segments, fits):
        self.segments, self.fits, self.searched = segments, fits, []

    def segments_at(self, jpeg, x, y, max_area=0.6):
        return self.segments

    def full_search(self, label, jpeg, k, bbox, outline=None):
        self.searched.append(tuple(bbox))
        iou, z = self.fits[tuple(bbox)]
        return PoseServiceResult(_pose(z), 0.9, "coarse+refine", 900.0, tuple(bbox), iou)


def test_the_region_the_model_fits_best_is_picked_not_the_smallest(monkeypatch):
    service = FakeService([SPECK, PANEL, CASE], {PANEL.bbox: (0.41, 0.6), CASE.bbox: (0.88, 0.6)})
    monkeypatch.setattr(vision, "_model_pose_client", service)
    events = []
    picked = vision.select_object_at(b"jpeg", 600, 150, "m1", "Case", K, events)
    assert picked is not None
    detection, pose = picked
    assert (detection.bbox.x2, detection.bbox.y2) == (791, 609)
    assert detection.confidence == 0.88 and pose is None  # MegaPose judged it; the tracker searches again
    assert SPECK.bbox not in service.searched  # too small to be the object: not searched
    assert "88%" in events[-1]


def test_a_fit_outside_the_working_distance_is_not_picked(monkeypatch):
    service = FakeService([PANEL, CASE], {PANEL.bbox: (0.5, 0.6), CASE.bbox: (0.95, 4.0)})  # "fits" 4 m away
    monkeypatch.setattr(vision, "_model_pose_client", service)
    picked = vision.select_object_at(b"jpeg", 600, 150, "m1", "Case", K, [])
    assert picked is not None and picked[0].bbox.y2 == 220


def test_nothing_under_the_click(monkeypatch):
    monkeypatch.setattr(vision, "_model_pose_client", FakeService([], {}))
    events = []
    assert vision.select_object_at(b"jpeg", 10, 10, "m1", "Case", K, events) is None
    assert "Nothing segmented" in events[0]


def test_a_seeded_detection_is_locked_on_the_next_step_even_while_tracking():
    session = ARSession("case", lambda frame: None, tracker=None, config=ARTrackingConfig())
    session.state = TrackingState.TRACKING
    detection = SegmentedObject(class_label="Case", confidence=0.88, bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10), polygon=[])
    events = []
    session.seed(detection, events)
    assert session.state == TrackingState.INITIALIZING and session._pending is detection
    assert session.detection_count == 1 and "selected by the operator" in events[0]


# ── with the model's geometry: the pose from the outline itself ─────────

from app.services.pose.outline_pose import outline_pose  # noqa: E402
from app.services.tracking.trackers import resting_side_ok  # noqa: E402

from glb_builder import box  # noqa: E402

CASE_V, CASE_F = box((0, 0, 0), (0.114, 0.035, 0.19))  # x, y (thin, "top"), z


def _look(eye):
    eye = np.asarray(eye, float)
    z = -eye / np.linalg.norm(eye)
    up = np.array([0, 1.0, 0]) if abs(z[1]) < 0.99 else np.array([0, 0, -1.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    t = np.eye(4)
    t[:3, :3] = np.stack([x, y, z])
    t[:3, 3] = -t[:3, :3] @ eye
    return t


def _outline(t_co):
    """The case's outline as the camera sees it (hull of its projected corners)."""
    import cv2

    cam = CASE_V @ t_co[:3, :3].T + t_co[:3, 3]
    px = cam @ K.T
    px = px[:, :2] / px[:, 2:3]
    return cv2.convexHull(px.astype(np.float32)).reshape(-1, 2)


def _iou_with_truth(found, truth):
    import cv2

    a, b = np.zeros((720, 1280), np.uint8), np.zeros((720, 1280), np.uint8)
    cv2.fillConvexPoly(a, np.round(_outline(found)).astype(np.int32), 1)
    cv2.fillConvexPoly(b, np.round(_outline(truth)).astype(np.int32), 1)
    return (a & b).sum() / (a | b).sum()


def test_the_outline_alone_gives_a_pose_that_covers_the_object():
    truth = _look((0.05, 0.30, 0.12))  # top face mostly towards the camera, seen at an angle
    found = outline_pose(CASE_V, _outline(truth), K, rotations=1500)
    assert found is not None and found.iou > 0.93
    assert _iou_with_truth(found.t_co, truth) > 0.9
    assert abs(found.t_co[2, 3] - truth[2, 3]) < 0.03  # the right distance, from the size


def test_the_resting_side_is_respected():
    truth = _look((0.05, 0.30, 0.12))
    found = outline_pose(CASE_V, _outline(truth), K, allowed=lambda t: resting_side_ok(t, "camera"), rotations=1500)
    assert found is not None and resting_side_ok(found.t_co, "camera")


def test_with_geometry_the_click_picks_by_outline_and_hands_over_the_pose(monkeypatch):
    truth = _look((0.05, 0.30, 0.12))
    outline = [tuple(p) for p in _outline(truth)]
    xs, ys = zip(*outline)
    whole = ImageSegment((min(xs), min(ys), max(xs), max(ys)), 0.1, outline)
    strip = ImageSegment((min(xs), min(ys), max(xs), min(ys) + 30), 0.02,
                         [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), min(ys) + 30), (min(xs), min(ys) + 30)])
    service = FakeService([strip, whole], {})
    monkeypatch.setattr(vision, "_model_pose_client", service)
    picked = vision.select_object_at(b"jpeg", xs[0], ys[0], "m1", "Case", K, [], vertices=CASE_V, resting="camera")
    assert picked is not None
    detection, pose = picked
    assert detection.bbox.y2 == max(ys) and pose is not None
    assert _iou_with_truth(pose, truth) > 0.9
    assert service.searched == []  # no MegaPose search needed
