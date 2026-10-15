"""3D edge alignment (app/services/pose/edge_align.py) on rendered images of a case-like box."""

import cv2
import numpy as np
import pytest

from app.services.pose.edge_align import EdgeModel, align, edge_image, visible_edge_points

from glb_builder import box, merge

K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
SHAPE = (720, 1280)

# A flat case, 19 x 3.5 x 11.4 cm (x, y = up, z), with a raised end block (a step = internal edges).
BODY = box((-0.02, 0.0, 0.0), (0.15, 0.035, 0.114))
BLOCK = box((0.075, 0.003, 0.0), (0.04, 0.041, 0.114))
VERTS, FACES = merge(BODY, BLOCK)
VERTS = VERTS - (VERTS.min(axis=0) + VERTS.max(axis=0)) / 2  # recentred, as the tracker's frame is


def look_at(eye, target=(0, 0, 0), up=(0, 1, 0)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    r = np.stack([x, y, z])
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = -r @ eye
    return t


TRUE_POSE = look_at((0.05, 0.36, 0.17))  # camera above, looking down at ~25 deg from vertical


def render(t_co, background=235):
    """Flat-shaded render: dark case on a light desk, each face lit by its angle to a light."""
    img = np.full(SHAPE, background, np.uint8)
    cam = VERTS @ t_co[:3, :3].T + t_co[:3, 3]
    px = cam @ K.T
    px = np.round(px[:, :2] / px[:, 2:3]).astype(np.int32)
    light = np.array([0.3, -0.8, -0.5])
    light /= np.linalg.norm(light)
    order = np.argsort(-cam[FACES].mean(axis=1)[:, 2])  # far to near
    for f in order:
        a, b, c = cam[FACES[f]]
        n = np.cross(b - a, c - a)
        n /= np.linalg.norm(n) or 1.0
        if n @ (-(a + b + c) / 3) <= 0:
            continue  # facing away
        shade = int(30 + 110 * max(0.0, float(n @ -light)))  # a dark case: faces 30..140 grey
        cv2.fillConvexPoly(img, px[FACES[f]], shade)
    rng = np.random.default_rng(0)
    return np.clip(img.astype(int) + rng.integers(-3, 4, SHAPE), 0, 255).astype(np.uint8)


def perturbed(t_co, tilt_deg=10.0, turn_deg=4.0, shift=(0.008, -0.006, 0.01)):
    out = t_co.copy()
    rot, _ = cv2.Rodrigues(np.radians([tilt_deg, turn_deg, 0.0]))
    out[:3, :3] = rot @ out[:3, :3]
    out[:3, 3] = out[:3, 3] + np.asarray(shift)
    return out


def rotation_error_deg(a, b):
    r = a[:3, :3] @ b[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


@pytest.fixture(scope="module")
def model():
    return EdgeModel.from_mesh(VERTS, FACES)


def test_edge_model_keeps_the_box_creases(model):
    assert 200 <= len(model.points) <= 6000
    assert model.crease.mean() > 0.9  # a box is all sharp edges
    # Seen from above, the top face's border and the step are visible; edges underneath aren't.
    assert 0.2 < visible_edge_points(model, TRUE_POSE).mean() < 0.9


def test_a_tilted_pose_is_pulled_back_onto_the_edges(model):
    image = render(TRUE_POSE)
    start = perturbed(TRUE_POSE)
    assert rotation_error_deg(start, TRUE_POSE) > 9
    fit = align(model, start, K, edge_image(image))
    assert fit is not None and fit.improved
    assert rotation_error_deg(fit.t_co, TRUE_POSE) < 2.0
    assert np.linalg.norm(fit.t_co[:3, 3] - TRUE_POSE[:3, 3]) < 0.005
    assert fit.score > 0.75  # a crease between two equally lit faces shows no edge: support < 1 even when exact


def test_works_when_part_of_the_object_is_cut_off_by_the_image_border(model):
    """The case running off the bottom of the frame — what tilted the overlay on the user's camera."""
    image = render(TRUE_POSE)
    cam = VERTS @ TRUE_POSE[:3, :3].T + TRUE_POSE[:3, 3]
    px = cam @ K.T
    bottom = int((px[:, 1] / px[:, 2]).max())
    image[bottom - 60 :] = 235  # the lowest 60 px of the case are outside the image
    fit = align(model, perturbed(TRUE_POSE), K, edge_image(image))
    assert fit is not None and fit.improved
    assert rotation_error_deg(fit.t_co, TRUE_POSE) < 3.0


def test_a_hand_over_one_end_is_ignored(model):
    image = render(TRUE_POSE)
    hand = np.zeros(SHAPE, bool)
    cam = VERTS @ TRUE_POSE[:3, :3].T + TRUE_POSE[:3, 3]
    px = cam @ K.T
    left = int((px[:, 0] / px[:, 2]).min())
    hand[:, left - 40 : left + 120] = True
    image[hand] = 150  # skin over the left end, with its own edges
    fit = align(model, perturbed(TRUE_POSE), K, edge_image(image, exclude=hand))
    assert fit is not None and fit.improved
    assert rotation_error_deg(fit.t_co, TRUE_POSE) < 3.0


def test_a_correct_pose_stays_put(model):
    fit = align(model, TRUE_POSE, K, edge_image(render(TRUE_POSE)))
    assert fit is not None
    assert rotation_error_deg(fit.t_co, TRUE_POSE) < 1.0


def test_no_edges_means_no_change(model):
    blank = np.full(SHAPE, 200, np.uint8)
    start = perturbed(TRUE_POSE)
    fit = align(model, start, K, edge_image(blank))
    assert fit is None or not fit.improved
    if fit is not None:
        assert np.allclose(fit.t_co, start)


# ── in the tracker: at the lock and on background corrections ───────────


def _frame():
    from app.services.pose.calibration import CameraCalibration
    from app.services.tracking.trackers import Frame

    gray = render(TRUE_POSE)
    calibration = CameraCalibration.approximate(SHAPE[1], SHAPE[0])
    calibration.camera_matrix = K.copy()
    return Frame(np.dstack([gray] * 3), calibration)


class TiltedMegaPose:
    """MegaPose's answers off by a 10 deg tilt (what the user's camera showed)."""

    def __init__(self, pose):
        self.pose = pose

    def full_search(self, label, jpeg, k, bbox, outline=None):
        from app.services.pose.model_pose_client import PoseServiceResult

        return PoseServiceResult(self.pose, 0.9, "coarse+refine", 900.0, (400, 150, 900, 560))

    def refine(self, label, jpeg, k, prev_pose, iterations):
        from app.services.pose.model_pose_client import PoseServiceResult

        return PoseServiceResult(self.pose, 0.9, "refine", 250.0, (400, 150, 900, 560))


def _tracker(service, model, **kw):
    from app.services.tracking.trackers import MegaPoseTracker

    return MegaPoseTracker(service, "case", refine_iterations=2, edge_model=model, async_refine=False, **kw)


def _detection():
    from app.schemas.vision import BoundingBox, SegmentedObject

    return SegmentedObject(class_label="case", confidence=0.8, bbox=BoundingBox(x1=400, y1=150, x2=900, y2=560), polygon=[])


def test_the_lock_is_pulled_onto_the_edges(model):
    m = _tracker(TiltedMegaPose(perturbed(TRUE_POSE)), model).initialize(_frame(), _detection())
    assert m.extra["edge_fit"]["adjusted"]
    assert rotation_error_deg(m.extra["t_camera_object"], TRUE_POSE) < 2.0
    assert m.extra["edge_fit"]["score"] > 0.75


def test_without_an_edge_model_the_lock_is_megaposes(model):
    m = _tracker(TiltedMegaPose(perturbed(TRUE_POSE)), None).initialize(_frame(), _detection())
    assert m.extra["edge_fit"] is None
    assert rotation_error_deg(m.extra["t_camera_object"], TRUE_POSE) > 9


def test_a_background_correction_is_edge_refined_too(model):
    """A refine off by a few degrees is put on the edges before it is blended in."""
    service = TiltedMegaPose(TRUE_POSE)
    tracker = _tracker(service, model, edge_gain_translation=1.0, edge_gain_rotation=1.0)
    tracker.commit(tracker.initialize(_frame(), _detection()))
    service.pose = perturbed(TRUE_POSE, tilt_deg=5.0, turn_deg=2.0, shift=(0.004, 0.0, 0.0))
    tracker._needs_settle = True  # ask for one correction on the next (still) frame
    tracker.update(_frame())
    m = tracker.update(_frame())
    assert m.extra["edge_fit"]["adjusted"]
    assert rotation_error_deg(m.extra["t_camera_object"], TRUE_POSE) < 2.0


def test_a_lock_already_on_the_edges_is_left_alone(model):
    m = _tracker(TiltedMegaPose(TRUE_POSE), model).initialize(_frame(), _detection())
    assert not m.extra["edge_fit"]["adjusted"]
    assert np.allclose(m.extra["t_camera_object"], TRUE_POSE)


def test_a_lock_from_a_given_pose_skips_the_search_and_is_put_on_the_edges(model):
    """The operator clicked the object: its outline gave a pose; MegaPose's full search isn't run."""

    class NoSearch(TiltedMegaPose):
        def full_search(self, *args, **kwargs):
            raise AssertionError("full search should not run")

    detection = _detection()
    detection.confidence = 0.95
    m = _tracker(NoSearch(TRUE_POSE), model).initialize(_frame(), detection, initial_pose=perturbed(TRUE_POSE, 6.0, 2.0, (0.004, 0, 0.005)))
    assert m.extra["edge_fit"]["adjusted"]
    assert rotation_error_deg(m.extra["t_camera_object"], TRUE_POSE) < 2.0
    assert m.confidence == pytest.approx(0.95)
