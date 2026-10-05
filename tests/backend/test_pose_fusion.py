"""Model-based tracking, FoundationPose++ style: optical flow every frame, MegaPose corrections in the background."""

import threading

import numpy as np
import pytest

from app.schemas.vision import BoundingBox, SegmentedObject
from app.services.pose.calibration import CameraCalibration
from app.services.pose.model_pose_client import PoseServiceResult
from app.services.tracking.trackers import Frame, MegaPoseTracker, blend_pose, propagate_pose

K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.5) -> np.ndarray:
    t = np.eye(4)
    t[:3, 3] = (x, y, z)
    return t


def _similarity(dx: float = 0.0, dy: float = 0.0, scale: float = 1.0, angle_deg: float = 0.0, about=(320, 240)) -> np.ndarray:
    a = np.radians(angle_deg)
    rs = scale * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = np.asarray(about, float)
    m = np.eye(3)
    m[:2, :2] = rs
    m[:2, 2] = c - rs @ c + (dx, dy)
    return m


def test_no_motion_keeps_the_pose():
    assert propagate_pose(_pose(0.02, -0.01), np.eye(3), K) == pytest.approx(_pose(0.02, -0.01))


def test_image_shift_moves_the_object_sideways_at_its_depth():
    # 50 px at f = 500 px and 0.5 m away is 5 cm.
    out = propagate_pose(_pose(), _similarity(dx=50), K)
    assert out[:3, 3] == pytest.approx((0.05, 0.0, 0.5))
    assert out[:3, :3] == pytest.approx(np.eye(3))


def test_growing_in_the_image_means_closer():
    out = propagate_pose(_pose(), _similarity(scale=1.25), K)
    assert out[2, 3] == pytest.approx(0.4)  # 0.5 / 1.25


def test_in_plane_turn_rotates_about_the_viewing_axis():
    out = propagate_pose(_pose(), _similarity(angle_deg=30), K)
    # Image x right / y down = camera x / y, so a +30 deg image rotation is +30 deg about camera z.
    a = np.radians(30)
    assert out[:3, :3] == pytest.approx(np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]))
    assert out[:3, 3] == pytest.approx((0, 0, 0.5))  # turning about its own center doesn't move it


# -- the tracker with a slow (background) MegaPose -------------------------

def _textured_frame(dx: int = 0) -> Frame:
    rng = np.random.default_rng(0)
    gray = rng.integers(100, 140, (480, 640)).astype(np.uint8)
    gray[160:320, 240:360] = np.kron(rng.integers(0, 255, (16, 12)), np.ones((10, 10))).astype(np.uint8)
    gray = np.roll(gray, dx, axis=1)
    return Frame(np.dstack([gray] * 3), CameraCalibration.approximate(640, 480))


class SlowPoseService:
    """Full search answers at once (locked on the object); each refine waits until the
    test releases it and reports `projected` as where it drew the model."""

    def __init__(self, refine_pose: np.ndarray, projected=(240, 160, 360, 320)):
        self.refine_pose = refine_pose
        self.projected = projected
        self.release = threading.Event()
        self.refines = 0

    def full_search(self, label, jpeg, k, bbox, outline=None):
        return PoseServiceResult(_pose(), 0.9, "coarse+refine", 900.0, (240, 160, 360, 320))  # lock on the object

    def refine(self, label, jpeg, k, prev_pose, iterations):
        self.refines += 1
        assert self.release.wait(5)
        return PoseServiceResult(self.refine_pose, 0.9, "refine", 250.0, self.projected)


def _locked(service, gain: float = 1.0) -> MegaPoseTracker:
    # Full gain: these tests check where a correction lands; blending is tested separately.
    tracker = MegaPoseTracker(
        service, "m1", refine_iterations=2, still_motion_px=1.5,
        correction_gain_translation=gain, correction_gain_rotation=gain,
    )
    detection = SegmentedObject(class_label="cup", confidence=0.8, bbox=BoundingBox(x1=240, y1=160, x2=360, y2=320), polygon=[])
    tracker.commit(tracker.initialize(_textured_frame(), detection))
    return tracker


def test_moving_frames_are_answered_by_flow_without_waiting_for_megapose():
    service = SlowPoseService(refine_pose=_pose())
    tracker = _locked(service)
    k = _textured_frame().calibration.camera_matrix
    m = tracker.update(_textured_frame(dx=10))  # moved: refine starts in the background, frame answered now
    assert service.refines == 1 and m.extra["refine_pending"]
    expected_x = 10 / k[0, 0] * 0.5
    assert m.extra["t_camera_mesh"][0, 3] == pytest.approx(expected_x, abs=1e-3)
    m = tracker.update(_textured_frame(dx=20))  # MegaPose still busy: flow carries on
    assert service.refines == 1
    assert m.extra["t_camera_mesh"][0, 3] == pytest.approx(2 * expected_x, abs=1e-3)
    service.release.set()


def test_a_late_correction_is_carried_forward_by_the_motion_since_it_was_asked():
    # MegaPose (asked at the 10 px frame) says the object is 1 cm further than flow thought.
    k = _textured_frame().calibration.camera_matrix
    asked_at = _pose(x=10 / k[0, 0] * 0.51, z=0.51)
    service = SlowPoseService(refine_pose=asked_at)
    tracker = _locked(service)
    tracker.update(_textured_frame(dx=10))  # refine asked here
    tracker.update(_textured_frame(dx=20))  # object kept moving while MegaPose worked
    service.release.set()
    for _ in range(100):  # let the background refine finish
        if tracker._pending[0].done():
            break
        threading.Event().wait(0.01)
    m = tracker.update(_textured_frame(dx=20))
    t = m.extra["t_camera_mesh"][:3, 3]
    # Corrected depth, and moved on by the 10 px that happened after it was asked: x = 20 px at 0.51 m.
    assert t[2] == pytest.approx(0.51, abs=1e-3)
    assert t[0] == pytest.approx(20 / k[0, 0] * 0.51, abs=1e-3)


def test_a_correction_that_misses_the_object_is_not_adopted():
    service = SlowPoseService(refine_pose=_pose(z=0.9), projected=(0, 0, 50, 50))  # nowhere near the object
    service.release.set()
    tracker = _locked(service)
    tracker.async_refine = False
    first = tracker.update(_textured_frame(dx=10))
    assert first.extra["t_camera_mesh"][2, 3] == pytest.approx(0.5)  # kept flow's pose
    assert first.confidence > 0.9
    tracker.update(_textured_frame(dx=20))  # second miss in a row: confidence drops so the session re-detects
    assert tracker.update(_textured_frame(dx=30)).confidence < 0.1


def test_blend_moves_part_way_in_position_and_along_the_rotation_arc():
    a = _pose(z=0.5)
    b = _pose(x=0.1, z=0.5)
    angle = np.radians(40)
    b[:3, :3] = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    out = blend_pose(a, b, translation_gain=0.5, rotation_gain=0.25)
    assert out[0, 3] == pytest.approx(0.05)
    assert np.degrees(np.arccos((np.trace(out[:3, :3]) - 1) / 2)) == pytest.approx(10.0)  # a quarter of 40 deg


def test_a_correction_only_pulls_part_way_so_one_bad_refine_cannot_jump_the_model():
    service = SlowPoseService(refine_pose=_pose(z=0.6))  # MegaPose says 10 cm further than flow
    service.release.set()
    tracker = _locked(service, gain=0.5)
    tracker.async_refine = False
    m = tracker.update(_textured_frame(dx=10))
    assert m.extra["t_camera_mesh"][2, 3] == pytest.approx(0.55, abs=1e-3)  # half way
