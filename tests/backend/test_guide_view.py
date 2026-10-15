"""Guide view: lock once the object is lined up with the model's outline (app/services/tracking/guide_view.py)."""

import numpy as np

from app.services.pose.edge_align import EdgeModel
from app.services.tracking.guide_view import GuideMatcher, default_guide_pose

from test_edge_align import FACES, K, SHAPE, TRUE_POSE, VERTS, perturbed, render, rotation_error_deg


def _model():
    return EdgeModel.from_mesh(VERTS, FACES)


def test_the_default_guide_shows_the_object_in_front_long_side_across():
    t = default_guide_pose(VERTS, K, (SHAPE[1], SHAPE[0]), resting="camera")
    cam = VERTS @ t[:3, :3].T + t[:3, 3]
    assert (cam[:, 2] > 0).all()
    px = cam @ K.T
    px = px[:, :2] / px[:, 2:3]
    width, height = px.max(axis=0) - px.min(axis=0)
    assert width > height  # the case's 19 cm side runs across the image
    assert 0.4 * SHAPE[0] < width < 0.9 * SHAPE[1]  # fills a good part of it
    assert abs(px[:, 0].mean() - K[0, 2]) < 60  # centred
    up = t[:3, 1]  # model +Y in the camera frame: the top tilts towards the camera, up in the image
    assert up[1] < 0 and up[2] < 0


def test_lined_up_with_the_guide_it_locks_after_two_agreeing_checks():
    model = _model()
    guide = perturbed(TRUE_POSE, tilt_deg=5.0, turn_deg=3.0, shift=(0.006, -0.004, 0.008))  # roughly lined up
    frame = render(TRUE_POSE)
    matcher = GuideMatcher(confirmations=2)
    assert matcher.poll(model, guide, frame, None, K, run_inline=True) is None  # first match: wait for a second
    assert matcher.last is not None and matcher.last.matched
    locked = matcher.poll(model, guide, frame, None, K, run_inline=True)
    assert locked is not None
    assert rotation_error_deg(locked.t_co, TRUE_POSE) < 2.0  # refined onto the edges, not the guide pose


def test_nothing_lined_up_never_locks():
    model = _model()
    blank = np.full(SHAPE, 200, np.uint8)
    matcher = GuideMatcher(confirmations=1)
    for _ in range(3):
        assert matcher.poll(model, TRUE_POSE, blank, None, K, run_inline=True) is None
    assert matcher.last is not None and not matcher.last.matched


def test_the_object_far_from_the_guide_does_not_lock():
    model = _model()
    frame = render(TRUE_POSE)
    far_guide = perturbed(TRUE_POSE, tilt_deg=0.0, turn_deg=0.0, shift=(0.12, 0.0, 0.0))  # outline 12 cm to the side
    matcher = GuideMatcher(confirmations=1)
    assert matcher.poll(model, far_guide, frame, None, K, run_inline=True) is None


def test_saving_the_resting_side_keeps_a_saved_guide_view(tmp_path, monkeypatch):
    import app.api.vision as vision

    monkeypatch.setattr(vision._settings, "station_settings_dir", str(tmp_path))
    guide = np.eye(4)
    guide[2, 3] = 0.4
    vision._write_station_settings("m1", vision.StationSettings(resting="camera", guide_pose=guide.tolist()))
    vision.put_station_settings("m1", vision.StationSettings(resting="upright"))
    saved = vision.station_settings("m1")
    assert saved.resting == "upright" and saved.guide_pose == guide.tolist()
    assert vision.reset_guide_view("m1").guide_pose is None


def test_a_lock_that_passed_the_guide_check_is_not_thrown_away_by_the_session():
    """Edge score 0.45 passes a 0.40 guide threshold; as a lock confidence it must clear 'lost' (0.5)."""
    from app.api.vision import guide_lock_confidence
    from app.services.tracking.ar_session import ARTrackingConfig

    config = ARTrackingConfig(good_confidence=0.7, lost_confidence=0.5)
    assert guide_lock_confidence(0.45, config.good_confidence) >= config.lost_confidence
    assert guide_lock_confidence(0.9, config.good_confidence) == 0.9
