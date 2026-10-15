"""Fixed-station mode (tracking/station.py, vision station-lock) and per-step inspection focus."""

import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import component_inspection, vision
from app.main import app
from app.services.tracking.ar_session import TrackingState
from app.services.tracking.station import StationLock

from test_component_inspection_api import FakeSession, frame, model  # noqa: F401  (fixture)
from test_component_inspection import jittered, learn, make_session, render
from test_component_projection import K
from glb_builder import plate_with_screws


def _scene(dx=0, hole=False):
    rng = np.random.default_rng(3)
    img = rng.integers(60, 90, (480, 640, 3), dtype=np.uint8)
    part = np.kron(rng.integers(100, 250, (16, 24)), np.ones((10, 10))).astype(np.uint8)
    img[160:320, 200:440] = part[..., None]
    if hole:
        cv2.circle(img, (260, 200), 8, (10, 10, 10), -1)  # a missing screw: a small local change
    return np.roll(img, dx, axis=1)


def _lock(img):
    return StationLock.create("m1", np.eye(4), (0, 0, -0.5), (0, 0, 0, 1), img, (200, 160, 440, 320), max_shift_px=4.0)


def test_same_scene_has_no_shift():
    lock = _lock(_scene())
    shift, _ = lock.shift(_scene())
    assert shift < 0.5


def test_a_moved_part_is_detected():
    lock = _lock(_scene())
    shift, _ = lock.shift(_scene(dx=12))
    assert shift == pytest.approx(12, abs=1.5) and shift > lock.max_shift_px


def test_a_missing_component_is_not_mistaken_for_movement():
    lock = _lock(_scene())
    shift, _ = lock.shift(_scene(hole=True))
    assert shift < lock.max_shift_px


def test_lock_round_trips_through_disk(tmp_path):
    lock = _lock(_scene())
    lock.save(tmp_path)
    loaded = StationLock.load(tmp_path, "m1")
    assert loaded is not None and loaded.roi == lock.roi and loaded.frame_size == (640, 480)
    assert loaded.shift(_scene())[0] < 0.5
    StationLock.remove(tmp_path, "m1")
    assert StationLock.load(tmp_path, "m1") is None


# ── through the AR frame path ───────────────────────────────────────────


@pytest.fixture
def station(model, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(vision._settings, "station_locks_dir", str(tmp_path / "locks"))
    monkeypatch.setattr(vision, "_station_locks", {})
    return model


def test_a_locked_station_reuses_the_pose_and_never_tracks(station, monkeypatch):
    session = FakeSession()
    t_co = jittered(0)
    img = render(_registry(), t_co)
    StationLock.create(str(station.id), t_co, (0, 0, -0.3), (0, 0, 0, 1), img, (300, 200, 900, 520)).save(
        vision._station_dir()
    )
    session.step = lambda f: (_ for _ in ()).throw(AssertionError("tracker must not run while the station is locked"))
    out = _frame(station, session, img, station_lock=True)
    assert out.state == "TRACKING" and out.station.locked and not out.station.moved
    assert out.position.z == pytest.approx(-0.3)


def test_moving_the_part_off_the_station_pose_stops_inspection(station):
    session = FakeSession()
    t_co = jittered(0)
    img = render(_registry(), t_co)
    StationLock.create(str(station.id), t_co, (0, 0, -0.3), (0, 0, 0, 1), img, (300, 200, 900, 520)).save(
        vision._station_dir()
    )
    moved = np.roll(img, 30, axis=1)
    out = _frame(station, session, moved, station_lock=True, inspection="inspect")
    assert out.state == "LOST" and out.station.moved
    assert any("moved" in e for e in out.events)
    assert "moved" in out.inspection.components[0].reason


def test_lock_endpoint_needs_healthy_tracking(station):
    vision._ar_sessions.pop("nobody", None)
    response = TestClient(app).post(
        "/api/vision/station-lock", json={"session_id": "nobody", "model_id": str(station.id), "image_base64": ""}
    )
    assert response.status_code == 409


# ── per-step focus ──────────────────────────────────────────────────────


def test_focus_inspects_only_the_steps_components(tmp_path):
    registry = _registry()
    session = make_session(registry, tmp_path, confirm_missing_after=3)
    learn(session, registry)
    for k in range(4):
        t_co = jittered(k)
        session.process(render(registry, t_co, {"SCREW_001", "SCREW_002"}, seed=900 + k), t_co, K, True, None,
                        now=300 + k * 0.3, focus=["SCREW_002"])
    statuses = {cid: d.state.status.value for cid, d in session.decisions.items()}
    assert statuses["SCREW_002"] == "MISSING_CONFIRMED"
    assert statuses["SCREW_001"] == "NOT_INSPECTED"  # missing too, but not this step's concern
    summary = session.summary()
    assert (summary.required, summary.missing, summary.verdict) == (1, 1, "FAIL")


_reg = None


def _registry():
    global _reg
    if _reg is None:
        from app.services.component_inspection.registry import registry_from_glb

        _reg = registry_from_glb("m", plate_with_screws(), 1.0)
    return _reg


def _frame(model, session, img, **fields):
    from app.schemas.pose import ARFrameRequest

    ok, png = cv2.imencode(".png", img)
    request = ARFrameRequest(session_id="s1", mode="model", image_base64="", model_id=str(model.id), **fields)
    return vision.process_ar_frame(request, session, model, png.tobytes(), 0.0, db=object())


def test_tracking_state_import_is_used():
    assert TrackingState.TRACKING.value == "TRACKING" and base64  # keeps imports honest
