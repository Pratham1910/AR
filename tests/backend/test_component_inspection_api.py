"""Component inspection through the live AR frame path (vision.process_ar_frame) and the registry endpoints."""

import base64
import json
import uuid
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import component_inspection, vision
from app.core.database import get_db
from app.main import app
from app.schemas.pose import ARFrameRequest
from app.services.pose.calibration import CameraCalibration
from app.services.tracking.ar_session import StepResult, TrackedObject, TrackingState

from glb_builder import plate_with_screws
from test_component_inspection import jittered, render
from test_component_projection import K

MODEL_ID = uuid.UUID("00000000-0000-0000-0000-00000000c0de")


@pytest.fixture
def model(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / "plate.glb").write_bytes(plate_with_screws())
    s = component_inspection._settings
    for name, value in {
        "models_3d_dir": str(models),
        "component_registry_dir": str(tmp_path / "registry"),
        "component_reference_dir": str(tmp_path / "reference"),
        "inspection_log_dir": str(tmp_path / "logs"),
        "part_calibration_dir": str(tmp_path / "calibration"),
        "inspection_interval_s": 0.0,
        "inspection_async": False,
        "hand_mask_method": "off",
    }.items():
        monkeypatch.setattr(s, name, value)
    monkeypatch.setattr(component_inspection, "named_parts", lambda db, m: {})
    monkeypatch.setattr(component_inspection, "_registry_cache", {})
    monkeypatch.setattr(component_inspection, "_sessions", {})
    monkeypatch.setattr(vision, "load_calibration", lambda *a, **k: CameraCalibration(K, np.zeros(5), 1280, 720))
    return SimpleNamespace(id=MODEL_ID, storage_key="plate.glb", scale=1.0, name="Plate", asset_id=uuid.uuid4(), component_id=None)


class FakeSession:
    """Stands in for the AR session: reports a given state and pose for every frame."""

    def __init__(self):
        self.state, self.t_co, self.monitoring = TrackingState.TRACKING, None, False
        self.obj = None
        self.config = SimpleNamespace(good_confidence=0.7, lost_confidence=0.5)
        self.frame_index = self.detection_runs = self.detection_count = self.tracking_frames = self.frames_since_detection = 0

    orientation = None

    def step(self, frame):
        obj = TrackedObject(1, "Plate", 0.9 if not self.monitoring else 0.6, 0, 0, 0.0, position=(0, 0, 0.3),
                            quaternion=(0, 0, 0, 1), extra={"t_camera_object": self.t_co, "orientation": self.orientation})
        return StepResult(self.state, True, self.monitoring, obj, None, False, True,
                          {"detection": 0, "initialization": 0, "tracking": 0, "refinement": 0}, [])


def frame(model, session, k, inspection, mode="model", removed=(), run="r1"):
    session.t_co = jittered(k)
    ok, jpeg = cv2.imencode(".png", render_plate(session.t_co, removed, k))
    request = ARFrameRequest(session_id="s1", mode=mode, image_base64="", model_id=str(model.id),
                             real_world_height_m=0.2, inspection=inspection, inspection_run=run)
    return vision.process_ar_frame(request, session, model, jpeg.tobytes(), 0.0, db=object())


_registry = None


def render_plate(t_co, removed, k):
    global _registry
    if _registry is None:
        from app.services.component_inspection.registry import registry_from_glb

        _registry = registry_from_glb("m", plate_with_screws(), 1.0)
    return render(_registry, t_co, removed, seed=k)


def test_inspection_off_adds_nothing_to_the_response(model):
    assert frame(model, FakeSession(), 0, "off").inspection is None


def test_learn_then_find_the_missing_screw_through_the_ar_frame_path(model, tmp_path):
    session = FakeSession()
    for k in range(8):
        out = frame(model, session, k, "learn").inspection
    assert out.mode == "learn" and out.components[0].learn_samples == 8
    out = frame(model, session, 8, "inspect", removed={"SCREW_004"}).inspection
    assert out.message.startswith("Learned the good state of 8/8")
    assert (tmp_path / "reference" / f"{MODEL_ID}.json").exists()
    for k in range(9, 16):
        out = frame(model, session, k, "inspect", removed={"SCREW_004"}).inspection
    by_id = {c.id: c for c in out.components}
    assert by_id["SCREW_004"].status == "MISSING_CONFIRMED"
    assert by_id["SCREW_004"].bbox_px and by_id["SCREW_004"].outline_px and by_id["SCREW_004"].last_valid
    assert by_id["SCREW_004"].verdicts[0].verifier == "reference"
    assert out.summary.verdict == "FAIL" and out.summary.missing == 1 and out.summary.present == 7
    logs = list((tmp_path / "logs" / str(MODEL_ID)).glob("s1-r1.jsonl"))
    assert logs and len(logs[0].read_text().splitlines()) == 16


def test_tracking_loss_and_low_confidence_are_reasons_not_defects(model):
    session = FakeSession()
    for k in range(8):
        frame(model, session, k, "learn")
    session.state = TrackingState.LOST
    out = frame(model, session, 8, "inspect", removed={"SCREW_001"}).inspection
    assert all(c.status == "UNCERTAIN" and c.visibility == "tracking" for c in out.components)
    assert out.components[0].visibility_reasons == ["tracking lost"]
    session.state, session.monitoring = TrackingState.TRACKING, True
    out = frame(model, session, 9, "inspect", removed={"SCREW_001"}).inspection
    assert "tracking confidence 0.60 below 0.70" in out.components[0].reason


def test_an_unconfirmed_orientation_is_a_reason_not_a_defect(model):
    """Phase 8: a symmetric part whose pose features don't settle which way round it is."""
    session = FakeSession()
    for k in range(8):
        frame(model, session, k, "learn")
    session.orientation = {"status": "ambiguous", "reason": "features not seen (covered, out of view, or too faint)"}
    for k in range(8, 16):
        out = frame(model, session, k, "inspect", removed={"SCREW_001"}).inspection
    assert all(c.status != "MISSING_CONFIRMED" for c in out.components)
    assert "orientation not confirmed" in out.components[0].reason
    # Without configured pose features it can't be confirmed at all: inspection runs as before.
    session.orientation = {"status": "ambiguous", "reason": "symmetric model and no pose features configured"}
    for k in range(16, 24):
        out = frame(model, session, k, "inspect", removed={"SCREW_001"}).inspection
    assert {c.id: c for c in out.components}["SCREW_001"].status == "MISSING_CONFIRMED"


def test_markerless_tracking_has_no_pose_to_inspect_with(model):
    out = frame(model, FakeSession(), 0, "inspect", mode="markerless").inspection
    assert "Model-based (CAD) tracking" in out.components[0].reason


def test_async_passes_never_block_the_frame_and_report_the_latest(model, monkeypatch):
    """With the worker on, a frame returns at once with the last finished pass."""
    import threading

    monkeypatch.setattr(component_inspection._settings, "inspection_async", True)
    gate = threading.Event()
    original = component_inspection.InspectionSession.process

    def slow(self, *args, **kwargs):
        gate.wait(5)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(component_inspection.InspectionSession, "process", slow)
    session = FakeSession()
    first = frame(model, session, 0, "inspect").inspection
    assert first.passes == 0 and not first.inspected_this_frame  # the pass is still running
    second = frame(model, session, 1, "inspect").inspection  # a frame during the pass: not queued
    assert second.passes == 0
    gate.set()
    live = component_inspection._sessions["s1"][1]
    for _ in range(100):
        if live.latest.passes == 1:
            break
        threading.Event().wait(0.05)
    assert live.latest.passes == 1


def test_a_new_run_starts_over(model):
    session = FakeSession()
    frame(model, session, 0, "inspect", run="a")
    assert frame(model, session, 1, "inspect", run="a").inspection.passes == 2
    assert frame(model, session, 2, "inspect", run="b").inspection.passes == 1


# ── registry endpoints ──────────────────────────────────────────────────


@pytest.fixture
def client(model):
    db = SimpleNamespace(get=lambda cls, key: model if key == MODEL_ID else None)
    app.dependency_overrides[get_db] = lambda: db
    yield TestClient(app)
    app.dependency_overrides.pop(get_db, None)


def test_registry_lists_components_and_nodes(client):
    body = client.get(f"/api/models3d/{MODEL_ID}/components").json()
    assert body["source"] == "auto" and len(body["components"]) == 8
    screw = body["components"][0]
    assert screw["id"] == "SCREW_001" and screw["node_name"] == "Screw_01" and screw["reference_samples"] == 0
    assert len(screw["node_matrix"]) == 4 and screw["center_m"] is not None
    assert {n["name"] for n in body["nodes"]} == {"Base_Plate", *(f"Screw_{i:02d}" for i in range(1, 9))}


def test_mapping_can_be_saved_and_removed(client, tmp_path):
    mapping = {"objectId": "PLATE_7", "components": [{"id": "TOP_LEFT", "meshName": "Screw_01", "type": "screw"}]}
    body = client.put(f"/api/models3d/{MODEL_ID}/components", json=mapping).json()
    assert body["source"] == "mapping" and body["object_id"] == "PLATE_7" and [c["id"] for c in body["components"]] == ["TOP_LEFT"]
    assert json.loads((tmp_path / "registry" / f"{MODEL_ID}.json").read_text()) == mapping
    body = client.delete(f"/api/models3d/{MODEL_ID}/components").json()
    assert body["source"] == "auto"


def test_an_invalid_mapping_is_rejected_and_the_old_one_kept(client, tmp_path):
    good = {"components": [{"id": "A", "meshName": "Screw_01"}]}
    client.put(f"/api/models3d/{MODEL_ID}/components", json=good)
    response = client.put(f"/api/models3d/{MODEL_ID}/components", json={"components": [{"id": "B"}]})
    assert response.status_code == 400 and "nodeIndex or meshName" in response.json()["detail"]
    assert json.loads((tmp_path / "registry" / f"{MODEL_ID}.json").read_text()) == good


def test_unknown_model_is_404(client):
    assert client.get(f"/api/models3d/{uuid.uuid4()}/components").status_code == 404


def test_frame_jpeg_helper_roundtrip():
    """Sanity: the frames these tests send decode like camera JPEGs do."""
    ok, png = cv2.imencode(".png", np.zeros((4, 4, 3), np.uint8))
    assert vision.decode_frame(base64.b64encode(png.tobytes()).decode()).shape == (4, 4, 3)
