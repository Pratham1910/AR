"""How a live session finds its object: by class, by 3D model, or from a box the user drew."""

import types

import numpy as np
import pytest
from fastapi import HTTPException

from app.api import vision
from app.schemas.pose import ARFrameRequest
from app.services.pose.calibration import CameraCalibration
from app.services.tracking.trackers import Frame

MODEL = types.SimpleNamespace(id="m1", scale=0.01, storage_key="missing.glb", name="Joystick")
FRAME = Frame(np.zeros((480, 640, 3), np.uint8), CameraCalibration.approximate(640, 480))


def _request(**kw) -> ARFrameRequest:
    base = {"session_id": f"s-{id(kw)}", "mode": "markerless", "image_base64": "", "real_world_height_m": 0.12}
    return ARFrameRequest(**{**base, **kw})


def test_a_drawn_box_is_the_first_lock_then_the_3d_model_refinds_the_object():
    refound = []
    detect = vision._box_detector((10.0, 20.0, 110.0, 220.0), "Joystick", lambda f: refound.append(f) or "by model")
    first = detect(FRAME)
    assert (first.bbox.x1, first.bbox.y2, first.confidence) == (10.0, 220.0, 1.0)
    assert [(p.x, p.y) for p in first.polygon] == [(10, 20), (110, 20), (110, 220), (10, 220)]
    assert detect(FRAME) == "by model" and len(refound) == 1


def test_without_a_model_a_lost_object_needs_a_new_box():
    detect = vision._box_detector((0, 0, 5, 5), "object", None)
    assert detect(FRAME) is not None
    assert detect(FRAME) is None


def test_box_finder_needs_the_box(monkeypatch):
    monkeypatch.setattr(vision, "_ar_model", lambda request, db: MODEL)
    with pytest.raises(HTTPException, match="Draw a box"):
        vision._ar_session_for(_request(detect_by="box"), db=None)


def test_markerless_by_3d_model_needs_a_model():
    with pytest.raises(HTTPException, match="needs a model"):
        vision._ar_session_for(_request(detect_by="model"), db=None)


def test_markerless_without_class_or_model_works_from_a_drawn_box():
    session, model = vision._ar_session_for(_request(detect_by="box", init_box=[1, 2, 30, 40]), db=None)
    assert model is None and session.class_label == "object"


def test_class_is_only_required_when_finding_by_class():
    with pytest.raises(HTTPException, match="Pick an object class"):
        vision._ar_session_for(_request(detect_by="class"), db=None)
