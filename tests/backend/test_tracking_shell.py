"""Optimize for tracking: the visible shell of a CAD model (app/services/model3d/tracking_shell.py, tracking_model.py)."""

import numpy as np
import pytest

from app.services.model3d import tracking_model
from app.services.model3d.glb_inspect import glb_part_meshes
from app.services.model3d.tracking_shell import build_tracking_glb

from glb_builder import Node, box, build_glb


def _enclosure():
    """A closed case with a drive inside it, a small screw head on the lid, and a part sticking out below."""
    return build_glb([
        Node("Case", mesh=box((0, 0, 0), (0.19, 0.035, 0.114))),
        Node("Drive", mesh=box((0, 0, 0), (0.14, 0.02, 0.10))),  # fully inside: never visible
        Node("Screw", mesh=box((0.08, 0.0185, 0.05), (0.004, 0.002, 0.004))),  # 4 mm head on the lid
        Node("Foot", mesh=box((0.0, -0.02, 0.0), (0.03, 0.005, 0.03))),  # below, outside the case
    ])


def test_the_hidden_drive_is_dropped_and_everything_visible_kept():
    shell, report = build_tracking_glb(_enclosure())
    by_name = {n["name"]: n for n in report.nodes}
    assert by_name["Drive"]["faces_after"] == 0
    assert by_name["Case"]["faces_after"] >= 10  # its outer faces (the bottom is partly behind the foot)
    assert by_name["Screw"]["faces_after"] >= 5  # tiny, but on the outside
    assert by_name["Foot"]["faces_after"] >= 5
    assert report.faces_after < report.faces_before
    names = {len(m.faces) for m in glb_part_meshes(shell).values()}
    assert len(glb_part_meshes(shell)) == 3 and 0 not in names  # a node with nothing visible isn't written


def test_the_frame_stays_the_originals():
    original = _enclosure()
    _, report = build_tracking_glb(original)
    every = np.concatenate([m.vertices for m in glb_part_meshes(original).values()])
    assert report.frame_center == pytest.approx(((every.min(0) + every.max(0)) / 2).tolist())


def test_a_shell_is_used_only_for_the_glb_it_was_built_from(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking_model.get_settings(), "models_3d_dir", str(tmp_path))
    glb = _enclosure()
    assert tracking_model.load("m1", glb) is None
    info = tracking_model.build("m1", glb)
    mesh, center, sha = tracking_model.load("m1", glb)
    assert sha == info["shell_sha256"] and center == info["frame_center"] and len(mesh) < len(glb)
    changed = build_glb([Node("Case", mesh=box((0, 0, 0), (0.2, 0.04, 0.12)))])
    assert tracking_model.load("m1", changed) is None and tracking_model.status("m1", changed)["stale"]
    tracking_model.remove("m1")
    assert tracking_model.status("m1", glb) is None


def test_registration_sends_the_shell_in_the_originals_frame_and_switches_cleanly(tmp_path, monkeypatch):
    import app.api.vision as vision
    from app.services.pose.model_pose_client import ModelPoseClient

    monkeypatch.setattr(tracking_model.get_settings(), "models_3d_dir", str(tmp_path))
    client = ModelPoseClient("http://pose.invalid", 5.0)
    sent = []
    monkeypatch.setattr(client, "_post", lambda path, payload: sent.append(payload) or {"offset_m": [0, 0, 0]})
    monkeypatch.setattr(vision, "_model_pose_client", client)
    glb = _enclosure()

    vision.register_model_mesh("m1", lambda: glb, 0.9739)
    vision.register_model_mesh("m1", lambda: glb, 0.9739)  # cached: not sent again
    assert len(sent) == 1 and "frame_center" not in sent[0]
    info = tracking_model.build("m1", glb)
    vision.register_model_mesh("m1", lambda: glb, 0.9739)  # the shell now: sent once more, with the frame
    assert len(sent) == 2 and sent[1]["frame_center"] == pytest.approx(info["frame_center"])
    assert len(sent[1]["glb_base64"]) < len(sent[0]["glb_base64"])
