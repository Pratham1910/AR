"""
Component inspection end to end (session.py) on rendered frames: a plate
with eight screws seen by a camera at known poses, screws removed and put
back, tracking lost, the image blurred, the plate seen from underneath.
Plus the individual verifiers (verifiers.py).
"""

import json

import cv2
import numpy as np
import pytest

from app.services.component_inspection.decision import ComponentStatus, DecisionConfig
from app.services.component_inspection.projection import project_component, to_camera, to_pixels
from app.services.component_inspection.registry import registry_from_glb
from app.services.component_inspection.session import InspectionConfig, InspectionSession
from app.services.component_inspection.verifiers import (
    CalibratedPresenceVerifier,
    ComponentDetection,
    DetectorVerifier,
    Observation,
    ReferenceAppearanceVerifier,
    ReferenceSet,
)
from app.services.vision.part_presence import PartRegion, PresenceCalibration

from glb_builder import plate_with_screws
from test_component_projection import K, SHAPE, look_at

S = ComponentStatus


@pytest.fixture(scope="module")
def registry():
    return registry_from_glb("m1", plate_with_screws(), scale=1.0)


def render(registry, t_co, removed=(), seed=0, blur=0):
    """A camera image of the plate at `t_co`: textured background and plate,
    screw heads bright with a dark slot; a removed screw leaves a dark hole."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(50, 90, (*SHAPE, 3), dtype=np.uint8)
    scene = registry.scene
    cam = to_camera(scene.vertices, t_co)
    px = np.round(to_pixels(cam, K)).astype(np.int32)
    plate = np.zeros(SHAPE, np.uint8)
    for f in np.flatnonzero(scene.labels == -1):
        cv2.fillConvexPoly(plate, px[scene.faces[f]], 1)
    texture = rng.integers(115, 150, (*SHAPE, 3), dtype=np.uint8)
    frame[plate.astype(bool)] = texture[plate.astype(bool)]
    for c in registry.components:
        p = project_component(c.geometry, t_co, K, SHAPE)
        cx, cy = (int(round(v)) for v in p.center_px)
        if c.id in removed:
            cv2.circle(frame, (cx, cy), max(3, (p.roi[2] - p.roi[0]) // 5), (20, 20, 20), -1)  # the empty threaded hole
            continue
        x1, y1, x2, y2 = p.roi
        region = frame[y1:y2, x1:x2]
        metal = rng.integers(195, 225, region.shape, dtype=np.uint8)
        region[p.mask] = metal[p.mask]
        half = (x2 - x1) // 3
        cv2.line(frame, (cx - half, cy), (cx + half, cy), (40, 40, 40), 2)  # slot
        cv2.line(frame, (cx, cy - half), (cx, cy + half), (40, 40, 40), 2)
    if blur:
        frame = cv2.GaussianBlur(frame, (0, 0), blur)
    return frame


def jittered(k):
    """Viewpoints a few degrees apart, as a hand-held camera wanders."""
    angle = np.radians(4 * np.sin(k))
    eye = (0.3 * np.sin(angle), 0.3 * np.cos(angle), 0.02 * np.cos(k))
    return look_at(eye)


def make_session(registry, tmp_path, references=None, **decision):
    config = InspectionConfig(interval_s=0.0, decision=DecisionConfig(**decision))
    verifier = ReferenceAppearanceVerifier(references)
    return InspectionSession("m1", registry, [verifier], config,
                             reference_path=tmp_path / "reference.json", log_path=tmp_path / "log.jsonl")


def learn(session, registry, frames=8):
    session.set_mode("learn")
    for k in range(frames):
        t_co = jittered(k)
        session.process(render(registry, t_co, seed=k), t_co, K, True, None, now=k * 0.3)
    return session.set_mode("inspect")


def inspect(session, registry, frames, start=0, removed=(), tracking_ok=True, blur=0, pose=None, t0=100.0):
    for k in range(start, start + frames):
        t_co = pose if pose is not None else jittered(k)
        session.process(render(registry, t_co, removed, seed=1000 + k, blur=blur), t_co, K, tracking_ok,
                        None if tracking_ok else "tracking lost", now=t0 + k * 0.3)
    return {cid: d.state.status for cid, d in session.decisions.items()}


# ── end to end ──────────────────────────────────────────────────────────


def test_learning_the_good_state_saves_a_reference(registry, tmp_path):
    session = make_session(registry, tmp_path)
    message = learn(session, registry)
    assert "Learned the good state of 8/8 components" in message
    saved = json.loads((tmp_path / "reference.json").read_text())
    assert saved["model_id"] == "m1" and set(saved["components"]) == {c.id for c in registry.components}
    # Learning never decides anything.
    assert all(d.state.status == S.NOT_INSPECTED for d in session.decisions.values())


def test_all_present_passes(registry, tmp_path):
    session = make_session(registry, tmp_path)
    learn(session, registry)
    statuses = inspect(session, registry, 6)
    assert set(statuses.values()) == {S.PRESENT}
    assert session.summary().verdict == "PASS"


def test_one_missing_screw_is_confirmed_and_named(registry, tmp_path):
    session = make_session(registry, tmp_path, confirm_missing_after=6)
    learn(session, registry)
    statuses = inspect(session, registry, 5, removed={"SCREW_004"})
    assert statuses["SCREW_004"] == S.MISSING_CANDIDATE  # not yet: 5 of 6
    statuses = inspect(session, registry, 3, start=5, removed={"SCREW_004"})
    assert statuses["SCREW_004"] == S.MISSING_CONFIRMED
    assert all(s == S.PRESENT for cid, s in statuses.items() if cid != "SCREW_004")  # one missing fails only itself
    summary = session.summary()
    assert (summary.verdict, summary.missing, summary.present) == ("FAIL", 1, 7)
    assert "consecutive valid observations" in session.decisions["SCREW_004"].state.reason
    assert "differs from the learned good state" in session.decisions["SCREW_004"].state.reason


def test_several_missing_screws(registry, tmp_path):
    session = make_session(registry, tmp_path, confirm_missing_after=4)
    learn(session, registry)
    removed = {"SCREW_001", "SCREW_006", "SCREW_008"}
    statuses = inspect(session, registry, 6, removed=removed)
    assert {cid for cid, s in statuses.items() if s == S.MISSING_CONFIRMED} == removed
    assert session.summary().missing == 3


def test_putting_the_screw_back_clears_it(registry, tmp_path):
    session = make_session(registry, tmp_path, confirm_missing_after=4, clear_missing_after=4)
    learn(session, registry)
    inspect(session, registry, 5, removed={"SCREW_002"})
    assert session.decisions["SCREW_002"].state.status == S.MISSING_CONFIRMED
    inspect(session, registry, 3, start=5)
    assert session.decisions["SCREW_002"].state.status == S.MISSING_CONFIRMED  # 3 of 4 needed
    inspect(session, registry, 1, start=8)
    assert session.decisions["SCREW_002"].state.status == S.PRESENT
    assert session.summary().verdict == "PASS"


def test_tracking_loss_does_not_produce_missing(registry, tmp_path):
    session = make_session(registry, tmp_path)
    learn(session, registry)
    # The pose is wrong (object moved, tracker lost it): the expected pixels show whatever is there now.
    wrong = look_at((0.05, 0.3, 0.0))
    statuses = inspect(session, registry, 12, removed={"SCREW_003"}, tracking_ok=False, pose=wrong)
    assert S.MISSING_CANDIDATE not in statuses.values() and S.MISSING_CONFIRMED not in statuses.values()
    assert session.decisions["SCREW_003"].state.status == S.UNCERTAIN
    assert "tracking lost" in session.decisions["SCREW_003"].state.reason


def test_blurred_frames_are_uncertain_not_missing(registry, tmp_path):
    session = make_session(registry, tmp_path)
    learn(session, registry)
    statuses = inspect(session, registry, 12, removed={"SCREW_005"}, blur=6)
    assert statuses["SCREW_005"] in (S.UNCERTAIN, S.NOT_INSPECTED)
    assert "blurred" in session.decisions["SCREW_005"].state.reason


def test_screws_seen_from_under_the_plate_are_occluded(registry, tmp_path):
    session = make_session(registry, tmp_path)
    learn(session, registry)
    below = look_at((0, -0.3, 0), up=(0, 0, 1))
    statuses = inspect(session, registry, 8, pose=below)
    assert set(statuses.values()) == {S.OCCLUDED}
    assert session.summary().verdict == "INCOMPLETE"


def test_new_viewpoint_outside_the_reference_is_uncertain(registry, tmp_path):
    session = make_session(registry, tmp_path)
    learn(session, registry)
    oblique = look_at((0.25, 0.17, 0.0))  # ~55 deg from the learned top-down views
    statuses = inspect(session, registry, 8, removed={"SCREW_001"}, pose=oblique)
    assert S.MISSING_CONFIRMED not in statuses.values()
    assert "learned view" in session.decisions["SCREW_001"].state.reason


def test_without_a_reference_nothing_is_called_missing(registry, tmp_path):
    session = make_session(registry, tmp_path)
    statuses = inspect(session, registry, 8, removed={"SCREW_001"})
    assert set(statuses.values()) == {S.UNCERTAIN}
    assert "learn the good state" in session.decisions["SCREW_001"].state.reason


def test_pause_keeps_results_and_stops_observing(registry, tmp_path):
    session = make_session(registry, tmp_path, confirm_missing_after=3)
    learn(session, registry)
    inspect(session, registry, 4, removed={"SCREW_007"})
    session.set_mode("paused")
    before = session.passes
    inspect(session, registry, 4)
    assert session.passes == before and session.decisions["SCREW_007"].state.status == S.MISSING_CONFIRMED


def test_every_pass_is_logged_with_its_evidence(registry, tmp_path):
    session = make_session(registry, tmp_path)
    learn(session, registry)
    inspect(session, registry, 2, removed={"SCREW_004"})
    lines = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    last = lines[-1]
    assert last["mode"] == "inspect" and len(last["components"]) == 8
    screw = next(c for c in last["components"] if c["id"] == "SCREW_004")
    assert screw["decision"] == "absent" and "differs" in screw["evidence"] and screw["metrics"]["area_px"] > 0


def test_inspection_rate_is_limited(registry, tmp_path):
    session = InspectionSession("m1", registry, [], InspectionConfig(interval_s=0.5))
    t_co = jittered(0)
    frame = render(registry, t_co)
    ran = [session.process(frame, t_co, K, True, None, now=t).ran for t in (0.0, 0.1, 0.4, 0.55, 0.9, 1.1)]
    assert ran == [True, False, False, True, False, True]


# ── verifiers on their own ──────────────────────────────────────────────


def _observation(registry, removed=(), i=0):
    t_co = jittered(0)
    frame = render(registry, t_co, removed)
    c = registry.components[i]
    return Observation(c, frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), project_component(c.geometry, t_co, K, SHAPE))


def test_calibrated_verifier_uses_the_saved_part_calibration(registry):
    node = registry.components[0].node_index
    cal = PresenceCalibration(region=PartRegion(0, 1, 0, 1), present_mean=180, absent_mean=110, present_samples=5,
                              absent_samples=5, present_std=8, absent_std=8, node_index=node, method="pose", feature="mean")
    verifier = CalibratedPresenceVerifier([cal])
    assert verifier.applies_to(registry.components[0]) and not verifier.applies_to(registry.components[1])
    assert verifier.verify(_observation(registry)).decision == "present"
    assert verifier.verify(_observation(registry, removed={"SCREW_001"})).decision == "absent"


def test_calibrated_verifier_doubts_what_looks_like_neither_state(registry):
    """E.g. a hand over the part: far darker than either calibrated state."""
    node = registry.components[0].node_index
    cal = PresenceCalibration(region=PartRegion(0, 1, 0, 1), present_mean=240, absent_mean=225, present_samples=5,
                              absent_samples=5, present_std=2, absent_std=2, node_index=node, method="pose", feature="mean")
    verdict = CalibratedPresenceVerifier([cal]).verify(_observation(registry, removed={"SCREW_001"}))
    assert verdict.decision == "uncertain" and "like neither" in verdict.reason


def test_box_method_calibrations_are_not_used_for_components(registry):
    cal = PresenceCalibration(region=PartRegion(0, 1, 0, 1), present_mean=1, absent_mean=2, present_samples=1,
                              absent_samples=1, node_index=registry.components[0].node_index, method="box")
    assert not CalibratedPresenceVerifier([cal]).applies_to(registry.components[0])


class FakeDetector:
    types = {"screw"}

    def __init__(self, hits):
        self.hits = hits
        self.crops = []

    def detect(self, crop):
        self.crops.append(crop.shape)
        return self.hits(crop)


def test_detector_hit_at_the_expected_place_is_present(registry):
    obs = _observation(registry)
    centred = lambda crop: [ComponentDetection("screw", 0.9, (crop.shape[1] / 2 - 5, crop.shape[0] / 2 - 5, crop.shape[1] / 2 + 5, crop.shape[0] / 2 + 5))]  # noqa: E731
    detector = FakeDetector(centred)
    verdict = DetectorVerifier(detector).verify(obs)
    assert verdict.decision == "present" and verdict.confidence == 0.9
    assert detector.crops[0][0] >= 64  # the crop has context around a small part


def test_detector_hit_elsewhere_or_nothing_is_only_weak_evidence(registry):
    obs = _observation(registry)
    corner = lambda crop: [ComponentDetection("screw", 0.9, (0, 0, 4, 4))]  # noqa: E731
    verdict = DetectorVerifier(FakeDetector(corner)).verify(obs)
    assert verdict.decision == "absent" and verdict.negative_only
    assert DetectorVerifier(FakeDetector(lambda crop: [])).verify(obs).negative_only


def test_reference_set_round_trips(tmp_path, registry):
    session = make_session(registry, tmp_path)
    learn(session, registry)
    loaded = ReferenceSet.load(tmp_path / "reference.json")
    assert loaded is not None and loaded.components["SCREW_001"].samples == 8
    assert ReferenceAppearanceVerifier(loaded).applies_to(registry.components[0])
