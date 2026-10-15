"""Orientation: symmetric twins (pose/symmetry.py) told apart by pose features (pose/pose_features.py,
pose/disambiguation.py) — the "circle at top-right shows at top-left" flip."""

import json

import numpy as np
import pytest

from app.eval.synthetic_scene import K, Mark, box_mesh, corner_circle, look_at, project, render, turned
from app.services.pose.disambiguation import choose_orientation, is_flip
from app.services.pose.edge_align import EdgeModel, _fit_score, align, edge_image
from app.services.pose.pose_features import PoseFeature, load_pose_features, observe_features, with_feature_edges
from app.services.pose.symmetry import find_symmetries, rotation_angle_deg, twins
from app.services.tracking.trackers import resting_side_ok

from glb_builder import Node, box, build_glb, merge

V, F = box_mesh()
MARK = corner_circle()
FEATURE = PoseFeature("UR_CIRCLE", "circle", MARK.center, MARK.normal, MARK.radius)
TRUTH = look_at((0.03, 0.30, 0.20))  # above and in front, looking down ~55 deg
FLIP = np.diag([-1.0, 1.0, -1.0])  # 180 deg about the box's up axis: same outline, circle at the far corner


def _twin(t):
    out = np.array(t, float)
    out[:3, :3] = out[:3, :3] @ FLIP
    return out


def _scene(t=TRUTH, marks=(MARK,), **kw):
    gray, hand = render(V, F, t, list(marks), **kw)
    return gray, edge_image(gray, exclude=hand)


# ── symmetry ─────────────────────────────────────────────────────────────


def test_a_plain_box_has_three_half_turn_twins():
    sym = find_symmetries(V, F)
    assert len(sym) == 4  # identity + 180 deg about x, y, z
    assert any(np.allclose(s, FLIP) for s in sym)


def test_a_cube_has_all_24_rotations():
    v, f = box((0, 0, 0), (0.1, 0.1, 0.1))
    assert len(find_symmetries(v, f)) == 24


def _stepped():
    v, f = merge(box((-0.02, 0.0, 0.0), (0.15, 0.035, 0.114)), box((0.075, 0.003, 0.0), (0.04, 0.041, 0.114)))
    return v - (v.min(axis=0) + v.max(axis=0)) / 2, f


def test_a_small_step_at_one_end_is_near_symmetric_but_not_exactly():
    """6 mm step on a 22 cm part: the outline barely changes turned round (so it counts), but strictly it isn't."""
    v, f = _stepped()
    assert len(find_symmetries(v, f)) > 1
    assert len(find_symmetries(v, f, tolerance=0.0)) == 1


def test_a_near_symmetric_part_is_turned_back_by_its_3d_edges():
    """No features configured: the step's edges decide."""
    v, f = _stepped()
    sym = find_symmetries(v, f)
    gray, _ = render(v, f, TRUTH, [])
    edges = edge_image(gray)
    model = EdgeModel.from_mesh(v, f)
    wrong = twins(TRUTH, sym)
    flipped = next(t for t, s in zip(wrong, sym) if not np.allclose(s, np.eye(3)) and resting_side_ok(t, "camera"))
    decision = choose_orientation(flipped, sym, [], K, gray, edges, model=model, allowed=lambda t: resting_side_ok(t, "camera"))
    assert decision.status == "confirmed" and rotation_angle_deg(decision.t_co, TRUTH) < 1.0


# ── the flip, reproduced ─────────────────────────────────────────────────


def test_the_flip_geometry_alone_cannot_see():
    """Baseline: outline and 3D edges fit the twin exactly as well as the truth."""
    gray, edges = _scene()
    model = EdgeModel.from_mesh(V, F)
    truth_score = _fit_score(model, TRUTH, K, edges, 3.0)[0]
    twin_score = _fit_score(model, _twin(TRUTH), K, edges, 3.0)[0]
    assert abs(truth_score - twin_score) < 0.02
    # The circle really is on the other side under the twin.
    assert np.linalg.norm(project(MARK.center[None], TRUTH) - project(MARK.center[None], _twin(TRUTH))) > 100


@pytest.mark.parametrize("turn_deg", [0, 90, 180, 270])
def test_the_circle_puts_the_box_the_right_way_round(turn_deg):
    """Box turned on the desk by 0/90/180/270 deg; the pose handed in is its 180-deg twin."""
    truth = turned(TRUTH, (0, 1, 0), turn_deg)
    gray, edges = _scene(truth)
    decision = choose_orientation(_twin(truth), find_symmetries(V, F), [FEATURE], K, gray, edges,
                                  allowed=lambda t: resting_side_ok(t, "camera"))
    assert decision.status == "confirmed" and decision.flipped
    assert rotation_angle_deg(decision.t_co, truth) < 1.0
    seen = decision.observations[0]
    assert seen.status == "detected" and seen.residual_px is not None and seen.residual_px < 2.0


def test_the_right_pose_is_kept():
    gray, edges = _scene()
    decision = choose_orientation(TRUTH, find_symmetries(V, F), [FEATURE], K, gray, edges)
    assert decision.status == "confirmed" and not decision.flipped


# ── never invent certainty ───────────────────────────────────────────────


def test_a_hand_over_the_circle_leaves_it_ambiguous_and_keeps_the_tracked_orientation():
    c = project(MARK.center[None], TRUTH)[0].astype(int)
    gray, edges = _scene(hand=(c[0] - 45, c[1] - 45, c[0] + 45, c[1] + 45))
    sym = find_symmetries(V, F)
    held = choose_orientation(_twin(TRUTH), sym, [FEATURE], K, gray, edges, reference=TRUTH)
    assert held.status == "ambiguous" and "not seen" in held.reason
    assert rotation_angle_deg(held.t_co, TRUTH) < 1.0  # the orientation it was tracking, not the twin
    assert held.observations[0].status == "occluded"  # where the held pose puts the circle: under the hand
    blind = choose_orientation(_twin(TRUTH), sym, [FEATURE], K, gray, edges)
    assert blind.status == "ambiguous" and not blind.flipped  # given pose kept, flagged, not "fixed" by guessing


def test_a_plain_box_stays_ambiguous():
    gray, edges = _scene(marks=())
    decision = choose_orientation(_twin(TRUTH), find_symmetries(V, F), [FEATURE], K, gray, edges)
    assert decision.status == "ambiguous"


def test_no_features_configured_is_reported_as_such():
    gray, edges = _scene()
    decision = choose_orientation(TRUTH, find_symmetries(V, F), [], K, gray, edges)
    assert decision.status == "ambiguous" and "no pose features" in decision.reason


def test_a_look_alike_circle_in_the_twin_position_is_not_proof():
    """Circles at both opposite corners: the box looks the same either way round."""
    other = Mark(-MARK.center * np.array([1, -1, 1]), MARK.normal, MARK.radius)
    other.center = np.array([-MARK.center[0], MARK.center[1], -MARK.center[2]])
    gray, edges = _scene(marks=(MARK, other))
    decision = choose_orientation(_twin(TRUTH), find_symmetries(V, F), [FEATURE], K, gray, edges)
    assert decision.status == "ambiguous" and "more than one" in decision.reason


def test_a_required_feature_not_seen_is_listed():
    c = project(MARK.center[None], TRUTH)[0].astype(int)
    gray, edges = _scene(hand=(c[0] - 45, c[1] - 45, c[0] + 45, c[1] + 45))
    required = PoseFeature("UR_CIRCLE", "circle", MARK.center, MARK.normal, MARK.radius, required_for_initialization=True)
    decision = choose_orientation(TRUTH, find_symmetries(V, F), [required], K, gray, edges)
    assert decision.required_unconfirmed == ["UR_CIRCLE"]


def test_low_contrast_still_decides_or_says_ambiguous_never_wrong():
    gray, edges = _scene(contrast=0.35, noise=6)
    decision = choose_orientation(_twin(TRUTH), find_symmetries(V, F), [FEATURE], K, gray, edges)
    if decision.status == "confirmed":
        assert rotation_angle_deg(decision.t_co, TRUTH) < 1.0
    else:
        assert decision.status == "ambiguous"


def test_is_flip_tells_a_twin_from_motion():
    sym = find_symmetries(V, F)
    assert is_flip(TRUTH, _twin(TRUTH), sym)
    assert not is_flip(TRUTH, turned(TRUTH, (0, 1, 0), 8), sym)  # a small real turn
    assert not is_flip(TRUTH, _twin(TRUTH), [np.eye(3)])  # no symmetry: no twin


# ── features from the mapping file; rims in the edge fit ─────────────────


def test_features_load_from_the_mapping_file(tmp_path):
    disc_v, disc_f = box((0.06, 0.0185, -0.029), (0.024, 0.002, 0.024))  # a thin disc-like node on the top face
    glb = build_glb([Node("Body", mesh=box((0, 0, 0), (0.19, 0.035, 0.114))), Node("Circle", mesh=(disc_v, disc_f))])
    mapping = tmp_path / "m.json"
    mapping.write_text(json.dumps({
        "components": [{"id": "MARK_1", "meshName": "Circle"}],
        "poseFeatures": [
            {"featureId": "BY_NODE", "type": "circle", "meshName": "Circle"},
            {"featureId": "BY_COMPONENT", "type": "node", "componentId": "MARK_1", "requiredForInitialization": True},
            {"featureId": "BY_POSITION", "type": "circle", "position": [0.06, 0.0175, -0.029], "normal": [0, 1, 0], "radius": 0.012},
            {"featureId": "BROKEN", "type": "circle", "meshName": "Nope"},
        ],
    }))
    features, warnings = load_pose_features(mapping, glb, scale=1.0)
    by_id = {f.feature_id: f for f in features}
    assert set(by_id) == {"BY_NODE", "BY_COMPONENT", "BY_POSITION"} and "BROKEN" in warnings[0]
    node = by_id["BY_NODE"]
    assert node.normal == pytest.approx([0, 1, 0]) and node.radius == pytest.approx(0.012)
    # Recentred like the tracked frame: the assembly spans y -0.0175..0.0195 -> centre 0.001.
    assert node.center == pytest.approx([0.06, 0.0175, -0.029], abs=1e-6)
    assert by_id["BY_COMPONENT"].required_for_initialization and by_id["BY_COMPONENT"].vertices is not None
    assert by_id["BY_POSITION"].center == pytest.approx([0.06, 0.0165, -0.029], abs=1e-6)


def test_no_pose_features_section_is_fine(tmp_path):
    mapping = tmp_path / "m.json"
    mapping.write_text(json.dumps({"components": []}))
    assert load_pose_features(mapping, None, 1.0) == ([], [])
    assert load_pose_features(None, None, 1.0) == ([], [])


def test_the_circle_rim_joins_the_edge_fit():
    model = with_feature_edges(EdgeModel.from_mesh(V, F), [FEATURE])
    assert len(model.points) == len(EdgeModel.from_mesh(V, F).points) + 64
    gray, edges = _scene()
    start = turned(TRUTH, (0, 1, 0), 3)
    start[:3, 3] += [0.004, 0.0, 0.003]
    fit = align(model, start, K, edges)
    assert fit is not None and rotation_angle_deg(fit.t_co, TRUTH) < 1.5


def test_observation_statuses():
    gray, edges = _scene()
    seen = observe_features([FEATURE], TRUTH, K, gray, edges)[0]
    assert seen.status == "detected" and seen.evidence > 0.6
    under = turned(TRUTH, (1, 0, 0), 180)  # box upside down: the circle faces the desk
    assert observe_features([FEATURE], under, K, gray, edges)[0].status == "facing_away"
    away = np.array(TRUTH)
    away[:3, 3] += [1.0, 0, 0]
    assert observe_features([FEATURE], away, K, gray, edges)[0].status == "outside"
    assert twins(TRUTH, [np.eye(3), FLIP])[1][:3, :3] == pytest.approx(TRUTH[:3, :3] @ FLIP)
