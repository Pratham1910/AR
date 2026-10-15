"""Projecting components at the tracked pose, and deciding whether they can be judged (projection.py, visibility.py)."""

import cv2
import numpy as np
import pytest

from app.services.component_inspection.projection import cad_visibility, project_component
from app.services.component_inspection.registry import registry_from_glb
from app.services.component_inspection.visibility import VisibilityConfig, assess

from glb_builder import SCREW_POSITIONS, plate_with_screws

K = np.array([[800.0, 0, 640], [0, 800.0, 360], [0, 0, 1]])
SHAPE = (720, 1280)
# The registry recentres on the assembly's box: plate -0.01..0, screws to +0.004 -> centre y -0.003.
CENTER_Y = -0.003


def look_at(eye, target=(0, 0, 0), up=(0, 0, -1)) -> np.ndarray:
    """T_camera_object for an OpenCV camera at `eye` (object frame) looking at `target`."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    r = np.stack([x, y, z])  # rows: camera axes in object coordinates
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = -r @ eye
    return t


def expected_pixel(t_co, point):
    cam = t_co[:3, :3] @ np.asarray(point, float) + t_co[:3, 3]
    return K[0, 0] * cam[0] / cam[2] + K[0, 2], K[1, 1] * cam[1] / cam[2] + K[1, 2]


@pytest.fixture(scope="module")
def registry():
    return registry_from_glb("m1", plate_with_screws(), scale=1.0)


def screw_center(i):
    x, z = SCREW_POSITIONS[i]
    return (x, 0.002 - CENTER_Y, z)


# ── projection ──────────────────────────────────────────────────────────


def test_component_centre_projects_where_the_pinhole_model_puts_it(registry):
    t_co = look_at((0, 0.3, 0))
    for i, component in enumerate(registry.components):
        p = project_component(component.geometry, t_co, K, SHAPE)
        assert p.center_px == pytest.approx(expected_pixel(t_co, screw_center(i)), abs=1e-3)  # GLB vertices are float32
        assert p.in_frame == 1.0 and not p.behind_camera
        # A 12 mm screw head ~0.3 m away covers ~32 px: its mask is that big.
        x1, y1, x2, y2 = p.roi
        assert 28 <= x2 - x1 <= 36 and 28 <= y2 - y1 <= 36
        assert p.facing == pytest.approx(0.5, abs=0.2)


def test_markers_follow_the_object_when_it_moves_and_turns(registry):
    """The same component, the object turned 90 degrees and shifted: its pixel follows the pose."""
    base = look_at((0, 0.3, 0))
    turn = np.eye(4)
    turn[:3, :3] = cv2.Rodrigues(np.array([0.0, np.pi / 2, 0.0]))[0]  # object turned about its up axis
    turn[:3, 3] = [0.02, 0.0, 0.01]
    t_co = base @ turn
    screw = registry.components[0]
    p = project_component(screw.geometry, t_co, K, SHAPE)
    assert p.center_px == pytest.approx(expected_pixel(t_co, screw_center(0)), abs=1e-3)
    assert p.center_px != pytest.approx(project_component(screw.geometry, base, K, SHAPE).center_px, abs=5)


def test_view_direction_is_from_the_component_to_the_camera(registry):
    p = project_component(registry.components[0].geometry, look_at((0, 0.3, 0)), K, SHAPE)
    assert p.view_dir[1] > 0.95  # camera straight above


def test_component_behind_the_camera_is_flagged(registry):
    t_co = look_at((0, 0.3, 0))
    t_co[2, 3] -= 1.0  # push the object behind the camera
    assert project_component(registry.components[0].geometry, t_co, K, SHAPE).behind_camera


# ── CAD occlusion ───────────────────────────────────────────────────────


def _rois(registry, t_co):
    out = {}
    for i, c in enumerate(registry.components):
        p = project_component(c.geometry, t_co, K, SHAPE)
        if p.roi:
            out[i] = p.roi
    return out


def test_screws_seen_from_above_are_in_plain_view(registry):
    t_co = look_at((0, 0.3, 0))
    fractions = cad_visibility(registry.scene, _rois(registry, t_co), t_co, K, SHAPE).fractions
    assert len(fractions) == 8 and min(fractions.values()) > 0.95


def test_screws_seen_from_under_the_plate_are_occluded(registry):
    t_co = look_at((0, -0.3, 0), up=(0, 0, 1))
    fractions = cad_visibility(registry.scene, _rois(registry, t_co), t_co, K, SHAPE).fractions
    assert len(fractions) == 8 and max(fractions.values()) < 0.05


def test_a_flush_component_is_not_hidden_by_the_surface_it_sits_in():
    """Countersunk screw head in the plate's top face: depth ties go to the component."""
    from glb_builder import Node, box, build_glb

    glb = build_glb([Node("Plate", mesh=box((0, -0.005, 0), (0.1, 0.01, 0.1))),
                     Node("Screw_Flush", mesh=box((0, -0.002, 0), (0.012, 0.004, 0.012)))])  # top face at y 0, same as the plate
    reg = registry_from_glb("m1", glb, scale=1.0)
    t_co = look_at((0, 0.3, 0))
    assert cad_visibility(reg.scene, _rois(reg, t_co), t_co, K, SHAPE).fractions[0] > 0.9


def test_too_many_triangles_is_reported_not_guessed(registry):
    t_co = look_at((0, 0.3, 0))
    assert cad_visibility(registry.scene, _rois(registry, t_co), t_co, K, SHAPE, max_faces=10) is None


# ── visibility decisions ────────────────────────────────────────────────


def _textured(shape=SHAPE, seed=0):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(40, 220, shape, dtype=np.uint8), (3, 3), 0)


def _assess(registry, t_co, gray=None, tracking_ok=True, cad=1.0, i=0, **config):
    p = project_component(registry.components[i].geometry, t_co, K, SHAPE)
    return assess(p, _textured() if gray is None else gray, tracking_ok, "tracking lost", cad, VisibilityConfig(**config))


def test_clear_view_is_observable(registry):
    v = _assess(registry, look_at((0, 0.3, 0)))
    assert v.observable and v.status == "visible" and v.reasons == []


def test_tracking_loss_makes_every_observation_invalid(registry):
    v = _assess(registry, look_at((0, 0.3, 0)), tracking_ok=False)
    assert not v.observable and v.status == "tracking" and v.reasons[0] == "tracking lost"


def test_partly_outside_the_image(registry):
    t_co = look_at((0, 0.3, 0))
    t_co[0, 3] += 0.27  # slide the object right: the rightmost screws leave the 1280 px frame
    v = _assess(registry, t_co, i=3)
    assert not v.observable and v.status == "out_of_frame"


def test_too_small_at_a_distance(registry):
    v = _assess(registry, look_at((0, 3.0, 0)))
    assert not v.observable and v.status == "too_small"


def test_hidden_behind_cad_geometry(registry):
    v = _assess(registry, look_at((0, 0.3, 0)), cad=0.2)
    assert not v.observable and v.status == "occluded" and "80% hidden" in v.reasons[0]


def test_blurred_dark_and_clipped_images(registry):
    t_co = look_at((0, 0.3, 0))
    flat = np.full(SHAPE, 128, np.uint8)  # no texture at all: nothing to see a screw by
    assert _assess(registry, t_co, gray=flat).status == "poor_image"
    dark = (_textured() // 16).astype(np.uint8)
    v = _assess(registry, t_co, gray=dark)
    assert not v.observable and any("too dark" in r for r in v.reasons)
    burnt = np.full(SHAPE, 255, np.uint8)
    burnt[::2] = 0
    assert any("clipped" in r for r in _assess(registry, t_co, gray=burnt).reasons)


def test_visible_mask_is_the_part_of_a_component_not_hidden():
    """A ring around a post, seen from the side and above: its far half is behind the post."""
    from glb_builder import Node, box, build_glb

    def band(radius, height, y, segments=24):
        # A thin open band (outer wall only): enough to show a hidden far side.
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        bottom = np.c_[radius * np.cos(angles), np.full(segments, y), radius * np.sin(angles)]
        v = np.r_[bottom, bottom + [0, height, 0]]
        f = []
        for k in range(segments):
            n = (k + 1) % segments
            f += [[k, n, segments + n], [k, segments + n, segments + k]]
        return v, np.array(f)

    glb = build_glb([Node("Post", mesh=box((0, 0.05, 0), (0.03, 0.1, 0.03))), Node("Seal_Ring", mesh=band(0.025, 0.005, 0.05))])
    reg = registry_from_glb("m1", glb, scale=1.0)
    t_co = look_at((0.0, 0.12, 0.25), target=(0, 0.05, 0), up=(0, 1, 0))
    p = project_component(reg.components[0].geometry, t_co, K, SHAPE)
    occlusion = cad_visibility(reg.scene, {0: p.roi}, t_co, K, SHAPE)
    visible = occlusion.visible_mask(0, p)
    # The band's near side is in view; where the far side shows above it, the post is in front.
    assert 0.4 < occlusion.fractions[0] < 0.95
    assert visible.sum() < p.mask.sum()
    assert visible.sum() / p.mask.sum() == pytest.approx(occlusion.fractions[0], abs=0.1)
