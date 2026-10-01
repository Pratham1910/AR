import json
import math
from pathlib import Path

import pytest

from app.services.model3d.glb_inspect import (
    GlbParseError,
    compute_glb_bounds,
    compute_scale_for_real_height,
    list_glb_parts,
)

BOTTLE_GLB_PATH = Path(__file__).resolve().parents[2] / "data" / "models" / "bottle.glb"


def test_compute_glb_bounds_matches_the_known_bottle_dimensions():
    """
    Regression check against Project.md's provided bottle.glb: its vertex
    accessor spans Y [-1.629, 3.643], and its one node ("Cylinder") is
    translated +0.99253 in Y, so the world-space box (what Three.js renders
    and recenters on) is Y [-0.637, 4.636]. Height, which the scale comes
    from, is the same either way.
    """
    data = BOTTLE_GLB_PATH.read_bytes()
    bounds = compute_glb_bounds(data)

    node_y = 0.9925339221954346
    assert bounds.min == pytest.approx((-1.0, -1.62907075881958 + node_y, -1.0), abs=1e-6)
    assert bounds.max == pytest.approx((1.0, 3.6430740356445312 + node_y, 1.0), abs=1e-6)
    assert bounds.height == pytest.approx(5.272144794464111, abs=1e-6)
    assert bounds.width == pytest.approx(2.0, abs=1e-6)


def test_compute_scale_for_real_height_matches_the_seeded_value():
    """seed_demo.py sets bottle.glb's scale from a measured 0.15m real height."""
    data = BOTTLE_GLB_PATH.read_bytes()
    scale = compute_scale_for_real_height(data, real_height_m=0.15)
    assert scale == pytest.approx(0.028451418890752752, rel=1e-6)


def _glb_from_json(doc: dict) -> bytes:
    payload = json.dumps(doc).encode()
    payload += b" " * (-len(payload) % 4)
    chunk = len(payload).to_bytes(4, "little") + b"JSON" + payload
    return b"glTF" + (2).to_bytes(4, "little") + (12 + len(chunk)).to_bytes(4, "little") + chunk


_UNIT_CUBE = {
    "accessors": [{"min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5]}],
    "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
}


def test_bounds_apply_node_trs_through_the_hierarchy():
    """Unit cube, child node scaled (0.1, 0.2, 0.3), rotated 90deg about X
    ((x, y, z) -> (x, -z, y)), translated +1 in X, under a parent scaled 2x.
    By hand: x in [1.9, 2.1], y in [-0.3, 0.3], z in [-0.2, 0.2]."""
    s = math.sqrt(0.5)
    doc = {
        **_UNIT_CUBE,
        "nodes": [
            {"children": [1], "scale": [2, 2, 2]},
            {"mesh": 0, "translation": [1, 0, 0], "rotation": [s, 0, 0, s], "scale": [0.1, 0.2, 0.3]},
        ],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    bounds = compute_glb_bounds(_glb_from_json(doc))
    assert bounds.min == pytest.approx((1.9, -0.3, -0.2), abs=1e-9)
    assert bounds.max == pytest.approx((2.1, 0.3, 0.2), abs=1e-9)
    assert compute_scale_for_real_height(_glb_from_json(doc), 0.6) == pytest.approx(1.0)


def test_bounds_apply_column_major_node_matrix():
    doc = {
        **_UNIT_CUBE,
        "nodes": [{"mesh": 0, "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 5, 0, 1]}],
        "scenes": [{"nodes": [0]}],
    }
    bounds = compute_glb_bounds(_glb_from_json(doc))
    assert bounds.min == pytest.approx((-0.5, 4.5, -0.5))
    assert bounds.max == pytest.approx((0.5, 5.5, 0.5))


def test_list_parts_gives_each_mesh_node_its_own_world_bounds():
    """Two-part assembly: a 'Body' at the origin and a 'Cap' moved up 0.6
    and scaled to half size — each part's box must reflect its own node
    transform, and together they make up the overall bounds."""
    doc = {
        **_UNIT_CUBE,
        "nodes": [
            {"name": "Body", "mesh": 0},
            {"name": "Cap", "mesh": 0, "translation": [0, 0.6, 0], "scale": [0.5, 0.5, 0.5]},
        ],
        "scenes": [{"nodes": [0, 1]}],
    }
    data = _glb_from_json(doc)
    parts = list_glb_parts(data)
    assert [(p.node_index, p.name) for p in parts] == [(0, "Body"), (1, "Cap")]
    assert parts[0].bounds.min == pytest.approx((-0.5, -0.5, -0.5))
    assert parts[1].bounds.min == pytest.approx((-0.25, 0.35, -0.25))
    assert parts[1].bounds.max == pytest.approx((0.25, 0.85, 0.25))
    assert compute_glb_bounds(data).max == pytest.approx((0.5, 0.85, 0.5))


def test_non_glb_data_raises():
    with pytest.raises(GlbParseError):
        compute_glb_bounds(b"this is not a glb file")


def test_glb_with_no_json_chunk_raises():
    header = b"glTF" + (2).to_bytes(4, "little") + (12).to_bytes(4, "little")
    with pytest.raises(GlbParseError):
        compute_glb_bounds(header)
