from pathlib import Path

import pytest

from app.services.model3d.glb_inspect import GlbParseError, compute_glb_bounds, compute_scale_for_real_height

BOTTLE_GLB_PATH = Path(__file__).resolve().parents[2] / "data" / "models" / "bottle.glb"


def test_compute_glb_bounds_matches_the_known_bottle_dimensions():
    """
    Regression check against the exact bounds manually confirmed earlier
    (Project.md's provided bottle.glb): min [-1, -1.629, -1], max [1, 3.643, 1].
    """
    data = BOTTLE_GLB_PATH.read_bytes()
    bounds = compute_glb_bounds(data)

    assert bounds.min == pytest.approx((-1.0, -1.62907075881958, -1.0), abs=1e-6)
    assert bounds.max == pytest.approx((1.0, 3.6430740356445312, 1.0), abs=1e-6)
    assert bounds.height == pytest.approx(5.272144794464111, abs=1e-6)
    assert bounds.width == pytest.approx(2.0, abs=1e-6)


def test_compute_scale_for_real_height_matches_the_seeded_value():
    """seed_demo.py sets bottle.glb's scale from a measured 0.15m real height."""
    data = BOTTLE_GLB_PATH.read_bytes()
    scale = compute_scale_for_real_height(data, real_height_m=0.15)
    assert scale == pytest.approx(0.028451418890752752, rel=1e-6)


def test_non_glb_data_raises():
    with pytest.raises(GlbParseError):
        compute_glb_bounds(b"this is not a glb file")


def test_glb_with_no_json_chunk_raises():
    header = b"glTF" + (2).to_bytes(4, "little") + (12).to_bytes(4, "little")
    with pytest.raises(GlbParseError):
        compute_glb_bounds(header)
