import numpy as np
import pytest

from app.schemas.vision import BoundingBox
from app.services.model3d.glb_inspect import GlbBounds, GlbPart
from app.services.vision.part_presence import PartRegion, PresenceCalibration, region_brightness

# The user's full-bottle.glb, in file units: body, cap on top, thin ring below the cap.
BODY = GlbPart(0, "Cylinder", GlbBounds(min=(-3.0, -5.1874, -3.0), max=(3.0, 14.4957, 3.0)))
CAP = GlbPart(1, "Cylinder.001", GlbBounds(min=(-1.953, 12.6341, -1.953), max=(1.953, 14.8126, 1.953)))
RING = GlbPart(2, "Cylinder.003", GlbBounds(min=(-2.358, 12.213, -2.05), max=(2.358, 12.5199, 2.05)))
PARTS = [BODY, CAP, RING]


def test_cap_region_is_the_top_eleven_percent_and_middle_of_the_bottle():
    r = PartRegion.from_parts(CAP, PARTS)
    # Assembly y -5.1874..14.8126 (20.0); cap 12.6341..14.8126 -> 0 .. 0.1089 of the height,
    # trimmed 10% of its own band each side.
    assert r.top == pytest.approx(0.01089, abs=1e-4)
    assert r.bottom == pytest.approx(0.09803, abs=1e-4)
    # Width -3..3; cap -1.953..1.953 -> 0.1745..0.8255, trimmed 10% of 0.651 each side.
    assert r.left == pytest.approx(0.2396, abs=1e-3)
    assert r.right == pytest.approx(0.7604, abs=1e-3)


def test_region_brightness_reads_only_the_part_region():
    frame = np.full((400, 200, 3), 200, np.uint8)  # bright background/body
    box = BoundingBox(x1=50, y1=0, x2=150, y2=400)
    r = PartRegion.from_parts(CAP, PARTS)
    x1, y1, x2, y2 = r.pixels(box)
    frame[y1:y2, x1:x2] = 40  # dark cap exactly where the region is
    assert region_brightness(frame, r, box) == pytest.approx(40, abs=1)


def test_region_off_frame_gives_none():
    frame = np.zeros((100, 100, 3), np.uint8)
    assert region_brightness(frame, PartRegion.from_parts(CAP, PARTS), BoundingBox(x1=200, y1=200, x2=300, y2=500)) is None


# Calibrated from the user's frames: cap on ~79, cap off (steel neck) ~159.
CAL = PresenceCalibration(PartRegion.from_parts(CAP, PARTS), 79.0, 159.0, 2, 3, 0.2, 2.6)


@pytest.mark.parametrize(
    "brightness, state",
    [(78.7, "present"), (95.0, "present"), (119.0, "uncertain"), (124.0, "uncertain"), (140.0, "absent"), (160.9, "absent")],
)
def test_classification_is_three_way(brightness, state):
    assert CAL.classify(brightness)[0] == state


def test_confidence_is_full_at_the_calibrated_means_and_zero_at_the_midpoint():
    assert CAL.classify(79.0)[1] == pytest.approx(1.0)
    assert CAL.classify(159.0)[1] == pytest.approx(1.0)
    assert CAL.classify(119.0)[1] == pytest.approx(0.0)


def test_works_the_other_way_round_for_a_light_part_on_a_dark_object():
    light_cap = PresenceCalibration(CAL.region, present_mean=180.0, absent_mean=60.0, present_samples=3, absent_samples=3)
    assert light_cap.classify(175.0)[0] == "present"
    assert light_cap.classify(65.0)[0] == "absent"


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "cal.json"
    CAL.save(path)
    loaded = PresenceCalibration.load(path)
    assert loaded == CAL and isinstance(loaded.region, PartRegion)
    assert PresenceCalibration.load(tmp_path / "missing.json") is None
