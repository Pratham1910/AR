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


# ── pose method: the part's exact pixels at the tracked pose ─────────────

from app.services.model3d.glb_inspect import GlbPartMesh  # noqa: E402
from app.services.vision.part_presence import best_feature, mask_features, project_part_mask  # noqa: E402

K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
# A 2 x 2 unit square (two triangles) centred on the assembly center (10, 0, 0).
SQUARE = GlbPartMesh(
    vertices=np.array([[9, -1, 0], [11, -1, 0], [11, 1, 0], [9, 1, 0]], float),
    faces=np.array([[0, 1, 2], [0, 2, 3]]),
)
CENTER = np.array([10.0, 0.0, 0.0])


def _at(z: float) -> np.ndarray:
    pose = np.eye(4)
    pose[2, 3] = z
    return pose


def test_part_mask_is_the_part_drawn_at_the_pose():
    # Scale 0.1: the square is 0.2 m wide, 1 m away -> 20 px wide around the principal point.
    mask = project_part_mask(SQUARE, CENTER, 0.1, _at(1.0), K, (100, 100))
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    assert (cols[0], cols[-1], rows[0], rows[-1]) == (40, 60, 40, 60)
    assert mask[40:61, 40:61].all()


def test_overlapping_triangles_do_not_cancel_into_holes():
    """cv2.fillPoly with all triangles at once uses even-odd filling: a front
    and back face over the same pixels would leave a hole."""
    doubled = GlbPartMesh(SQUARE.vertices, np.vstack([SQUARE.faces, SQUARE.faces[:, ::-1]]))
    assert project_part_mask(doubled, CENTER, 0.1, _at(1.0), K, (100, 100))[50, 50]


def test_triangles_behind_the_camera_are_dropped():
    assert not project_part_mask(SQUARE, CENTER, 0.1, _at(-1.0), K, (100, 100)).any()


def test_features_need_enough_visible_pixels():
    frame = np.full((100, 100, 3), 200, np.uint8)
    tiny = np.zeros((100, 100), bool)
    tiny[0:3, 0:3] = True
    assert mask_features(frame, tiny) is None
    big = np.zeros((100, 100), bool)
    big[20:80, 20:80] = True
    assert mask_features(frame, big) == pytest.approx({"mean": 200, "p10": 200, "p90": 200, "std": 0})


def test_calibration_keeps_the_feature_that_separates_on_from_off():
    """The user's flask: similar mean brightness either way, but a black cap is
    uniform (low spread) while the threaded steel neck is not."""
    present = [{"mean": 70, "p10": lo, "p90": p, "std": s} for lo, p, s in ((48, 90, 21), (55, 110, 24), (42, 94, 27))]
    absent = [
        {"mean": m, "p10": lo, "p90": p, "std": s} for m, lo, p, s in ((72, 50, 100, 36), (90, 38, 160, 40), (140, 60, 193, 45))
    ]
    feature, stats = best_feature(present, absent)
    assert feature == "std"
    assert stats["present_mean"] == pytest.approx(24) and stats["absent_mean"] == pytest.approx(40.33, abs=0.01)


def test_files_saved_before_the_pose_method_load_as_box_mean(tmp_path):
    path = tmp_path / "old.json"
    old = {k: v for k, v in CAL.__dict__.items() if k not in ("method", "feature")}
    old["region"] = CAL.region.__dict__
    path.write_text(__import__("json").dumps(old))
    loaded = PresenceCalibration.load(path)
    assert (loaded.method, loaded.feature) == ("box", "mean")
