"""
ArUco product tags (docs/marker-tags.md), end to end without a camera:
synthetic frames with real rendered tags read by the real ArUco detector,
and a colour-blob stand-in for YOLO that reports what it is actually shown —
so "an object outside the tags is never detected" is tested on pixels, not
assumed.

The API tests at the bottom need PostgreSQL and run only with
MIGRATION_TEST_DATABASE_URL set (they create and drop their own database).
"""

import base64
import uuid

import cv2
import numpy as np
import pytest

from app.models.enums import MarkerCorner
from app.schemas.vision import BoundingBox, SegmentedObject, Vector2
from app.services.markers.geometry import (
    QUAD_ORDER,
    enclosed_region,
    is_clockwise_convex,
    points_inside,
    polygon_inside_fraction,
)
from app.services.markers.identification import (
    LookupStatus,
    SetStatus,
    TagAssignment,
    find_four_tag_set,
    lookup_by_any_tag,
)
from app.services.markers.mapping import MappingError, suggest_free_ids, validate_tag_set
from app.services.markers.region_detection import RegionStatus, detect_in_region, masked_crop
from app.services.markers.tags import TagSighting, dictionary_size, read_tags, render_sheet_png, render_tag_png
from app.services.markers.tracking import FourTagTracker, ProductLookupSession, TrackState
from app.services.pose.aruco_pose import ArucoPoseEstimator

DICTIONARY = "DICT_4X4_50"
detect = ArucoPoseEstimator(DICTIONARY, 0.05).detect_markers
TL, TR, BR, BL = QUAD_ORDER

BOTTLE_IDS = {TL: 0, TR: 1, BR: 2, BL: 3}
CUP_IDS = {TL: 4, TR: 5, BR: 6, BL: 7}
REGISTRY = {i: TagAssignment("bottle", c) for c, i in BOTTLE_IDS.items()} | {
    i: TagAssignment("cup", c) for c, i in CUP_IDS.items()
}

TAG = 70  # printed tag side in the synthetic frames, px
BLUE, RED = (255, 0, 0), (0, 0, 255)  # BGR: what the stand-in detector calls "bottle" / "cup"


# --- Synthetic frames ---------------------------------------------------------


def _paste_tag(frame: np.ndarray, marker_id: int, x: int, y: int) -> None:
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id, TAG)
    frame[y : y + TAG, x : x + TAG] = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)


def tagged_frame(
    ids: dict[MarkerCorner, int] = BOTTLE_IDS,
    box: tuple[int, int, int, int] = (300, 150, 900, 600),  # where the four tags sit: x0, y0, x1, y1
    skip: tuple[MarkerCorner, ...] = (),
    size: tuple[int, int] = (720, 1280),
) -> np.ndarray:
    frame = np.full((*size, 3), 235, dtype=np.uint8)
    x0, y0, x1, y1 = box
    positions = {TL: (x0, y0), TR: (x1 - TAG, y0), BR: (x1 - TAG, y1 - TAG), BL: (x0, y1 - TAG)}
    for corner, marker_id in ids.items():
        if corner not in skip:
            _paste_tag(frame, marker_id, *positions[corner])
    return frame


def blob_segmenter(seen: list[np.ndarray] | None = None):
    """Stand-in for YOLO: blue blobs are 'bottle', red blobs 'cup', with the
    blob's real contour as the mask. Records every image it is given."""

    def segment(image: np.ndarray) -> list[SegmentedObject]:
        if seen is not None:
            seen.append(image.copy())
        objects = []
        for color, label in ((BLUE, "bottle"), (RED, "cup")):
            mask = cv2.inRange(image, np.array(color) - 20, np.array(color) + 20)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 50:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                objects.append(
                    SegmentedObject(
                        class_label=label,
                        confidence=0.9,
                        bbox=BoundingBox(x1=x, y1=y, x2=x + w, y2=y + h),
                        polygon=[Vector2(x=float(px), y=float(py)) for px, py in contour.reshape(-1, 2)],
                    )
                )
        return objects

    return segment


def sightings_of(frame: np.ndarray) -> list[TagSighting]:
    found, duplicates = read_tags(detect, frame)
    assert duplicates == []
    return found


def complete_set(frame: np.ndarray):
    found = find_four_tag_set(sightings_of(frame), REGISTRY, [], 6.0)
    assert found.status is SetStatus.COMPLETE, found
    return found


# --- Tags: reading and printing ----------------------------------------------------


def test_reads_every_tag_with_its_real_corners():
    sightings = sightings_of(tagged_frame())
    assert [s.marker_id for s in sightings] == [0, 1, 2, 3]
    top_left = next(s for s in sightings if s.marker_id == 0)
    assert np.allclose(top_left.corners.min(axis=0), (300, 150), atol=1.5)
    assert np.allclose(top_left.corners.max(axis=0), (300 + TAG - 1, 150 + TAG - 1), atol=1.5)


def test_every_marker_in_the_frame_is_returned_not_just_the_first():
    """The single-marker pose path (/api/vision/pose) uses the first marker
    only; the tag path must get all of them, with their own corners."""
    found = detect(tagged_frame())
    assert sorted(i for i, _ in found) == [0, 1, 2, 3]
    assert all(c.shape == (4, 2) for _, c in found)


def test_a_tag_printed_twice_is_reported_not_guessed():
    frame = tagged_frame()
    _paste_tag(frame, 0, 600, 300)
    sightings, duplicates = read_tags(detect, frame)
    assert duplicates == [0]
    assert 0 not in [s.marker_id for s in sightings]


def test_printable_tag_and_sheet_are_readable_by_the_detector():
    assert dictionary_size(DICTIONARY) == 50
    tag = cv2.imdecode(np.frombuffer(render_tag_png(DICTIONARY, 17, 300, "Bottle  Top-left  id 17"), np.uint8), cv2.IMREAD_COLOR)
    assert [i for i, _ in detect(tag)] == [17]
    sheet_png = render_sheet_png(DICTIONARY, {TL: 10, TR: 11, BR: 12, BL: 13}, 200, "BOTTLE-001")
    sheet = cv2.imdecode(np.frombuffer(sheet_png, np.uint8), cv2.IMREAD_COLOR)
    by_id = {i: c.mean(axis=0) for i, c in detect(sheet)}
    assert sorted(by_id) == [10, 11, 12, 13]
    # Each tag is printed in its own corner position.
    assert by_id[10][0] < by_id[11][0] and by_id[10][1] < by_id[13][1] and by_id[12][0] > by_id[13][0]


# --- Mapping rules -------------------------------------------------------------------


def test_mapping_accepts_four_distinct_free_ids_and_a_products_own_ids():
    taken = {0: ("bottle", "Bottle"), 1: ("bottle", "Bottle")}
    validate_tag_set({TL: 0, TR: 1, BR: 8, BL: 9}, "bottle", taken, 50)  # rearranging its own ids is fine


@pytest.mark.parametrize(
    ("requested", "conflict", "text"),
    [
        ({TL: 0, TR: 0, BR: 2, BL: 3}, False, "both"),
        ({TL: 0, TR: 1, BR: 2, BL: 50}, False, "outside"),
        ({TL: 0, TR: 1, BR: 2}, False, "missing"),
        ({TL: 4, TR: 1, BR: 2, BL: 3}, True, "already assigned to Cup"),
    ],
)
def test_mapping_refuses_duplicates_out_of_range_incomplete_and_other_products_ids(requested, conflict, text):
    with pytest.raises(MappingError) as err:
        validate_tag_set(requested, "bottle", {4: ("cup", "Cup")}, 50)
    assert err.value.conflict is conflict
    assert text in str(err.value)


def test_suggests_the_lowest_unused_ids():
    assert suggest_free_ids({0, 1, 3}, 50) == [2, 4, 5, 6]
    assert suggest_free_ids(set(range(48)), 50) == [48, 49]


# --- Identification ------------------------------------------------------------------


def test_single_tag_lookup():
    frame = tagged_frame(skip=(TR, BR, BL))
    assert lookup_by_any_tag(sightings_of(frame), REGISTRY).asset_id == "bottle"
    assert lookup_by_any_tag([], REGISTRY).status is LookupStatus.NO_MARKERS
    unknown = sightings_of(tagged_frame(ids={TL: 30}))
    assert lookup_by_any_tag(unknown, REGISTRY).status is LookupStatus.UNKNOWN_MARKERS
    both = tagged_frame(ids={TL: 0})
    _paste_tag(both, 5, 600, 300)
    lookup = lookup_by_any_tag(sightings_of(both), REGISTRY)
    assert lookup.status is LookupStatus.AMBIGUOUS and lookup.asset_ids == ["bottle", "cup"]


def test_four_tags_of_one_product_give_a_region_along_their_outer_corners():
    found = complete_set(tagged_frame())
    assert found.asset_id == "bottle"
    assert np.allclose(found.region, [[300, 150], [899, 150], [899, 599], [300, 599]], atol=1.5)


def test_missing_corners_are_named():
    found = find_four_tag_set(sightings_of(tagged_frame(skip=(BR,))), REGISTRY, [], 6.0)
    assert found.status is SetStatus.INCOMPLETE and found.missing == [BR]


def test_tags_of_two_products_are_never_combined():
    frame = tagged_frame(skip=(BR, BL))
    _paste_tag(frame, CUP_IDS[BR], 830, 530)
    _paste_tag(frame, CUP_IDS[BL], 300, 530)
    found = find_four_tag_set(sightings_of(frame), REGISTRY, [], 6.0)
    assert found.status is SetStatus.MIXED_PRODUCTS


def test_a_foreign_tag_inside_the_set_blocks_it():
    frame = tagged_frame()
    _paste_tag(frame, CUP_IDS[TL], 560, 340)
    found = find_four_tag_set(sightings_of(frame), REGISTRY, [], 6.0)
    assert found.status is SetStatus.MIXED_PRODUCTS and "inside" in found.problem


def test_two_complete_sets_are_reported_not_picked_between():
    frame = tagged_frame(box=(20, 20, 560, 400))
    cup = tagged_frame(ids=CUP_IDS, box=(640, 300, 1260, 700))
    frame[cup != 235] = cup[cup != 235]
    assert find_four_tag_set(sightings_of(frame), REGISTRY, [], 6.0).status is SetStatus.MULTIPLE_PRODUCTS


def test_swapped_corners_and_bunched_tags_are_invalid_layouts():
    swapped = tagged_frame(ids={TL: 1, TR: 0, BR: 2, BL: 3})
    found = find_four_tag_set(sightings_of(swapped), REGISTRY, [], 6.0)
    assert found.status is SetStatus.LAYOUT_INVALID and "corners" in found.problem
    assert found.region is not None  # still drawn, so the operator can see what is wrong
    bunched = tagged_frame(box=(500, 300, 500 + 2 * TAG + 10, 300 + 2 * TAG + 10))
    found = find_four_tag_set(sightings_of(bunched), REGISTRY, [], 6.0)
    assert found.status is SetStatus.LAYOUT_INVALID and "tag areas" in found.problem


def test_duplicate_mapped_id_stops_the_set():
    found = find_four_tag_set(sightings_of(tagged_frame()), REGISTRY, [2], 6.0)
    assert found.status is SetStatus.DUPLICATE_MARKERS


# --- Geometry under perspective ----------------------------------------------------------


def _warp(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[120, 40], [w - 60, 110], [w - 180, h - 30], [60, h - 90]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, matrix, (w, h), borderValue=(235, 235, 235)), matrix


def test_region_follows_perspective_not_the_bounding_box():
    frame, matrix = _warp(tagged_frame())
    found = complete_set(frame)
    expected = cv2.perspectiveTransform(np.float32([[[300, 150]], [[899, 150]], [[899, 599]], [[300, 599]]]), matrix).reshape(4, 2)
    assert np.allclose(found.region, expected, atol=3.0)
    x, y, w, h = cv2.boundingRect(found.region.astype(np.float32))
    box_area = w * h
    region_area = cv2.contourArea(found.region.astype(np.float32))
    assert region_area < 0.95 * box_area  # a real quadrilateral, not its bounding box
    assert is_clockwise_convex(found.region)


def test_inside_tests():
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
    assert points_inside(np.array([[50, 50], [150, 50]]), square).tolist() == [True, False]
    assert polygon_inside_fraction(np.array([[10, 10], [40, 10], [40, 40], [10, 40]], dtype=float), square) == pytest.approx(1.0)
    straddling = np.array([[80, 10], [120, 10], [120, 50], [80, 50]], dtype=float)
    assert polygon_inside_fraction(straddling, square) == pytest.approx(0.5, abs=0.05)


# --- Finding the object inside the region ------------------------------------------------


def test_the_detector_sees_only_the_region_and_a_band_around_it():
    frame = tagged_frame()
    cv2.circle(frame, (600, 375), 90, BLUE, -1)  # the bottle, between the tags
    cv2.circle(frame, (1180, 375), 80, BLUE, -1)  # another bottle elsewhere in view
    cv2.circle(frame, (60, 680), 40, RED, -1)  # and a cup
    found = complete_set(frame)
    seen: list[np.ndarray] = []

    result = detect_in_region(frame, found.region, blob_segmenter(seen), "bottle", 0.8, context=0.5)

    assert result.status is RegionStatus.FOUND
    assert len(result.accepted) == 1 and result.rejected == []
    # Mapped back to frame coordinates: the mask is the circle at (600, 375).
    b = result.best.obj.bbox
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((510, 285, 691, 466), abs=2)
    # What the detector was given: the whole area between the tags (not a
    # single tag) plus a band of a quarter of its size, nothing beyond.
    x, y, w, h = result.search_rect
    assert (x, x + w) == pytest.approx((150, 1050), abs=2) and (y, y + h) == pytest.approx((37, 713), abs=2)
    blue = cv2.inRange(seen[0], np.array(BLUE) - 20, np.array(BLUE) + 20)
    assert cv2.countNonZero(blue) == pytest.approx(np.pi * 90**2, rel=0.05)  # only the bottle between the tags
    assert cv2.countNonZero(cv2.inRange(seen[0], np.array(RED) - 20, np.array(RED) + 20)) == 0


def test_a_similar_object_next_to_the_tags_is_seen_and_rejected():
    frame = tagged_frame()
    cv2.circle(frame, (600, 375), 90, BLUE, -1)
    cv2.circle(frame, (990, 375), 40, BLUE, -1)  # just outside the right tags, in the context band
    result = detect_in_region(frame, complete_set(frame).region, blob_segmenter(), "bottle", 0.8)
    assert result.status is RegionStatus.FOUND and len(result.accepted) == 1
    assert len(result.rejected) == 1 and result.rejected[0].inside < 0.1
    assert result.best.obj.bbox.x1 == pytest.approx(510, abs=2)


def test_an_object_reaching_in_from_outside_is_rejected():
    frame = tagged_frame()
    cv2.rectangle(frame, (820, 330), (1100, 420), BLUE, -1)  # mostly outside the right edge
    result = detect_in_region(frame, complete_set(frame).region, blob_segmenter(), "bottle", 0.8)
    assert result.status is RegionStatus.NOT_FOUND
    # The context band lets the detector see it crossing the edge, so its
    # mask is measured whole: about a third of it is inside.
    assert len(result.rejected) == 1 and result.rejected[0].inside == pytest.approx(80 / 231, abs=0.05)


def test_a_mask_leaking_outside_the_region_is_rejected():
    frame = tagged_frame()
    found = complete_set(frame)

    def leaky(image: np.ndarray) -> list[SegmentedObject]:
        # A mask (in the crop's coordinates, which start at (150, 38)) that
        # covers frame x 250-750, y 100-300: it runs past the region's top
        # and left edges (x 300, y 150).
        poly = [Vector2(x=100, y=62), Vector2(x=600, y=62), Vector2(x=600, y=262), Vector2(x=100, y=262)]
        return [SegmentedObject(class_label="bottle", confidence=0.9, bbox=BoundingBox(x1=100, y1=62, x2=600, y2=262), polygon=poly)]

    result = detect_in_region(frame, found.region, leaky, "bottle", 0.8)
    assert result.status is RegionStatus.NOT_FOUND
    assert result.rejected[0].inside == pytest.approx(450 * 150 / (500 * 200), abs=0.03)


def test_the_wrong_object_between_the_tags_is_named():
    frame = tagged_frame()
    cv2.circle(frame, (600, 375), 80, RED, -1)
    result = detect_in_region(frame, complete_set(frame).region, blob_segmenter(), "bottle", 0.8)
    assert result.status is RegionStatus.WRONG_OBJECT and result.best.obj.class_label == "cup"


def test_without_a_detection_class_the_most_prominent_object_inside_is_taken():
    frame = tagged_frame()
    cv2.circle(frame, (600, 375), 120, RED, -1)
    cv2.circle(frame, (420, 260), 25, BLUE, -1)
    result = detect_in_region(frame, complete_set(frame).region, blob_segmenter(), None, 0.8)
    assert result.status is RegionStatus.FOUND and result.best.obj.class_label == "cup"


def test_object_found_inside_a_perspective_region():
    base = tagged_frame()
    cv2.circle(base, (600, 375), 90, BLUE, -1)
    cv2.circle(base, (1180, 375), 60, BLUE, -1)  # outside
    frame, matrix = _warp(base)
    result = detect_in_region(frame, complete_set(frame).region, blob_segmenter(), "bottle", 0.8)
    assert result.status is RegionStatus.FOUND and len(result.accepted) == 1
    centre = cv2.perspectiveTransform(np.float32([[[600, 375]]]), matrix).reshape(2)
    b = result.best.obj.bbox
    assert ((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2) == pytest.approx(tuple(centre), abs=6)


def test_masked_crop_blanks_everything_outside_the_quadrilateral():
    frame = np.full((200, 200, 3), 50, dtype=np.uint8)
    region = np.array([[50, 20], [180, 60], [150, 180], [20, 150]], dtype=float)
    crop, (x, y, w, h) = masked_crop(frame, region)
    assert (x, y, w, h) == (20, 20, 161, 161)
    assert tuple(crop[0, 0]) == (114, 114, 114)  # corner of the rectangle, outside the quad
    assert tuple(crop[80, 80]) == (50, 50, 50)  # inside


# --- Tracking over frames ----------------------------------------------------------------


def _frame_with_bottle(dx: int = 0, skip=()) -> np.ndarray:
    frame = tagged_frame(box=(300 + dx, 150, 900 + dx, 600), skip=skip)
    cv2.circle(frame, (600 + dx, 375), 90, BLUE, -1)
    return frame


def _step(tracker: FourTagTracker, frame: np.ndarray, now: float, segment=None):
    sightings, duplicates = read_tags(detect, frame)
    located = tracker.locate(sightings, REGISTRY, duplicates, now)
    detection = None
    if located.found.status is SetStatus.COMPLETE:
        detection = detect_in_region(frame, located.found.region, segment or blob_segmenter(), "bottle", 0.8)
    return located, tracker.finish(located, detection, now)


def test_tracking_needs_consecutive_agreeing_frames():
    tracker = FourTagTracker(hold_ms=1500, confirm_frames=3, min_visible=2, min_region_in_tags=6.0)
    states = [_step(tracker, _frame_with_bottle(), t)[1].state for t in (0, 100, 200, 300)]
    assert states == [TrackState.VERIFYING, TrackState.VERIFYING, TrackState.TRACKING, TrackState.TRACKING]
    empty = tagged_frame()  # tags still there, bottle removed
    assert _step(tracker, empty, 400)[1].state is TrackState.TRACKING  # held briefly
    outcome = _step(tracker, empty, 2000)[1]
    assert outcome.state is TrackState.OBJECT_NOT_FOUND and outcome.lost


def test_a_covered_tag_is_reprojected_from_the_visible_ones_while_tracking():
    tracker = FourTagTracker(hold_ms=1500, confirm_frames=3, min_visible=2, min_region_in_tags=6.0)
    for t in (0, 100, 200):
        _step(tracker, _frame_with_bottle(), t)
    # Product moved 40 px right and the bottom-right tag is covered.
    located, outcome = _step(tracker, _frame_with_bottle(dx=40, skip=(BR,)), 300)
    assert located.estimated == [BR]
    assert outcome.state is TrackState.TRACKING
    estimated = located.found.tags[BR].corners
    assert np.allclose(estimated.mean(axis=0), (900 + 40 - TAG / 2, 600 - TAG / 2), atol=2.0)
    # ...but only for hold_ms after the last time all four were read.
    located, outcome = _step(tracker, _frame_with_bottle(dx=40, skip=(BR,)), 1900)
    assert located.estimated == [] and outcome.state is TrackState.INCOMPLETE and outcome.lost


def test_a_product_is_never_acquired_from_estimated_tags():
    tracker = FourTagTracker(hold_ms=1500, confirm_frames=3, min_visible=2, min_region_in_tags=6.0)
    _step(tracker, _frame_with_bottle(), 0)  # verifying, not yet tracking
    located, outcome = _step(tracker, _frame_with_bottle(skip=(BR,)), 100)
    assert located.estimated == [] and outcome.state is TrackState.INCOMPLETE


def test_single_tag_lookup_needs_the_same_product_in_consecutive_frames():
    session = ProductLookupSession(required=2)
    session.record("bottle")
    assert not session.identified
    session.record("cup")
    assert not session.identified
    session.record("cup")
    assert session.identified
    session.record(None)
    assert not session.identified


# --- API against a real database (optional) ----------------------------------------------

from app.core.config import get_settings  # noqa: E402

_server_url = get_settings().migration_test_database_url
needs_database = pytest.mark.skipif(not _server_url, reason="set MIGRATION_TEST_DATABASE_URL to run the API tests")


@pytest.fixture
def client():
    from pathlib import Path

    import sqlalchemy as sa
    from alembic import command
    from alembic.config import Config
    from fastapi.testclient import TestClient
    from sqlalchemy.engine import make_url
    from sqlalchemy.orm import sessionmaker

    from app.api import marker_tags
    from app.core.database import get_db
    from app.main import app

    server = make_url(_server_url)
    name = f"tvasta_tagtest_{uuid.uuid4().hex[:12]}"
    admin = sa.create_engine(server, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    url = server.set(database=name).render_as_string(hide_password=False)
    cfg = Config()
    cfg.set_main_option("script_location", str(Path(__file__).resolve().parents[2] / "backend" / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    command.upgrade(cfg, "head")
    engine = sa.create_engine(url)
    Session = sessionmaker(bind=engine)

    def override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    original = marker_tags.segment_objects
    marker_tags.segment_objects = blob_segmenter()
    marker_tags._trackers.clear()
    marker_tags._lookups.clear()
    try:
        yield TestClient(app)
    finally:
        marker_tags.segment_objects = original
        app.dependency_overrides.pop(get_db, None)
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def _b64(frame: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".png", frame)[1].tobytes()).decode()


@needs_database
def test_api_mapping_is_persisted_and_conflicts_are_refused(client):
    bottle = client.post("/api/assets", json={"name": "BOTTLE-001"}).json()["id"]
    cup = client.post("/api/assets", json={"name": "CUP-001"}).json()["id"]

    assert client.get("/api/marker-tags/suggest").json()["ids"] == [0, 1, 2, 3]
    saved = client.put(f"/api/marker-tags/sets/{bottle}", json={"top_left": 0, "top_right": 1, "bottom_right": 2, "bottom_left": 3})
    assert saved.status_code == 200, saved.text
    assert client.get("/api/marker-tags/suggest").json()["ids"] == [4, 5, 6, 7]

    clash = client.put(f"/api/marker-tags/sets/{cup}", json={"top_left": 3, "top_right": 5, "bottom_right": 6, "bottom_left": 7})
    assert clash.status_code == 409 and "BOTTLE-001" in clash.json()["detail"]
    same = client.put(f"/api/marker-tags/sets/{cup}", json={"top_left": 5, "top_right": 5, "bottom_right": 6, "bottom_left": 7})
    assert same.status_code == 400

    sets = client.get("/api/marker-tags/sets").json()
    assert [(s["asset_name"], s["top_left"], s["bottom_left"]) for s in sets] == [("BOTTLE-001", 0, 3)]
    assert client.get("/api/marker-tags/config").json()["used_ids"] == [0, 1, 2, 3]

    png = client.get(f"/api/marker-tags/sets/{bottle}/sheet.png")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    assert client.get("/api/marker-tags/markers/0.png?download=true").headers["content-disposition"].startswith("attachment")

    assert client.delete(f"/api/marker-tags/sets/{bottle}").status_code == 204
    assert client.get("/api/marker-tags/sets").json() == []


@needs_database
def test_api_identify_and_track(client):
    bottle = client.post("/api/assets", json={"name": "BOTTLE-001"}).json()["id"]
    client.put(f"/api/marker-tags/sets/{bottle}", json={"top_left": 0, "top_right": 1, "bottom_right": 2, "bottom_left": 3})

    one_tag = _b64(tagged_frame(skip=(TR, BR, BL)))
    first = client.post("/api/marker-tags/identify", json={"session_id": "s", "image_base64": one_tag}).json()
    assert first["state"] == "IDENTIFYING"
    second = client.post("/api/marker-tags/identify", json={"session_id": "s", "image_base64": one_tag}).json()
    assert second["state"] == "IDENTIFIED" and second["product"]["asset_name"] == "BOTTLE-001"
    assert second["product"]["model"] is None and second["product"]["procedures"] == []

    frame = _frame_with_bottle()
    cv2.circle(frame, (1180, 375), 80, BLUE, -1)  # a second bottle, away from the tags
    states = []
    for _ in range(3):
        r = client.post("/api/marker-tags/track", json={"session_id": "t", "image_base64": _b64(frame)}).json()
        states.append(r["state"])
    # Nothing but frames: tags -> arrangement -> object -> tracking, no other call needed.
    assert states == ["VERIFYING", "VERIFYING", "TRACKING"]
    assert r["arrangement"] == "valid"
    assert [(c["corner"], c["marker_id"], c["status"]) for c in r["corners"]] == [
        ("top_left", 0, "seen"),
        ("top_right", 1, "seen"),
        ("bottom_right", 2, "seen"),
        ("bottom_left", 3, "seen"),
    ]
    assert r["corners"][0]["center_px"] == pytest.approx({"x": 334.5, "y": 184.5}, abs=1.5)
    assert r["tracking"] and r["product"]["asset_name"] == "BOTTLE-001"
    assert len(r["region"]) == 4 and len(r["search_area"]) == 4 and r["rejected"] == []
    centre_x = (r["object"]["bbox"][0] + r["object"]["bbox"][2]) / 2
    assert centre_x == pytest.approx(600, abs=3)  # the bottle between the tags, not the one at x=1150

    incomplete = client.post("/api/marker-tags/track", json={"session_id": "u", "image_base64": _b64(tagged_frame(skip=(BL,)))}).json()
    assert incomplete["state"] == "INCOMPLETE" and incomplete["missing_corners"] == ["bottom_left"]
    assert incomplete["arrangement"] == "incomplete" and incomplete["object"] is None and incomplete["yolo_ms"] is None
    assert [c["status"] for c in incomplete["corners"]] == ["seen", "seen", "seen", "missing"]
    swapped = client.post(
        "/api/marker-tags/track",
        json={"session_id": "v", "image_base64": _b64(tagged_frame(ids={TL: 1, TR: 0, BR: 2, BL: 3}))},
    ).json()
    assert swapped["state"] == "LAYOUT_INVALID" and swapped["arrangement"] == "invalid"
    assert swapped["region"] is not None and swapped["yolo_ms"] is None  # drawn, but never searched
    assert client.delete("/api/marker-tags/track/t").status_code == 204
