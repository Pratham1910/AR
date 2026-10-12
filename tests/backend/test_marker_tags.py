"""
Four-corner ArUco product tags: detection on synthetic frames (so no printed
tags or camera are needed), product identification, YOLO verification on
hand-made detections, the scan's memory between frames, and the mapping rules.
"""

import cv2
import numpy as np
import pytest

from app.models.enums import MarkerCorner
from app.schemas.vision import BoundingBox, SegmentedObject
from app.services.markers.identification import (
    IdentificationStatus,
    LayoutLimits,
    TagAssignment,
    TagLookupStatus,
    identify_product,
    lookup_product_by_tag,
)
from app.services.markers.registry import TagSetError, suggest_free_ids, validate_tag_set
from app.services.markers.scan_session import ScanSession
from app.services.markers.tags import (
    MarkerDetection,
    detect_tags,
    dictionary_size,
    render_sheet_png,
    render_tag_png,
)
from app.services.markers.verification import (
    VerificationStatus,
    crop_to_region,
    to_frame_coordinates,
    verify_product,
)
from app.services.pose.aruco_pose import ArucoPoseEstimator

DICTIONARY = "DICT_4X4_50"
TL, TR, BL, BR = MarkerCorner.TOP_LEFT, MarkerCorner.TOP_RIGHT, MarkerCorner.BOTTOM_LEFT, MarkerCorner.BOTTOM_RIGHT

# Two products: PUMP has tags 0-3, VALVE 4-7.
REGISTRY = {
    0: TagAssignment("pump", TL),
    1: TagAssignment("pump", TR),
    2: TagAssignment("pump", BL),
    3: TagAssignment("pump", BR),
    4: TagAssignment("valve", TL),
    5: TagAssignment("valve", TR),
    6: TagAssignment("valve", BL),
    7: TagAssignment("valve", BR),
}

detect = ArucoPoseEstimator(DICTIONARY, 0.05).detect_markers


def _tag(marker_id: int, x: float, y: float, side: float = 40.0) -> MarkerDetection:
    """A tag as the detector would report it: square, top-left corner at (x, y)."""
    return MarkerDetection(marker_id, np.array([[x, y], [x + side, y], [x + side, y + side], [x, y + side]], dtype=float))


def _set(ids: tuple[int, int, int, int], x: float = 100, y: float = 100, w: float = 300, h: float = 200) -> list[MarkerDetection]:
    """Four tags at the corners of a w x h rectangle, ids given as (TL, TR, BL, BR)."""
    tl, tr, bl, br = ids
    return [_tag(tl, x, y), _tag(tr, x + w, y), _tag(bl, x, y + h), _tag(br, x + w, y + h)]


def _scene(placements: dict[int, tuple[int, int]], tag_px: int = 80, size: tuple[int, int] = (480, 640)) -> np.ndarray:
    """A light frame with generated tags pasted at the given top-left positions."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    frame = np.full((*size, 3), 220, dtype=np.uint8)
    for marker_id, (x, y) in placements.items():
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, tag_px)
        frame[y : y + tag_px, x : x + tag_px] = marker[..., None]
    return frame


FOUR_CORNERS = {0: (60, 40), 1: (500, 40), 2: (60, 360), 3: (500, 360)}


# --- Detection --------------------------------------------------------------


def test_detects_all_four_tags_in_a_frame():
    tags, duplicates = detect_tags(detect, _scene(FOUR_CORNERS))
    assert sorted(t.marker_id for t in tags) == [0, 1, 2, 3]
    assert duplicates == []
    assert tags[0].side_px == pytest.approx(80, abs=2)


@pytest.mark.parametrize("angle_deg", [30, 90, 180, 250])
def test_detects_tags_on_a_rotated_product(angle_deg):
    frame = _scene(FOUR_CORNERS, size=(640, 640))
    rotation = cv2.getRotationMatrix2D((320, 320), angle_deg, 0.7)
    rotated = cv2.warpAffine(frame, rotation, (640, 640), borderValue=(220, 220, 220))
    tags, _ = detect_tags(detect, rotated)
    assert sorted(t.marker_id for t in tags) == [0, 1, 2, 3]
    # ...and the rotated set still identifies the product: order is checked
    # by the direction of travel around the outline, not by image position.
    assert identify_product(tags, REGISTRY).status is IdentificationStatus.CANDIDATE


def test_detects_tags_seen_at_an_angle():
    frame = _scene(FOUR_CORNERS)
    flat = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])
    tilted = np.float32([[90, 60], [560, 20], [620, 460], [40, 420]])  # far edge shorter: viewed from the side
    warped = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(flat, tilted), (640, 480), borderValue=(220, 220, 220))
    tags, _ = detect_tags(detect, warped)
    assert sorted(t.marker_id for t in tags) == [0, 1, 2, 3]
    assert identify_product(tags, REGISTRY).status is IdentificationStatus.CANDIDATE


def test_detects_tags_in_dim_uneven_light():
    frame = _scene(FOUR_CORNERS).astype(np.float32)
    shade = np.linspace(0.12, 0.6, frame.shape[1], dtype=np.float32)[None, :, None]  # dark on the left, brighter right
    dim = np.clip(frame * shade, 0, 255).astype(np.uint8)
    tags, _ = detect_tags(detect, dim)
    assert sorted(t.marker_id for t in tags) == [0, 1, 2, 3]


def test_reports_a_tag_id_that_appears_twice():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    frame = _scene({0: (60, 40), 1: (500, 40)})
    frame[360:440, 60:140] = cv2.aruco.generateImageMarker(dictionary, 1, 80)[..., None]  # a second copy of tag 1
    tags, duplicates = detect_tags(detect, frame)
    assert duplicates == [1]
    assert sorted(t.marker_id for t in tags) == [0, 1]


def test_a_partly_covered_tag_is_not_reported():
    frame = _scene(FOUR_CORNERS)
    frame[360:440, 500:545] = 220  # hide over half of tag 3
    tags, _ = detect_tags(detect, frame)
    assert sorted(t.marker_id for t in tags) == [0, 1, 2]


def test_generated_tags_are_printable_and_readable():
    tile = cv2.imdecode(np.frombuffer(render_tag_png(DICTIONARY, 7, 300, "PUMP Top-Left id 7"), np.uint8), cv2.IMREAD_COLOR)
    assert [marker_id for marker_id, _ in detect(tile)] == [7]

    sheet_png = render_sheet_png(DICTIONARY, {TL: 10, TR: 11, BL: 12, BR: 13}, 250, "PUMP-001")
    sheet = cv2.imdecode(np.frombuffer(sheet_png, np.uint8), cv2.IMREAD_COLOR)
    tags, _ = detect_tags(detect, sheet)
    assert sorted(t.marker_id for t in tags) == [10, 11, 12, 13]
    # The sheet lays the tags out in their corner positions, but an uncut
    # sheet shown to the camera is not a product: the tags touch each other.
    registry = {10: TagAssignment("p", TL), 11: TagAssignment("p", TR), 12: TagAssignment("p", BL), 13: TagAssignment("p", BR)}
    on_sheet = identify_product(tags, registry)
    assert on_sheet.status is IdentificationStatus.LAYOUT_INVALID
    assert "too close" in on_sheet.layout_problem
    # Only spacing is wrong — the corner order on the sheet is the registered one.
    assert identify_product(tags, registry, LayoutLimits(min_side_in_tags=1.0)).status is IdentificationStatus.CANDIDATE


def test_dictionary_size():
    assert dictionary_size("DICT_4X4_50") == 50
    assert dictionary_size("DICT_6X6_250") == 250


# --- Identification ---------------------------------------------------------


def test_four_tags_in_their_corners_identify_the_product():
    result = identify_product(_set((0, 1, 2, 3)), REGISTRY)
    assert result.status is IdentificationStatus.CANDIDATE
    assert result.asset_id == "pump"
    # Quad is TL, TR, BR, BL tag centers.
    assert result.quad.tolist() == [[120, 120], [420, 120], [420, 320], [120, 320]]
    assert result.tag_side_px == pytest.approx(40)


def test_nothing_in_view():
    assert identify_product([], REGISTRY).status is IdentificationStatus.NO_MARKERS


def test_only_unregistered_tags():
    result = identify_product([_tag(30, 10, 10), _tag(31, 200, 10)], REGISTRY)
    assert result.status is IdentificationStatus.UNKNOWN_MARKERS
    assert result.unknown_ids == [30, 31]


def test_three_of_four_tags_is_incomplete_and_names_the_missing_corner():
    result = identify_product(_set((0, 1, 2, 3))[:3], REGISTRY)
    assert result.status is IdentificationStatus.INCOMPLETE
    assert result.asset_id == "pump"
    assert result.missing_corners == [BR]
    assert result.quad is None


def test_tags_of_two_products_are_never_combined_into_one():
    # Two pump tags + two valve tags at the four corners of one rectangle.
    mixed = [_tag(0, 100, 100), _tag(1, 400, 100), _tag(6, 100, 300), _tag(7, 400, 300)]
    result = identify_product(mixed, REGISTRY)
    assert result.status is IdentificationStatus.MIXED_PRODUCTS
    assert result.quad is None
    assert set([result.asset_id, *result.other_asset_ids]) == {"pump", "valve"}


def test_a_complete_set_with_another_products_tag_nearby_is_still_a_candidate():
    result = identify_product(_set((0, 1, 2, 3)) + [_tag(4, 700, 100)], REGISTRY)
    assert result.status is IdentificationStatus.CANDIDATE
    assert result.asset_id == "pump"
    assert result.other_asset_ids == ["valve"]


def test_another_products_tag_inside_the_set_blocks_it():
    result = identify_product(_set((0, 1, 2, 3)) + [_tag(4, 250, 200)], REGISTRY)
    assert result.status is IdentificationStatus.MIXED_PRODUCTS
    assert result.other_asset_ids == ["valve"]


def test_two_complete_products_in_view_are_not_resolved_by_guessing():
    result = identify_product(_set((0, 1, 2, 3)) + _set((4, 5, 6, 7), x=600), REGISTRY)
    assert result.status is IdentificationStatus.MULTIPLE_PRODUCTS
    assert result.asset_id is None
    assert result.other_asset_ids == ["pump", "valve"]


@pytest.mark.parametrize(
    "ids",
    [
        (1, 0, 2, 3),  # top two swapped
        (0, 1, 3, 2),  # bottom two swapped
        (3, 1, 2, 0),  # diagonal swapped
        (1, 0, 3, 2),  # mirrored left-right
    ],
)
def test_tags_on_the_wrong_corners_are_rejected(ids):
    result = identify_product(_set(ids), REGISTRY)
    assert result.status is IdentificationStatus.LAYOUT_INVALID
    assert "registered corners" in result.layout_problem


def test_one_tag_much_larger_than_the_others_is_rejected():
    tags = _set((0, 1, 2, 3))
    tags[3] = _tag(3, 400, 300, side=130)  # e.g. a different printout held closer to the camera
    result = identify_product(tags, REGISTRY)
    assert result.status is IdentificationStatus.LAYOUT_INVALID
    assert "apparent size" in result.layout_problem


def test_tags_bunched_together_are_rejected():
    result = identify_product(_set((0, 1, 2, 3), w=50, h=50), REGISTRY)
    assert result.status is IdentificationStatus.LAYOUT_INVALID
    assert "too close" in result.layout_problem


def test_layout_limits_are_configurable():
    tags = _set((0, 1, 2, 3), w=50, h=50)
    relaxed = LayoutLimits(min_side_in_tags=1.0)
    assert identify_product(tags, REGISTRY, relaxed).status is IdentificationStatus.CANDIDATE


def test_a_duplicated_registered_tag_blocks_identification():
    result = identify_product(_set((0, 1, 2, 3)), REGISTRY, duplicate_ids=[2])
    assert result.status is IdentificationStatus.DUPLICATE_MARKERS
    assert result.duplicate_ids == [2]


# --- Product scan: lookup by any one tag --------------------------------------


@pytest.mark.parametrize("marker_id, product", [(0, "pump"), (1, "pump"), (2, "pump"), (3, "pump"), (6, "valve")])
def test_any_one_registered_tag_names_its_product(marker_id, product):
    result = lookup_product_by_tag([_tag(marker_id, 100, 100)], REGISTRY)
    assert result.status is TagLookupStatus.FOUND
    assert result.asset_id == product
    assert result.matched_ids == [marker_id]


def test_several_tags_of_one_product_name_it():
    result = lookup_product_by_tag(_set((0, 1, 2, 3))[:2], REGISTRY)
    assert result.status is TagLookupStatus.FOUND and result.asset_id == "pump" and result.matched_ids == [0, 1]


def test_an_unknown_tag_selects_no_product():
    result = lookup_product_by_tag([_tag(30, 100, 100)], REGISTRY)
    assert result.status is TagLookupStatus.UNKNOWN_MARKERS
    assert result.asset_id is None and result.unknown_ids == [30]


def test_an_unknown_tag_next_to_a_known_one_does_not_change_the_answer():
    result = lookup_product_by_tag([_tag(30, 100, 100), _tag(5, 300, 100)], REGISTRY)
    assert result.status is TagLookupStatus.FOUND and result.asset_id == "valve" and result.unknown_ids == [30]


def test_tags_of_two_products_are_ambiguous_not_a_pick():
    result = lookup_product_by_tag([_tag(0, 100, 100), _tag(4, 300, 100)], REGISTRY)
    assert result.status is TagLookupStatus.AMBIGUOUS
    assert result.asset_id is None and result.asset_ids == ["pump", "valve"]


def test_lookup_with_nothing_in_view():
    assert lookup_product_by_tag([], REGISTRY).status is TagLookupStatus.NO_MARKERS


@pytest.mark.parametrize("count", [1, 2, 3])
def test_what_satisfies_the_lookup_never_satisfies_tracking(count):
    """The two modes are separate: one to three tags name a product for the
    product scan, and are still not a candidate for live tracking."""
    tags = _set((0, 1, 2, 3))[:count]
    assert lookup_product_by_tag(tags, REGISTRY).status is TagLookupStatus.FOUND
    strict = identify_product(tags, REGISTRY)
    assert strict.status is IdentificationStatus.INCOMPLETE
    assert strict.quad is None and strict.region is None


# --- YOLO verification ------------------------------------------------------

# The area enclosed by PUMP's four tags as _set() places them: their outer corners.
REGION = np.array([[100, 100], [440, 100], [440, 340], [100, 340]], dtype=float)
CLASSES = ["bottle", "cup", "person"]


def _object(class_label: str, x1: float, y1: float, x2: float, y2: float, confidence: float = 0.9) -> SegmentedObject:
    return SegmentedObject(
        class_label=class_label, confidence=confidence, bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2), polygon=[]
    )


def test_the_region_is_the_outline_around_all_four_tags():
    result = identify_product(_set((0, 1, 2, 3)), REGISTRY)
    xs, ys = result.region[:, 0], result.region[:, 1]
    # From the tags' OUTER edges (tags are 40px, placed at 100..440 x 100..340), not their centers.
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (100, 440, 100, 340)
    assert cv2.contourArea(result.region.astype(np.float32)) == pytest.approx(340 * 240)


def test_region_follows_a_tilted_product():
    tags = [_tag(0, 200, 60), _tag(1, 460, 160), _tag(3, 380, 400), _tag(2, 110, 300)]  # a rotated rectangle
    region = identify_product(tags, REGISTRY).region
    inside = lambda x, y: cv2.pointPolygonTest(region.astype(np.float32), (x, y), False) >= 0
    assert inside(300, 240)  # the middle of the product
    assert not inside(120, 80) and not inside(480, 420)  # corners of the bounding box, outside the tags


def test_detector_only_gets_the_pixels_inside_the_tags():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = (0, 0, 255)  # everything red...
    region = np.array([[200, 60], [500, 160], [420, 440], [110, 340]], dtype=float)  # a tilted quadrilateral
    crop, (x0, y0) = crop_to_region(frame, region)

    assert (x0, y0) == (110, 60)
    assert crop.shape[:2] == (381, 391)  # the region's bounding rectangle, not the frame
    # ...but in what the detector receives, red survives only inside the region.
    red = np.all(crop == (0, 0, 255), axis=-1)
    blank = np.all(crop == 114, axis=-1)
    assert np.all(red | blank)
    for x, y, expect_red in [(300, 250, True), (115, 65, False), (495, 435, False), (480, 170, True)]:
        assert red[y - y0, x - x0] == expect_red
    assert red.sum() == pytest.approx(cv2.contourArea(region.astype(np.float32)), rel=0.02)
    assert np.all(frame == (0, 0, 255))  # the caller's frame is left untouched


def test_region_at_the_frame_edge_is_clipped():
    frame = np.full((100, 100, 3), 50, dtype=np.uint8)
    crop, origin = crop_to_region(frame, np.array([[-20, -10], [60, -10], [60, 70], [-20, 70]], dtype=float))
    assert origin == (0, 0) and crop.shape[:2] == (71, 61)


def test_detections_on_the_crop_are_moved_back_into_the_frame():
    moved = to_frame_coordinates([_object("bottle", 10, 20, 50, 80)], (110, 60))[0]
    assert (moved.bbox.x1, moved.bbox.y1, moved.bbox.x2, moved.bbox.y2) == (120, 80, 160, 140)


def test_expected_object_inside_the_tags_is_verified():
    result = verify_product("bottle", CLASSES, [_object("bottle", 200, 150, 340, 300)], REGION)
    assert result.status is VerificationStatus.VERIFIED
    assert result.inside == pytest.approx(1.0)


def test_a_different_object_inside_the_tags_is_a_wrong_object():
    result = verify_product("bottle", CLASSES, [_object("cup", 200, 150, 340, 300)], REGION)
    assert result.status is VerificationStatus.WRONG_OBJECT
    assert result.obj.class_label == "cup"


def test_expected_object_outside_the_tags_does_not_verify():
    # Even if such a detection were handed in, an object elsewhere in the frame can't verify the product.
    result = verify_product("bottle", CLASSES, [_object("bottle", 700, 150, 840, 300)], REGION)
    assert result.status is VerificationStatus.NOT_FOUND


def test_a_large_object_that_merely_contains_the_tags_does_not_verify():
    # e.g. a "person" expected class with someone standing behind the tagged area.
    behind = _object("person", 0, 0, 900, 700)  # the region is ~13% of this box
    assert verify_product("person", CLASSES, [behind], REGION).status is VerificationStatus.NOT_FOUND


def test_wrong_object_inside_is_not_excused_by_the_right_one_outside():
    objects = [_object("bottle", 700, 150, 840, 300), _object("cup", 200, 150, 340, 300)]
    assert verify_product("bottle", CLASSES, objects, REGION).status is VerificationStatus.WRONG_OBJECT


def test_right_object_inside_wins_over_other_objects_inside():
    objects = [_object("cup", 120, 120, 200, 200), _object("bottle", 200, 150, 340, 300)]
    assert verify_product("bottle", CLASSES, objects, REGION).status is VerificationStatus.VERIFIED


def test_nothing_detected_is_not_found():
    assert verify_product("bottle", CLASSES, [], REGION).status is VerificationStatus.NOT_FOUND


def test_a_product_without_a_detection_class_cannot_be_verified():
    assert verify_product(None, CLASSES, [_object("bottle", 200, 150, 340, 300)], REGION).status is VerificationStatus.NO_CLASS


def test_a_class_the_detector_does_not_know_cannot_be_verified():
    result = verify_product("pcb", CLASSES, [_object("bottle", 200, 150, 340, 300)], REGION)
    assert result.status is VerificationStatus.UNKNOWN_CLASS


def test_object_straddling_the_edge_of_the_region_needs_most_of_itself_inside():
    half_in = _object("bottle", 370, 150, 510, 300)  # half of the box is inside the region
    assert verify_product("bottle", CLASSES, [half_in], REGION, min_inside=0.6).status is VerificationStatus.NOT_FOUND
    assert verify_product("bottle", CLASSES, [half_in], REGION, min_inside=0.4).status is VerificationStatus.VERIFIED


# --- Scan session -----------------------------------------------------------


def test_a_tag_lost_for_a_moment_is_held_then_expires():
    session = ScanSession(hold_ms=500, confirm_frames=3)
    full = _set((0, 1, 2, 3))
    assert len(session.with_held(full, now_ms=0)) == 4

    with_one_missing = session.with_held(full[:3], now_ms=200)
    assert sorted(t.marker_id for t in with_one_missing) == [0, 1, 2, 3]
    assert [t.marker_id for t in with_one_missing if t.held] == [3]
    assert identify_product(with_one_missing, REGISTRY).status is IdentificationStatus.CANDIDATE

    expired = session.with_held(full[:3], now_ms=600)
    assert sorted(t.marker_id for t in expired) == [0, 1, 2]
    assert identify_product(expired, REGISTRY).status is IdentificationStatus.INCOMPLETE


def test_confirmation_needs_consecutive_verified_frames_of_one_product():
    session = ScanSession(confirm_frames=3)
    assert session.record("pump") == 1 and not session.is_confirmed()
    assert session.record("pump") == 2 and not session.is_confirmed()
    assert session.record(None) == 0  # one failed frame starts over
    assert session.record("pump") == 1
    assert session.record("valve") == 1  # a different product starts over too
    session.record("valve")
    assert session.record("valve") == 3 and session.is_confirmed()
    assert session.record(None) == 0 and not session.is_confirmed()


# --- Mapping rules ----------------------------------------------------------

TAKEN = {0: ("pump", "PUMP-001"), 1: ("pump", "PUMP-001"), 2: ("pump", "PUMP-001"), 3: ("pump", "PUMP-001")}


def test_a_fresh_set_of_four_ids_is_accepted():
    validate_tag_set({TL: 4, TR: 5, BL: 6, BR: 7}, "valve", TAKEN, 50)


def test_a_product_may_rearrange_its_own_ids():
    validate_tag_set({TL: 3, TR: 2, BL: 1, BR: 0}, "pump", TAKEN, 50)


def test_an_id_of_another_product_is_a_conflict_that_names_it():
    with pytest.raises(TagSetError) as error:
        validate_tag_set({TL: 4, TR: 5, BL: 2, BR: 7}, "valve", TAKEN, 50)
    assert error.value.conflict
    assert "2 (PUMP-001)" in str(error.value)


def test_the_same_id_on_two_corners_is_rejected():
    with pytest.raises(TagSetError) as error:
        validate_tag_set({TL: 4, TR: 4, BL: 6, BR: 7}, "valve", TAKEN, 50)
    assert not error.value.conflict
    assert "more than once" in str(error.value)


def test_ids_outside_the_dictionary_are_rejected():
    with pytest.raises(TagSetError, match="0 to 49"):
        validate_tag_set({TL: 4, TR: 5, BL: 6, BR: 50}, "valve", TAKEN, 50)
    with pytest.raises(TagSetError):
        validate_tag_set({TL: -1, TR: 5, BL: 6, BR: 7}, "valve", TAKEN, 50)


def test_a_set_must_cover_every_corner():
    with pytest.raises(TagSetError, match="bottom_right"):
        validate_tag_set({TL: 4, TR: 5, BL: 6}, "valve", TAKEN, 50)


def test_suggested_ids_skip_used_ones_and_run_out_gracefully():
    assert suggest_free_ids({0, 1, 2, 3, 5}, 50) == [4, 6, 7, 8]
    assert suggest_free_ids(set(range(48)), 50) == [48, 49]  # fewer than four left
