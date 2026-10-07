"""
Automatic product scan: product verification (marker identity vs what the
object detector sees), positioning guidance (where the product is relative to
the scan box, and which way to move it — including on a mirrored preview) and
the scan gate (stability, confirmation, re-arming).
Pure logic on hand-built markers/detections — no camera, model or database.
"""

import math

import pytest

from app.services.markers.detector import MarkerDetection
from app.services.scan.positioning import ScanGateConfig, analyze_position, scan_area
from app.services.scan.product_verifier import ProductDetection, Verification, VerificationStatus, verify_product
from app.services.scan.scan_gate import ScanGate, ScanObservation, ScanState

W, H = 1280, 720
CONFIG = ScanGateConfig()  # scan box 768 x 576 px, centered on (640, 360)
CLASSES = ["bottle", "cup", "bus"]


def _marker(
    cx: float = 640, cy: float = 360, side: float = 120, marker_id: int = 1, squash: float = 1.0, roll_deg: float = 0.0
) -> MarkerDetection:
    """A marker facing the camera; squash < 1 narrows it as if turned away, roll spins it in the image."""
    hw, hh = side * squash / 2, side / 2
    cos, sin = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return MarkerDetection(marker_id, tuple((cx + x * cos - y * sin, cy + x * sin + y * cos) for x, y in corners))


def _box(
    label: str = "bottle", confidence: float = 0.9, around: tuple[float, float] = (640, 360), half: float = 200
) -> ProductDetection:
    x, y = around
    return ProductDetection(label, confidence, (x - half, y - half, x + half, y + half))


def _verify(detections, expected="bottle", marker=None, **kwargs) -> Verification:
    return verify_product(marker or _marker(), expected, CLASSES, detections, 0.5, **kwargs)


def _match(around: tuple[float, float] = (640, 360), half: float = 200) -> Verification:
    return Verification(VerificationStatus.MATCH, "bottle", _box(around=around, half=half))


def _obs(marker=None, count=1, known=True, verification=None, mirrored=False) -> ScanObservation:
    marker = marker if marker is not None or count != 1 else _marker()
    return ScanObservation(
        W,
        H,
        marker_count=count,
        marker=marker if count == 1 else None,
        known=known,
        verification=verification or _match(),
        mirrored=mirrored,
    )


def _hints(marker=None, product=None, mirrored=False, config=CONFIG) -> list[str]:
    box = product.bbox if product is not None else None
    return list(analyze_position(marker or _marker(), box, W, H, config, mirrored).hints)


def _product_at(x: float, y: float, half: float = 150):
    """A product box at (x, y) with its marker on it."""
    return {"marker": _marker(cx=x, cy=y, side=100), "product": _box(around=(x, y), half=half)}


# --- product verification ---


def test_expected_class_at_the_marker_is_a_match():
    result = _verify([_box("cup", 0.95), _box("bottle", 0.8)])

    assert result.status is VerificationStatus.MATCH
    assert result.detection.class_label == "bottle"


def test_other_class_at_the_marker_is_a_mismatch_naming_what_was_seen():
    result = _verify([_box("bus")])

    assert (result.status, result.expected_class, result.detection.class_label) == (
        VerificationStatus.MISMATCH,
        "bottle",
        "bus",
    )


def test_class_names_compare_case_insensitively():
    assert _verify([_box("Bottle")], expected="BOTTLE").status is VerificationStatus.MATCH


def test_nothing_at_the_marker_is_no_product():
    assert _verify([]).status is VerificationStatus.NO_PRODUCT
    # The right class elsewhere in the frame is not this marker's product.
    assert _verify([_box("bottle", around=(150, 150), half=60)]).status is VerificationStatus.NO_PRODUCT


def test_marker_need_not_be_centered_on_or_even_inside_the_product_box():
    on_its_corner = _marker(cx=460, cy=180)
    assert _verify([_box("bottle")], marker=on_its_corner).status is VerificationStatus.MATCH
    # Marker center 80 px outside the box edge, within one marker side (120 px).
    assert _verify([_box("bottle", around=(360, 360), half=200)]).status is VerificationStatus.MATCH


def test_association_reach_is_configurable():
    beside = [_box("bottle", around=(360, 360), half=200)]  # 80 px from the marker center

    assert _verify(beside, reach=0.5).status is VerificationStatus.NO_PRODUCT  # 60 px
    assert _verify(beside, reach=1.0).status is VerificationStatus.MATCH  # 120 px


def test_weak_sighting_of_the_expected_class_is_uncertain_not_a_match():
    assert _verify([_box("bottle", 0.3)]).status is VerificationStatus.UNCERTAIN
    # A weak sighting of some other class proves nothing either way.
    assert _verify([_box("bus", 0.3)]).status is VerificationStatus.NO_PRODUCT
    # A confident other class still rejects, however weakly the expected one also shows.
    assert _verify([_box("bottle", 0.3), _box("bus", 0.9)]).status is VerificationStatus.MISMATCH


def test_ignored_classes_are_not_evidence():
    assert _verify([_box("person")], ignored_classes=frozenset({"person"})).status is VerificationStatus.NO_PRODUCT
    both = [_box("person"), _box("bottle")]
    assert _verify(both, ignored_classes=frozenset({"person"})).status is VerificationStatus.MATCH


def test_product_the_model_cannot_recognise_is_unverifiable_not_matched():
    assert _verify([_box("bottle")], expected=None).status is VerificationStatus.UNVERIFIABLE
    assert _verify([_box("bottle")], expected="pump").status is VerificationStatus.UNVERIFIABLE


# --- positioning: centering ---


def test_scan_box_is_centered_and_sized_from_config():
    assert scan_area(W, H, CONFIG) == pytest.approx((256, 72, 1024, 648))
    assert scan_area(W, H, ScanGateConfig(area_width=0.5, area_height=0.5)) == pytest.approx((320, 180, 960, 540))


def test_well_placed_product_needs_no_guidance():
    position = analyze_position(_marker(), _box().bbox, W, H, CONFIG)

    assert position.hints == ()
    assert (position.distance, position.inside_area, position.complete) == ("ok", True, True)
    assert (position.offset_x, position.offset_y) == pytest.approx((0, 0))


def test_product_left_of_target_is_told_to_move_right():
    assert _hints(**_product_at(360, 360)) == ["move_right"]


def test_product_right_of_target_is_told_to_move_left():
    assert _hints(**_product_at(920, 360)) == ["move_left"]


def test_product_above_target_is_told_to_move_down():
    assert _hints(**_product_at(640, 230)) == ["move_down"]


def test_product_below_target_is_told_to_move_up():
    assert _hints(**_product_at(640, 490)) == ["move_up"]


def test_diagonal_offset_gives_both_directions():
    assert _hints(**_product_at(920, 230)) == ["move_left", "move_down"]


def test_offsets_are_measured_from_the_scan_box_center_in_box_sizes():
    position = analyze_position(_marker(cx=360, cy=490, side=100), _box(around=(360, 490), half=150).bbox, W, H, CONFIG)

    assert position.target_center == pytest.approx((640, 360))
    assert position.product_center == (360.0, 490.0)
    assert round(position.offset_x, 4) == round(-280 / 768, 4)  # negative = left in the camera image
    assert round(position.offset_y, 4) == round(130 / 576, 4)  # positive = below


def test_small_offsets_within_tolerance_are_left_alone():
    assert _hints(**_product_at(640 + 100, 360 - 70)) == []  # 0.13 and 0.12 of the box, tolerance 0.15
    tighter = ScanGateConfig(center_tolerance=0.05)
    assert _hints(**_product_at(640 + 100, 360 - 70), config=tighter) == ["move_left", "move_down"]


def test_the_product_box_is_what_gets_centered_not_the_marker():
    # Marker on the product's top-left corner, product itself centered: nothing to fix.
    assert _hints(marker=_marker(cx=500, cy=230, side=80), product=_box()) == []
    # Marker dead center, but the product it sits on is off to the right.
    assert _hints(marker=_marker(side=80), product=_box(around=(820, 360), half=190)) == ["move_left"]


def test_marker_alone_is_centered_until_a_product_is_found():
    assert _hints(marker=_marker(cx=200)) == ["move_right"]
    assert _hints(marker=_marker(cx=1100)) == ["move_left"]


# --- positioning: mirrored preview ---


def test_mirrored_preview_swaps_left_and_right_only():
    # Left in the camera image is the RIGHT side of a mirrored preview, so to
    # reach the center the user sees it travel left.
    assert _hints(**_product_at(360, 360), mirrored=True) == ["move_left"]
    assert _hints(**_product_at(920, 360), mirrored=True) == ["move_right"]
    assert _hints(**_product_at(640, 230), mirrored=True) == ["move_down"]
    assert _hints(**_product_at(640, 490), mirrored=True) == ["move_up"]


def test_hint_always_points_from_where_the_user_sees_it_toward_the_center():
    for mirrored in (False, True):
        for image_x in (360, 920):
            seen_x = W - image_x if mirrored else image_x  # where it appears in the preview
            (hint,) = _hints(**_product_at(image_x, 360), mirrored=mirrored)
            assert hint == ("move_right" if seen_x < W / 2 else "move_left")


def test_mirroring_changes_no_measurement():
    plain = analyze_position(_marker(cx=360), _box(around=(360, 360)).bbox, W, H, CONFIG, mirrored=False)
    flipped = analyze_position(_marker(cx=360), _box(around=(360, 360)).bbox, W, H, CONFIG, mirrored=True)

    assert (plain.offset_x, plain.product_center, plain.distance) == (
        flipped.offset_x,
        flipped.product_center,
        flipped.distance,
    )


# --- positioning: apparent size, completeness, orientation ---


def test_product_too_small_is_told_to_move_closer():
    position = analyze_position(_marker(side=30), _box(half=60).bbox, W, H, CONFIG)

    assert (position.distance, position.hints) == ("too_far", ("move_closer",))


def test_product_too_large_is_told_to_move_farther():
    bigger_than_box = analyze_position(_marker(), _box(half=320).bbox, W, H, CONFIG)  # 640 px tall, box is 576
    huge_marker = analyze_position(_marker(side=400), None, W, H, CONFIG)

    assert (bigger_than_box.distance, bigger_than_box.hints) == ("too_close", ("move_farther",))
    assert (huge_marker.distance, huge_marker.hints) == ("too_close", ("move_farther",))


def test_small_marker_is_fine_once_the_product_fills_the_box():
    # A big product with a small marker: moving closer would push it out of the box.
    assert _hints(marker=_marker(side=30), product=_box(half=200)) == []
    assert _hints(marker=_marker(side=30)) == ["move_closer"]  # no product box to go by


def test_size_limits_are_configurable():
    strict = ScanGateConfig(min_marker_size=0.2, max_product_fill=0.5)

    assert _hints(marker=_marker(side=120), product=_box(half=100), config=strict) == ["move_closer"]
    assert _hints(marker=_marker(side=160), product=_box(half=200), config=strict) == ["move_farther"]


def test_product_cut_off_by_the_frame_edge_is_asked_to_be_shown_completely():
    cut_off = ProductDetection("bottle", 0.9, (0.0, 200.0, 420.0, 600.0))

    position = analyze_position(_marker(cx=300, cy=400), cut_off.bbox, W, H, CONFIG)

    assert position.complete is False
    assert position.hints == ("show_complete_product",)  # its center and size mean nothing while cut off


def test_marker_outside_the_scan_box_gets_guidance():
    position = analyze_position(_marker(cx=150), None, W, H, CONFIG)

    assert position.inside_area is False
    assert position.hints == ("move_right",)


def test_product_poking_out_of_the_scan_box_while_roughly_centered_is_asked_to_center():
    # 520 px tall, 50 px low: within the centering tolerance, but its bottom edge crosses the box.
    tall = ProductDetection("bottle", 0.9, (540.0, 150.0, 740.0, 670.0))

    position = analyze_position(_marker(cy=400), tall.bbox, W, H, CONFIG)

    assert (position.inside_area, position.hints) == (False, ("center",))


def test_tilted_marker_is_asked_to_straighten():
    assert _hints(marker=_marker(squash=0.8), product=_box()) == []
    assert _hints(marker=_marker(squash=0.4), product=_box()) == ["straighten"]


def test_in_plane_rotation_is_only_checked_when_configured():
    spun = _marker(roll_deg=45)

    assert round(analyze_position(spun, None, W, H, CONFIG).roll_deg) == 45
    assert _hints(marker=spun, product=_box()) == []  # any rotation allowed by default
    assert _hints(marker=spun, product=_box(), config=ScanGateConfig(max_roll_deg=20)) == ["straighten"]


# --- scan gate ---


def test_no_marker_is_searching():
    result = ScanGate().update(_obs(count=0), now_ms=0)

    assert (result.state, result.problem) == (ScanState.SEARCHING, "no_marker")


def test_multiple_and_unknown_markers_do_not_progress():
    gate = ScanGate()
    for t in range(0, 3000, 100):
        assert gate.update(_obs(count=2), now_ms=t).problem == "multiple_markers"
    for t in range(3000, 6000, 100):
        result = gate.update(_obs(known=False), now_ms=t)
        assert (result.state, result.problem) == (ScanState.MARKER_DETECTED, "unknown_marker")


def test_missing_uncertain_wrong_or_unverifiable_product_never_confirms():
    for verification, state, problem in [
        (Verification(VerificationStatus.NO_PRODUCT, "bottle"), ScanState.MARKER_DETECTED, "no_product"),
        (Verification(VerificationStatus.UNVERIFIABLE, None), ScanState.MARKER_DETECTED, "unverifiable"),
        (Verification(VerificationStatus.UNCERTAIN, "bottle", _box(confidence=0.3)), ScanState.VERIFYING, "uncertain"),
        (Verification(VerificationStatus.MISMATCH, "bottle", _box("bus")), ScanState.MISMATCH, "mismatch"),
    ]:
        gate = ScanGate()
        for t in range(0, 3000, 100):
            result = gate.update(_obs(verification=verification), now_ms=t)
            assert (result.state, result.problem) == (state, problem)


def test_badly_placed_product_gets_guidance_and_never_confirms():
    gate = ScanGate()
    off_left = _obs(marker=_marker(cx=360, side=100), verification=_match(around=(360, 360), half=150))
    for t in range(0, 3000, 100):
        result = gate.update(off_left, now_ms=t)
        assert (result.state, result.hints) == (ScanState.POSITIONING, ("move_right",))
        assert result.positioning.offset_x < 0


def test_gate_guidance_follows_the_mirrored_preview():
    off_left = dict(marker=_marker(cx=360, side=100), verification=_match(around=(360, 360), half=150))

    assert ScanGate().update(_obs(**off_left), now_ms=0).hints == ("move_right",)
    assert ScanGate().update(_obs(**off_left, mirrored=True), now_ms=0).hints == ("move_left",)


def test_correctly_positioned_product_is_ready_then_confirms_by_itself_after_the_hold():
    gate = ScanGate()
    states = [gate.update(_obs(), now_ms=t) for t in range(0, 1000, 100)]

    assert states[0].state is ScanState.HOLD_STEADY  # ready: every check passes, hold starts
    assert states[0].stages == {"marker": True, "product": True, "verified": True, "position": True, "steady": True}
    assert [s.state for s in states[1:8]] == [ScanState.SCANNING] * 7
    assert [round(s.progress, 3) for s in states[1:8]] == [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    assert states[8].state is ScanState.CONFIRMED  # 800 ms after the first good frame
    assert (states[8].marker_id, states[8].confirmation) == (1, 1)


def test_hold_duration_is_configurable():
    gate = ScanGate(ScanGateConfig(hold_ms=300))

    assert [gate.update(_obs(), now_ms=t).state for t in (0, 100, 200, 300)][-1] is ScanState.CONFIRMED


def test_any_interruption_restarts_the_hold():
    gate = ScanGate()
    for t in range(0, 700, 100):
        gate.update(_obs(), now_ms=t)
    gate.update(_obs(marker=_marker(side=30), verification=_match(half=60)), now_ms=700)  # drifted too far

    resumed = [gate.update(_obs(), now_ms=t).state for t in range(800, 1700, 100)]
    assert resumed[:-1] == [ScanState.HOLD_STEADY] + [ScanState.SCANNING] * 7
    assert resumed[-1] is ScanState.CONFIRMED


def test_moving_product_resets_stability_and_does_not_confirm():
    gate = ScanGate()
    for t in range(0, 500, 100):
        gate.update(_obs(), now_ms=t)
    assert gate.update(_obs(), now_ms=500).progress > 0.5

    # 30 px per 100 ms on a 120 px marker = 2.5 marker sides per second.
    for i, t in enumerate(range(600, 3000, 100)):
        result = gate.update(_obs(marker=_marker(cx=620 + 30 * (i % 3))), now_ms=t)
        assert (result.state, result.problem, result.progress) == (ScanState.HOLD_STEADY, "moving", 0.0)
        assert result.speed > CONFIG.max_speed

    # Still again: the hold starts over from zero rather than resuming.
    after = [gate.update(_obs(marker=_marker(cx=620)), now_ms=t) for t in range(3100, 4100, 100)]
    assert after[0].progress == 0.0
    assert after[-1].state is ScanState.CONFIRMED


def test_small_jitter_still_counts_as_steady():
    gate = ScanGate()
    states = [gate.update(_obs(marker=_marker(cx=640 + (i % 2) * 3)), now_ms=i * 100).state for i in range(9)]

    assert states[-1] is ScanState.CONFIRMED


def test_detector_blinking_briefly_does_not_restart_the_hold_but_a_mismatch_does():
    gate = ScanGate()
    missing = Verification(VerificationStatus.NO_PRODUCT, "bottle")
    gate.update(_obs(), now_ms=0)
    gate.update(_obs(), now_ms=100)
    assert gate.update(_obs(verification=missing), now_ms=200).state is ScanState.SCANNING  # within the grace
    assert gate.update(_obs(verification=missing), now_ms=700).state is ScanState.MARKER_DETECTED  # gone too long

    gate.reset()
    gate.update(_obs(), now_ms=0)
    gate.update(_obs(), now_ms=100)
    wrong = Verification(VerificationStatus.MISMATCH, "bottle", _box("bus"))
    assert gate.update(_obs(verification=wrong), now_ms=200).state is ScanState.MISMATCH


def test_swapping_markers_mid_hold_restarts_it():
    gate = ScanGate()
    for t in range(0, 700, 100):
        gate.update(_obs(), now_ms=t)

    swapped = gate.update(_obs(marker=_marker(marker_id=2)), now_ms=700)
    assert swapped.state is not ScanState.CONFIRMED
    assert gate.update(_obs(marker=_marker(marker_id=2)), now_ms=800).state is not ScanState.CONFIRMED


def _confirm(gate: ScanGate, start_ms: int = 0) -> int:
    for t in range(start_ms, start_ms + 900, 100):
        result = gate.update(_obs(), now_ms=t)
    assert result.state is ScanState.CONFIRMED
    return start_ms + 800


def test_confirmed_product_staying_in_the_scan_box_is_not_scanned_again():
    gate = ScanGate()
    done = _confirm(gate)

    for t in range(done + 100, done + 10_000, 100):
        result = gate.update(_obs(), now_ms=t)
        assert (result.state, result.confirmation) == (ScanState.CONFIRMED, 1)


def test_brief_dropout_does_not_rearm_the_scanner():
    gate = ScanGate()
    done = _confirm(gate)

    assert gate.update(_obs(count=0), now_ms=done + 500).state is ScanState.CONFIRMED  # under rearm_ms
    for t in range(done + 600, done + 3000, 100):
        assert gate.update(_obs(), now_ms=t).confirmation == 1


def test_scanner_rearms_once_the_product_has_left_the_scan_box():
    gate = ScanGate()
    done = _confirm(gate)
    outside = _obs(marker=_marker(cx=100, side=100), verification=_match(around=(100, 360), half=90))

    assert gate.update(outside, now_ms=done + 500).state is ScanState.CONFIRMED
    assert gate.update(outside, now_ms=done + 1200).state is not ScanState.CONFIRMED  # gone for over rearm_ms
    assert gate.update(_obs(count=0), now_ms=done + 1300).state is ScanState.SEARCHING

    # Brought back: a full new hold, and a new confirmation number.
    again = [gate.update(_obs(), now_ms=t) for t in range(done + 2000, done + 2900, 100)]
    assert again[0].state is ScanState.HOLD_STEADY
    assert (again[-1].state, again[-1].confirmation) == (ScanState.CONFIRMED, 2)


def test_reset_clears_a_confirmation_immediately():
    gate = ScanGate()
    done = _confirm(gate)

    gate.reset()
    assert gate.update(_obs(count=0), now_ms=done + 100).state is ScanState.SEARCHING


def test_far_away_product_the_detector_cannot_confirm_is_told_to_move_closer():
    tiny = _marker(side=30)
    weak = Verification(VerificationStatus.UNCERTAIN, "bottle", _box(confidence=0.3, half=60))
    missing = Verification(VerificationStatus.NO_PRODUCT, "bottle")

    assert ScanGate().update(_obs(marker=tiny, verification=weak), now_ms=0).hints == ("move_closer",)
    assert ScanGate().update(_obs(marker=tiny, verification=missing), now_ms=0).hints == ("move_closer",)
    # At a normal size there is nothing to suggest but waiting / showing the product.
    assert ScanGate().update(_obs(verification=weak), now_ms=0).hints == ()
