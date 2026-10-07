"""
Marker detection and presence tracking on synthetic frames (no camera or
database needed) — see tests/backend/test_aruco_pose.py for the pose side.
"""

import cv2
import numpy as np
import pytest

from app.services.markers.aruco_detector import ArucoDetector, aruco_dictionary_names
from app.services.markers.detector import MarkerDetection, MarkerDetector
from app.services.markers.factory import build_marker_detector
from app.services.markers.presence import MarkerPresenceTracker
from app.workers.generate_marker import render_marker

WIDTH, HEIGHT = 960, 480
MARKER_PX = 160


def _frame_with_markers(placements: dict[int, tuple[int, int]], dictionary: str = "DICT_4X4_50") -> np.ndarray:
    """A light gray frame with each marker id drawn at its (x, y) top-left pixel."""
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    frame = np.full((HEIGHT, WIDTH, 3), 200, dtype=np.uint8)
    for marker_id, (x, y) in placements.items():
        marker = cv2.aruco.generateImageMarker(aruco_dictionary, marker_id, MARKER_PX)
        frame[y : y + MARKER_PX, x : x + MARKER_PX] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return frame


def test_factory_builds_an_aruco_detector_behind_the_generic_interface():
    detector = build_marker_detector("aruco", "DICT_4X4_50")

    assert isinstance(detector, MarkerDetector)
    assert (detector.family, detector.dictionary, detector.marker_count) == ("aruco", "DICT_4X4_50", 50)


def test_unknown_family_and_dictionary_are_rejected():
    with pytest.raises(ValueError, match="Unknown marker family"):
        build_marker_detector("qr", "DICT_4X4_50")
    with pytest.raises(ValueError, match="Unknown ArUco dictionary"):
        build_marker_detector("aruco", "DICT_NOPE")


def test_dictionary_is_configurable():
    assert "DICT_6X6_250" in aruco_dictionary_names()
    frame = _frame_with_markers({5: (100, 100)}, dictionary="DICT_6X6_250")

    assert [d.marker_id for d in ArucoDetector("DICT_6X6_250").detect(frame)] == [5]
    assert ArucoDetector("DICT_4X4_50").detect(frame) == []


def test_no_marker_returns_empty_list():
    blank = np.full((HEIGHT, WIDTH, 3), 200, dtype=np.uint8)

    assert ArucoDetector("DICT_4X4_50").detect(blank) == []


def test_detects_id_and_four_corners_in_marker_order():
    x, y = 300, 120
    detections = ArucoDetector("DICT_4X4_50").detect(_frame_with_markers({7: (x, y)}))

    assert len(detections) == 1
    assert detections[0].marker_id == 7
    expected = [(x, y), (x + MARKER_PX, y), (x + MARKER_PX, y + MARKER_PX), (x, y + MARKER_PX)]
    assert np.allclose(detections[0].corners_px, expected, atol=2.0)  # top-left, clockwise
    assert np.allclose(detections[0].center_px, (x + MARKER_PX / 2, y + MARKER_PX / 2), atol=2.0)


def test_detects_multiple_markers_in_one_frame():
    frame = _frame_with_markers({0: (60, 60), 1: (400, 200), 42: (740, 280)})

    detections = ArucoDetector("DICT_4X4_50").detect(frame)

    assert sorted(d.marker_id for d in detections) == [0, 1, 42]
    left_edges = {d.marker_id: min(x for x, _ in d.corners_px) for d in detections}
    assert left_edges[0] < left_edges[1] < left_edges[42]


def _detection(marker_id: int) -> MarkerDetection:
    return MarkerDetection(marker_id, ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))


def test_marker_leaving_the_frame_is_held_then_dropped():
    tracker = MarkerPresenceTracker(hold_ms=1000)

    seen = tracker.update([_detection(3)], now_ms=0)
    assert [(t.detection.marker_id, t.visible) for t in seen] == [(3, True)]

    held = tracker.update([], now_ms=400)
    assert [(t.detection.marker_id, t.visible, t.ms_since_seen) for t in held] == [(3, False, 400)]
    assert held[0].detection.corners_px == _detection(3).corners_px  # last known position

    assert tracker.update([], now_ms=1500) == []


def test_marker_returning_within_the_hold_is_visible_again():
    tracker = MarkerPresenceTracker(hold_ms=1000)
    tracker.update([_detection(3)], now_ms=0)
    tracker.update([], now_ms=900)

    back = tracker.update([_detection(3)], now_ms=950)
    assert [(t.detection.marker_id, t.visible) for t in back] == [(3, True)]
    # The hold restarts from the latest sighting.
    assert [t.visible for t in tracker.update([], now_ms=1900)] == [False]


def test_markers_are_held_independently():
    tracker = MarkerPresenceTracker(hold_ms=1000)
    tracker.update([_detection(0), _detection(1)], now_ms=0)

    result = tracker.update([_detection(1)], now_ms=500)

    assert {t.detection.marker_id: t.visible for t in result} == {1: True, 0: False}


def test_generated_marker_is_detectable_on_any_background():
    # As shown full-screen in an image viewer: the file centered on black.
    marker = cv2.cvtColor(render_marker("DICT_4X4_50", 1, 180), cv2.COLOR_GRAY2BGR)
    screen = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    y, x = (HEIGHT - marker.shape[0]) // 2, (WIDTH - marker.shape[1]) // 2
    screen[y : y + marker.shape[0], x : x + marker.shape[1]] = marker

    detector = ArucoDetector("DICT_4X4_50")
    assert [d.marker_id for d in detector.detect(marker)] == [1]  # the file itself
    assert [d.marker_id for d in detector.detect(screen)] == [1]
