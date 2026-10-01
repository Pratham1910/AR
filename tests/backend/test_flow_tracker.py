"""Optical-flow box tracker on synthetic frames with known motion."""

import cv2
import numpy as np
import pytest

from app.schemas.vision import BoundingBox
from app.services.tracking.flow_tracker import FlowBoxTracker

H, W = 480, 640
_rng = np.random.default_rng(0)
BACKGROUND = _rng.integers(100, 140, (H, W)).astype(np.uint8)  # low-contrast noise: few strong corners
OBJECT = np.kron(_rng.integers(0, 255, (16, 12)), np.ones((10, 10))).astype(np.uint8)  # 160x120 textured patch


def frame(cx: float, cy: float, scale: float = 1.0) -> np.ndarray:
    img = BACKGROUND.copy()
    patch = cv2.resize(OBJECT, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    h, w = patch.shape
    x, y = int(round(cx - w / 2)), int(round(cy - h / 2))
    img[y : y + h, x : x + w] = patch
    return cv2.GaussianBlur(img, (3, 3), 0)


def locked_tracker() -> FlowBoxTracker:
    tracker = FlowBoxTracker()
    assert tracker.initialize(frame(300, 240), BoundingBox(x1=240, y1=160, x2=360, y2=320), None)
    return tracker


def center_and_height(bbox: BoundingBox) -> tuple[float, float, float]:
    return (bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2, bbox.y2 - bbox.y1


def test_follows_translation_without_redetecting():
    tracker = locked_tracker()
    for step in range(1, 11):
        result = tracker.update(frame(300 + 6 * step, 240 - 3 * step))
        assert result.ok and result.confidence > 0.9
    cx, cy, height = center_and_height(result.bbox)
    assert cx == pytest.approx(360, abs=1.0)
    assert cy == pytest.approx(210, abs=1.0)
    assert height == pytest.approx(160, abs=1.0)


def test_growing_object_grows_the_box_which_is_what_carries_depth():
    tracker = locked_tracker()
    for scale in (1.05, 1.10, 1.15, 1.20):
        result = tracker.update(frame(300, 240, scale))
    _, _, height = center_and_height(result.bbox)
    assert height == pytest.approx(160 * 1.20, rel=0.02)


def test_object_disappearing_drops_confidence_and_fails():
    tracker = locked_tracker()
    assert tracker.update(frame(304, 240)).ok
    result = tracker.update(BACKGROUND.copy())
    assert not result.ok
    assert result.confidence < 0.4


def test_featureless_object_cannot_be_locked():
    blank = np.full((H, W), 120, dtype=np.uint8)
    assert not FlowBoxTracker().initialize(blank, BoundingBox(x1=240, y1=160, x2=360, y2=320), None)


def test_update_before_initialize_is_a_miss():
    assert not FlowBoxTracker().update(BACKGROUND.copy()).ok
