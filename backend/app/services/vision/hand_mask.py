"""
Where are the user's hands? A per-frame mask, so hand pixels can be kept out
of the tracked object's features, its outline and component checks.

Two sources, combined (union):

- MediaPipe Hands (21 landmarks per hand, CPU, ~35 ms at 1280x720): the mask
  is drawn from the landmarks — palm polygon plus thick finger segments — so
  it can't spill onto the object being held. It can miss a hand gripping an
  object seen from behind (the palm detector needs to see the hand's shape).
- The existing COCO YOLO segmentation's "person" masks: catches arms and
  gripping hands MediaPipe misses, but on a hand holding a bottle it also
  covered half the bottle (measured on the user's screenshot, 2026-10-13).

Union errs towards marking too much as hand. That is the safe side for every
consumer here: the tracker keeps fewer object points (and holds its pose if
too few remain) and component inspection says OCCLUDED rather than MISSING.
Missing a hand is the error that hurts (hand features dragging the pose).

HandMaskService runs this off the tracking thread at a capped rate; the
tracker uses the newest mask that isn't too old, grown by a margin for the
motion since it was taken.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("uvicorn.error").getChild("hands")

# MediaPipe hand landmark indices: palm outline and the finger chains.
_PALM = (0, 1, 2, 5, 9, 13, 17)
_FINGERS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))

PersonSegmenter = Callable[[np.ndarray], list[np.ndarray]]  # frame -> polygons ((N, 2) px) of people


@dataclass
class HandMask:
    mask: np.ndarray  # bool, frame-sized: hand / arm pixels (with margin)
    hands: int  # hands MediaPipe found
    persons: int  # person regions from the segmentation model
    sources: list[str]  # which sources produced it ("mediapipe", "person")
    taken_at: float  # monotonic time of the frame it was computed on
    elapsed_ms: float
    landmarks: list[np.ndarray] = field(default_factory=list)  # (21, 2) px per hand, for debug drawing

    def outlines(self, simplify_px: float = 3.0) -> list[list[tuple[float, float]]]:
        contours, _ = cv2.findContours(self.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [
            [(float(p[0][0]), float(p[0][1])) for p in cv2.approxPolyDP(c, simplify_px, True)]
            for c in contours
            if cv2.contourArea(c) > 50
        ]


def landmark_mask(landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """A hand's pixels from its 21 landmarks (px): palm polygon + fingers as
    thick segments, sized by the hand's own scale in the image."""
    mask = np.zeros(shape, np.uint8)
    pts = np.round(landmarks).astype(np.int32)
    palm_width = float(np.linalg.norm(landmarks[5] - landmarks[17]))
    hand_size = max(palm_width, float(np.linalg.norm(landmarks[0] - landmarks[9])) * 0.8, 6.0)
    finger = max(3, int(round(hand_size * 0.32)))
    cv2.fillPoly(mask, [cv2.convexHull(pts[list(_PALM)])], 1)
    for chain in _FINGERS:
        for a, b in zip(chain, chain[1:]):
            cv2.line(mask, tuple(pts[a]), tuple(pts[b]), 1, finger)
        cv2.circle(mask, tuple(pts[chain[-1]]), finger // 2 + 1, 1, -1)
    cv2.line(mask, tuple(pts[0]), tuple(pts[1]), 1, finger)
    return mask.astype(bool)


class MediaPipeHands:
    """MediaPipe Tasks hand landmarker (IMAGE mode: every call independent)."""

    def __init__(self, model_path: str, max_hands: int = 2, min_confidence: float = 0.4, max_side_px: int = 640):
        from mediapipe.tasks.python import BaseOptions, vision  # optional dependency, imported only when used

        if not Path(model_path).exists():
            raise FileNotFoundError(f"hand landmark model not found: {model_path}")
        self._landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=max_hands,
                min_hand_detection_confidence=min_confidence,
                min_hand_presence_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
            )
        )
        self.max_side_px = max_side_px
        self._lock = threading.Lock()  # the landmarker isn't safe to call from two threads at once

    def landmarks(self, frame_bgr: np.ndarray) -> list[np.ndarray]:
        import mediapipe as mp

        height, width = frame_bgr.shape[:2]
        scale = min(1.0, self.max_side_px / max(height, width))
        small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame_bgr
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        with self._lock:
            result = self._landmarker.detect(image)
        return [np.array([[p.x * width, p.y * height] for p in hand]) for hand in result.hand_landmarks]


class HandSegmenter:
    def __init__(
        self,
        hands: MediaPipeHands | None,
        persons: PersonSegmenter | None,
        margin_px: int = 6,
    ):
        self.hands = hands
        self.persons = persons
        self.margin_px = margin_px

    @property
    def available(self) -> bool:
        return self.hands is not None or self.persons is not None

    def segment(self, frame_bgr: np.ndarray, taken_at: float) -> HandMask:
        started = time.perf_counter()
        shape = frame_bgr.shape[:2]
        mask = np.zeros(shape, bool)
        sources: list[str] = []
        landmarks: list[np.ndarray] = []
        persons = 0
        if self.hands is not None:
            landmarks = self.hands.landmarks(frame_bgr)
            for hand in landmarks:
                mask |= landmark_mask(hand, shape)
            sources.append("mediapipe")
        if self.persons is not None:
            canvas = np.zeros(shape, np.uint8)
            polygons = self.persons(frame_bgr)
            for polygon in polygons:
                cv2.fillPoly(canvas, [np.round(polygon).astype(np.int32)], 1)
            persons = len(polygons)
            mask |= canvas.astype(bool)
            sources.append("person")
        if self.margin_px > 0 and mask.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.margin_px + 1, 2 * self.margin_px + 1))
            mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
        return HandMask(mask, len(landmarks), persons, sources, taken_at, (time.perf_counter() - started) * 1000, landmarks)


_mask_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hand-mask")


class HandMaskService:
    """Hand masks for one live session, computed on a worker thread.

    submit() starts a mask for this frame if none is running and the last one
    is at least `interval_ms` old; latest() returns the newest finished mask
    if it is at most `max_age_ms` old, grown by `motion_margin_px` per 100 ms
    of age (the hand kept moving since)."""

    def __init__(
        self,
        segmenter: HandSegmenter,
        interval_ms: float = 100.0,
        max_age_ms: float = 400.0,
        motion_margin_px: float = 8.0,
        run_async: bool = True,
    ):
        self.segmenter = segmenter
        self.interval_ms = interval_ms
        self.max_age_ms = max_age_ms
        self.motion_margin_px = motion_margin_px
        self.run_async = run_async
        self._latest: HandMask | None = None
        self._busy = False
        self._last_submit = -1e9
        self._lock = threading.Lock()
        self.error: str | None = None

    def submit(self, frame_bgr: np.ndarray, now: float) -> None:
        with self._lock:
            if self._busy or (now - self._last_submit) * 1000 < self.interval_ms:
                return
            self._busy = True
            self._last_submit = now
        if self.run_async:
            _mask_pool.submit(self._run, frame_bgr, now)
        else:
            self._run(frame_bgr, now)

    def _run(self, frame_bgr: np.ndarray, now: float) -> None:
        try:
            result = self.segmenter.segment(frame_bgr, now)
            self._latest = result
            self.error = None
        except Exception as exc:  # noqa: BLE001 - a failing mask must not break tracking; it is reported
            if self.error != str(exc):
                log.exception("[HANDS] hand mask failed")
            self.error = str(exc)
        finally:
            self._busy = False

    def latest(self, now: float) -> HandMask | None:
        mask = self._latest
        if mask is None:
            return None
        age_ms = (now - mask.taken_at) * 1000
        if age_ms > self.max_age_ms:
            return None
        grow = int(round(self.motion_margin_px * age_ms / 100.0))
        if grow <= 0 or not mask.mask.any():
            return mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        grown = cv2.dilate(mask.mask.astype(np.uint8), kernel).astype(bool)
        return HandMask(grown, mask.hands, mask.persons, mask.sources, mask.taken_at, mask.elapsed_ms, mask.landmarks)
