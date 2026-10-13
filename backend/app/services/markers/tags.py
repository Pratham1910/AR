"""
ArUco product tags as images: reading them out of a camera frame, and drawing
printable ones (docs/marker-tags.md).

Reading answers only "which tag ids are where". Which product a tag belongs
to is identification.py's job, and what object sits between a product's tags
is region_detection.py's. The detector itself is the one ArucoPoseEstimator
already owns (passed in as `detect`), so tags are read with the same
dictionary and parameters as the single-marker pose path.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from app.models.enums import MarkerCorner
from app.services.pose.aruco_pose import ARUCO_DICTIONARIES

RawDetector = Callable[[np.ndarray], list[tuple[int, np.ndarray]]]

CORNER_LABELS = {
    MarkerCorner.TOP_LEFT: "Top-left",
    MarkerCorner.TOP_RIGHT: "Top-right",
    MarkerCorner.BOTTOM_RIGHT: "Bottom-right",
    MarkerCorner.BOTTOM_LEFT: "Bottom-left",
}


@dataclass(frozen=True)
class TagSighting:
    """One tag in one frame."""

    marker_id: int
    corners: np.ndarray  # 4x2 pixel corners, cv2.aruco order (tag's own TL, TR, BR, BL)
    # Not read in this frame: re-projected from the other tags of a locked
    # product while it was covered (tracking.py).
    estimated: bool = False

    @property
    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)

    @property
    def area_px(self) -> float:
        return float(cv2.contourArea(self.corners.astype(np.float32)))


def dictionary_size(dictionary_name: str) -> int:
    """How many tag ids the dictionary has (ids are 0..size-1)."""
    return int(cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary_name]).bytesList.shape[0])


def read_tags(detect: RawDetector, frame_bgr: np.ndarray) -> tuple[list[TagSighting], list[int]]:
    """
    (tags read, ids read more than once).

    Two passes: the frame as it is, then contrast-equalized (CLAHE), which
    recovers tags in shadow or under uneven light that the plain pass misses;
    the second pass only adds ids the first did not find. An id read twice in
    one pass is two physical copies of the same tag in view — reported rather
    than resolved, since which copy marks the corner is then unknowable.
    """
    first = detect(frame_bgr)
    counts = Counter(marker_id for marker_id, _ in first)
    duplicates = sorted(i for i, n in counts.items() if n > 1)
    found = {marker_id: corners for marker_id, corners in first if counts[marker_id] == 1}

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    second = detect(cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR))
    second_counts = Counter(marker_id for marker_id, _ in second)
    for marker_id, corners in second:
        if marker_id in found or marker_id in duplicates:
            continue
        if second_counts[marker_id] > 1:
            duplicates.append(marker_id)
            continue
        found[marker_id] = corners

    sightings = [TagSighting(marker_id=i, corners=np.asarray(c, dtype=np.float64)) for i, c in sorted(found.items())]
    return sightings, sorted(set(duplicates))


# --- Printable tags ----------------------------------------------------------

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _tag_image(dictionary_name: str, marker_id: int, size_px: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary_name])
    return cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)


def _with_margin_and_caption(tag: np.ndarray, caption: str) -> np.ndarray:
    """White quiet zone (a quarter of the tag side — detection needs one) and
    a caption under it naming the tag."""
    size = tag.shape[0]
    margin = max(size // 4, 20)
    caption_h = max(size // 6, 30)
    canvas = np.full((size + 2 * margin + caption_h, size + 2 * margin), 255, dtype=np.uint8)
    canvas[margin : margin + size, margin : margin + size] = tag
    scale = max(size / 600.0, 0.4)
    (text_w, text_h), _ = cv2.getTextSize(caption, _FONT, scale, 1)
    while text_w > canvas.shape[1] - 10 and scale > 0.3:
        scale *= 0.9
        (text_w, text_h), _ = cv2.getTextSize(caption, _FONT, scale, 1)
    org = ((canvas.shape[1] - text_w) // 2, margin + size + (caption_h + text_h) // 2)
    cv2.putText(canvas, caption, org, _FONT, scale, 0, max(1, int(round(scale * 2))), cv2.LINE_AA)
    return canvas


def _png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return encoded.tobytes()


def render_tag_png(dictionary_name: str, marker_id: int, size_px: int, caption: str) -> bytes:
    """One printable tag: the tag, a white margin, and a caption under it."""
    return _png(_with_margin_and_caption(_tag_image(dictionary_name, marker_id, size_px), caption))


def render_sheet_png(dictionary_name: str, ids: dict[MarkerCorner, int], size_px: int, title: str) -> bytes:
    """A product's four tags on one page, each in its corner position, so the
    printed sheet shows how they go around the product."""
    cells = {
        corner: _with_margin_and_caption(_tag_image(dictionary_name, marker_id, size_px), f"{CORNER_LABELS[corner]}  id {marker_id}")
        for corner, marker_id in ids.items()
    }
    cell_h, cell_w = next(iter(cells.values())).shape
    gap = size_px // 2
    title_h = max(size_px // 4, 40)
    sheet = np.full((title_h + 2 * cell_h + gap, 2 * cell_w + gap), 255, dtype=np.uint8)
    positions = {
        MarkerCorner.TOP_LEFT: (0, 0),
        MarkerCorner.TOP_RIGHT: (0, 1),
        MarkerCorner.BOTTOM_LEFT: (1, 0),
        MarkerCorner.BOTTOM_RIGHT: (1, 1),
    }
    for corner, cell in cells.items():
        row, col = positions[corner]
        y = title_h + row * (cell_h + gap)
        x = col * (cell_w + gap)
        sheet[y : y + cell_h, x : x + cell_w] = cell
    scale = max(size_px / 500.0, 0.5)
    text = f"{title}  ({dictionary_name}) - place the object between the four tags"
    (text_w, text_h), _ = cv2.getTextSize(text, _FONT, scale, 1)
    while text_w > sheet.shape[1] - 20 and scale > 0.3:
        scale *= 0.9
        (text_w, text_h), _ = cv2.getTextSize(text, _FONT, scale, 1)
    cv2.putText(sheet, text, (10, (title_h + text_h) // 2), _FONT, scale, 0, max(1, int(round(scale * 2))), cv2.LINE_AA)
    return _png(sheet)
