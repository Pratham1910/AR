"""
ArUco product tags: finding them in a frame, and drawing printable ones.

Detection answers only "which tag ids are where" — which product a tag
belongs to is identification.py's job, and whether the right object is there
is verification.py's. The actual detector is the one ArucoPoseEstimator
already owns (passed in as `detect`), so tags are read with the same
dictionary and parameters as the marker pose path.
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
    MarkerCorner.TOP_LEFT: "Top-Left",
    MarkerCorner.TOP_RIGHT: "Top-Right",
    MarkerCorner.BOTTOM_LEFT: "Bottom-Left",
    MarkerCorner.BOTTOM_RIGHT: "Bottom-Right",
}


@dataclass(frozen=True)
class MarkerDetection:
    marker_id: int
    corners: np.ndarray  # 4x2 pixel corners, in cv2.aruco's order
    # Not seen in this frame: carried over from a moment ago (scan_session.py).
    held: bool = False

    @property
    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)

    @property
    def side_px(self) -> float:
        """Mean edge length — the tag's apparent size."""
        return float(np.linalg.norm(self.corners - np.roll(self.corners, -1, axis=0), axis=1).mean())


def dictionary_size(dictionary_name: str) -> int:
    """How many distinct tag ids the dictionary has (ids are 0..size-1)."""
    return int(cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary_name]).bytesList.shape[0])


def detect_tags(detect: RawDetector, frame_bgr: np.ndarray) -> tuple[list[MarkerDetection], list[int]]:
    """
    (tags found, ids seen more than once). Two passes: the frame as it is,
    then contrast-equalized (CLAHE), which recovers tags in shadow or under
    uneven light that the plain pass misses; the second pass only adds ids the
    first didn't find. An id seen twice in one pass is two physical copies of
    the same tag in view — reported, since which one is "the" corner is then
    ambiguous.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    first = detect(gray)
    duplicates = sorted(marker_id for marker_id, n in Counter(i for i, _ in first).items() if n > 1)
    found = {marker_id: corners for marker_id, corners in reversed(first)}
    equalized = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    for marker_id, corners in detect(equalized):
        found.setdefault(marker_id, corners)
    return [MarkerDetection(marker_id, np.asarray(c, dtype=np.float64)) for marker_id, c in found.items()], duplicates


def _encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode the tag image as PNG")
    return encoded.tobytes()


def _tag_tile(dictionary_name: str, marker_id: int, size_px: int, caption: str) -> np.ndarray:
    """One tag on white with a quiet zone and a caption underneath. The white
    margin is not decoration: a tag cut out or printed flush against a dark
    edge has no border contrast and is not detected."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary_name])
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    margin = max(size_px // 6, 8)
    caption_height = max(size_px // 7, 24)
    tile = np.full((size_px + 2 * margin + caption_height, size_px + 2 * margin), 255, dtype=np.uint8)
    tile[margin : margin + size_px, margin : margin + size_px] = marker
    scale = caption_height / 55.0
    thickness = max(1, round(scale * 2))
    (text_width, _), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    if text_width > tile.shape[1] - 8:  # long product names: shrink to fit the tile
        scale *= (tile.shape[1] - 8) / text_width
        thickness = max(1, round(scale * 2))
    cv2.putText(
        tile,
        caption,
        (4, size_px + 2 * margin + caption_height * 2 // 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        0,
        thickness,
        cv2.LINE_AA,
    )
    return tile


def render_tag_png(dictionary_name: str, marker_id: int, size_px: int, caption: str | None = None) -> bytes:
    return _encode_png(_tag_tile(dictionary_name, marker_id, size_px, caption or f"id {marker_id}"))


def render_sheet_png(
    dictionary_name: str, tags: dict[MarkerCorner, int], size_px: int, product_name: str | None = None
) -> bytes:
    """All four tags on one page, laid out as they go on the product."""
    rows = []
    for left, right in ((MarkerCorner.TOP_LEFT, MarkerCorner.TOP_RIGHT), (MarkerCorner.BOTTOM_LEFT, MarkerCorner.BOTTOM_RIGHT)):
        rows.append(
            np.hstack(
                [_tag_tile(dictionary_name, tags[c], size_px, f"{CORNER_LABELS[c]}  id {tags[c]}") for c in (left, right)]
            )
        )
    sheet = np.vstack(rows)
    if product_name:
        header = np.full((max(size_px // 6, 30), sheet.shape[1]), 255, dtype=np.uint8)
        scale = header.shape[0] / 50.0
        cv2.putText(
            header,
            f"{product_name}  ({dictionary_name})",
            (8, header.shape[0] * 3 // 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            0,
            max(1, round(scale * 2)),
            cv2.LINE_AA,
        )
        sheet = np.vstack([header, sheet])
    return _encode_png(sheet)
