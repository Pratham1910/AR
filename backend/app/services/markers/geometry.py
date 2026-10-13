"""
The area a product's four corner tags enclose, and whether a detected object
lies inside it (docs/marker-tags.md, "How the region is built").

Everything here works on the tags' real pixel corners, so the region follows
the product's rotation and the camera's perspective: four tags seen at an
angle enclose a general quadrilateral, not their axis-aligned bounding box.
Pure geometry on numpy arrays — no images are decoded and no model runs.
"""

import cv2
import numpy as np

from app.models.enums import MarkerCorner
from app.services.markers.tags import TagSighting

# Walking around the product: clockwise as the camera sees it (image y points down).
QUAD_ORDER = (MarkerCorner.TOP_LEFT, MarkerCorner.TOP_RIGHT, MarkerCorner.BOTTOM_RIGHT, MarkerCorner.BOTTOM_LEFT)


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """z of (a-o) x (b-o); positive = clockwise turn in image coordinates."""
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def is_clockwise_convex(points: np.ndarray) -> bool:
    """True if walking the points in order turns clockwise (as seen on screen)
    at every vertex. Holds at any rotation or perspective of a real
    rectangle; fails if two corners are swapped or the set is mirrored."""
    n = len(points)
    return all(_cross(points[i], points[(i + 1) % n], points[(i + 2) % n]) > 0 for i in range(n))


def enclosed_region(tags: dict[MarkerCorner, TagSighting]) -> np.ndarray:
    """
    The quadrilateral the four tags enclose, as 4x2 pixel points in QUAD_ORDER.

    Each vertex is that tag's outermost corner — the one farthest from the
    middle of the four tags — so the region runs along the tags' outer edges
    and the tags themselves are inside it. It is built from the detected
    corner coordinates, not from a bounding box, so it is a true (possibly
    skewed) quadrilateral under perspective.
    """
    middle = np.mean([tags[c].center for c in QUAD_ORDER], axis=0)
    vertices = []
    for corner in QUAD_ORDER:
        tag_corners = tags[corner].corners
        vertices.append(tag_corners[int(np.argmax(np.linalg.norm(tag_corners - middle, axis=1)))])
    return np.asarray(vertices, dtype=np.float64)


def layout_problem(tags: dict[MarkerCorner, TagSighting], min_region_in_tags: float) -> str | None:
    """Why four tags cannot be a product's corners, or None if they can."""
    centers = np.asarray([tags[c].center for c in QUAD_ORDER])
    if not is_clockwise_convex(centers):
        return (
            "the tags are not in their registered corners (going top-left, top-right, bottom-right, "
            "bottom-left must run clockwise) — check each tag is on the corner printed under it"
        )
    region = enclosed_region(tags)
    if not is_clockwise_convex(region):
        return "the tags' outer corners do not form a convex outline"
    tag_area = float(np.mean([tags[c].area_px for c in QUAD_ORDER]))
    region_area = float(cv2.contourArea(region.astype(np.float32)))
    if tag_area <= 0 or region_area < min_region_in_tags * tag_area:
        return (
            f"the tags enclose only {region_area / max(tag_area, 1e-9):.1f} tag areas "
            f"(at least {min_region_in_tags:g} needed) — place them around the object, not next to each other"
        )
    return None


def points_inside(points: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Boolean per point: inside (or on the edge of) the region."""
    contour = region.astype(np.float32).reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0 for x, y in points], dtype=bool)


def polygon_inside_fraction(polygon: np.ndarray, region: np.ndarray) -> float:
    """
    Share of the polygon's area (an object's segmentation mask outline) that
    lies inside the region, 0..1. Both are rasterized on the same pixel grid,
    limited to their joint bounding box, so the cost is the size of the
    object, not of the frame.
    """
    if len(polygon) < 3:
        return 0.0
    both = np.vstack([polygon, region])
    x0, y0 = np.floor(both.min(axis=0)).astype(int)
    x1, y1 = np.ceil(both.max(axis=0)).astype(int)
    shape = (y1 - y0 + 1, x1 - x0 + 1)
    offset = np.array([x0, y0], dtype=np.float64)
    obj = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(obj, [np.round(polygon - offset).astype(np.int32)], 1)
    area = int(obj.sum())
    if area == 0:
        return 0.0
    inside = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(inside, [np.round(region - offset).astype(np.int32)], 1)
    return float(np.logical_and(obj, inside).sum()) / area
