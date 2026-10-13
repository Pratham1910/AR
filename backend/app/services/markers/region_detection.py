"""
Find the physical object that sits between a product's four tags
(docs/marker-tags.md, "Finding the object").

The tags only say where the product *should* be; this finds what is actually
there with the existing YOLO segmentation model, and only there:

1. The detector is given the area between the tags plus a context band
   around it (the quadrilateral scaled up about its centre by
   `1 + context`), with every pixel outside that set to neutral gray. Objects
   elsewhere in the camera view are never seen. The crop is the whole area
   between the tags, never a single tag.
   The band matters: blanking exactly at the region's edge would cut an
   object that only reaches into the region down to the sliver inside, and
   that sliver would then look entirely inside. With the band the detector
   sees the crossing object whole (or most of it) and step 2 rejects it.
2. Each detection's segmentation mask is measured against the quadrilateral
   itself and kept only if most of it lies inside (`min_inside`). Objects in
   the band, and large objects that merely reach into the region, are
   rejected and reported as such.
3. Coordinates are mapped back to the full camera frame by adding the crop's
   offset — a pure translation, so masks line up with the video exactly.

No perspective rectification: warping the region to a rectangle would
stretch the object into a shape the COCO-trained model has not seen, and the
mask-against-quadrilateral test (2) already handles any perspective. The
module never runs a model itself (`segment` is passed in) and never decides
PASS/FAIL — it reports which object is inside the tags.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from app.schemas.vision import BoundingBox, SegmentedObject, Vector2
from app.services.markers.geometry import polygon_inside_fraction

Segment = Callable[[np.ndarray], list[SegmentedObject]]

# Blanked-out pixels: the mid-gray YOLO pads images with, so the cut edge
# reads as "no image here" rather than as a dark shape.
_BLANK = 114


class RegionStatus(str, Enum):
    FOUND = "FOUND"  # the expected object (or, with no expected class, an object) is inside the tags
    WRONG_OBJECT = "WRONG_OBJECT"  # objects inside, none of the product's detection class
    NOT_FOUND = "NOT_FOUND"  # nothing detected inside the tags


@dataclass
class RegionObject:
    obj: SegmentedObject  # in full-frame pixel coordinates
    inside: float  # share of its mask inside the region, 0..1


@dataclass
class RegionDetection:
    status: RegionStatus
    best: RegionObject | None = None  # the object the verdict is about
    accepted: list[RegionObject] = field(default_factory=list)  # mask mostly inside the region
    rejected: list[RegionObject] = field(default_factory=list)  # detected, but mostly outside — ignored
    search_rect: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h of what the detector was given
    search_polygon: np.ndarray | None = None  # the unblanked part of it (region + context band)


def search_area(region: np.ndarray, context: float) -> np.ndarray:
    """The quadrilateral the detector is shown: the region scaled up about its
    centre by 1 + context (same shape, same perspective)."""
    centre = region.mean(axis=0)
    return centre + (region - centre) * (1.0 + context)


def masked_crop(frame_bgr: np.ndarray, region: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """The quadrilateral's bounding rectangle (clipped to the frame) with
    everything outside the quadrilateral blanked, and that rectangle as
    (x, y, w, h)."""
    height, width = frame_bgr.shape[:2]
    x, y, w, h = cv2.boundingRect(np.round(region).astype(np.int32))
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0, 3), dtype=frame_bgr.dtype), (x0, y0, 0, 0)
    crop = frame_bgr[y0:y1, x0:x1].copy()
    keep = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(keep, [np.round(region - (x0, y0)).astype(np.int32)], 1)
    crop[keep == 0] = _BLANK
    return crop, (x0, y0, x1 - x0, y1 - y0)


def _to_frame(obj: SegmentedObject, dx: int, dy: int) -> SegmentedObject:
    return SegmentedObject(
        class_label=obj.class_label,
        confidence=obj.confidence,
        bbox=BoundingBox(x1=obj.bbox.x1 + dx, y1=obj.bbox.y1 + dy, x2=obj.bbox.x2 + dx, y2=obj.bbox.y2 + dy),
        polygon=[Vector2(x=p.x + dx, y=p.y + dy) for p in obj.polygon],
    )


def _outline(obj: SegmentedObject) -> np.ndarray:
    """The mask outline, or the box when the model gave no mask."""
    if len(obj.polygon) >= 3:
        return np.asarray([[p.x, p.y] for p in obj.polygon], dtype=np.float64)
    b = obj.bbox
    return np.asarray([[b.x1, b.y1], [b.x2, b.y1], [b.x2, b.y2], [b.x1, b.y2]], dtype=np.float64)


def _area(obj: SegmentedObject) -> float:
    return float(cv2.contourArea(_outline(obj).astype(np.float32)))


def detect_in_region(
    frame_bgr: np.ndarray,
    region: np.ndarray,
    segment: Segment,
    expected_class: str | None,
    min_inside: float,
    context: float = 0.5,
) -> RegionDetection:
    """
    The object inside the tag region. With `expected_class` (the product's
    detection class) only that class can be FOUND; without one, the most
    prominent object inside the region is taken and its class reported.
    """
    area = search_area(region, context)
    crop, rect = masked_crop(frame_bgr, area)
    if crop.size == 0:
        return RegionDetection(RegionStatus.NOT_FOUND, search_rect=rect, search_polygon=area)
    dx, dy = rect[0], rect[1]

    accepted: list[RegionObject] = []
    rejected: list[RegionObject] = []
    for raw in segment(crop):
        obj = _to_frame(raw, dx, dy)
        inside = polygon_inside_fraction(_outline(obj), region)
        (accepted if inside >= min_inside else rejected).append(RegionObject(obj, inside))

    if not accepted:
        return RegionDetection(RegionStatus.NOT_FOUND, rejected=rejected, search_rect=rect, search_polygon=area)

    if expected_class:
        wanted = expected_class.strip().lower()
        matches = [r for r in accepted if r.obj.class_label.lower() == wanted]
        if not matches:
            other = max(accepted, key=lambda r: r.obj.confidence)
            return RegionDetection(RegionStatus.WRONG_OBJECT, other, accepted, rejected, rect, area)
        best = max(matches, key=lambda r: r.obj.confidence)
    else:
        # Most prominent: confidence weighted by size, so a confident speck
        # does not beat the large object the tags were placed around.
        best = max(accepted, key=lambda r: r.obj.confidence * _area(r.obj))
    return RegionDetection(RegionStatus.FOUND, best, accepted, rejected, rect, area)
