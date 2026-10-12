"""
Is the object inside the four tags really the product they claim?

Tags only say what *should* be there — a tag sheet can sit under the wrong
part. So the object detector is shown ONLY the area the tags enclose
(crop_to_region blanks everything else before it runs), and the product's
configured detection class has to be found in there. An object of the right
class anywhere else in the camera view is never seen, so it cannot verify a
product it isn't part of.

Pure: takes an image region, detections and geometry, returns a verdict. It
never runs the detector itself (the API does, on the crop this module
prepares) and is not a QA PASS/FAIL — it only decides whether the product may
be identified.
"""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from app.schemas.vision import BoundingBox, SegmentedObject, Vector2

# What blanked-out pixels are set to: the mid-gray YOLO itself pads images
# with, so the cut edge reads as "no image here" rather than as a dark shape.
_BLANK = 114


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"  # expected class detected inside the tags
    WRONG_OBJECT = "WRONG_OBJECT"  # a different class is inside the tags, the expected one is not
    NOT_FOUND = "NOT_FOUND"  # nothing detected inside the tags
    NO_CLASS = "NO_CLASS"  # the product has no detection class configured — cannot verify
    UNKNOWN_CLASS = "UNKNOWN_CLASS"  # its class is not one the detector knows — cannot verify


@dataclass
class Verification:
    status: VerificationStatus
    expected_class: str | None
    # The object the verdict is about: the match, or the wrong object found instead.
    obj: SegmentedObject | None = None
    inside: float | None = None  # share of that object's box lying inside the tag region, 0..1


def crop_to_region(frame_bgr: np.ndarray, region: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """
    The part of the frame the detector may look at, and where it sits in the
    frame: the region's bounding rectangle, with every pixel outside the
    region itself blanked. Returns (image, (x, y) of its top-left in the frame).

    Blanking, not just cropping: the tags are rarely square to the camera, so
    the bounding rectangle of a tilted region still has corners that lie
    outside the tags, and whatever is there must not be detectable either.
    """
    height, width = frame_bgr.shape[:2]
    outline = np.round(region).astype(np.int32)
    x0, y0 = np.clip(outline.min(axis=0), 0, [width - 1, height - 1])
    x1, y1 = np.clip(outline.max(axis=0) + 1, [x0 + 1, y0 + 1], [width, height])
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, outline, 255)
    crop = frame_bgr[y0:y1, x0:x1].copy()
    crop[mask[y0:y1, x0:x1] == 0] = _BLANK
    return crop, (int(x0), int(y0))


def to_frame_coordinates(objects: list[SegmentedObject], origin: tuple[int, int]) -> list[SegmentedObject]:
    """Detections made on a crop, moved back into the full frame's pixels."""
    dx, dy = origin
    return [
        SegmentedObject(
            class_label=o.class_label,
            confidence=o.confidence,
            bbox=BoundingBox(x1=o.bbox.x1 + dx, y1=o.bbox.y1 + dy, x2=o.bbox.x2 + dx, y2=o.bbox.y2 + dy),
            polygon=[Vector2(x=p.x + dx, y=p.y + dy) for p in o.polygon],
        )
        for o in objects
    ]


def share_inside(obj: SegmentedObject, region: np.ndarray) -> float:
    """How much of the object's box lies inside the tag region (0..1). Divided
    by the BOX's area on purpose: an object that merely overlaps the region
    while mostly lying outside it scores low, however big the region is."""
    b = obj.bbox
    box_area = max(0.0, (b.x2 - b.x1) * (b.y2 - b.y1))
    if box_area <= 0.0:
        return 0.0
    box = np.array([[b.x1, b.y1], [b.x2, b.y1], [b.x2, b.y2], [b.x1, b.y2]], dtype=np.float32)
    shared, _ = cv2.intersectConvexConvex(region.astype(np.float32), box)
    return min(1.0, float(shared) / box_area)


def verify_product(
    expected_class: str | None,
    known_classes: list[str],
    objects: list[SegmentedObject],
    region: np.ndarray,
    min_inside: float = 0.6,
) -> Verification:
    """
    `objects`: what the detector found, in full-frame pixels — in the live
    scan these come from the crop_to_region image only. `region`: the area
    the product's tags enclose (identification.enclosed_region).

    The inside-share test is applied regardless of where the detections came
    from, so the rule "the object must be inside the tags" holds on its own
    and does not rest on the caller having cropped.
    """
    if not expected_class:
        return Verification(VerificationStatus.NO_CLASS, None)
    if expected_class not in known_classes:
        return Verification(VerificationStatus.UNKNOWN_CLASS, expected_class)

    inside = [(share_inside(obj, region), obj) for obj in objects]
    inside = [pair for pair in inside if pair[0] >= min_inside]

    def best(class_matches: bool) -> tuple[float, SegmentedObject] | None:
        candidates = [pair for pair in inside if (pair[1].class_label == expected_class) == class_matches]
        return max(candidates, key=lambda pair: (pair[1].confidence, pair[0])) if candidates else None

    expected = best(True)
    if expected is not None:
        return Verification(VerificationStatus.VERIFIED, expected_class, expected[1], expected[0])
    other = best(False)
    if other is not None:
        return Verification(VerificationStatus.WRONG_OBJECT, expected_class, other[1], other[0])
    return Verification(VerificationStatus.NOT_FOUND, expected_class)
