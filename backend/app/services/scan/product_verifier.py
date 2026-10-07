"""
Checks that the physical object in front of the camera is the product its
marker claims. The marker gives the identity (and so the expected detector
class); this compares that against what the object detector actually sees at
the marker. It runs no model itself: detections are passed in, so the same
check works with the stock COCO model or a custom-trained product model.
"""

from dataclasses import dataclass
from enum import Enum

from app.services.markers.detector import MarkerDetection


@dataclass(frozen=True)
class ProductDetection:
    class_label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixels


class VerificationStatus(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"  # an object is at the marker, but of another class
    UNCERTAIN = "uncertain"  # the expected class is at the marker, but below the confidence required
    NO_PRODUCT = "no_product"  # nothing detected at the marker
    UNVERIFIABLE = "unverifiable"  # the detector has no class for this product


@dataclass(frozen=True)
class Verification:
    status: VerificationStatus
    expected_class: str | None
    detection: ProductDetection | None = None  # the object the verdict is about


def _side_px(marker: MarkerDetection) -> float:
    c = marker.corners_px
    return sum(((c[i][0] - c[i - 1][0]) ** 2 + (c[i][1] - c[i - 1][1]) ** 2) ** 0.5 for i in range(4)) / 4


def verify_product(
    marker: MarkerDetection,
    expected_class: str | None,
    known_classes: list[str],
    detections: list[ProductDetection],
    min_confidence: float,
    ignored_classes: frozenset[str] = frozenset(),
    reach: float = 1.0,
) -> Verification:
    """
    An object counts as "at the marker" when its box, grown by `reach` marker
    sides (the marker may sit anywhere on the product, or just beside it),
    contains the marker's center. The expected class there, at
    `min_confidence` or better, is a match; otherwise the tightest other
    confident box there is what the product was mistaken for. Detections
    below `min_confidence` never confirm or reject anything: the expected
    class seen only that weakly is reported as uncertain.
    """
    known = {c.lower() for c in known_classes}
    if expected_class is None or expected_class.lower() not in known:
        return Verification(VerificationStatus.UNVERIFIABLE, expected_class)

    cx, cy = marker.center_px
    reach_px = reach * _side_px(marker)
    near = [
        d
        for d in detections
        if d.class_label.lower() not in ignored_classes
        and d.bbox[0] - reach_px <= cx <= d.bbox[2] + reach_px
        and d.bbox[1] - reach_px <= cy <= d.bbox[3] + reach_px
    ]
    at_marker = [d for d in near if d.confidence >= min_confidence]
    matches = [d for d in at_marker if d.class_label.lower() == expected_class.lower()]
    if matches:
        return Verification(VerificationStatus.MATCH, expected_class, max(matches, key=lambda d: d.confidence))
    if at_marker:
        tightest = min(at_marker, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
        return Verification(VerificationStatus.MISMATCH, expected_class, tightest)
    weak = [d for d in near if d.class_label.lower() == expected_class.lower()]
    if weak:
        return Verification(VerificationStatus.UNCERTAIN, expected_class, max(weak, key=lambda d: d.confidence))
    return Verification(VerificationStatus.NO_PRODUCT, expected_class)
