"""
Which product do the tags in view identify?

A product is a candidate only when all four of its registered tags are seen
AND they sit where their corner names say. Everything short of that is a
named reason, never a guess: tags of two products are not combined, three
tags are not "close enough", and a tag stuck on the wrong corner is caught by
the layout check.

Pure geometry on tag ids and pixel corners — no images, no database, no YOLO.
Whether the right object is actually between the tags is verification.py.
"""

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from app.models.enums import MarkerCorner
from app.services.markers.tags import MarkerDetection

# Walking the product's outline: this order is clockwise as seen by the camera.
QUAD_ORDER = (MarkerCorner.TOP_LEFT, MarkerCorner.TOP_RIGHT, MarkerCorner.BOTTOM_RIGHT, MarkerCorner.BOTTOM_LEFT)


class IdentificationStatus(str, Enum):
    NO_MARKERS = "NO_MARKERS"  # nothing in view
    UNKNOWN_MARKERS = "UNKNOWN_MARKERS"  # tags in view, none registered to a product
    DUPLICATE_MARKERS = "DUPLICATE_MARKERS"  # the same tag id twice in one frame
    INCOMPLETE = "INCOMPLETE"  # some of one product's tags, not all four
    MIXED_PRODUCTS = "MIXED_PRODUCTS"  # tags of different products, no complete set / a foreign tag inside the set
    MULTIPLE_PRODUCTS = "MULTIPLE_PRODUCTS"  # two or more complete sets in view
    LAYOUT_INVALID = "LAYOUT_INVALID"  # all four tags, but not arranged as registered
    CANDIDATE = "CANDIDATE"  # one product, four tags, valid layout — go on to verification


@dataclass(frozen=True)
class TagAssignment:
    asset_id: str
    corner: MarkerCorner


@dataclass(frozen=True)
class LayoutLimits:
    max_size_ratio: float = 2.5
    max_opposite_side_ratio: float = 2.5
    min_side_in_tags: float = 1.5


@dataclass
class Identification:
    status: IdentificationStatus
    asset_id: str | None = None  # the candidate, or the product the visible tags mostly belong to
    # Tag centers in QUAD_ORDER, only when all four of asset_id's tags are seen.
    quad: np.ndarray | None = None
    tag_side_px: float | None = None  # mean apparent tag size of the set
    # The area the four tags enclose, tags included: the convex outline around
    # all 16 tag corners (Nx2 pixels). This is the only part of the image the
    # object detector is shown (verification.crop_to_region).
    region: np.ndarray | None = None
    missing_corners: list[MarkerCorner] = field(default_factory=list)
    other_asset_ids: list[str] = field(default_factory=list)
    unknown_ids: list[int] = field(default_factory=list)
    duplicate_ids: list[int] = field(default_factory=list)
    layout_problem: str | None = None


def check_layout(quad: np.ndarray, tag_sides_px: list[float], limits: LayoutLimits) -> str | None:
    """
    Why four tag centers (in QUAD_ORDER) can't be the corners they claim to
    be, or None if they can.

    The order test is the one that catches a wrong tag on a corner: going
    top-left -> top-right -> bottom-right -> bottom-left must turn the same
    way (clockwise in the image) at every corner. That holds however the
    product is rotated or tilted, and fails when two tags are swapped (the
    outline crosses itself) or the set is seen mirrored / from behind.
    """
    edges = np.roll(quad, -1, axis=0) - quad
    following = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]  # z of each corner's cross product
    if not all(t > 0 for t in turns):
        return "The tags are not in their registered corners (two may be swapped, or the product is seen mirrored)."

    if max(tag_sides_px) > limits.max_size_ratio * min(tag_sides_px):
        return "The tags differ too much in apparent size to be on one product face."

    top, right, bottom, left = (float(np.linalg.norm(e)) for e in edges)
    for a, b in ((top, bottom), (left, right)):
        if max(a, b) > limits.max_opposite_side_ratio * min(a, b):
            return "The tags form too skewed a shape — view the product more squarely."

    if min(top, right, bottom, left) < limits.min_side_in_tags * float(np.mean(tag_sides_px)):
        return "The tags are too close together to mark a product's corners."
    return None


def enclosed_region(tags: list[MarkerDetection]) -> np.ndarray:
    """The convex outline around every corner of the given tags — what "inside
    the four tags" means: from each tag's outer edges inward."""
    points = np.concatenate([t.corners for t in tags]).astype(np.float32)
    return cv2.convexHull(points).reshape(-1, 2).astype(np.float64)


def identify_product(
    detections: list[MarkerDetection],
    registry: dict[int, TagAssignment],
    limits: LayoutLimits = LayoutLimits(),
    duplicate_ids: list[int] | None = None,
) -> Identification:
    """`registry` maps tag id -> (product, corner) for the dictionary in use."""
    unknown_ids = sorted(d.marker_id for d in detections if d.marker_id not in registry)
    by_asset: dict[str, dict[MarkerCorner, MarkerDetection]] = {}
    for detection in detections:
        assignment = registry.get(detection.marker_id)
        if assignment is not None:
            by_asset.setdefault(assignment.asset_id, {})[assignment.corner] = detection

    registered_duplicates = sorted(i for i in (duplicate_ids or []) if i in registry)
    if registered_duplicates:
        return Identification(
            IdentificationStatus.DUPLICATE_MARKERS,
            asset_id=registry[registered_duplicates[0]].asset_id,
            duplicate_ids=registered_duplicates,
            unknown_ids=unknown_ids,
        )

    if not by_asset:
        status = IdentificationStatus.UNKNOWN_MARKERS if unknown_ids else IdentificationStatus.NO_MARKERS
        return Identification(status, unknown_ids=unknown_ids)

    complete = [asset_id for asset_id, corners in by_asset.items() if len(corners) == 4]
    # Most tags first; ties broken by id so the answer doesn't flicker between frames.
    ranked = sorted(by_asset, key=lambda asset_id: (-len(by_asset[asset_id]), asset_id))

    if len(complete) > 1:
        return Identification(
            IdentificationStatus.MULTIPLE_PRODUCTS,
            asset_id=None,
            other_asset_ids=sorted(complete),
            unknown_ids=unknown_ids,
        )

    if not complete:
        best = ranked[0]
        others = [a for a in ranked if a != best]
        return Identification(
            IdentificationStatus.MIXED_PRODUCTS if others else IdentificationStatus.INCOMPLETE,
            asset_id=best,
            missing_corners=[c for c in QUAD_ORDER if c not in by_asset[best]],
            other_asset_ids=others,
            unknown_ids=unknown_ids,
        )

    asset_id = complete[0]
    corners = by_asset[asset_id]
    quad = np.array([corners[c].center for c in QUAD_ORDER], dtype=np.float64)
    tag_sides = [corners[c].side_px for c in QUAD_ORDER]
    result = Identification(
        IdentificationStatus.CANDIDATE,
        asset_id=asset_id,
        quad=quad,
        tag_side_px=float(np.mean(tag_sides)),
        region=enclosed_region([corners[c] for c in QUAD_ORDER]),
        other_asset_ids=[a for a in ranked if a != asset_id],
        unknown_ids=unknown_ids,
    )

    problem = check_layout(quad, tag_sides, limits)
    if problem is not None:
        result.status, result.layout_problem = IdentificationStatus.LAYOUT_INVALID, problem
        return result

    # Another product's tag elsewhere in view is just another product nearby;
    # one INSIDE this product's four corners means the set can't be trusted.
    contour = quad.astype(np.float32)
    intruders = [
        other
        for other in result.other_asset_ids
        if any(
            cv2.pointPolygonTest(contour, (float(d.center[0]), float(d.center[1])), False) >= 0
            for d in by_asset[other].values()
        )
    ]
    if intruders:
        result.status, result.other_asset_ids = IdentificationStatus.MIXED_PRODUCTS, intruders
    return result


# --- Product lookup by any one tag ------------------------------------------
# A separate, deliberately lenient question from identify_product above:
# "which product does this tag belong to?", for opening its 3D model or
# procedure. It proves nothing about what is physically in view — no layout,
# no object check — and must not be used to confirm a product for tracking.


class TagLookupStatus(str, Enum):
    NO_MARKERS = "NO_MARKERS"  # nothing in view
    UNKNOWN_MARKERS = "UNKNOWN_MARKERS"  # tags in view, none mapped to a product
    AMBIGUOUS = "AMBIGUOUS"  # tags of more than one product in view
    FOUND = "FOUND"  # at least one tag, all mapped ones belonging to the same product


@dataclass
class TagLookup:
    status: TagLookupStatus
    asset_id: str | None = None
    matched_ids: list[int] = field(default_factory=list)  # this product's tags in view
    unknown_ids: list[int] = field(default_factory=list)
    asset_ids: list[str] = field(default_factory=list)  # every product with a tag in view


def lookup_product_by_tag(detections: list[MarkerDetection], registry: dict[int, TagAssignment]) -> TagLookup:
    """
    The product any ONE registered tag in view belongs to. Unmapped tags are
    reported but never turned into a product, and tags of two products are
    not resolved by picking one — the caller is told it is ambiguous.
    """
    unknown_ids = sorted({d.marker_id for d in detections if d.marker_id not in registry})
    by_asset: dict[str, list[int]] = {}
    for detection in detections:
        assignment = registry.get(detection.marker_id)
        if assignment is not None:
            by_asset.setdefault(assignment.asset_id, []).append(detection.marker_id)

    if not by_asset:
        status = TagLookupStatus.UNKNOWN_MARKERS if unknown_ids else TagLookupStatus.NO_MARKERS
        return TagLookup(status, unknown_ids=unknown_ids)
    asset_ids = sorted(by_asset)
    if len(asset_ids) > 1:
        return TagLookup(TagLookupStatus.AMBIGUOUS, unknown_ids=unknown_ids, asset_ids=asset_ids)
    asset_id = asset_ids[0]
    return TagLookup(
        TagLookupStatus.FOUND,
        asset_id=asset_id,
        matched_ids=sorted(set(by_asset[asset_id])),
        unknown_ids=unknown_ids,
        asset_ids=asset_ids,
    )
