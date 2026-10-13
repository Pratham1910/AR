"""
Which product do the tags in view belong to? (docs/marker-tags.md)

Two questions, deliberately answered separately:

- lookup_by_any_tag: the single-tag product scan. One registered tag is
  enough to name its product, so its 3D model and procedure can be opened.
  A lookup, not proof the product is physically there.
- find_four_tag_set: four-tag tracking. A product qualifies only when all
  four of its tags are in view, on their registered corners, around a
  plausible area. Anything short of that is a named reason, never a guess:
  tags of two products are not combined and three tags are not "close
  enough" (tracking.py may re-project a covered tag of an already locked
  product, which is a different, time-limited thing).

Pure: tag ids and pixel corners in, a verdict out. No database, no images.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from app.models.enums import MarkerCorner
from app.services.markers.geometry import QUAD_ORDER, enclosed_region, layout_problem, points_inside
from app.services.markers.tags import TagSighting


@dataclass(frozen=True)
class TagAssignment:
    asset_id: str
    corner: MarkerCorner


Registry = dict[int, TagAssignment]  # marker id -> the product corner it marks


# --- Single tag: product scan -------------------------------------------------


class LookupStatus(str, Enum):
    NO_MARKERS = "NO_MARKERS"
    UNKNOWN_MARKERS = "UNKNOWN_MARKERS"  # tags in view, none mapped to a product
    AMBIGUOUS = "AMBIGUOUS"  # mapped tags of more than one product in view
    FOUND = "FOUND"


@dataclass
class TagLookup:
    status: LookupStatus
    asset_id: str | None = None
    matched_ids: list[int] = field(default_factory=list)  # mapped tags of asset_id in view
    unknown_ids: list[int] = field(default_factory=list)
    asset_ids: list[str] = field(default_factory=list)  # every product with a tag in view


def lookup_by_any_tag(sightings: list[TagSighting], registry: Registry) -> TagLookup:
    if not sightings:
        return TagLookup(LookupStatus.NO_MARKERS)
    unknown = sorted(s.marker_id for s in sightings if s.marker_id not in registry)
    by_asset: dict[str, list[int]] = {}
    for s in sightings:
        if s.marker_id in registry:
            by_asset.setdefault(registry[s.marker_id].asset_id, []).append(s.marker_id)
    if not by_asset:
        return TagLookup(LookupStatus.UNKNOWN_MARKERS, unknown_ids=unknown)
    if len(by_asset) > 1:
        return TagLookup(LookupStatus.AMBIGUOUS, unknown_ids=unknown, asset_ids=sorted(by_asset))
    asset_id, ids = next(iter(by_asset.items()))
    return TagLookup(LookupStatus.FOUND, asset_id=asset_id, matched_ids=sorted(ids), unknown_ids=unknown, asset_ids=[asset_id])


# --- Four tags: tracking ------------------------------------------------------


class SetStatus(str, Enum):
    NO_MARKERS = "NO_MARKERS"
    UNKNOWN_MARKERS = "UNKNOWN_MARKERS"
    DUPLICATE_MARKERS = "DUPLICATE_MARKERS"  # one of the mapped ids appears twice in the frame
    INCOMPLETE = "INCOMPLETE"  # some of one product's tags
    MIXED_PRODUCTS = "MIXED_PRODUCTS"  # tags of several products and no complete set, or a foreign tag inside the set
    MULTIPLE_PRODUCTS = "MULTIPLE_PRODUCTS"  # more than one complete set in view
    LAYOUT_INVALID = "LAYOUT_INVALID"  # all four, but not arranged as registered
    COMPLETE = "COMPLETE"  # one product, four tags, valid layout: region usable


@dataclass
class FourTagSet:
    status: SetStatus
    # The product the verdict is about: the complete one, or the one most of
    # the mapped tags in view belong to.
    asset_id: str | None = None
    tags: dict[MarkerCorner, TagSighting] = field(default_factory=dict)  # asset_id's tags in view, by corner
    missing: list[MarkerCorner] = field(default_factory=list)
    region: np.ndarray | None = None  # 4x2, QUAD_ORDER; only when COMPLETE (or for LAYOUT_INVALID, for drawing)
    unknown_ids: list[int] = field(default_factory=list)
    duplicate_ids: list[int] = field(default_factory=list)
    other_asset_ids: list[str] = field(default_factory=list)
    problem: str | None = None


def _by_asset(sightings: list[TagSighting], registry: Registry) -> dict[str, dict[MarkerCorner, TagSighting]]:
    grouped: dict[str, dict[MarkerCorner, TagSighting]] = {}
    for s in sightings:
        if s.marker_id in registry:
            assignment = registry[s.marker_id]
            grouped.setdefault(assignment.asset_id, {})[assignment.corner] = s
    return grouped


def check_set(
    asset_id: str,
    tags: dict[MarkerCorner, TagSighting],
    others: list[TagSighting],
    min_region_in_tags: float,
) -> FourTagSet:
    """Verdict on one product's four tags (all corners present): layout, and
    no other product's tag inside the area they enclose."""
    problem = layout_problem(tags, min_region_in_tags)
    if problem is not None:
        # The boundary is still returned so the operator sees what is wrong;
        # only a COMPLETE set's region is ever handed to the detector.
        return FourTagSet(SetStatus.LAYOUT_INVALID, asset_id, tags, region=enclosed_region(tags), problem=problem)
    region = enclosed_region(tags)
    if others:
        inside = points_inside(np.asarray([s.center for s in others]), region)
        if inside.any():
            ids = [s.marker_id for s, hit in zip(others, inside) if hit]
            return FourTagSet(
                SetStatus.MIXED_PRODUCTS,
                asset_id,
                tags,
                region=region,
                problem=f"tag(s) {ids} of another product are inside this product's four tags",
            )
    return FourTagSet(SetStatus.COMPLETE, asset_id, tags, region=region)


def find_four_tag_set(
    sightings: list[TagSighting],
    registry: Registry,
    duplicate_ids: list[int],
    min_region_in_tags: float,
) -> FourTagSet:
    if not sightings and not duplicate_ids:
        return FourTagSet(SetStatus.NO_MARKERS)
    unknown = sorted(s.marker_id for s in sightings if s.marker_id not in registry)
    mapped_duplicates = [i for i in duplicate_ids if i in registry]
    if mapped_duplicates:
        return FourTagSet(SetStatus.DUPLICATE_MARKERS, unknown_ids=unknown, duplicate_ids=mapped_duplicates)

    grouped = _by_asset(sightings, registry)
    if not grouped:
        return FourTagSet(SetStatus.UNKNOWN_MARKERS, unknown_ids=unknown)

    complete = [a for a, tags in grouped.items() if len(tags) == len(QUAD_ORDER)]
    if len(complete) > 1:
        return FourTagSet(SetStatus.MULTIPLE_PRODUCTS, unknown_ids=unknown, other_asset_ids=sorted(complete))
    if len(complete) == 1:
        asset_id = complete[0]
        others = [s for s in sightings if s.marker_id in registry and registry[s.marker_id].asset_id != asset_id]
        result = check_set(asset_id, grouped[asset_id], others, min_region_in_tags)
        result.unknown_ids = unknown
        result.other_asset_ids = sorted(a for a in grouped if a != asset_id)
        return result

    # No complete set: report on the product with the most tags in view.
    asset_id = max(grouped, key=lambda a: (len(grouped[a]), a))
    tags = grouped[asset_id]
    missing = [c for c in QUAD_ORDER if c not in tags]
    status = SetStatus.MIXED_PRODUCTS if len(grouped) > 1 else SetStatus.INCOMPLETE
    return FourTagSet(
        status,
        asset_id,
        tags,
        missing=missing,
        unknown_ids=unknown,
        other_asset_ids=sorted(a for a in grouped if a != asset_id),
    )
