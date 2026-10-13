"""
Rules for assigning tag ids to a product's four corners (docs/marker-tags.md).

The database enforces the same two rules with unique constraints
(app/models/marker_tag.py); checking them here first turns a clash into a
message that names the product already holding the id, instead of an
IntegrityError.
"""

from app.models.enums import MarkerCorner
from app.services.markers.tags import CORNER_LABELS


class MappingError(ValueError):
    def __init__(self, message: str, conflict: bool = False):
        super().__init__(message)
        self.conflict = conflict  # True: clashes with another product (HTTP 409), else invalid input (400)


def validate_tag_set(
    requested: dict[MarkerCorner, int],
    asset_id: str,
    taken: dict[int, tuple[str, str]],  # marker id -> (asset id, asset name) for every mapped tag
    dictionary_size: int,
) -> None:
    missing = [CORNER_LABELS[c] for c in MarkerCorner if c not in requested]
    if missing:
        raise MappingError(f"Every corner needs a tag id; missing: {', '.join(missing)}.")
    out_of_range = sorted({i for i in requested.values() if not 0 <= i < dictionary_size})
    if out_of_range:
        raise MappingError(f"Tag id(s) {out_of_range} are outside this dictionary (0-{dictionary_size - 1}).")
    seen: dict[int, MarkerCorner] = {}
    for corner, marker_id in requested.items():
        if marker_id in seen:
            raise MappingError(
                f"Tag id {marker_id} is given for both {CORNER_LABELS[seen[marker_id]]} and {CORNER_LABELS[corner]} — "
                "each corner needs its own tag."
            )
        seen[marker_id] = corner
    for corner, marker_id in requested.items():
        owner = taken.get(marker_id)
        if owner is not None and owner[0] != asset_id:
            raise MappingError(
                f"Tag id {marker_id} ({CORNER_LABELS[corner]}) is already assigned to {owner[1]}.", conflict=True
            )


def suggest_free_ids(taken_ids: set[int], dictionary_size: int, count: int = 4) -> list[int]:
    """The lowest `count` ids nobody uses (fewer if the dictionary runs out)."""
    return [i for i in range(dictionary_size) if i not in taken_ids][:count]
