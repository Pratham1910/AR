"""
Rules for assigning tag ids to products — the checks behind the mapping API,
kept free of the database so they're testable on plain dicts.

A valid set is four different ids from the dictionary in use, none of which
already marks a corner of another product.
"""

from app.models.enums import MarkerCorner


class TagSetError(ValueError):
    """The requested tag set is not acceptable. `conflict` is True when the
    reason is another product's existing mapping (HTTP 409 rather than 400)."""

    def __init__(self, message: str, conflict: bool = False):
        super().__init__(message)
        self.conflict = conflict


def validate_tag_set(
    requested: dict[MarkerCorner, int],
    asset_id: str,
    taken: dict[int, tuple[str, str]],
    dictionary_size: int,
) -> None:
    """`taken`: tag id -> (asset id, asset name) for every mapped tag in this
    dictionary, including this product's own current set (which it may reuse
    or rearrange)."""
    missing = [c.value for c in MarkerCorner if c not in requested]
    if missing:
        raise TagSetError(f"A product needs a tag on every corner; missing: {', '.join(missing)}")

    ids = list(requested.values())
    out_of_range = sorted({i for i in ids if not 0 <= i < dictionary_size})
    if out_of_range:
        raise TagSetError(
            f"Tag id(s) {out_of_range} are not in this dictionary — it has ids 0 to {dictionary_size - 1}."
        )

    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise TagSetError(f"Each corner needs its own tag: id(s) {repeated} are used more than once.")

    clashes = [(i, taken[i][1]) for i in sorted(ids) if i in taken and taken[i][0] != asset_id]
    if clashes:
        listed = ", ".join(f"{i} ({name})" for i, name in clashes)
        raise TagSetError(f"Already assigned to another product: {listed}", conflict=True)


def suggest_free_ids(taken_ids: set[int], dictionary_size: int, count: int = 4) -> list[int]:
    """The lowest `count` unused ids, or fewer if the dictionary is nearly full."""
    free = [i for i in range(dictionary_size) if i not in taken_ids]
    return free[:count]
