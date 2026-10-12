"""
MarkerTag — one printed ArUco tag on one corner of a product (Asset).

A product carries four tags, one per corner. The detector only reports tag
ids; these rows are what turn four ids into "this product, and these are its
corners", and through the Asset into its Model3D and Procedures. Adding a
product is four rows here (PUT /api/marker-tags/sets/{asset_id}), never a
code change.

The two unique constraints are the mapping's integrity rules, enforced by the
database as well as the API: a tag id identifies exactly one product corner,
and a product has exactly one tag per corner. `dictionary` is part of both
because the same id in another dictionary is a different physical tag.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class MarkerTag(Base):
    __tablename__ = "marker_tags"
    __table_args__ = (
        UniqueConstraint("dictionary", "marker_id", name="uq_marker_tag_marker"),
        UniqueConstraint("asset_id", "dictionary", "corner", name="uq_marker_tag_asset_corner"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("assets.id"), nullable=False)
    dictionary: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g. "DICT_4X4_50"
    marker_id: Mapped[int] = mapped_column(Integer, nullable=False)
    # A MarkerCorner value. A plain string rather than a Postgres enum type:
    # the API validates it, and a new value then needs no ALTER TYPE.
    corner: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    asset: Mapped["Asset"] = relationship()
