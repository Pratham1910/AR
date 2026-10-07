"""
MarkerBinding — which product (Asset) a printed fiducial marker identifies.

The detector only ever reports a marker id; this table is what turns that id
into a product, and through the Asset into its Model3D and Procedures. Adding
a marker or a product is a row here (PUT /api/markers/bindings), never a code
change. family + dictionary are part of the key because the same id in a
different dictionary is a different physical marker.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class MarkerBinding(Base):
    __tablename__ = "marker_bindings"
    __table_args__ = (UniqueConstraint("family", "dictionary", "marker_id", name="uq_marker_binding"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    family: Mapped[str] = mapped_column(String(32), nullable=False)  # e.g. "aruco"
    dictionary: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g. "DICT_4X4_50"
    marker_id: Mapped[int] = mapped_column(Integer, nullable=False)
    asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("assets.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    asset: Mapped["Asset"] = relationship()
