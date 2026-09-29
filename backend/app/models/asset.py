"""
Asset and Component — the physical/digital things a procedure operates on.

Component supports a parent/child tree (e.g. PCB -> Connector_J1) matching the
3D hierarchy described in Project.md #20, even though Phase 1 doesn't yet load
3D models. component_id_str is the stable, human-meaningful identifier used by
procedure JSON (Project.md #6) and eventually CAD/BOM/S1000D cross-references
(Project.md #38).
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    components: Mapped[list["Component"]] = relationship(back_populates="asset", cascade="all, delete-orphan")
    procedures: Mapped[list["Procedure"]] = relationship(back_populates="asset")


class Component(Base):
    __tablename__ = "components"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("assets.id"), nullable=False)
    parent_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("components.id"), nullable=True)

    component_id_str: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    class_label: Mapped[str] = mapped_column(String(64), nullable=False)  # maps to a YOLO class

    asset: Mapped["Asset"] = relationship(back_populates="components")
    parent: Mapped["Component | None"] = relationship(remote_side=[id])
