"""
State, Procedure, ProcedureRevision.

The procedure is modelled as a State/Action graph, not a flat list of steps
(Project.md #5, #6, #51) — States are first-class rows here, and Steps
(app/models/step.py) reference startingState/expectedState by FK, which is
what lets a run resume mid-procedure or branch later.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.enums import RevisionStatus


class State(Base):
    """A discrete, observable state of an asset/component (e.g. COVER_OPEN)."""

    __tablename__ = "states"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    state_id_str: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)


class Procedure(Base):
    __tablename__ = "procedures"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    procedure_id_str: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("assets.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)

    asset: Mapped["Asset"] = relationship(back_populates="procedures")
    revisions: Mapped[list["ProcedureRevision"]] = relationship(
        back_populates="procedure", cascade="all, delete-orphan"
    )


class ProcedureRevision(Base):
    __tablename__ = "procedure_revisions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    procedure_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("procedures.id"), nullable=False)
    revision_label: Mapped[str] = mapped_column(String(16), nullable=False)  # e.g. "A"
    status: Mapped[RevisionStatus] = mapped_column(
        Enum(RevisionStatus, name="revision_status"), default=RevisionStatus.DRAFT, nullable=False
    )
    created_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    procedure: Mapped["Procedure"] = relationship(back_populates="revisions")
    steps: Mapped[list["Step"]] = relationship(
        back_populates="revision", cascade="all, delete-orphan", order_by="Step.order_index"
    )

    @property
    def step_count(self) -> int:
        return len(self.steps)
