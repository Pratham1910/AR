"""
InspectionRun / InspectionStep / Observation.

An InspectionRun is one operator's execution of a procedure revision against
a physical asset. Each InspectionStep stores the QA engine's decision
(Project.md #8) for one Step, and never overwrites the original AI observation
even on manual override (Project.md #44) — manual_* fields sit alongside it.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.enums import InspectionRunStatus, QAResult


class InspectionRun(Base):
    __tablename__ = "inspection_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("assets.id"), nullable=False)
    procedure_revision_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("procedure_revisions.id"), nullable=False)
    operator: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[InspectionRunStatus] = mapped_column(
        Enum(InspectionRunStatus, name="inspection_run_status"), default=InspectionRunStatus.IN_PROGRESS
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    procedure_revision: Mapped["ProcedureRevision"] = relationship()
    steps: Mapped[list["InspectionStep"]] = relationship(
        back_populates="inspection_run", cascade="all, delete-orphan"
    )


class InspectionStep(Base):
    __tablename__ = "inspection_steps"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    inspection_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("inspection_runs.id"), nullable=False)
    step_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("steps.id"), nullable=False)

    result: Mapped[QAResult] = mapped_column(Enum(QAResult, name="qa_result"), default=QAResult.NOT_EVALUATED)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    observed_state_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("states.id"), nullable=True)
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Manual override (Project.md #44) — original AI result is never overwritten.
    manual_override_result: Mapped[QAResult | None] = mapped_column(Enum(QAResult, name="qa_result"), nullable=True)
    manual_override_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    manual_override_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    manual_override_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    inspection_run: Mapped["InspectionRun"] = relationship(back_populates="steps")
    step: Mapped["Step"] = relationship()
    observed_state: Mapped["State | None"] = relationship()
    observations: Mapped[list["Observation"]] = relationship(
        back_populates="inspection_step", cascade="all, delete-orphan"
    )


class Observation(Base):
    """Raw vision output for one InspectionStep — never itself a PASS/FAIL (Project.md #3)."""

    __tablename__ = "observations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    inspection_step_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("inspection_steps.id"), nullable=False)
    detected_components: Mapped[dict] = mapped_column(JSONB, default=dict)
    raw_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    frame_ref: Mapped[str | None] = mapped_column(String, nullable=True)  # MinIO key of the source frame

    inspection_step: Mapped["InspectionStep"] = relationship(back_populates="observations")
