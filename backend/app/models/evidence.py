"""Evidence — the traceable artifact backing a QA decision (Project.md #33)."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.enums import EvidenceType


class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    inspection_step_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("inspection_steps.id"), nullable=False)
    storage_key: Mapped[str] = mapped_column(String, nullable=False)  # MinIO object key
    type: Mapped[EvidenceType] = mapped_column(Enum(EvidenceType, name="evidence_type"), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    inspection_step: Mapped["InspectionStep"] = relationship()
