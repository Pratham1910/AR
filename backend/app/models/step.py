"""
Step and ValidationRule.

A Step is one edge of the procedure's State -> Action -> Validation -> State
graph (Project.md #5): it points at a startingState and an expectedState by
FK (not by embedding state names), which is what makes "resume mid-procedure"
(Project.md #50) a lookup rather than a rewrite.
"""

import uuid

from sqlalchemy import Enum, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.enums import ActionType, ValidationMethod


class Step(Base):
    __tablename__ = "steps"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    revision_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("procedure_revisions.id"), nullable=False)
    step_id_str: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g. STEP-001
    title: Mapped[str] = mapped_column(String(255), nullable=False)

    starting_state_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("states.id"), nullable=False)
    expected_state_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("states.id"), nullable=False)

    action_type: Mapped[ActionType] = mapped_column(Enum(ActionType, name="action_type"), nullable=False)
    target_component_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("components.id"), nullable=True)

    order_index: Mapped[int] = mapped_column(Integer, nullable=False)

    revision: Mapped["ProcedureRevision"] = relationship(back_populates="steps")
    starting_state: Mapped["State"] = relationship(foreign_keys=[starting_state_id])
    expected_state: Mapped["State"] = relationship(foreign_keys=[expected_state_id])
    target_component: Mapped["Component | None"] = relationship()
    validation_rules: Mapped[list["ValidationRule"]] = relationship(
        back_populates="step", cascade="all, delete-orphan"
    )


class ValidationRule(Base):
    """
    One vision method a step requires (Project.md #6's validation.methods).
    `config` carries method-specific parameters (thresholds, tolerances - #31).
    """

    __tablename__ = "validation_rules"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    step_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("steps.id"), nullable=False)
    method: Mapped[ValidationMethod] = mapped_column(Enum(ValidationMethod, name="validation_method"), nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, default=dict)

    step: Mapped["Step"] = relationship(back_populates="validation_rules")
