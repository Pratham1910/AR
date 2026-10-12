"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-09-29

The Phase 1 schema, written out explicitly and frozen.

This used to be `Base.metadata.create_all()`, which builds whatever the models
look like *today* rather than what they looked like at this revision. Once
later revisions added `components.cad_node_id` and `models_3d`, a fresh
database got them here and `alembic upgrade head` then failed in 0002/0003
with DuplicateColumn. A migration must not import `app.models`: the tables,
columns and enum labels below are the ones the models had at commit 54baf08
(checked by diffing `pg_dump --schema-only` against `create_all` run on that
commit's models), and they never change again. New schema goes in a new
revision.

Constraints are left unnamed on purpose so PostgreSQL gives them the same
default names (`<table>_pkey`, `<table>_<column>_fkey`, `<table>_<column>_key`)
that `create_all` gave them on databases built before this rewrite.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


# Labels are the Python enum member *names*, which is what SQLAlchemy's Enum
# stores (app/models/enums.py). create_type=False: the types are created and
# dropped explicitly below, so each is emitted exactly once even though
# qa_result is used by two columns.
_ENUMS = {
    "revision_status": ("DRAFT", "PUBLISHED"),
    "action_type": ("REMOVE", "INSTALL", "OPEN", "CLOSE", "CONNECT", "DISCONNECT", "INSPECT"),
    "validation_method": ("OBJECT_DETECTION", "STATE_CLASSIFICATION", "TRACKING", "POSE"),
    "inspection_run_status": ("IN_PROGRESS", "COMPLETED", "ABORTED"),
    "qa_result": ("PASS", "FAIL", "UNCERTAIN", "NOT_EVALUATED", "MANUAL_REVIEW"),
    "evidence_type": ("IMAGE", "VIDEO", "JSON"),
    "user_role": (
        "ADMIN",
        "ENGINEER",
        "TECHNICAL_AUTHOR",
        "QA_INSPECTOR",
        "MAINTAINER",
        "OPERATOR",
        "REVIEWER",
    ),
}

# Children before parents, for downgrade.
_TABLES_IN_DROP_ORDER = (
    "evidence",
    "observations",
    "inspection_steps",
    "inspection_runs",
    "validation_rules",
    "steps",
    "procedure_revisions",
    "procedures",
    "components",
    "audit_logs",
    "users",
    "states",
    "assets",
)


def _enum(name: str) -> postgresql.ENUM:
    return postgresql.ENUM(*_ENUMS[name], name=name, create_type=False)


def _uuid() -> postgresql.UUID:
    return postgresql.UUID(as_uuid=True)


def upgrade() -> None:
    bind = op.get_bind()
    for name in _ENUMS:
        # checkfirst=False: a type that already exists means this database was
        # not empty, which should stop the upgrade rather than be papered over.
        _enum(name).create(bind, checkfirst=False)

    op.create_table(
        "assets",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "states",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column("state_id_str", sa.String(length=64), nullable=False, unique=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
    )
    op.create_table(
        "users",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False, unique=True),
        sa.Column("role", _enum("user_role"), nullable=False),
    )
    op.create_table(
        "audit_logs",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column("user_id", _uuid(), nullable=True),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("entity_type", sa.String(length=64), nullable=False),
        sa.Column("entity_id", sa.String(length=64), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("details", postgresql.JSONB(), nullable=False),
    )
    op.create_table(
        "components",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column("asset_id", _uuid(), sa.ForeignKey("assets.id"), nullable=False),
        sa.Column("parent_id", _uuid(), sa.ForeignKey("components.id"), nullable=True),
        sa.Column("component_id_str", sa.String(length=64), nullable=False, unique=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("class_label", sa.String(length=64), nullable=False),
    )
    op.create_table(
        "procedures",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column("procedure_id_str", sa.String(length=64), nullable=False, unique=True),
        sa.Column("asset_id", _uuid(), sa.ForeignKey("assets.id"), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
    )
    op.create_table(
        "procedure_revisions",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column("procedure_id", _uuid(), sa.ForeignKey("procedures.id"), nullable=False),
        sa.Column("revision_label", sa.String(length=16), nullable=False),
        sa.Column("status", _enum("revision_status"), nullable=False),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "steps",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column("revision_id", _uuid(), sa.ForeignKey("procedure_revisions.id"), nullable=False),
        sa.Column("step_id_str", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("starting_state_id", _uuid(), sa.ForeignKey("states.id"), nullable=False),
        sa.Column("expected_state_id", _uuid(), sa.ForeignKey("states.id"), nullable=False),
        sa.Column("action_type", _enum("action_type"), nullable=False),
        sa.Column("target_component_id", _uuid(), sa.ForeignKey("components.id"), nullable=True),
        sa.Column("order_index", sa.Integer(), nullable=False),
    )
    op.create_table(
        "validation_rules",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column("step_id", _uuid(), sa.ForeignKey("steps.id"), nullable=False),
        sa.Column("method", _enum("validation_method"), nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=False),
    )
    op.create_table(
        "inspection_runs",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column("asset_id", _uuid(), sa.ForeignKey("assets.id"), nullable=False),
        sa.Column("procedure_revision_id", _uuid(), sa.ForeignKey("procedure_revisions.id"), nullable=False),
        sa.Column("operator", sa.String(length=255), nullable=True),
        sa.Column("status", _enum("inspection_run_status"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "inspection_steps",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column("inspection_run_id", _uuid(), sa.ForeignKey("inspection_runs.id"), nullable=False),
        sa.Column("step_id", _uuid(), sa.ForeignKey("steps.id"), nullable=False),
        sa.Column("result", _enum("qa_result"), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("observed_state_id", _uuid(), sa.ForeignKey("states.id"), nullable=True),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("manual_override_result", _enum("qa_result"), nullable=True),
        sa.Column("manual_override_by", sa.String(length=255), nullable=True),
        sa.Column("manual_override_reason", sa.String(), nullable=True),
        sa.Column("manual_override_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "observations",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column("inspection_step_id", _uuid(), sa.ForeignKey("inspection_steps.id"), nullable=False),
        sa.Column("detected_components", postgresql.JSONB(), nullable=False),
        sa.Column("raw_confidence", sa.Float(), nullable=True),
        sa.Column("frame_ref", sa.String(), nullable=True),
    )
    op.create_table(
        "evidence",
        sa.Column("id", _uuid(), primary_key=True),
        sa.Column("inspection_step_id", _uuid(), sa.ForeignKey("inspection_steps.id"), nullable=False),
        sa.Column("storage_key", sa.String(), nullable=False),
        sa.Column("type", _enum("evidence_type"), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    # Drops exactly what upgrade() created, by name. The old `drop_all` dropped
    # every table of the current models, including ones later revisions own.
    for table in _TABLES_IN_DROP_ORDER:
        op.drop_table(table)
    bind = op.get_bind()
    for name in _ENUMS:
        _enum(name).drop(bind, checkfirst=False)
