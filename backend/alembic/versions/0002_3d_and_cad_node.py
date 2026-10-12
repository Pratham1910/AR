"""add models_3d table and components.cad_node_id

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-29

Phase 4 (3D) additions (Project.md #20, #58).

`models_3d` is created explicitly in the shape it had when this revision was
written (commit ec45fc1) — without `scale` and the anchor offset, which belong
to 0003 and 0004. It used to be created with `Base.metadata.create_all()`,
which builds the table from today's model, so 0003 then failed with
DuplicateColumn on `scale`.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "models_3d",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("asset_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("assets.id"), nullable=False),
        sa.Column("component_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("components.id"), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("format", sa.String(length=16), nullable=False),
        sa.Column("storage_key", sa.String(length=512), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.add_column("components", sa.Column("cad_node_id", sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column("components", "cad_node_id")
    op.drop_table("models_3d")
