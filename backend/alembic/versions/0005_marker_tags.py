"""add marker_tags (four ArUco tags per product)

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-09

See app/models/marker_tag.py.

Unconditional: 0001 is a fixed Phase 1 baseline now, so this revision is the
only thing that ever creates `marker_tags` and its downgrade drops only that.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "marker_tags",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("asset_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("assets.id"), nullable=False),
        sa.Column("dictionary", sa.String(length=64), nullable=False),
        sa.Column("marker_id", sa.Integer(), nullable=False),
        sa.Column("corner", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("dictionary", "marker_id", name="uq_marker_tag_marker"),
        sa.UniqueConstraint("asset_id", "dictionary", "corner", name="uq_marker_tag_asset_corner"),
    )


def downgrade() -> None:
    op.drop_table("marker_tags")
