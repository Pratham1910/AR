"""add models_3d table and components.cad_node_id

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-29

Phase 4 (3D) additions (Project.md #20, #58). New tables are created via
metadata.create_all (consistent with 0001); the new column on an existing
table is added explicitly since create_all does not alter existing tables.
"""

from alembic import op
import sqlalchemy as sa

import app.models  # noqa: F401 - populates Base.metadata
from app.core.database import Base

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)  # creates models_3d
    op.add_column("components", sa.Column("cad_node_id", sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column("components", "cad_node_id")
    op.drop_table("models_3d")
