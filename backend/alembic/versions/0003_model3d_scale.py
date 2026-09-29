"""add models_3d.scale

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-29

Fixes a real bug: bottle.glb's mesh is ~2m wide x ~5.3m tall in glTF units
(an unscaled Blender export), but AR overlays place models directly at
real-world positions in meters — without a per-model scale correction, the
overlay camera ends up inside the mesh at any realistic close-range distance
and nothing renders (backface-culled). See app/models/model3d.py.
"""

from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("models_3d", sa.Column("scale", sa.Float(), nullable=False, server_default="1.0"))


def downgrade() -> None:
    op.drop_column("models_3d", "scale")
