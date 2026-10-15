"""add models_3d anchor offset (position + rotation)

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-30

Fixes a real registration bug: pose estimation (marker or feature-tracking)
gives the pose of a flat reference patch on the object (a marker's face, or a
photographed label), not the 3D model's own origin/orientation. Placing the
model directly at that pose is visibly wrong unless the model's own local
origin happens to already coincide with the patch's position/orientation.
This offset is calibrated once per model (nudged into alignment in the AR
overlay UI, then saved) and composed onto the tracked pose every frame.
See app/models/model3d.py.
"""

from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def _add_column_if_missing(table: str, column: sa.Column) -> None:
    # 0001 builds the schema with create_all from the *current* models, so on
    # a fresh database this column already exists; only databases created
    # before it was added to the model actually need the ALTER.
    existing = {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}
    if column.name not in existing:
        op.add_column(table, column)


def upgrade() -> None:
    _add_column_if_missing("models_3d", sa.Column("anchor_offset_x", sa.Float(), nullable=False, server_default="0.0"))
    _add_column_if_missing("models_3d", sa.Column("anchor_offset_y", sa.Float(), nullable=False, server_default="0.0"))
    _add_column_if_missing("models_3d", sa.Column("anchor_offset_z", sa.Float(), nullable=False, server_default="0.0"))
    _add_column_if_missing("models_3d", sa.Column("anchor_rotation_x", sa.Float(), nullable=False, server_default="0.0"))
    _add_column_if_missing("models_3d", sa.Column("anchor_rotation_y", sa.Float(), nullable=False, server_default="0.0"))
    _add_column_if_missing("models_3d", sa.Column("anchor_rotation_z", sa.Float(), nullable=False, server_default="0.0"))
    _add_column_if_missing("models_3d", sa.Column("anchor_rotation_w", sa.Float(), nullable=False, server_default="1.0"))


def downgrade() -> None:
    op.drop_column("models_3d", "anchor_rotation_w")
    op.drop_column("models_3d", "anchor_rotation_z")
    op.drop_column("models_3d", "anchor_rotation_y")
    op.drop_column("models_3d", "anchor_rotation_x")
    op.drop_column("models_3d", "anchor_offset_z")
    op.drop_column("models_3d", "anchor_offset_y")
    op.drop_column("models_3d", "anchor_offset_x")
