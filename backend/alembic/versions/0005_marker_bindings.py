"""add marker_bindings (marker id -> asset)

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-03

See app/models/marker.py.
"""

from alembic import op

from app.models.marker import MarkerBinding

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # checkfirst: 0001 builds the schema with create_all from the *current*
    # models, so on a fresh database this table already exists.
    MarkerBinding.__table__.create(bind=op.get_bind(), checkfirst=True)


def downgrade() -> None:
    op.drop_table("marker_bindings")
