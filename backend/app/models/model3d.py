"""
Model3D — a GLB/glTF asset attached to an Asset (and optionally scoped to one
Component), per Project.md #20/#7. Kept deliberately thin in Phase 4: the
authoritative geometry/hierarchy lives inside the GLB file itself (loaded
client-side by Three.js); this row is just the pointer + digital-thread
identifiers, per Project.md #38 (stable IDs across CAD/BOM/S1000D/physical).
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Model3D(Base):
    __tablename__ = "models_3d"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("assets.id"), nullable=False)
    component_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("components.id"), nullable=True)

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    format: Mapped[str] = mapped_column(String(16), default="glb", nullable=False)
    # Path served under the backend's /static/models mount (Project.md #20/#66
    # — binary geometry is a served file, not a DB blob; Phase 8 can move this
    # to MinIO/S3 behind the same storage_key convention as Evidence).
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    # Multiplier converting the GLB's own mesh units into real-world meters
    # (Project.md #26's coordinate-system discipline extended to model
    # authoring scale). AR/registration overlays place this model directly
    # at a real-world position in meters — if the GLB wasn't authored at
    # 1 unit = 1 meter (e.g. an unscaled Blender export), the overlay is
    # comically wrong-sized without this correction. Pure viewing
    # (ThreeViewer.tsx) auto-frames relative to the model's own bounding box
    # and doesn't need it; AR overlays (RegistrationOverlay.tsx) do.
    scale: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    asset: Mapped["Asset"] = relationship()
    component: Mapped["Component | None"] = relationship()
