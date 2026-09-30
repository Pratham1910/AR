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

    # Anchor offset: the model's own local transform relative to the tracked
    # reference frame (an ArUco marker's face, or a feature-tracked reference
    # photo's plane), in the reference frame's own local coordinates.
    # Registration/tracking gives you the pose of a FLAT PATCH on the object
    # (a marker, or a label photo) — it does not know where that patch sits
    # relative to the 3D model's own origin/orientation, so without this
    # offset the model gets planted directly at the patch's pose, which is
    # visibly wrong for anything but a perfectly flat, centrally-labeled
    # object. This offset is calibrated once per model (nudge in the AR
    # overlay UI until it lines up, then save) and reused every frame:
    # model_world_pose = reference_plane_pose ∘ anchor_offset.
    anchor_offset_x: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    anchor_offset_y: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    anchor_offset_z: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    anchor_rotation_x: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    anchor_rotation_y: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    anchor_rotation_z: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    anchor_rotation_w: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    asset: Mapped["Asset"] = relationship()
    component: Mapped["Component | None"] = relationship()
