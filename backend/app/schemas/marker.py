"""Marker -> product API schemas (app/api/markers.py)."""

import uuid
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.pose import Vector2


class MarkerConfigOut(BaseModel):
    family: str
    dictionary: str
    marker_count: int | None  # valid ids are 0..marker_count-1
    hold_ms: float


class MarkerBindingIn(BaseModel):
    marker_id: int = Field(ge=0)
    asset_id: uuid.UUID


class MarkerBindingOut(BaseModel):
    id: uuid.UUID
    family: str
    dictionary: str
    marker_id: int
    asset_id: uuid.UUID
    asset_name: str


class MarkerDetectRequest(BaseModel):
    image_base64: str
    # Reuse one id per camera stream to have markers held briefly after they
    # leave the frame; omit for a stateless single-frame detection.
    session_id: str | None = None


class MarkerProductRef(BaseModel):
    asset_id: uuid.UUID
    name: str


class DetectedMarkerOut(BaseModel):
    marker_id: int
    # Image pixels, in the marker's own corner order (top-left, top-right,
    # bottom-right, bottom-left of the printed pattern). Deliberately no
    # metric position here: that needs a calibrated camera (/api/vision/pose).
    corners_px: list[Vector2]
    center_px: Vector2
    visible: bool  # False = not in this frame, held from a moment ago
    ms_since_seen: float
    status: Literal["known", "unknown"]  # unknown = no product bound to this id
    product: MarkerProductRef | None = None


class MarkerDetectResponse(BaseModel):
    family: str
    dictionary: str
    frame_width: int
    frame_height: int
    markers: list[DetectedMarkerOut]


class ProductStepOut(BaseModel):
    step_id_str: str
    title: str
    action_type: str
    component: str | None
    starting_state: str
    expected_state: str


class ProductProcedureOut(BaseModel):
    procedure_id: uuid.UUID
    procedure_id_str: str
    title: str
    revision_id: uuid.UUID
    revision_label: str
    steps: list[ProductStepOut]


class ProductOut(BaseModel):
    """What a scanned marker leads to. 3D models come from /api/models3d?asset_id=."""

    asset_id: uuid.UUID
    name: str
    description: str | None
    procedures: list[ProductProcedureOut]  # latest published revision of each


class MarkerScanRequest(BaseModel):
    image_base64: str
    session_id: str  # one scan (hold timer, confirmation latch) per camera stream
    # True when the user sees this frame flipped left-right (a mirrored
    # preview), so "move left / right" are given for what they actually see.
    mirrored: bool = False


class ScanGeometryOut(BaseModel):
    """
    What the guidance was computed from (app/services/scan/positioning.py).
    All relative — shares of the frame or the scan box, in camera-image
    coordinates (+x right, +y down, before any mirroring). No real distances.
    """

    target_center: Vector2  # scan box center, frame pixels
    marker_center: Vector2
    product_center: Vector2 | None = None
    offset_x: float  # product (else marker) center minus target, in scan-box widths
    offset_y: float  # ... in scan-box heights
    marker_size: float  # marker side / frame short side
    product_fill: float | None = None  # product box / scan box
    distance: Literal["too_far", "ok", "too_close"]  # from apparent size only
    squareness: float  # 1 = marker facing the camera
    roll_deg: float
    inside_area: bool
    complete: bool  # product box not cut off by the frame edge
    speed: float | None = None  # marker sides per second


class ScanOut(BaseModel):
    """Where the automatic scan stands on this frame (app/services/scan/scan_gate.py)."""

    state: Literal[
        "SEARCHING", "MARKER_DETECTED", "VERIFYING", "MISMATCH", "POSITIONING", "HOLD_STEADY", "SCANNING", "CONFIRMED"
    ]
    message: str  # what to tell the user
    # no_marker | multiple_markers | unknown_marker | unverifiable | no_product | uncertain | mismatch | moving
    problem: str | None = None
    # show_complete_product | move_closer | move_farther | move_left | move_right | move_up | move_down | center |
    # straighten | hold_steady. Left/right are as seen in the preview (see MarkerScanRequest.mirrored).
    hints: list[str] = []
    progress: float = 0.0  # 0..1 through the hold
    stages: dict[str, bool]  # marker, product, verified, position, steady
    confirmation: int = 0  # changes each time a new scan is confirmed; the same number = still the same scan
    marker_id: int | None = None
    product: MarkerProductRef | None = None  # what the marker says this is
    expected_class: str | None = None  # the detector class that product should show up as
    detected_class: str | None = None  # what the detector saw at the marker
    detected_confidence: float | None = None
    product_bbox: list[float] | None = None  # [x1, y1, x2, y2] of that object
    scan_area: list[float]  # [x1, y1, x2, y2] of the scan box, frame pixels
    geometry: ScanGeometryOut | None = None  # whenever a single marker is in view


class MarkerScanResponse(MarkerDetectResponse):
    scan: ScanOut
