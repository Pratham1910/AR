"""ArUco product-tag API schemas: tag sets (mapping) and the product scan."""

import uuid

from pydantic import BaseModel

from app.schemas.pose import Vector2


class TagSetIn(BaseModel):
    """The four tag ids of one product, by corner."""

    top_left: int
    top_right: int
    bottom_left: int
    bottom_right: int


class TagSetOut(BaseModel):
    model_config = {"protected_namespaces": ()}  # model_id / model_name mean the 3D model

    asset_id: uuid.UUID
    asset_name: str
    dictionary: str
    top_left: int
    top_right: int
    bottom_left: int
    bottom_right: int
    # What the scan will check with YOLO: the detection class of the product's
    # newest 3D model that has one. None = the product can be recognized by
    # its tags but never verified, so a scan will not confirm it.
    detection_class: str | None
    model_id: uuid.UUID | None
    model_name: str | None


class TagConfigOut(BaseModel):
    dictionary: str
    dictionary_size: int  # tag ids are 0..dictionary_size-1
    used_ids: list[int]
    free_ids: int
    confirm_frames: int


class SuggestedIdsOut(BaseModel):
    dictionary: str
    ids: list[int]  # four when available, in corner order TL, TR, BL, BR


class ScanRequest(BaseModel):
    image_base64: str
    session_id: str


class ScannedTagOut(BaseModel):
    marker_id: int
    corners_px: list[Vector2]
    held: bool  # not in this frame; remembered from a moment ago
    asset_id: uuid.UUID | None  # None = not registered to any product
    asset_name: str | None
    corner: str | None


class ScanVerificationOut(BaseModel):
    status: str  # VerificationStatus
    expected_class: str | None
    detected_class: str | None  # the object this verdict is about
    confidence: float | None
    inside: float | None  # share of that object's box inside the tag region, 0..1
    bbox: list[float] | None  # [x1, y1, x2, y2]
    polygon: list[Vector2] | None


class ScanProcedureOut(BaseModel):
    procedure_id: uuid.UUID
    procedure_id_str: str
    title: str
    published_revision_id: uuid.UUID | None


class ScanProductOut(BaseModel):
    """The identified product and what to load for it."""

    model_config = {"protected_namespaces": ()}  # model_id / model_name mean the 3D model

    asset_id: uuid.UUID
    asset_name: str
    detection_class: str | None
    model_id: uuid.UUID | None
    model_name: str | None
    model_url: str | None
    has_model_procedure: bool  # a Vishwa procedure is attached to that model
    procedures: list[ScanProcedureOut]  # the asset's inspection procedures
    # The product's top-left tag: the one marker /api/vision/pose should lock
    # onto, so the 3D overlay doesn't hop between the four.
    pose_marker_id: int


class IdentifyResponse(BaseModel):
    """Product scan: which product does a tag in view belong to? Any one
    registered tag is enough; nothing about the physical object is checked."""

    # NO_MARKERS | UNKNOWN_MARKERS | AMBIGUOUS | IDENTIFYING | IDENTIFIED
    state: str
    message: str
    identified: bool
    tags: list[ScannedTagOut]
    candidate_asset_id: uuid.UUID | None
    candidate_asset_name: str | None
    stable_frames: int
    required_frames: int
    product: ScanProductOut | None  # set only once identified
    dictionary: str
    frame_width: int
    frame_height: int


class ScanResponse(BaseModel):
    # NO_MARKERS | UNKNOWN_MARKERS | DUPLICATE_MARKERS | INCOMPLETE |
    # MIXED_PRODUCTS | MULTIPLE_PRODUCTS | LAYOUT_INVALID | CANNOT_VERIFY |
    # WRONG_OBJECT | OBJECT_NOT_FOUND | VERIFYING | CONFIRMED
    state: str
    message: str
    confirmed: bool
    tags: list[ScannedTagOut]
    # The product the tags point to, before/without confirmation — a
    # candidate only; `product` is the confirmed identity.
    candidate_asset_id: uuid.UUID | None
    candidate_asset_name: str | None
    missing_corners: list[str]
    quad: list[Vector2] | None  # tag centers TL, TR, BR, BL when all four are seen
    # The area enclosed by the four tags (outline in frame pixels) — the only
    # part of the image YOLO was shown. None until all four tags are seen.
    region: list[Vector2] | None
    verification: ScanVerificationOut | None  # None = YOLO was not run this frame
    stable_frames: int
    required_frames: int
    product: ScanProductOut | None
    dictionary: str
    frame_width: int
    frame_height: int
    markers_ms: float
    yolo_ms: float | None
