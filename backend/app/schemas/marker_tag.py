"""ArUco product-tag API schemas: tag mappings, the single-tag product scan
and four-tag tracking (app/api/marker_tags.py, docs/marker-tags.md)."""

import uuid

from pydantic import BaseModel

from app.schemas.vision import Vector2


class TagConfigOut(BaseModel):
    dictionary: str  # ARUCO_DICTIONARY — every mapping and printed tag uses it
    dictionary_size: int
    supported_dictionaries: list[str]
    used_ids: list[int]
    free_ids: int
    identify_frames: int
    confirm_frames: int
    hold_ms: float


class TagSetIn(BaseModel):
    """The four tag ids of one product, by corner."""

    top_left: int
    top_right: int
    bottom_right: int
    bottom_left: int


class TagSetOut(BaseModel):
    model_config = {"protected_namespaces": ()}  # model_* means the 3D model here

    asset_id: uuid.UUID
    asset_name: str
    dictionary: str
    top_left: int
    top_right: int
    bottom_right: int
    bottom_left: int
    # The class four-tag tracking expects inside the tags (the detection class
    # of the product's newest 3D model that has one). None: any object inside
    # the tags is accepted and its class reported.
    detection_class: str | None
    model_name: str | None
    has_model_procedure: bool
    procedure_count: int


class SuggestedIdsOut(BaseModel):
    dictionary: str
    ids: list[int]


class ProductModelOut(BaseModel):
    id: uuid.UUID
    name: str
    url: str  # /static/models/<file>, relative to the API
    scale: float


class ProductProcedureOut(BaseModel):
    """An inspection procedure of the product (Procedure inspection tab)."""

    id: uuid.UUID
    procedure_id_str: str
    title: str
    published_revision_id: uuid.UUID | None  # newest published revision; None = nothing runnable yet
    step_count: int


class ProductOut(BaseModel):
    """What a tag identified, and what can be opened for it."""

    model_config = {"protected_namespaces": ()}

    asset_id: uuid.UUID
    asset_name: str
    description: str | None
    detection_class: str | None
    model: ProductModelOut | None  # newest 3D model -> "View 3D model"
    # Newest model with an attached (Vishwa) procedure -> "View procedure".
    procedure_model: ProductModelOut | None
    procedure_title: str | None
    procedures: list[ProductProcedureOut]


class FrameRequest(BaseModel):
    session_id: str  # one per camera / browser tab
    image_base64: str


class TagOut(BaseModel):
    marker_id: int
    corners_px: list[Vector2]  # the tag's own corners, cv2.aruco order
    asset_id: uuid.UUID | None  # None: id not mapped to any product
    asset_name: str | None
    corner: str | None
    estimated: bool = False  # covered this frame; re-projected from the product's visible tags


class IdentifyResponse(BaseModel):
    state: str  # NO_MARKERS | UNKNOWN_MARKERS | AMBIGUOUS | IDENTIFYING | IDENTIFIED
    message: str
    identified: bool
    tags: list[TagOut]
    candidate_asset_id: uuid.UUID | None
    candidate_asset_name: str | None
    stable_frames: int
    required_frames: int
    product: ProductOut | None  # only once identified
    frame_width: int
    frame_height: int


class DetectedObjectOut(BaseModel):
    class_label: str
    confidence: float
    bbox: list[float]  # x1, y1, x2, y2, full-frame pixels
    polygon: list[Vector2]  # segmentation mask outline, full-frame pixels
    inside: float  # share of the mask inside the tag region, 0..1
    held: bool = False  # not detected this frame; kept from a moment ago


class CornerStatusOut(BaseModel):
    """One of the candidate product's four registered tags, this frame."""

    corner: str
    marker_id: int  # the id registered for this corner
    status: str  # "seen" | "estimated" (covered, re-projected) | "missing"
    center_px: Vector2 | None  # where it is in the frame (None when missing)


class TrackResponse(BaseModel):
    state: str  # TrackState (app/services/markers/tracking.py)
    message: str
    tracking: bool
    asset_id: uuid.UUID | None
    asset_name: str | None
    tags: list[TagOut]
    corners: list[CornerStatusOut]  # the candidate product's four tags, TL, TR, BR, BL; empty without a candidate
    # "valid": four tags on their corners, region usable, detection runs;
    # "invalid": four tags but a bad arrangement (arrangement_problem says why);
    # "incomplete": fewer than four; None: no product's tags in view.
    arrangement: str | None
    arrangement_problem: str | None
    missing_corners: list[str]
    estimated_corners: list[str]  # covered tags bridged this frame
    # The quadrilateral through the four tags' outer corners (TL, TR, BR, BL):
    # the detection region when arrangement is "valid", drawn only otherwise.
    region: list[Vector2] | None
    search_rect: list[int] | None  # x, y, w, h of the image handed to YOLO
    search_area: list[Vector2] | None  # the part of it not blanked: region + context band
    expected_class: str | None
    object: DetectedObjectOut | None  # the object inside the tags (or the wrong one found there)
    rejected: list[DetectedObjectOut]  # detections mostly outside the region — ignored
    stable_frames: int
    required_frames: int
    product: ProductOut | None  # once tracking
    frame_width: int
    frame_height: int
    markers_ms: float
    yolo_ms: float | None
