"""Component registry and live component-inspection results (app/services/component_inspection)."""

from typing import Literal

from pydantic import BaseModel

ComponentStatusOut = Literal[
    "NOT_INSPECTED", "PRESENT", "MISSING_CANDIDATE", "MISSING_CONFIRMED", "UNCERTAIN", "OCCLUDED"
]


class ComponentOut(BaseModel):
    """One registry entry: what is inspected, and where it is on the object."""

    id: str
    type: str
    display_name: str
    required: bool
    inspection_mode: str
    source: Literal["mapping", "named-part", "name-match"]
    node_index: int | None
    node_name: str | None
    piece: int | None  # which body of a combined mesh node
    mesh_status: Literal["ok", "missing", "ambiguous", "piece-missing", "empty"]
    problem: str | None
    # In the tracked object's frame (the overlay group's local frame), metres.
    center_m: list[float] | None
    size_m: list[float] | None
    node_matrix: list[list[float]] | None  # the GLB node's transform in that frame (row-major 4x4)
    color: str | None = None  # "#rrggbb" chosen in the mapping; None = the UI's own pick
    reference_samples: int = 0  # frames the good state was learned from (0 = not learned)
    calibrated: bool = False  # has a two-state part calibration


class NodeOut(BaseModel):
    node_index: int
    name: str
    sanitized_name: str  # as Three.js names it
    pieces: int  # disconnected bodies in its mesh
    faces: int
    component_ids: list[str]


class ComponentRegistryOut(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    object_id: str
    source: Literal["mapping", "auto"]
    mapping_path: str | None
    components: list[ComponentOut]
    nodes: list[NodeOut]
    warnings: list[str]
    detector: str | None  # the component detector in use, if any


class VerdictOut(BaseModel):
    verifier: str
    decision: Literal["present", "absent", "uncertain"]
    confidence: float
    reason: str


class ComponentResultOut(BaseModel):
    id: str
    status: ComponentStatusOut
    reason: str
    required: bool
    score: float | None  # smoothed probability the component is present, over valid observations
    confidence: float | None  # of the latest evidence
    observable: bool  # could be judged in the latest inspected frame
    visibility: str | None  # visible / tracking / out_of_frame / too_small / occluded / facing_away / poor_image
    visibility_reasons: list[str]
    last_valid: str | None  # ISO time of the last valid observation
    valid_observations: int
    # Debug, in the frame's pixels (latest inspected frame).
    center_px: list[float] | None
    bbox_px: list[int] | None
    outline_px: list[list[float]] | None
    verdicts: list[VerdictOut]
    history: list[str]
    learn_samples: int | None = None  # "learn" mode: clear views collected so far


class InspectionSummaryOut(BaseModel):
    required: int
    present: int
    missing: int
    candidates: int
    uncertain: int
    not_inspected: int
    verdict: Literal["PASS", "FAIL", "INCOMPLETE"]


class ComponentInspectionOut(BaseModel):
    run_id: str
    mode: Literal["inspect", "learn", "paused"]
    passes: int  # inspection passes so far in this run
    inspected_this_frame: bool
    summary: InspectionSummaryOut
    components: list[ComponentResultOut]
    message: str | None = None  # e.g. what "learn" saved, on the frame it finished
    notes: list[str] = []  # e.g. CAD occlusion not evaluated, registry warnings
