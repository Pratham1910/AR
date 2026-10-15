"""
Is the component there? Pluggable checks on the component's own pixels at
the tracked pose (projection.py), combined into one piece of evidence.

Every verifier answers present / absent / uncertain with a confidence and a
reason; the decision engine (decision.py) turns a run of them into a status.

- CalibratedPresenceVerifier: the existing two-state part calibration
  ("cap-on" / "cap-off" frames, app/services/vision/part_presence.py).
- ReferenceAppearanceVerifier: one-state. The operator shows the system the
  assembly with everything installed ("Learn good state"); a component whose
  pixels later stop looking like that is flagged. No missing-part examples
  or training needed — and it can't tell *why* something looks different,
  which is why it only ever proposes "absent" for the decision engine to
  confirm over several frames, and stands down outside the viewpoints it
  learned.
- DetectorVerifier: a trained component detector (screws, caps, ...) run on
  a crop around the expected location. Not finding one is weak evidence
  (`negative_only`): it never makes a component absent on its own.

The stock COCO detector has no screw/bolt/nut/cap classes, so it isn't used
here; DetectorVerifier takes a custom model (ComponentDetector) when one is
configured (Settings.component_detector_weights).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import cv2
import numpy as np

from app.services.component_inspection.projection import Projection
from app.services.component_inspection.registry import RegisteredComponent
from app.services.vision.part_presence import PresenceCalibration

Decision = Literal["present", "absent", "uncertain"]


@dataclass
class Observation:
    """One component in one frame, as the verifiers see it."""

    component: RegisteredComponent
    frame_bgr: np.ndarray
    gray: np.ndarray
    projection: Projection  # with roi and mask (only observable components reach the verifiers)
    # The part of projection.mask where the component is the nearest CAD
    # surface (None: occlusion not evaluated, use the whole mask).
    visible_mask: np.ndarray | None = None
    _features: dict[str, float] | None = None
    _full_features: dict[str, float] | None = None

    @property
    def features(self) -> dict[str, float]:
        """Measured on what can be seen of the component, not its hidden far side."""
        if self._features is None:
            mask = self.visible_mask if self.visible_mask is not None and self.visible_mask.any() else None
            self._features = appearance_features(self.frame_bgr, self.gray, self.projection, mask)
        return self._features

    @property
    def full_features(self) -> dict[str, float]:
        """Measured on the whole projected component — as part calibrations are."""
        if self._full_features is None:
            self._full_features = appearance_features(self.frame_bgr, self.gray, self.projection)
        return self._full_features


@dataclass
class Verdict:
    verifier: str
    decision: Decision
    p_present: float  # 0..1
    confidence: float  # 0..1: how far from the decision boundary
    reason: str
    negative_only: bool = False  # "didn't find it", not "found it missing"
    details: dict[str, float] = field(default_factory=dict)


class ComponentVerifier(Protocol):
    name: str

    def applies_to(self, component: RegisteredComponent) -> bool: ...

    def verify(self, observation: Observation) -> Verdict: ...


# ── appearance features ─────────────────────────────────────────────────

FEATURES = ("mean", "std", "p10", "p90", "edges", "a", "b")
# Spread floors: a component that looked identical in every reference frame
# still varies this much with noise, so smaller deviations aren't anomalies.
FEATURE_FLOORS = {"mean": 6.0, "std": 4.0, "p10": 8.0, "p90": 8.0, "edges": 0.05, "a": 3.0, "b": 3.0}


def appearance_features(
    frame_bgr: np.ndarray, gray: np.ndarray, projection: Projection, mask: np.ndarray | None = None
) -> dict[str, float]:
    """Grey-level statistics, edge density and colour (Lab a/b) of the
    component's pixels (`mask`, roi-sized; default: all of projection.mask)."""
    assert projection.roi is not None and projection.mask is not None
    x1, y1, x2, y2 = projection.roi
    mask = projection.mask if mask is None else mask
    g = gray[y1:y2, x1:x2]
    pixels = g[mask].astype(float)
    # Edges: a screw head's slot/rim, a connector's pins — and their absence over a bare hole.
    pad = 2
    gy1, gx1 = max(0, y1 - pad), max(0, x1 - pad)
    around = gray[gy1 : y2 + pad, gx1 : x2 + pad]
    median = float(np.median(around)) if around.size else 0.0
    edges = cv2.Canny(around, max(5.0, 0.66 * median), max(15.0, 1.33 * median)) > 0
    edges = edges[y1 - gy1 : y1 - gy1 + (y2 - y1), x1 - gx1 : x1 - gx1 + (x2 - x1)]
    lab = cv2.cvtColor(np.ascontiguousarray(frame_bgr[y1:y2, x1:x2][mask])[:, None, :], cv2.COLOR_BGR2LAB)[:, 0].astype(float)
    return {
        "mean": float(pixels.mean()),
        "std": float(pixels.std()),
        "p10": float(np.percentile(pixels, 10)),
        "p90": float(np.percentile(pixels, 90)),
        "edges": float(edges[mask].mean()),
        "a": float(lab[:, 1].mean() - 128.0),
        "b": float(lab[:, 2].mean() - 128.0),
    }


# ── calibrated two-state check (existing part calibration) ──────────────


class CalibratedPresenceVerifier:
    """The part's saved present/absent calibration (Parts panel -> Calibrate),
    for components that are a whole GLB node calibrated with the "pose" method."""

    name = "calibrated"
    OUT_OF_RANGE_SPREADS = 4.0  # further than this beyond both calibrated states: looks like neither

    def __init__(self, calibrations: list[PresenceCalibration]):
        self._by_node = {c.node_index: c for c in calibrations if c.method == "pose"}

    def applies_to(self, component: RegisteredComponent) -> bool:
        return component.piece is None and component.node_index in self._by_node

    def verify(self, observation: Observation) -> Verdict:
        cal = self._by_node[observation.component.node_index]
        # The calibration measured the whole drawn part (part_presence.project_part_mask), so measure the same.
        value = observation.full_features.get(cal.feature)
        if value is None:
            return Verdict(self.name, "uncertain", 0.5, 0.0, f"feature {cal.feature!r} not measured")
        details = {"value": value, "present_mean": cal.present_mean, "absent_mean": cal.absent_mean}
        lo_mean, hi_mean = sorted((cal.present_mean, cal.absent_mean))
        lo_std = cal.present_std if lo_mean == cal.present_mean else cal.absent_std
        hi_std = cal.absent_std if hi_mean == cal.absent_mean else cal.present_std
        if value < lo_mean - self.OUT_OF_RANGE_SPREADS * max(lo_std, 3.0) or value > hi_mean + self.OUT_OF_RANGE_SPREADS * max(hi_std, 3.0):
            return Verdict(self.name, "uncertain", 0.5, 0.0,
                           f"{cal.feature} {value:.0f} is like neither calibrated state "
                           f"(present {cal.present_mean:.0f}, absent {cal.absent_mean:.0f}) — something in the way?", details=details)
        state, confidence = cal.classify(value)
        gap = cal.absent_mean - cal.present_mean
        t = (value - cal.present_mean) / gap if abs(gap) > 1e-6 else 0.5
        p_present = float(np.clip(1.0 - t, 0.0, 1.0))
        reason = f"{cal.feature} {value:.0f}: present ~{cal.present_mean:.0f}, absent ~{cal.absent_mean:.0f}"
        return Verdict(self.name, state, p_present, confidence, reason, details=details)


# ── one-state reference ("learn good state") ────────────────────────────


@dataclass
class ComponentReference:
    samples: int
    mean: dict[str, float]
    std: dict[str, float]
    views: list[list[float]]  # unit view directions (object frame) it was learned from


@dataclass
class ReferenceSet:
    model_id: str
    components: dict[str, ComponentReference]
    created: str = ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> ReferenceSet | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return ReferenceSet(
            model_id=data["model_id"],
            components={k: ComponentReference(**v) for k, v in data["components"].items()},
            created=data.get("created", ""),
        )


class ReferenceLearner:
    """Collects observable components' features while the operator shows the
    good state, from several viewpoints."""

    MIN_SAMPLES = 5
    VIEW_SPACING_DEG = 10.0  # keep one view direction per this many degrees

    def __init__(self) -> None:
        self._samples: dict[str, list[dict[str, float]]] = {}
        self._views: dict[str, list[np.ndarray]] = {}

    def add(self, observation: Observation) -> None:
        cid = observation.component.id
        self._samples.setdefault(cid, []).append(observation.features)
        views = self._views.setdefault(cid, [])
        view = observation.projection.view_dir
        if all(_angle_deg(view, v) > self.VIEW_SPACING_DEG for v in views):
            views.append(view)

    def counts(self) -> dict[str, int]:
        return {cid: len(s) for cid, s in self._samples.items()}

    def finish(self, model_id: str, created: str) -> ReferenceSet:
        components = {}
        for cid, samples in self._samples.items():
            if len(samples) < self.MIN_SAMPLES:
                continue
            components[cid] = ComponentReference(
                samples=len(samples),
                mean={f: float(np.mean([s[f] for s in samples])) for f in FEATURES},
                std={f: float(np.std([s[f] for s in samples])) for f in FEATURES},
                views=[[float(x) for x in v] for v in self._views[cid]],
            )
        return ReferenceSet(model_id, components, created)


def _angle_deg(a: np.ndarray, b) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(a, np.asarray(b, float)), -1.0, 1.0))))


class ReferenceAppearanceVerifier:
    name = "reference"

    def __init__(
        self,
        references: ReferenceSet | None,
        present_z: float = 3.0,
        absent_z: float = 6.0,
        max_view_angle_deg: float = 35.0,
    ):
        self.references = references
        self.present_z = present_z
        self.absent_z = absent_z
        self.max_view_angle_deg = max_view_angle_deg

    def applies_to(self, component: RegisteredComponent) -> bool:
        return self.references is not None and component.id in self.references.components

    def verify(self, observation: Observation) -> Verdict:
        assert self.references is not None
        ref = self.references.components[observation.component.id]
        nearest = min((_angle_deg(observation.projection.view_dir, v) for v in ref.views), default=180.0)
        if nearest > self.max_view_angle_deg:
            return Verdict(self.name, "uncertain", 0.5, 0.0,
                           f"viewpoint {nearest:.0f} deg from any learned view — learn the good state from here too",
                           details={"view_angle_deg": nearest})
        features = observation.features
        z = {f: abs(features[f] - ref.mean[f]) / max(ref.std[f], FEATURE_FLOORS[f]) for f in FEATURES}
        worst = max(z, key=z.get)
        details = {f"z_{f}": v for f, v in z.items()} | {"view_angle_deg": nearest}
        if z[worst] <= self.present_z:
            confidence = float(np.clip(1.0 - z[worst] / self.present_z, 0.0, 1.0)) * 0.5 + 0.5
            return Verdict(self.name, "present", 0.5 + 0.5 * confidence, confidence,
                           f"matches the learned good state (largest deviation {worst} {z[worst]:.1f} sd)", details=details)
        described = f"{worst} {features[worst]:.2f} vs {ref.mean[worst]:.2f} ±{max(ref.std[worst], FEATURE_FLOORS[worst]):.2f}"
        if z[worst] >= self.absent_z:
            confidence = float(np.clip((z[worst] - self.absent_z) / self.absent_z, 0.0, 1.0)) * 0.5 + 0.5
            return Verdict(self.name, "absent", 0.5 - 0.5 * confidence, confidence,
                           f"differs from the learned good state: {described}", details=details)
        return Verdict(self.name, "uncertain", 0.5, 0.0, f"somewhat unlike the good state: {described}", details=details)


# ── trained component detector ──────────────────────────────────────────


@dataclass
class ComponentDetection:
    component_type: str  # mapped onto registry component types ("screw", "cap", ...)
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in the crop's pixels


class ComponentDetector(Protocol):
    """A model that finds components of given types in an image crop."""

    types: set[str]

    def detect(self, crop_bgr: np.ndarray) -> list[ComponentDetection]: ...


class UltralyticsComponentDetector:
    """A custom-trained YOLO model (ultralytics is already a dependency).
    `class_map` maps the model's class names to component types."""

    def __init__(self, weights: str, class_map: dict[str, str], min_confidence: float = 0.25):
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._class_map = class_map
        self.types = set(class_map.values())
        self.min_confidence = min_confidence

    def detect(self, crop_bgr: np.ndarray) -> list[ComponentDetection]:
        result = self._model(crop_bgr, conf=self.min_confidence, verbose=False)[0]
        found = []
        for box in result.boxes:
            name = result.names[int(box.cls)]
            if name in self._class_map:
                found.append(ComponentDetection(self._class_map[name], float(box.conf), tuple(float(v) for v in box.xyxy[0])))
        return found


class DetectorVerifier:
    name = "detector"

    def __init__(self, detector: ComponentDetector, context: float = 1.0, min_crop_px: int = 64):
        self.detector = detector
        self.context = context  # crop = the component's box grown by this much on each side
        self.min_crop_px = min_crop_px

    def applies_to(self, component: RegisteredComponent) -> bool:
        return component.type in self.detector.types

    def verify(self, observation: Observation) -> Verdict:
        assert observation.projection.roi is not None
        x1, y1, x2, y2 = observation.projection.roi
        h, w = observation.frame_bgr.shape[:2]
        gx = max(int((x2 - x1) * self.context), (self.min_crop_px - (x2 - x1)) // 2)
        gy = max(int((y2 - y1) * self.context), (self.min_crop_px - (y2 - y1)) // 2)
        cx1, cy1, cx2, cy2 = max(0, x1 - gx), max(0, y1 - gy), min(w, x2 + gx), min(h, y2 + gy)
        detections = self.detector.detect(observation.frame_bgr[cy1:cy2, cx1:cx2])
        # A hit counts if its centre falls on the expected place (the projected box, grown by half).
        mx, my = (x2 - x1) * 0.5, (y2 - y1) * 0.5
        hits = [
            d for d in detections
            if d.component_type == observation.component.type
            and x1 - mx <= cx1 + (d.box[0] + d.box[2]) / 2 <= x2 + mx
            and y1 - my <= cy1 + (d.box[1] + d.box[3]) / 2 <= y2 + my
        ]
        if hits:
            best = max(hits, key=lambda d: d.confidence)
            return Verdict(self.name, "present", 0.5 + 0.5 * best.confidence, best.confidence,
                           f"detector found a {best.component_type} there ({best.confidence:.0%})")
        return Verdict(self.name, "absent", 0.35, 0.3,
                       f"detector found no {observation.component.type} at the expected place", negative_only=True)


# ── combining ───────────────────────────────────────────────────────────


@dataclass
class Evidence:
    decision: Decision
    p_present: float
    confidence: float
    reason: str
    verdicts: list[Verdict]


def combine(verdicts: list[Verdict], min_confidence: float = 0.25) -> Evidence:
    """Present if a confident check says so and none confidently disagrees;
    absent only on positive evidence of absence — a detector merely not
    finding the part needs another check agreeing."""
    if not verdicts:
        return Evidence("uncertain", 0.5, 0.0,
                        "no presence check available — learn the good state or calibrate this part", [])
    confident = [v for v in verdicts if v.confidence >= min_confidence and v.decision != "uncertain"]
    present = [v for v in confident if v.decision == "present"]
    absent = [v for v in confident if v.decision == "absent"]
    solid_absent = [v for v in absent if not v.negative_only]
    p = float(np.mean([v.p_present for v in verdicts]))
    if present and absent:
        reasons = "; ".join(f"{v.verifier}: {v.reason}" for v in present + absent)
        return Evidence("uncertain", p, 0.0, f"checks disagree — {reasons}", verdicts)
    if present:
        best = max(present, key=lambda v: v.confidence)
        return Evidence("present", p, best.confidence, f"{best.verifier}: {best.reason}", verdicts)
    if solid_absent:
        best = max(solid_absent, key=lambda v: v.confidence)
        others = [v for v in absent if v is not best]
        extra = "".join(f"; {v.verifier}: {v.reason}" for v in others)
        return Evidence("absent", p, best.confidence, f"{best.verifier}: {best.reason}{extra}", verdicts)
    if absent:
        return Evidence("uncertain", p, 0.0, f"{absent[0].verifier}: {absent[0].reason} (not enough on its own)", verdicts)
    best = max(verdicts, key=lambda v: len(v.reason))
    return Evidence("uncertain", p, 0.0, f"{best.verifier}: {best.reason}", verdicts)
