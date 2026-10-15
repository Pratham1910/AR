"""
Runs component inspection on the live AR session's frames.

Given each frame with the pose the AR session computed for that same frame
(T_camera_object) and its camera matrix, at most every `interval_s`:

    project every required component -> CAD occlusion -> visibility checks
    -> verifiers on the observable ones -> per-component decision

Modes: "inspect" (the above), "learn" (collect the good state's appearance
for ReferenceAppearanceVerifier; no decisions), "paused" (keep results,
observe nothing). Each pass is appended to a JSON-lines evidence log, and
every status change is logged with its reason.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from app.services.component_inspection.decision import ComponentDecision, DecisionConfig, InspectionSummary, summarize
from app.services.component_inspection.projection import Projection, cad_visibility, project_component
from app.services.component_inspection.registry import ComponentRegistry
from app.services.component_inspection.verifiers import (
    ComponentVerifier,
    Evidence,
    Observation,
    ReferenceAppearanceVerifier,
    ReferenceLearner,
    ReferenceSet,
    combine,
)
from app.services.component_inspection.visibility import Visibility, VisibilityConfig, assess

log = logging.getLogger("app.inspection")

InspectionMode = Literal["inspect", "learn", "paused"]


@dataclass
class InspectionConfig:
    interval_s: float = 0.25  # inspection passes at most this often (tracking runs every frame regardless)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    visibility: VisibilityConfig = field(default_factory=VisibilityConfig)
    occlusion_max_faces: int = 60000
    occlusion_depth_bias_m: float = 5e-4


@dataclass
class ComponentFrame:
    """What the latest pass saw for one component (debug view)."""

    projection: Projection | None = None
    visibility: Visibility | None = None
    evidence: Evidence | None = None


@dataclass
class PassResult:
    ran: bool
    message: str | None = None  # e.g. what "learn" collected, once it finishes


class InspectionSession:
    def __init__(
        self,
        model_id: str,
        registry: ComponentRegistry,
        verifiers: list[ComponentVerifier],
        config: InspectionConfig | None = None,
        reference_path: Path | None = None,
        log_path: Path | None = None,
        run_id: str = "",
    ):
        self.model_id = model_id
        self.run_id = run_id
        self.registry = registry
        self.verifiers = verifiers
        self.config = config or InspectionConfig()
        self.reference_path = reference_path
        self.log_path = log_path
        self.mode: InspectionMode = "inspect"
        self.decisions = {c.id: ComponentDecision(self.config.decision) for c in registry.components}
        self.frames: dict[str, ComponentFrame] = {c.id: ComponentFrame() for c in registry.components}
        self.learner: ReferenceLearner | None = None
        self.last_pass_s: float | None = None
        self.passes = 0
        self.occlusion_note: str | None = None
        self.focus: set[str] | None = None  # the current procedure step's components, when set
        self.started_wall = datetime.now(timezone.utc)
        for c in registry.components:
            if not c.required:
                self.decisions[c.id].state.reason = "optional — not part of this inspection"
            elif not c.inspectable:
                self.decisions[c.id].state.reason = f"cannot inspect: {c.problem}"

    # ── mode ────────────────────────────────────────────────────────────

    def set_mode(self, mode: InspectionMode) -> str | None:
        """Switches mode; leaving "learn" saves what was learned and returns a summary."""
        if mode == self.mode:
            return None
        message = None
        if self.mode == "learn" and self.learner is not None:
            message = self._finish_learning()
        if mode == "learn":
            self.learner = ReferenceLearner()
        self.mode = mode
        log.info("[INSPECTION] %s: mode -> %s", self.model_id[:8], mode)
        return message

    def _finish_learning(self) -> str:
        assert self.learner is not None
        counts = self.learner.counts()
        learned = self.learner.finish(self.model_id, datetime.now(timezone.utc).isoformat())
        self.learner = None
        required = [c.id for c in self.registry.components if c.required and c.inspectable]
        missing = [cid for cid in required if cid not in learned.components]
        if not learned.components:
            return (f"Nothing learned: no component was seen clearly at least {ReferenceLearner.MIN_SAMPLES} times. "
                    "Keep the object tracked and in view while learning.")
        previous = ReferenceSet.load(self.reference_path) if self.reference_path else None
        if previous is not None and previous.model_id == self.model_id:
            # Learning again adds to (replaces per component) what was learned before.
            previous.components.update(learned.components)
            learned.components = previous.components
        if self.reference_path is not None:
            learned.save(self.reference_path)
        for verifier in self.verifiers:
            if isinstance(verifier, ReferenceAppearanceVerifier):
                verifier.references = learned
        text = f"Learned the good state of {len([c for c in required if c in learned.components])}/{len(required)} components"
        if missing:
            shown = ", ".join(f"{cid} ({counts.get(cid, 0)} views)" for cid in missing[:6])
            text += f"; not enough clear views of {shown}{'…' if len(missing) > 6 else ''}"
        log.info("[INSPECTION] %s: %s", self.model_id[:8], text)
        return text

    # ── per frame ───────────────────────────────────────────────────────

    def process(
        self,
        frame_bgr: np.ndarray,
        t_co: np.ndarray | None,
        camera_matrix: np.ndarray,
        tracking_ok: bool,
        tracking_reason: str | None,
        now: float | None = None,
        hand_mask: np.ndarray | None = None,
        focus: list[str] | None = None,
    ) -> PassResult:
        """`hand_mask`: hand pixels in this frame (Phase 7). A component a hand
        covers is OCCLUDED for the frame; evidence never comes from hand pixels."""
        now = time.monotonic() if now is None else now
        if self.mode == "paused":
            return PassResult(False)
        if self.last_pass_s is not None and now - self.last_pass_s < self.config.interval_s:
            return PassResult(False)
        self.last_pass_s = now
        self.passes += 1

        self.focus = set(focus) if focus else None
        targets = [
            (i, c) for i, c in enumerate(self.registry.components)
            if c.required and c.inspectable and (self.focus is None or c.id in self.focus)
        ]
        shape = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if t_co is None:
            for _, c in targets:
                self.frames[c.id] = ComponentFrame()
                visibility = Visibility(False, "tracking", [tracking_reason or "no pose for this frame"])
                self._observe(c.id, now, visibility, None, None)
            self._write_log(now, tracking_ok)
            return PassResult(True)

        projections = {c.id: project_component(c.geometry, t_co, camera_matrix, shape) for _, c in targets}
        rois = {i: projections[c.id].roi for i, c in targets if projections[c.id].roi is not None}
        occlusion = cad_visibility(
            self.registry.scene, rois, t_co, camera_matrix, shape,
            depth_bias_m=self.config.occlusion_depth_bias_m, max_faces=self.config.occlusion_max_faces,
        )
        self.occlusion_note = (
            None if occlusion is not None
            else f"CAD occlusion not evaluated: more than {self.config.occlusion_max_faces} triangles in view (using facing only)"
        )

        for i, c in targets:
            projection = projections[c.id]
            cad = visible = None
            if occlusion is not None:
                cad = occlusion.fractions.get(i)
                visible = occlusion.visible_mask(i, projection)
            hand_cover = None
            if hand_mask is not None and projection.roi is not None and projection.mask is not None:
                x1, y1, x2, y2 = projection.roi
                seen = visible if visible is not None else projection.mask
                on_hand = seen & hand_mask[y1:y2, x1:x2]
                hand_cover = float(on_hand.sum() / max(1, seen.sum()))
                visible = seen & ~on_hand  # evidence only from pixels that are neither hidden nor hand
            visibility = assess(projection, gray, tracking_ok, tracking_reason, cad, self.config.visibility,
                                int(visible.sum()) if visible is not None else None, hand_cover)
            evidence = None
            if visibility.observable:
                observation = Observation(c, frame_bgr, gray, projection, visible)
                if self.mode == "learn":
                    assert self.learner is not None
                    self.learner.add(observation)
                else:
                    verdicts = [v.verify(observation) for v in self.verifiers if v.applies_to(c)]
                    evidence = combine(verdicts)
            self.frames[c.id] = ComponentFrame(projection, visibility, evidence)
            if self.mode == "inspect":
                self._observe(c.id, now, visibility, evidence, projection.view_dir)
        self._write_log(now, tracking_ok)
        return PassResult(True)

    def _observe(self, cid: str, now: float, visibility: Visibility, evidence: Evidence | None, view_dir) -> None:
        decision = self.decisions[cid]
        before = decision.state.status
        if decision.observe(now, visibility, evidence, view_dir):
            log.info("[INSPECTION] %s %s: %s -> %s (%s)", self.model_id[:8], cid, before.value,
                     decision.state.status.value, decision.state.reason)

    def summary(self) -> InspectionSummary:
        # A required component that can't be inspected (mesh not found) stays
        # NOT_INSPECTED, so the object can't pass while it's unresolved.
        components = [c for c in self.registry.components if self.focus is None or c.id in self.focus]
        return summarize([(c.required, self.decisions[c.id].state.status) for c in components])

    def wall_time(self, monotonic_s: float | None) -> str | None:
        """A monotonic timestamp from this session as ISO wall-clock time, for display."""
        if monotonic_s is None:
            return None
        offset = time.time() - time.monotonic()
        return datetime.fromtimestamp(monotonic_s + offset, timezone.utc).isoformat()

    def _write_log(self, now: float, tracking_ok: bool) -> None:
        if self.log_path is None:
            return
        entry = {
            "t": round(now, 3),
            "mode": self.mode,
            "tracking_ok": tracking_ok,
            "components": [
                {
                    "id": cid,
                    "status": self.decisions[cid].state.status.value,
                    "observable": f.visibility.observable if f.visibility else False,
                    "visibility": f.visibility.status if f.visibility else None,
                    "why_not": f.visibility.reasons if f.visibility and not f.visibility.observable else None,
                    "decision": f.evidence.decision if f.evidence else None,
                    "p_present": round(f.evidence.p_present, 3) if f.evidence else None,
                    "evidence": f.evidence.reason if f.evidence else None,
                    "metrics": {k: round(v, 3) for k, v in f.visibility.metrics.items()} if f.visibility else None,
                }
                for cid, f in self.frames.items()
                if f.visibility is not None
            ],
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        except OSError as exc:  # the log is evidence for debugging; losing it must be visible
            log.warning("[INSPECTION] could not write %s: %s", self.log_path, exc)
