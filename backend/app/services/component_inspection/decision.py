"""
Per-component inspection status from a stream of observations.

    NOT_INSPECTED      no valid observation yet
    PRESENT            confirmed present (confirm_present_after valid "present" in a row)
    MISSING_CANDIDATE  valid "absent" observations so far, fewer than needed to confirm
    MISSING_CONFIRMED  confirm_missing_after valid "absent" in a row
    UNCERTAIN          observable, but the checks can't tell (or no check is set up)
    OCCLUDED           can't be seen from here (hidden by other CAD parts / facing away)

Rules:
- Only valid observations count (visibility.py). Tracking loss, blur,
  out-of-frame etc. never move a component towards missing or present; they
  only explain why it is not being judged right now.
- A confirmed result is sticky: it changes only on the opposite evidence,
  repeated (`clear_missing_after` presents to clear a confirmed missing),
  never on a single frame — no flicker between present and missing.
- A missing streak must be consistent: it restarts on a "present"
  observation, when it goes stale (no valid observation for
  `stale_after_s`), and when the viewpoint moves more than
  `viewpoint_reset_deg` from where it started — confirmation then needs a
  fresh run of observations from the new viewpoint.
- Each component is independent; one missing screw changes nothing else.

Every status carries the reason it was given.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from app.services.component_inspection.verifiers import Evidence
from app.services.component_inspection.visibility import Visibility


class ComponentStatus(str, Enum):
    NOT_INSPECTED = "NOT_INSPECTED"
    PRESENT = "PRESENT"
    MISSING_CANDIDATE = "MISSING_CANDIDATE"
    MISSING_CONFIRMED = "MISSING_CONFIRMED"
    UNCERTAIN = "UNCERTAIN"
    OCCLUDED = "OCCLUDED"


@dataclass
class DecisionConfig:
    confirm_missing_after: int = 6  # valid "absent" observations in a row
    confirm_present_after: int = 3
    clear_missing_after: int = 4  # valid "present" observations to clear a confirmed missing
    uncertain_after: int = 3  # valid but undecidable observations before showing UNCERTAIN
    stale_after_s: float = 5.0  # a streak with no valid observation for this long restarts
    viewpoint_reset_deg: float = 30.0


@dataclass
class ComponentState:
    status: ComponentStatus = ComponentStatus.NOT_INSPECTED
    reason: str = "not observed yet"
    confirmed: ComponentStatus | None = None  # last confirmed result (PRESENT / MISSING_CONFIRMED)
    absent_streak: int = 0
    present_streak: int = 0
    uncertain_streak: int = 0
    streak_view: np.ndarray | None = None
    last_valid_s: float | None = None  # time of the last valid observation
    last_observed_s: float | None = None
    p_present: float | None = None  # smoothed evidence score over valid observations
    observable: bool = False  # in the latest frame
    valid_observations: int = 0
    history: list[str] = field(default_factory=list)  # status changes with reasons, newest last


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


class ComponentDecision:
    """The state machine for one component."""

    HISTORY = 20

    def __init__(self, config: DecisionConfig | None = None):
        self.config = config or DecisionConfig()
        self.state = ComponentState()

    def _set(self, status: ComponentStatus, reason: str, now: float) -> bool:
        s = self.state
        changed = status != s.status
        s.status, s.reason = status, reason
        if changed:
            s.history = (s.history + [f"{now:.2f}s {status.value}: {reason}"])[-self.HISTORY :]
        return changed

    def _restart_streaks(self) -> None:
        s = self.state
        s.absent_streak = s.present_streak = s.uncertain_streak = 0
        s.streak_view = None

    def _settled(self, now: float, fallback_reason: str) -> bool:
        """Back to the last confirmed result (or nothing) after a streak ends."""
        s = self.state
        if s.confirmed is not None:
            return self._set(s.confirmed, f"{fallback_reason}; keeping the confirmed result", now)
        return self._set(ComponentStatus.NOT_INSPECTED, fallback_reason, now)

    def observe(
        self,
        now: float,
        visibility: Visibility,
        evidence: Evidence | None,
        view_dir: np.ndarray | None,
    ) -> bool:
        """One frame's observation. Returns whether the status changed."""
        cfg, s = self.config, self.state
        s.last_observed_s = now
        s.observable = visibility.observable

        if s.last_valid_s is not None and now - s.last_valid_s > cfg.stale_after_s and (s.absent_streak or s.present_streak):
            self._restart_streaks()
            if s.status == ComponentStatus.MISSING_CANDIDATE:
                self._settled(now, f"no valid view for {cfg.stale_after_s:.0f} s; missing evidence expired")

        if not visibility.observable or evidence is None:
            reason = "; ".join(visibility.reasons) or "not observable"
            if s.status in (ComponentStatus.PRESENT, ComponentStatus.MISSING_CONFIRMED):
                return False  # confirmed results stand; the reason it isn't re-checked is in `observable`
            if s.status == ComponentStatus.MISSING_CANDIDATE:
                s.reason = f"missing {s.absent_streak}/{cfg.confirm_missing_after}, on hold: {reason}"
                return False
            occluded = visibility.status in ("occluded", "facing_away")
            return self._set(ComponentStatus.OCCLUDED if occluded else ComponentStatus.UNCERTAIN, reason, now)

        # A valid observation.
        s.valid_observations += 1
        s.last_valid_s = now
        s.p_present = evidence.p_present if s.p_present is None else 0.7 * s.p_present + 0.3 * evidence.p_present

        if view_dir is not None and s.streak_view is not None and s.absent_streak:
            moved = _angle_deg(view_dir, s.streak_view)
            if moved > cfg.viewpoint_reset_deg:
                self._restart_streaks()
                if s.status == ComponentStatus.MISSING_CANDIDATE:
                    self._settled(now, f"viewpoint moved {moved:.0f} deg; re-evaluating from here")

        if evidence.decision == "absent":
            s.present_streak = s.uncertain_streak = 0
            if s.absent_streak == 0:
                s.streak_view = view_dir
            s.absent_streak += 1
            if s.confirmed == ComponentStatus.MISSING_CONFIRMED:
                return self._set(
                    ComponentStatus.MISSING_CONFIRMED,
                    f"absent in {s.absent_streak} consecutive valid observations — {evidence.reason}",
                    now,
                )
            if s.absent_streak >= cfg.confirm_missing_after:
                s.confirmed = ComponentStatus.MISSING_CONFIRMED
                return self._set(
                    ComponentStatus.MISSING_CONFIRMED,
                    f"absent in {s.absent_streak} consecutive valid observations — {evidence.reason}",
                    now,
                )
            return self._set(
                ComponentStatus.MISSING_CANDIDATE,
                f"absent {s.absent_streak}/{cfg.confirm_missing_after} — {evidence.reason}",
                now,
            )

        if evidence.decision == "present":
            s.absent_streak = s.uncertain_streak = 0
            s.streak_view = None
            s.present_streak += 1
            needed = cfg.clear_missing_after if s.confirmed == ComponentStatus.MISSING_CONFIRMED else cfg.confirm_present_after
            if s.status == ComponentStatus.MISSING_CANDIDATE and s.confirmed != ComponentStatus.MISSING_CONFIRMED:
                return self._settled(now, f"seen present again — {evidence.reason}")
            if s.confirmed == ComponentStatus.PRESENT or s.present_streak >= needed:
                s.confirmed = ComponentStatus.PRESENT
                return self._set(ComponentStatus.PRESENT, evidence.reason, now)
            if s.confirmed == ComponentStatus.MISSING_CONFIRMED:
                return self._set(
                    ComponentStatus.MISSING_CONFIRMED,
                    f"present {s.present_streak}/{needed} needed to clear — {evidence.reason}",
                    now,
                )
            return self._set(s.status if s.status != ComponentStatus.OCCLUDED else ComponentStatus.NOT_INSPECTED,
                             f"present {s.present_streak}/{needed} — {evidence.reason}", now)

        # Valid view, but the checks can't tell: doesn't break a missing streak, doesn't confirm anything.
        s.uncertain_streak += 1
        if s.status == ComponentStatus.MISSING_CANDIDATE or s.confirmed is not None:
            s.reason = f"{s.reason.split(' | ')[0]} | latest view inconclusive: {evidence.reason}"
            return False
        if s.uncertain_streak >= cfg.uncertain_after:
            return self._set(ComponentStatus.UNCERTAIN, evidence.reason, now)
        return False


@dataclass
class InspectionSummary:
    required: int
    present: int
    missing: int
    candidates: int
    uncertain: int  # UNCERTAIN + OCCLUDED
    not_inspected: int
    verdict: str  # PASS / FAIL / INCOMPLETE


def summarize(states: list[tuple[bool, ComponentStatus]]) -> InspectionSummary:
    """(required, status) per component -> counts and the object's verdict.
    PASS only when every required component is confirmed present."""
    required = [status for req, status in states if req]
    count = lambda *kinds: sum(1 for s in required if s in kinds)  # noqa: E731
    missing = count(ComponentStatus.MISSING_CONFIRMED)
    present = count(ComponentStatus.PRESENT)
    verdict = "FAIL" if missing else "PASS" if required and present == len(required) else "INCOMPLETE"
    return InspectionSummary(
        required=len(required),
        present=present,
        missing=missing,
        candidates=count(ComponentStatus.MISSING_CANDIDATE),
        uncertain=count(ComponentStatus.UNCERTAIN, ComponentStatus.OCCLUDED),
        not_inspected=count(ComponentStatus.NOT_INSPECTED),
        verdict=verdict,
    )
