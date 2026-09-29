"""
State detection (Project.md #18): "what discrete, observable state is the
target component in?" — deliberately simpler than general action recognition.

Phase 1 uses a config-driven presence/absence rule per component (does the
detector see the component's class or not), which is enough to distinguish
e.g. PCB_INSTALLED vs PCB_REMOVED without a trained state classifier. This is
swappable later for a learned per-component state classifier (Project.md #62)
without changing the QA engine or procedure engine, which only see the
resulting (state_id, confidence) pair.
"""

from dataclasses import dataclass

from app.models.enums import ActionType
from app.schemas.vision import Detection

# Whether a step's *expected* (post-action) state implies the target
# component's class should still be visually detectable, or that it should
# have disappeared from view. Used by the QA engine to decide whether the
# "required component detected" gate applies (Project.md #8) — e.g. a
# REMOVE step's success condition is the component's *absence*, so PASS must
# not require it to be detected.
_ABSENCE_EXPECTING_ACTIONS = {ActionType.REMOVE, ActionType.DISCONNECT, ActionType.OPEN}


def action_expects_presence(action_type: ActionType) -> bool:
    return action_type not in _ABSENCE_EXPECTING_ACTIONS


@dataclass
class ComponentStateRule:
    """
    Config for one target component (Project.md #62 - configuration-driven,
    not hard-coded to one assembly).
    """

    component_id: str
    class_label: str
    present_state_id: str
    absent_state_id: str


class StateEstimationError(ValueError):
    pass


class StateEstimator:
    model_version = "presence-absence-rule-v1"

    def __init__(self, rules: list[ComponentStateRule]):
        self._rules_by_component: dict[str, ComponentStateRule] = {r.component_id: r for r in rules}

    def estimate(self, detections: list[Detection], target_component_id: str) -> tuple[str, float]:
        """Returns (state_id, confidence). Raises if no rule is configured for the component."""
        rule = self._rules_by_component.get(target_component_id)
        if rule is None:
            raise StateEstimationError(f"No state rule configured for component {target_component_id!r}")

        matches = [d for d in detections if d.class_label == rule.class_label]
        if matches:
            confidence = max(d.confidence for d in matches)
            return rule.present_state_id, confidence

        # Absence is itself an observation, not a failure to observe — but we
        # report lower confidence than a direct detection would, since a
        # negative is inherently less certain from a single frame.
        return rule.absent_state_id, 0.90
