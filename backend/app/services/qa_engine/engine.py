"""
Deterministic QA engine (Project.md #8, #30, #31).

IMPORTANT: AI confidence must NOT automatically equal QA approval. This module
applies fixed, configured rules to vision output — it never runs inference and
never calls a model itself (Project.md #3, #74). Inputs come from the vision
engine (detections) and state-detection engine (observed state); output is
always one of PASS / FAIL / UNCERTAIN / NOT_EVALUATED / MANUAL_REVIEW.
"""

from dataclasses import dataclass

from app.models.enums import QAResult, ValidationMethod
from app.schemas.inspection import ValidationDetail
from app.schemas.vision import Detection


@dataclass
class QAThresholds:
    """Configured, not learned — Project.md #62 (configuration-driven behavior)."""

    detection_confidence_threshold: float = 0.80
    state_confidence_threshold: float = 0.90


@dataclass
class QAInput:
    expected_state_id: str
    observed_state_id: str | None
    state_confidence: float | None
    detections: list[Detection]
    required_component_class: str | None
    required_methods: list[ValidationMethod]
    tracking_stable: bool | None = None
    pose_valid: bool | None = None
    # Whether the step's expected state implies the target component should
    # still be visually present (Project.md #8's PCB-removed example shows
    # the opposite case: "PCB detected = false" is the PASS condition). See
    # app/services/state_detection/state_engine.action_expects_presence.
    expects_presence: bool = True


@dataclass
class QADecision:
    result: QAResult
    confidence: float
    validation: ValidationDetail
    reason: str | None


class QAEngine:
    """Evaluates one step's observation against its requirement."""

    def __init__(self, thresholds: QAThresholds | None = None):
        self.thresholds = thresholds or QAThresholds()

    def evaluate(self, qa_input: QAInput) -> QADecision:
        # 1. Was the required component actually detected?
        object_detected = self._required_component_detected(qa_input)

        # 2. Did OBJECT_DETECTION confidence clear the deterministic threshold?
        detection_confidence_ok = self._max_relevant_confidence(qa_input) >= self.thresholds.detection_confidence_threshold

        # 3. Did state classification clear its threshold and match expectation?
        state_confidence = qa_input.state_confidence or 0.0
        state_confidence_ok = state_confidence >= self.thresholds.state_confidence_threshold
        state_matched = qa_input.observed_state_id == qa_input.expected_state_id

        validation = ValidationDetail(
            objectDetected=object_detected,
            trackingStable=qa_input.tracking_stable,
            poseValid=qa_input.pose_valid,
            stateMatched=state_matched,
        )

        # If a required method never produced a usable observation at all,
        # we cannot make a determination -> NOT_EVALUATED, not FAIL.
        if qa_input.observed_state_id is None:
            return QADecision(
                result=QAResult.NOT_EVALUATED,
                confidence=0.0,
                validation=validation,
                reason="No state observation was produced for this step.",
            )

        if ValidationMethod.TRACKING in qa_input.required_methods and qa_input.tracking_stable is False:
            return QADecision(
                result=QAResult.UNCERTAIN,
                confidence=state_confidence,
                validation=validation,
                reason="Tracking was unstable; result cannot be trusted.",
            )

        if ValidationMethod.POSE in qa_input.required_methods and qa_input.pose_valid is False:
            return QADecision(
                result=QAResult.UNCERTAIN,
                confidence=state_confidence,
                validation=validation,
                reason="Pose estimation failed or exceeded tolerance.",
            )

        # Deterministic PASS rule (Project.md #8):
        #   state confidence > threshold AND state matches expected AND
        #   pose/tracking valid where required. The detection-confidence /
        #   object-detected gate only applies when the expected state implies
        #   the component should still be visible — for an absence-expecting
        #   action (REMOVE/DISCONNECT/OPEN) the PASS condition is precisely
        #   that the component is *not* detected, matching Project.md #8's
        #   "PCB detected = false ... -> PASS" example.
        presence_requirement_met = (not qa_input.expects_presence) or (object_detected and detection_confidence_ok)
        if presence_requirement_met and state_confidence_ok and state_matched:
            return QADecision(
                result=QAResult.PASS,
                confidence=state_confidence,
                validation=validation,
                reason=None,
            )

        if not state_matched:
            return QADecision(
                result=QAResult.FAIL,
                confidence=state_confidence,
                validation=validation,
                reason=self._fail_reason(qa_input),
            )

        # State matched but confidence was borderline -> ask a human, don't
        # silently pass or fail.
        return QADecision(
            result=QAResult.UNCERTAIN,
            confidence=state_confidence,
            validation=validation,
            reason="Confidence below configured threshold; manual review recommended.",
        )

    def _required_component_detected(self, qa_input: QAInput) -> bool:
        if qa_input.required_component_class is None:
            return True
        return any(d.class_label == qa_input.required_component_class for d in qa_input.detections)

    def _max_relevant_confidence(self, qa_input: QAInput) -> float:
        relevant = [
            d.confidence
            for d in qa_input.detections
            if qa_input.required_component_class is None or d.class_label == qa_input.required_component_class
        ]
        return max(relevant, default=0.0)

    def _fail_reason(self, qa_input: QAInput) -> str:
        """
        The explanation must come from validation data — never hallucinated
        (Project.md #43).
        """
        detected = self._required_component_detected(qa_input)
        if qa_input.required_component_class and qa_input.expects_presence and not detected:
            return f"Required component '{qa_input.required_component_class}' was not detected."
        if qa_input.required_component_class and not qa_input.expects_presence and detected:
            return f"Component '{qa_input.required_component_class}' is still present; it was expected to be removed/disconnected."
        return (
            f"Observed state '{qa_input.observed_state_id}' does not match "
            f"expected state '{qa_input.expected_state_id}'."
        )
