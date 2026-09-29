from app.models.enums import QAResult, ValidationMethod
from app.schemas.vision import BoundingBox, Detection
from app.services.qa_engine.engine import QAEngine, QAInput, QAThresholds

ENGINE = QAEngine(QAThresholds(detection_confidence_threshold=0.80, state_confidence_threshold=0.90))


def _detection(class_label: str, confidence: float) -> Detection:
    return Detection(class_label=class_label, confidence=confidence, bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1))


def test_pass_when_state_matches_and_confidence_high():
    result = ENGINE.evaluate(
        QAInput(
            expected_state_id="STATE-PCB-REMOVED",
            observed_state_id="STATE-PCB-REMOVED",
            state_confidence=0.96,
            detections=[_detection("pcb", 0.92)],
            required_component_class="pcb",
            required_methods=[ValidationMethod.OBJECT_DETECTION, ValidationMethod.STATE_CLASSIFICATION],
        )
    )
    assert result.result == QAResult.PASS
    assert result.validation.stateMatched is True


def test_fail_when_state_does_not_match_expected():
    result = ENGINE.evaluate(
        QAInput(
            expected_state_id="STATE-CONNECTOR-DISCONNECTED",
            observed_state_id="STATE-CONNECTOR-CONNECTED",
            state_confidence=0.95,
            detections=[_detection("connector", 0.9)],
            required_component_class="connector",
            required_methods=[ValidationMethod.OBJECT_DETECTION],
        )
    )
    assert result.result == QAResult.FAIL
    assert "does not match" in result.reason


def test_uncertain_when_confidence_below_threshold_but_state_matches():
    result = ENGINE.evaluate(
        QAInput(
            expected_state_id="STATE-PCB-REMOVED",
            observed_state_id="STATE-PCB-REMOVED",
            state_confidence=0.55,
            detections=[_detection("pcb", 0.55)],
            required_component_class="pcb",
            required_methods=[ValidationMethod.STATE_CLASSIFICATION],
        )
    )
    assert result.result == QAResult.UNCERTAIN


def test_not_evaluated_when_no_observation_was_made():
    result = ENGINE.evaluate(
        QAInput(
            expected_state_id="STATE-PCB-REMOVED",
            observed_state_id=None,
            state_confidence=None,
            detections=[],
            required_component_class="pcb",
            required_methods=[ValidationMethod.OBJECT_DETECTION],
        )
    )
    assert result.result == QAResult.NOT_EVALUATED


def test_uncertain_when_tracking_required_but_unstable():
    result = ENGINE.evaluate(
        QAInput(
            expected_state_id="STATE-PCB-REMOVED",
            observed_state_id="STATE-PCB-REMOVED",
            state_confidence=0.95,
            detections=[_detection("pcb", 0.95)],
            required_component_class="pcb",
            required_methods=[ValidationMethod.TRACKING],
            tracking_stable=False,
        )
    )
    assert result.result == QAResult.UNCERTAIN


def test_pass_when_absence_expected_and_component_not_detected():
    """REMOVE-style step: PASS condition is the component being *absent*
    (Project.md #8's PCB-removed example), so the detection gate must not
    block PASS just because nothing of that class was seen."""
    result = ENGINE.evaluate(
        QAInput(
            expected_state_id="STATE-SCREWS-REMOVED",
            observed_state_id="STATE-SCREWS-REMOVED",
            state_confidence=0.90,
            detections=[],
            required_component_class="screw",
            required_methods=[ValidationMethod.OBJECT_DETECTION, ValidationMethod.STATE_CLASSIFICATION],
            expects_presence=False,
        )
    )
    assert result.result == QAResult.PASS


def test_fail_when_absence_expected_but_component_still_present():
    result = ENGINE.evaluate(
        QAInput(
            expected_state_id="STATE-SCREWS-REMOVED",
            observed_state_id="STATE-COVER-CLOSED",
            state_confidence=0.95,
            detections=[_detection("screw", 0.9)],
            required_component_class="screw",
            required_methods=[ValidationMethod.OBJECT_DETECTION],
            expects_presence=False,
        )
    )
    assert result.result == QAResult.FAIL
    assert "still present" in result.reason


def test_fail_reason_names_missing_required_component():
    result = ENGINE.evaluate(
        QAInput(
            expected_state_id="STATE-SCREWS-REMOVED",
            observed_state_id="STATE-COVER-CLOSED",
            state_confidence=0.95,
            detections=[],
            required_component_class="screw",
            required_methods=[ValidationMethod.OBJECT_DETECTION],
        )
    )
    assert result.result == QAResult.FAIL
    assert "screw" in result.reason
