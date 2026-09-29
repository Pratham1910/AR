import pytest

from app.models.enums import ActionType
from app.schemas.vision import BoundingBox, Detection
from app.services.state_detection.state_engine import (
    ComponentStateRule,
    StateEstimationError,
    StateEstimator,
    action_expects_presence,
)

RULE = ComponentStateRule(
    component_id="PCB-001",
    class_label="pcb",
    present_state_id="STATE-PCB-INSTALLED",
    absent_state_id="STATE-PCB-REMOVED",
)


def _detection(class_label: str, confidence: float = 0.9) -> Detection:
    return Detection(class_label=class_label, confidence=confidence, bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1))


def test_presence_maps_to_present_state():
    estimator = StateEstimator([RULE])
    state_id, confidence = estimator.estimate([_detection("pcb", 0.87)], "PCB-001")
    assert state_id == "STATE-PCB-INSTALLED"
    assert confidence == 0.87


def test_absence_maps_to_absent_state():
    estimator = StateEstimator([RULE])
    state_id, confidence = estimator.estimate([_detection("connector")], "PCB-001")
    assert state_id == "STATE-PCB-REMOVED"


def test_unconfigured_component_raises():
    estimator = StateEstimator([RULE])
    with pytest.raises(StateEstimationError):
        estimator.estimate([], "UNKNOWN-COMPONENT")


@pytest.mark.parametrize(
    "action_type,expected",
    [
        (ActionType.REMOVE, False),
        (ActionType.DISCONNECT, False),
        (ActionType.OPEN, False),
        (ActionType.INSTALL, True),
        (ActionType.CONNECT, True),
        (ActionType.CLOSE, True),
        (ActionType.INSPECT, True),
    ],
)
def test_action_expects_presence(action_type, expected):
    assert action_expects_presence(action_type) is expected
