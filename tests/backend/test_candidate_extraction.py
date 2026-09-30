"""
Verifies the temporal-segmentation/debounce logic directly against a
sequence of synthetic frames and a controllable fake detector — no real
video file or trained model needed to test the *logic*; test_extraction.py
covers the separate concern of actually reading a video file.
"""

import numpy as np

from app.schemas.vision import BoundingBox, Detection
from app.services.state_detection.state_engine import ComponentStateRule
from app.services.video.candidate_extraction import DebounceConfig, extract_candidates_from_frames
from app.services.vision.detector import Detector

PRESENT_FRAME = np.full((4, 4, 3), 255, dtype=np.uint8)
ABSENT_FRAME = np.zeros((4, 4, 3), dtype=np.uint8)

PCB_RULE = ComponentStateRule(
    component_id="PCB-001",
    class_label="pcb",
    present_state_id="STATE-PCB-INSTALLED",
    absent_state_id="STATE-PCB-REMOVED",
)


class FakeDetector(Detector):
    """Frame brightness stands in for 'is the target object visible' — fully deterministic and controllable."""

    model_version = "fake-detector-v0"

    def __init__(self, class_label: str = "pcb"):
        self._class_label = class_label

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame.mean() >= 128:
            return [Detection(class_label=self._class_label, confidence=0.9, bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1))]
        return []


def _numbered(frames: list[np.ndarray]) -> list[tuple[int, np.ndarray]]:
    return list(enumerate(frames))


def test_detects_a_single_removal_transition():
    sequence = [PRESENT_FRAME] * 3 + [ABSENT_FRAME] * 3
    candidates = extract_candidates_from_frames(_numbered(sequence), FakeDetector(), [PCB_RULE], source_video="test.mp4")

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.componentId == "PCB-001"
    assert candidate.startState == "STATE-PCB-INSTALLED"
    assert candidate.endState == "STATE-PCB-REMOVED"
    assert candidate.action == "REMOVE"
    assert candidate.source_video == "test.mp4"
    assert candidate.status.value == "pending_review"


def test_reinstall_after_removal_produces_two_candidates_in_order():
    sequence = [PRESENT_FRAME] * 3 + [ABSENT_FRAME] * 3 + [PRESENT_FRAME] * 3
    candidates = extract_candidates_from_frames(_numbered(sequence), FakeDetector(), [PCB_RULE])

    assert len(candidates) == 2
    assert candidates[0].action == "REMOVE"
    assert candidates[1].action == "INSTALL"


def test_single_frame_flicker_is_not_a_transition_with_debounce():
    # A single absent frame in the middle of an otherwise-present sequence.
    sequence = [PRESENT_FRAME] * 3 + [ABSENT_FRAME] + [PRESENT_FRAME] * 3
    candidates = extract_candidates_from_frames(
        _numbered(sequence), FakeDetector(), [PCB_RULE], debounce=DebounceConfig(min_consecutive_frames=2)
    )

    assert candidates == []


def test_flicker_without_debounce_is_detected_as_two_transitions():
    """Contrast case confirming debounce is actually doing something above."""
    sequence = [PRESENT_FRAME] * 3 + [ABSENT_FRAME] + [PRESENT_FRAME] * 3
    candidates = extract_candidates_from_frames(
        _numbered(sequence), FakeDetector(), [PCB_RULE], debounce=DebounceConfig(min_consecutive_frames=1)
    )

    assert len(candidates) == 2


def test_no_transition_when_state_never_changes():
    sequence = [PRESENT_FRAME] * 5
    candidates = extract_candidates_from_frames(_numbered(sequence), FakeDetector(), [PCB_RULE])
    assert candidates == []


def test_independent_components_tracked_separately():
    """Two components' transitions must not interfere with each other."""

    class MultiDetector(Detector):
        model_version = "multi"

        def detect(self, frame: np.ndarray) -> list[Detection]:
            dets = []
            if frame[..., 0].mean() >= 128:  # red channel -> pcb
                dets.append(Detection(class_label="pcb", confidence=0.9, bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1)))
            if frame[..., 1].mean() >= 128:  # green channel -> screw
                dets.append(Detection(class_label="screw", confidence=0.9, bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1)))
            return dets

    screw_rule = ComponentStateRule(
        component_id="SCREW-001",
        class_label="screw",
        present_state_id="STATE-SCREWS-PRESENT",
        absent_state_id="STATE-SCREWS-REMOVED",
    )

    def frame(red: int, green: int) -> np.ndarray:
        f = np.zeros((4, 4, 3), dtype=np.uint8)
        f[..., 0] = red
        f[..., 1] = green
        return f

    # PCB present+present -> removed+removed; screws stay present throughout.
    sequence = [frame(255, 255)] * 3 + [frame(0, 255)] * 3

    candidates = extract_candidates_from_frames(_numbered(sequence), MultiDetector(), [PCB_RULE, screw_rule])

    assert len(candidates) == 1
    assert candidates[0].componentId == "PCB-001"
    assert candidates[0].startState == "STATE-PCB-INSTALLED"
    assert candidates[0].endState == "STATE-PCB-REMOVED"
