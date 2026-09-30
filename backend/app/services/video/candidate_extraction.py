"""
Video-to-procedure candidate extraction (Project.md #28/#29, Phase 3):

    Video -> frame extraction -> detection -> state estimation ->
    temporal segmentation -> candidate steps -> human review

This module never publishes a procedure — it only proposes
`app.schemas.procedure.CandidateStep`s (status always `pending_review`),
per Project.md #28's explicit requirement that video-derived candidates
require human approval before becoming a production procedure. A separate
authoring workflow (not built here) is responsible for turning approved
candidates into a real `ProcedureDefinition` via the existing
`POST /api/procedures` endpoint.

Deliberately reuses the same `ComponentStateRule`/`StateEstimator` as the
live inspection flow (app/services/state_detection/state_engine.py) rather
than a separate video-specific state model — one state-detection
implementation, two callers (live camera, recorded video).
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from app.schemas.procedure import CandidateStep
from app.services.state_detection.state_engine import ComponentStateRule, StateEstimationError, StateEstimator
from app.services.vision.detector import Detector


@dataclass
class DebounceConfig:
    """
    A state must be observed this many consecutive sampled frames before
    it's accepted as "the" current state for that component — filters
    single-frame detector flicker (a missed detection, a motion blur frame)
    out of the transition timeline, rather than treating every noisy blip as
    a real state change.
    """

    min_consecutive_frames: int = 2


def extract_candidates_from_frames(
    frames: Iterable[tuple[int, np.ndarray]],
    detector: Detector,
    rules: list[ComponentStateRule],
    source_video: str | None = None,
    debounce: DebounceConfig | None = None,
) -> list[CandidateStep]:
    """
    Runs `detector` over each frame, estimates each rule's (component's)
    state, and emits one CandidateStep per debounced state transition per
    component, in the order frames were supplied.
    """
    debounce = debounce or DebounceConfig()
    estimator = StateEstimator(rules)

    confirmed_state: dict[str, str | None] = {rule.component_id: None for rule in rules}
    pending_state: dict[str, str | None] = {rule.component_id: None for rule in rules}
    pending_run_length: dict[str, int] = {rule.component_id: 0 for rule in rules}

    candidates: list[CandidateStep] = []

    for _frame_index, frame in frames:
        detections = detector.detect(frame)

        for rule in rules:
            try:
                observed_state, confidence = estimator.estimate(detections, rule.component_id)
            except StateEstimationError:
                continue

            if observed_state == pending_state[rule.component_id]:
                pending_run_length[rule.component_id] += 1
            else:
                pending_state[rule.component_id] = observed_state
                pending_run_length[rule.component_id] = 1

            if pending_run_length[rule.component_id] < debounce.min_consecutive_frames:
                continue  # not yet confirmed — could still be a flicker

            if confirmed_state[rule.component_id] is None:
                confirmed_state[rule.component_id] = observed_state  # first confirmed state, not a transition
                continue

            if confirmed_state[rule.component_id] == observed_state:
                continue  # no change

            action = "INSTALL" if observed_state == rule.present_state_id else "REMOVE"
            candidates.append(
                CandidateStep(
                    componentId=rule.component_id,
                    startState=confirmed_state[rule.component_id],
                    action=action,
                    endState=observed_state,
                    source_video=source_video,
                    confidence=confidence,
                )
            )
            confirmed_state[rule.component_id] = observed_state

    return candidates


def extract_candidates_from_video(
    video_path: str,
    detector: Detector,
    rules: list[ComponentStateRule],
    sample_every_n_frames: int = 15,
    debounce: DebounceConfig | None = None,
) -> list[CandidateStep]:
    from app.services.video.extraction import extract_frames  # local import: keeps this module testable without a real video file

    frames = extract_frames(video_path, sample_every_n_frames)
    return extract_candidates_from_frames(frames, detector, rules, source_video=str(video_path), debounce=debounce)
