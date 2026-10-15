"""Missing-component decisions: the per-component state machine (decision.py) and evidence combining (verifiers.combine)."""

import numpy as np
import pytest

from app.services.component_inspection.decision import ComponentDecision, ComponentStatus, DecisionConfig, summarize
from app.services.component_inspection.verifiers import Evidence, Verdict, combine
from app.services.component_inspection.visibility import Visibility

S = ComponentStatus
VISIBLE = Visibility(True, "visible")
UP = np.array([0.0, 1.0, 0.0])


def ev(decision, p=None):
    p = {"present": 0.9, "absent": 0.1, "uncertain": 0.5}[decision] if p is None else p
    return Evidence(decision, p, 0.8, f"looks {decision}", [])


def run(machine, decisions, start=0.0, step=0.25, view=UP, visibility=VISIBLE):
    t = start
    for d in decisions:
        machine.observe(t, visibility, ev(d) if visibility.observable else None, view)
        t += step
    return t


def test_starts_not_inspected():
    assert ComponentDecision().state.status == S.NOT_INSPECTED


def test_missing_needs_repeated_valid_observations():
    m = ComponentDecision(DecisionConfig(confirm_missing_after=6))
    run(m, ["absent"] * 5)
    assert m.state.status == S.MISSING_CANDIDATE and "5/6" in m.state.reason
    run(m, ["absent"], start=2.0)
    assert m.state.status == S.MISSING_CONFIRMED
    assert "6 consecutive valid observations" in m.state.reason
    assert m.state.history[-1].endswith(m.state.reason)


def test_present_is_confirmed_after_a_few_observations():
    m = ComponentDecision(DecisionConfig(confirm_present_after=3))
    run(m, ["present"] * 2)
    assert m.state.status == S.NOT_INSPECTED
    run(m, ["present"], start=1.0)
    assert m.state.status == S.PRESENT


def test_a_single_present_frame_breaks_a_missing_streak():
    m = ComponentDecision(DecisionConfig(confirm_missing_after=6))
    run(m, ["absent"] * 4 + ["present"] + ["absent"] * 4)
    assert m.state.status == S.MISSING_CANDIDATE and m.state.absent_streak == 4


def test_invalid_observations_never_count():
    """Tracking loss / blur between absent frames: neither confirms nor clears."""
    m = ComponentDecision(DecisionConfig(confirm_missing_after=3))
    lost = Visibility(False, "tracking", ["tracking lost"])
    t = run(m, ["absent"] * 2)
    t = run(m, ["absent"] * 10, start=t, visibility=lost)
    assert m.state.status == S.MISSING_CANDIDATE and m.state.absent_streak == 2 and "on hold: tracking lost" in m.state.reason
    run(m, ["absent"], start=t)
    assert m.state.status == S.MISSING_CONFIRMED


def test_never_seen_component_is_occluded_or_uncertain_not_missing():
    m = ComponentDecision()
    run(m, ["absent"] * 20, visibility=Visibility(False, "occluded", ["70% hidden behind other parts"]))
    assert m.state.status == S.OCCLUDED and m.state.absent_streak == 0
    run(m, ["absent"] * 20, start=10, visibility=Visibility(False, "poor_image", ["blurred"]))
    assert m.state.status == S.UNCERTAIN and "blurred" in m.state.reason


def test_confirmed_results_survive_occlusion_and_tracking_loss():
    m = ComponentDecision(DecisionConfig(confirm_missing_after=3))
    t = run(m, ["absent"] * 3)
    t = run(m, ["present"] * 5, start=t, visibility=Visibility(False, "tracking", ["lost"]))
    assert m.state.status == S.MISSING_CONFIRMED and not m.state.observable


def test_confirmed_missing_needs_several_presents_to_clear_no_flicker():
    m = ComponentDecision(DecisionConfig(confirm_missing_after=3, clear_missing_after=4))
    t = run(m, ["absent"] * 3)
    statuses = []
    for d in ["present", "absent", "present", "present", "present"]:
        m.observe(t, VISIBLE, ev(d), UP)
        statuses.append(m.state.status)
        t += 0.25
    assert statuses == [S.MISSING_CONFIRMED] * 5  # one stray frame each way changes nothing
    run(m, ["present"], start=t)
    assert m.state.status == S.PRESENT


def test_confirmed_present_needs_a_full_missing_run_to_flip():
    m = ComponentDecision(DecisionConfig(confirm_present_after=2, confirm_missing_after=4))
    t = run(m, ["present"] * 2)
    t = run(m, ["absent"] * 3, start=t)
    assert m.state.status == S.MISSING_CANDIDATE
    run(m, ["present"], start=t)
    assert m.state.status == S.PRESENT and "keeping the confirmed result" in m.state.reason


def test_stale_missing_evidence_expires():
    m = ComponentDecision(DecisionConfig(confirm_missing_after=6, stale_after_s=5))
    run(m, ["absent"] * 3)
    lost = Visibility(False, "tracking", ["lost"])
    m.observe(10.0, lost, None, UP)
    # The candidate is dropped (and why is in the history); right now it can't be judged.
    assert m.state.status == S.UNCERTAIN and m.state.absent_streak == 0
    assert any("missing evidence expired" in h for h in m.state.history)


def test_viewpoint_change_restarts_a_missing_streak():
    m = ComponentDecision(DecisionConfig(confirm_missing_after=4, viewpoint_reset_deg=30))
    t = run(m, ["absent"] * 3, view=UP)
    side = np.array([np.sin(np.radians(45)), np.cos(np.radians(45)), 0.0])
    run(m, ["absent"], start=t, view=side)
    assert m.state.status == S.MISSING_CANDIDATE and m.state.absent_streak == 1
    assert any("viewpoint moved 45 deg" in h for h in m.state.history)


def test_small_viewpoint_drift_keeps_the_streak():
    m = ComponentDecision(DecisionConfig(confirm_missing_after=4, viewpoint_reset_deg=30))
    for k in range(4):
        a = np.radians(5 * k)
        m.observe(k * 0.25, VISIBLE, ev("absent"), np.array([np.sin(a), np.cos(a), 0.0]))
    assert m.state.status == S.MISSING_CONFIRMED


def test_undecidable_views_show_uncertain_but_do_not_break_a_streak():
    m = ComponentDecision(DecisionConfig(uncertain_after=3, confirm_missing_after=3))
    run(m, ["uncertain"] * 3)
    assert m.state.status == S.UNCERTAIN
    m2 = ComponentDecision(DecisionConfig(confirm_missing_after=3))
    run(m2, ["absent", "absent", "uncertain", "absent"])
    assert m2.state.status == S.MISSING_CONFIRMED


def test_components_are_independent():
    a, b = ComponentDecision(DecisionConfig(confirm_missing_after=2)), ComponentDecision(DecisionConfig(confirm_present_after=2))
    run(a, ["absent"] * 2)
    run(b, ["present"] * 2)
    assert (a.state.status, b.state.status) == (S.MISSING_CONFIRMED, S.PRESENT)


# ── verdict ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "states, verdict",
    [
        ([(True, S.PRESENT), (True, S.PRESENT), (False, S.MISSING_CONFIRMED)], "PASS"),  # optional parts don't count
        ([(True, S.PRESENT), (True, S.MISSING_CONFIRMED)], "FAIL"),
        ([(True, S.PRESENT), (True, S.NOT_INSPECTED)], "INCOMPLETE"),  # never pass with an uninspected part
        ([(True, S.PRESENT), (True, S.UNCERTAIN)], "INCOMPLETE"),
        ([(True, S.PRESENT), (True, S.MISSING_CANDIDATE)], "INCOMPLETE"),
        ([], "INCOMPLETE"),
    ],
)
def test_object_verdict(states, verdict):
    assert summarize(states).verdict == verdict


def test_summary_counts():
    s = summarize([(True, S.PRESENT), (True, S.MISSING_CONFIRMED), (True, S.OCCLUDED), (True, S.UNCERTAIN), (True, S.NOT_INSPECTED)])
    assert (s.required, s.present, s.missing, s.uncertain, s.not_inspected) == (5, 1, 1, 2, 1)


# ── combining verifier verdicts ─────────────────────────────────────────


def v(name, decision, confidence=0.8, negative_only=False):
    p = {"present": 0.9, "absent": 0.1, "uncertain": 0.5}[decision]
    return Verdict(name, decision, p, confidence, f"{name} says {decision}", negative_only)


def test_no_check_configured_is_uncertain_and_says_what_to_do():
    e = combine([])
    assert e.decision == "uncertain" and "learn the good state" in e.reason


def test_detector_silence_alone_is_not_missing():
    assert combine([v("detector", "absent", 0.3, negative_only=True)]).decision == "uncertain"


def test_detector_silence_plus_a_real_absent_check_is_missing():
    e = combine([v("detector", "absent", 0.3, negative_only=True), v("reference", "absent")])
    assert e.decision == "absent" and "reference" in e.reason and "detector" in e.reason


def test_disagreeing_checks_are_uncertain():
    e = combine([v("detector", "present"), v("reference", "absent")])
    assert e.decision == "uncertain" and "disagree" in e.reason


def test_low_confidence_votes_do_not_decide():
    assert combine([v("reference", "absent", confidence=0.1)]).decision == "uncertain"
    assert combine([v("calibrated", "present")]).decision == "present"
