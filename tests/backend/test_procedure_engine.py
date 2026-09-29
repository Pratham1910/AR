import pytest

from app.services.procedure_engine.engine import ProcedureGraph, ProcedureGraphError
from app.schemas.procedure import ProcedureDefinition

VALID_DEFINITION = {
    "procedureId": "TEST-001",
    "revision": "A",
    "title": "Test Procedure",
    "assetId": "ASSET-001",
    "states": [
        {"id": "STATE-A", "name": "A"},
        {"id": "STATE-B", "name": "B"},
        {"id": "STATE-C", "name": "C"},
    ],
    "steps": [
        {
            "id": "STEP-001",
            "title": "First",
            "startingState": "STATE-A",
            "action": {"type": "REMOVE"},
            "target": {"componentId": "COMP-1"},
            "expectedState": "STATE-B",
            "validation": {"methods": ["OBJECT_DETECTION"]},
        },
        {
            "id": "STEP-002",
            "title": "Second",
            "startingState": "STATE-B",
            "action": {"type": "OPEN"},
            "target": {"componentId": "COMP-2"},
            "expectedState": "STATE-C",
            "validation": {"methods": ["STATE_CLASSIFICATION"]},
        },
    ],
}


def _graph() -> ProcedureGraph:
    return ProcedureGraph(ProcedureDefinition.model_validate(VALID_DEFINITION))


def test_first_step_is_authored_first_step():
    graph = _graph()
    assert graph.first_step().id == "STEP-001"


def test_next_step_advances_in_order():
    graph = _graph()
    nxt = graph.next_step("STEP-001")
    assert nxt is not None
    assert nxt.id == "STEP-002"


def test_next_step_returns_none_after_last_step():
    graph = _graph()
    assert graph.next_step("STEP-002") is None


def test_resume_candidates_finds_step_matching_observed_state():
    graph = _graph()
    candidates = graph.resume_candidates("STATE-B")
    assert [s.id for s in candidates] == ["STEP-002"]


def test_resume_candidates_empty_for_terminal_state():
    graph = _graph()
    assert graph.resume_candidates("STATE-C") == []


def test_invalid_starting_state_rejected():
    bad = {**VALID_DEFINITION, "steps": [{**VALID_DEFINITION["steps"][0], "startingState": "STATE-UNKNOWN"}]}
    with pytest.raises(ProcedureGraphError):
        ProcedureGraph(ProcedureDefinition.model_validate(bad))


def test_invalid_expected_state_rejected():
    bad = {**VALID_DEFINITION, "steps": [{**VALID_DEFINITION["steps"][0], "expectedState": "STATE-UNKNOWN"}]}
    with pytest.raises(ProcedureGraphError):
        ProcedureGraph(ProcedureDefinition.model_validate(bad))
