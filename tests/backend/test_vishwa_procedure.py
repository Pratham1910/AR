import pytest

from app.services.model3d.vishwa_procedure import (
    ProcedureFormatError,
    expected_vision_facts,
    normalize_package,
    summarize,
)


def _procedure():
    # Shape of Vishwa's flask-cap.procedure.json (trimmed).
    return {
        "schemaVersion": "2.0.0",
        "id": "flask-cap-removal",
        "title": "Flask cap: remove and refit",
        "stateModels": [
            {
                "id": "cap",
                "states": [
                    {"id": "CAP_ON", "allOf": [{"kind": "partPresent", "part": "Cap", "source": "vision"}]},
                    {"id": "CAP_OFF", "allOf": [{"kind": "partAbsent", "part": "Cap", "source": "vision"}]},
                ],
            }
        ],
        "steps": [
            {
                "id": "s1",
                "expectedState": [{"modelId": "cap", "stateId": "CAP_OFF"}],
                "actions": [
                    {"id": "a1", "type": "rotate", "targetParts": ["Cap"], "axis": [0, 1, 0], "angleDeg": 720},
                    {"id": "a2", "type": "translate", "targetParts": ["Cap"], "by": [0, 5, 0]},
                ],
            },
            {
                "id": "s2",
                "expectedState": [{"modelId": "cap", "stateId": "CAP_ON"}],
                "actions": [{"id": "a3", "type": "highlight", "targetParts": ["Cap", "Ring"]}],
            },
        ],
    }


def test_accepts_exported_package_and_bare_procedure_alike():
    wrapped = normalize_package({"schemaVersion": "2.0.0", "procedure": _procedure()})
    bare = normalize_package(_procedure())
    assert wrapped == bare
    assert wrapped["procedure"]["id"] == "flask-cap-removal"


def test_keeps_fields_tvasta_does_not_use():
    proc = _procedure()
    proc["stepGraph"] = {"nodes": ["s1", "s2"], "edges": []}
    assert normalize_package(proc)["procedure"]["stepGraph"] == proc["stepGraph"]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda p: p.update(schemaVersion="1.0.0"), "schema version"),
        (lambda p: p.update(steps=[]), "no steps"),
        (lambda p: p["steps"][0].pop("id"), "Step 1 has no id"),
        (lambda p: p["steps"][1]["actions"].append({"id": "x"}), "malformed actions"),
    ],
)
def test_rejects_what_tvasta_cannot_play(mutate, message):
    proc = _procedure()
    mutate(proc)
    with pytest.raises(ProcedureFormatError, match=message):
        normalize_package(proc)


def test_rejects_non_object():
    with pytest.raises(ProcedureFormatError):
        normalize_package([1, 2])


def test_expected_vision_facts_follow_the_state_model():
    proc = _procedure()
    assert expected_vision_facts(proc, proc["steps"][0]) == [{"kind": "partAbsent", "part": "Cap"}]
    assert expected_vision_facts(proc, proc["steps"][1]) == [{"kind": "partPresent", "part": "Cap"}]
    assert expected_vision_facts(proc, {"id": "s3", "expectedState": [{"modelId": "nope", "stateId": "X"}]}) == []


def test_summary_lists_animated_and_camera_checked_parts():
    s = summarize(normalize_package(_procedure()))
    assert (s.steps, s.actions) == (2, 3)
    assert s.target_parts == ["Cap", "Ring"]
    assert s.vision_parts == ["Cap"]
