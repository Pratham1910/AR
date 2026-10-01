"""
Procedures authored in Vishwa (the 3D authoring tool) and exported as
`.procedure.json` (procedure schema v2): steps, each with part animations
(translate / rotate / highlight …) and an expected end state whose facts can
be `partPresent` / `partAbsent` checked by the camera.

TVASTA stores one per 3D model and plays it over the tracked object. This
module only checks the structure TVASTA relies on; Vishwa's own validator is
the authority on everything else, so unknown fields are kept untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VISION_FACT_KINDS = ("partPresent", "partAbsent")


class ProcedureFormatError(ValueError):
    pass


@dataclass
class ProcedureSummary:
    procedure_id: str
    title: str
    steps: int
    actions: int
    target_parts: list[str] = field(default_factory=list)  # parts the animations move / highlight
    vision_parts: list[str] = field(default_factory=list)  # parts whose presence the camera checks


def normalize_package(data: Any) -> dict[str, Any]:
    """
    Accepts an exported package ({"schemaVersion", "procedure": {...}}) or a
    bare procedure, and returns the package form. Raises ProcedureFormatError
    with a message fit for the user when it isn't a Vishwa v2 procedure.
    """
    if not isinstance(data, dict):
        raise ProcedureFormatError("Not a procedure: expected a JSON object")
    procedure = data.get("procedure") if isinstance(data.get("procedure"), dict) else data
    version = str(procedure.get("schemaVersion") or data.get("schemaVersion") or "")
    if not version.startswith("2."):
        raise ProcedureFormatError(
            f"Unsupported procedure schema version '{version or 'missing'}' (expected 2.x, exported from Vishwa)"
        )
    if not isinstance(procedure.get("id"), str) or not procedure["id"]:
        raise ProcedureFormatError("Procedure has no id")
    steps = procedure.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ProcedureFormatError("Procedure has no steps")
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or not isinstance(step.get("id"), str):
            raise ProcedureFormatError(f"Step {i + 1} has no id")
        actions = step.get("actions", [])
        if not isinstance(actions, list) or not all(
            isinstance(a, dict) and isinstance(a.get("type"), str) for a in actions
        ):
            raise ProcedureFormatError(f"Step '{step['id']}' has malformed actions")
    return {"schemaVersion": version, "procedure": procedure}


def summarize(package: dict[str, Any]) -> ProcedureSummary:
    procedure = package["procedure"]
    steps = procedure["steps"]
    targets: list[str] = []
    for step in steps:
        for action in step.get("actions", []):
            for name in action.get("targetParts", []) or []:
                if name not in targets:
                    targets.append(name)
    vision: list[str] = []
    for step in steps:
        for fact in expected_vision_facts(procedure, step):
            if fact["part"] not in vision:
                vision.append(fact["part"])
    return ProcedureSummary(
        procedure_id=procedure["id"],
        title=procedure.get("title") or procedure["id"],
        steps=len(steps),
        actions=sum(len(s.get("actions", [])) for s in steps),
        target_parts=targets,
        vision_parts=vision,
    )


def expected_vision_facts(procedure: dict[str, Any], step: dict[str, Any]) -> list[dict[str, str]]:
    """The camera-checkable facts of a step's expected end state: [{"kind": "partAbsent", "part": "Cap"}]."""
    models = {m.get("id"): m for m in procedure.get("stateModels", []) or [] if isinstance(m, dict)}
    facts: list[dict[str, str]] = []
    for ref in step.get("expectedState", []) or []:
        model = models.get(ref.get("modelId"))
        state = next((s for s in (model or {}).get("states", []) if s.get("id") == ref.get("stateId")), None)
        for fact in (state or {}).get("allOf", []) or []:
            if fact.get("kind") in VISION_FACT_KINDS and isinstance(fact.get("part"), str):
                facts.append({"kind": fact["kind"], "part": fact["part"]})
    return facts
