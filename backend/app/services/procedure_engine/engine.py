"""
Procedure engine: the STATE -> ACTION -> VALIDATION -> STATE graph
(Project.md #5, #51).

This module knows nothing about vision, cameras, or QA rules (Project.md #74)
— it only knows the procedure's structure: which step follows which state,
and how to resume execution from an arbitrary observed state (Project.md #50).
"""

import json
from pathlib import Path

from app.schemas.procedure import ProcedureDefinition, StepDefinition


class ProcedureGraphError(ValueError):
    """Raised when a procedure definition is structurally invalid."""


class ProcedureGraph:
    """
    An in-memory, queryable form of a ProcedureDefinition.

    Modelled as a graph keyed by state, not a flat step list, so that future
    branching/optional/recovery steps (Project.md #51) don't require a
    different representation — they just add more edges out of a state.
    """

    def __init__(self, definition: ProcedureDefinition):
        self.definition = definition
        self._validate()
        # steps ordered as authored; also indexed by id and by starting state
        self._steps_by_id: dict[str, StepDefinition] = {s.id: s for s in definition.steps}
        self._steps_by_starting_state: dict[str, list[StepDefinition]] = {}
        for step in definition.steps:
            self._steps_by_starting_state.setdefault(step.startingState, []).append(step)

    def _validate(self) -> None:
        state_ids = self.definition.state_ids()
        for step in self.definition.steps:
            if step.startingState not in state_ids:
                raise ProcedureGraphError(
                    f"Step {step.id} references unknown startingState {step.startingState!r}"
                )
            if step.expectedState not in state_ids:
                raise ProcedureGraphError(
                    f"Step {step.id} references unknown expectedState {step.expectedState!r}"
                )

    @classmethod
    def from_file(cls, path: str | Path) -> "ProcedureGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(ProcedureDefinition.model_validate(data))

    @property
    def steps(self) -> list[StepDefinition]:
        return self.definition.steps

    def first_step(self) -> StepDefinition:
        if not self.definition.steps:
            raise ProcedureGraphError("Procedure has no steps")
        return self.definition.steps[0]

    def get_step(self, step_id: str) -> StepDefinition:
        try:
            return self._steps_by_id[step_id]
        except KeyError as exc:
            raise ProcedureGraphError(f"Unknown step id {step_id!r}") from exc

    def next_step(self, current_step_id: str) -> StepDefinition | None:
        """The next step in authored order, or None if current_step_id was last."""
        ids = [s.id for s in self.definition.steps]
        idx = ids.index(current_step_id)
        if idx + 1 >= len(ids):
            return None
        return self.definition.steps[idx + 1]

    def steps_from_state(self, state_id: str) -> list[StepDefinition]:
        """
        All steps whose startingState matches state_id — the candidate resume
        points for a technician who begins mid-procedure (Project.md #50).
        """
        return self._steps_by_starting_state.get(state_id, [])

    def resume_candidates(self, observed_state_id: str) -> list[StepDefinition]:
        """
        Given a physically-observed state, return the step(s) that would
        naturally follow it. The caller (API layer) is responsible for
        surfacing confidence and asking for confirmation (Project.md #50) —
        this engine never silently assumes.
        """
        return self.steps_from_state(observed_state_id)
