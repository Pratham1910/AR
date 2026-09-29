# Procedure Engine

`backend/app/services/procedure_engine/engine.py`

## Model

A procedure is a **State → Action → Validation → State** graph (Project.md
§5), not a flat step list. `ProcedureGraph` wraps a `ProcedureDefinition`
(the JSON format in Project.md §6, `backend/app/schemas/procedure.py`) and
answers structural questions:

- `first_step()` — the authored entry point.
- `next_step(current_step_id)` — linear advance, used by the demo UI.
- `steps_from_state(state_id)` / `resume_candidates(observed_state_id)` — all
  steps that could follow a given state. This is what makes "operator opens
  the app mid-procedure" (§50) tractable: observe the physical state, look up
  candidate steps, and — critically — **surface the match and its confidence
  to the operator rather than silently resuming** (the engine deliberately
  returns candidates, not a decision).

## Validation

`ProcedureGraph._validate()` rejects a definition whose steps reference a
`startingState`/`expectedState` not declared in `states` — this runs at load
time (`ProcedureGraph.from_file` / `ProcedureGraph(definition)`), not at
runtime, and is separate from the QA engine's runtime validation.

## Persistence

The JSON format is the interchange/authoring format. `backend/app/api/procedures.py`
converts a `ProcedureDefinition` into `Procedure` / `ProcedureRevision` /
`State` / `Step` / `ValidationRule` rows on `POST /api/procedures`, and
`POST /api/procedures/{id}/publish` moves the latest draft revision to
`published` — only published revisions should back an `InspectionRun`.

## What this module deliberately does not do

No camera access, no model inference, no PASS/FAIL logic (Project.md §74) —
see `docs/vision-pipeline.md` and `docs/qa-engine.md`.
