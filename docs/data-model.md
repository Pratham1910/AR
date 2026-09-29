# Data Model

Source of truth: `backend/app/models/*.py` (SQLAlchemy). This doc explains the
*why*, not a field-by-field dump — read the models for exact columns.

## Entity relationships

```
Asset ──< Component (self-referential parent/child tree)
Asset ──< Procedure ──< ProcedureRevision ──< Step ──< ValidationRule
                                          Step >── State (starting/expected, FK)
                                          Step >── Component (target, FK)

InspectionRun ─→ ProcedureRevision (which version was executed)
InspectionRun ──< InspectionStep ──< Observation
                              InspectionStep ──< Evidence
                              InspectionStep >── State (observed, FK)
```

## Design decisions

- **States are rows, not strings embedded in steps** (Project.md §5, §51): a
  Step references `starting_state_id` / `expected_state_id` by FK. This is
  what lets a procedure be a graph rather than a flat list — resuming
  mid-procedure (§50) is a lookup (`steps_from_state`), not a rewrite.
- **QAResult has 5 values, never just PASS/FAIL** (§8): `PASS`, `FAIL`,
  `UNCERTAIN`, `NOT_EVALUATED`, `MANUAL_REVIEW`.
- **Manual override never overwrites the AI result** (§44): `InspectionStep`
  keeps `result` (the AI/rule decision) and separate `manual_override_*`
  columns.
- **Binary media never lives in Postgres** (§65, §66): `Evidence.storage_key`
  and `Component`/asset image/video references point at MinIO object keys;
  only metadata is relational.
- **`ProcedureRevision.status`** (draft/published) makes publishing an
  explicit, auditable act (§39, §67) — an `InspectionRun` can only reference
  a revision, and nothing prevents referencing a draft today, but the
  authoring UI (future) should only offer published revisions.
- **`ValidationRule.config` is JSONB** so tolerance/threshold parameters
  (§31) are configuration, not schema changes, per component/step.

## Migrations

`backend/alembic/versions/0001_initial_schema.py` creates every table via
`Base.metadata.create_all` rather than hand-written `op.create_table()` calls,
so it can never drift from the SQLAlchemy models. Future schema changes
should be normal Alembic autogenerate revisions layered on top.
