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

`alembic upgrade head` works on an empty database and on one at any earlier
revision. Each revision owns exactly the schema objects it creates:

| Revision | Creates |
|---|---|
| `0001` | The Phase 1 schema: 13 tables and 7 enum types, as explicit `op.create_table()` calls |
| `0002` | `models_3d` (as it was in Phase 4) and `components.cad_node_id` |
| `0003` | `models_3d.scale` |
| `0004` | `models_3d.anchor_offset_*` / `anchor_rotation_*` |
| `0005` | `marker_tags` |

Rules for new revisions:

- **Never import `app.models` or call `Base.metadata.create_all()` in a
  migration.** That builds the schema from the models as they are when the
  migration *runs*, not as they were when it was written. `0001`/`0002` did
  this originally, so a fresh database already had every later column and the
  upgrade failed with `DuplicateColumn`. Write explicit `op.*` calls
  (`alembic revision --autogenerate` produces them); a migration is a frozen
  record and is not edited once another database has run it.
- **A column added as `NOT NULL` to an existing table needs a
  `server_default`** so existing rows get a value (`0003`, `0004`). That is why
  those columns have a database default the models do not declare — the only
  intended difference between the migrated schema and `Base.metadata`.
- `downgrade()` drops what its own `upgrade()` created, by name.

`alembic/env.py` takes a PostgreSQL advisory lock for the duration of a run,
so two `alembic upgrade head` processes started together take turns instead of
racing. `tests/backend/test_migrations.py` covers all of the above; its
real-database half runs when `MIGRATION_TEST_DATABASE_URL` is set (see
[docker.md](docker.md#migrations)).

**Databases created before this was fixed.** The old workaround was
`alembic upgrade 0001 && alembic stamp head`: such a database has the full
schema and is recorded at the head of that time, so `alembic upgrade head`
simply applies whatever is newer. A database that has tables but **no**
`alembic_version` table cannot be upgraded — Alembic has no way to know what
is in it — and the upgrade stops with `DuplicateObject`/`DuplicateTable`
without changing anything.
