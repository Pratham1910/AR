# CLAUDE.md

TVASTA — industrial AR procedure & QA platform. A procedure is an executable
`State → Action → Validation → State` graph, verified against a physical asset
by computer vision. FastAPI backend, React + Three.js frontend, PostgreSQL,
MinIO, and an optional GPU pose service.

Full walkthrough: [docs/PROJECT_ONBOARDING.md](docs/PROJECT_ONBOARDING.md).
Design brief the code cites as `Project.md #N`: [Project.md](Project.md).

## Commands

All backend commands run **from `backend/`** (paths and `.env` are resolved
relative to it).

```bash
# Infra (see "This machine" below before running this)
docker compose up -d postgres minio

# Backend — Python 3.11 (the pins have no wheels for newer interpreters)
cd backend
uv venv --python 3.11 --seed .venv && source .venv/bin/activate
uv pip install -r requirements.txt          # or: pip install -r requirements.txt
alembic upgrade head                        # fresh or existing database
python -m app.workers.seed_demo             # idempotent demo data
uvicorn app.main:app --reload --reload-dir app --port 8000

# Whole stack in containers instead (Python 3.11 image; docs/docker.md)
docker compose up -d                        # postgres, minio, migrate, backend, frontend
# ./backend and ./frontend are bind-mounted and hot-reload; add --build only
# after changing requirements.txt, constraints.txt, package.json or a Dockerfile

# Tests (no DB, MinIO, GPU, or camera needed)
cd backend && python -m pytest -q                       # all
python -m pytest ../tests/backend/test_qa_engine.py -q  # one file
python -m pytest -q -k "resume"                         # by name

# Frontend
cd frontend
npm install
npm run dev      # http://localhost:5173
npm run build    # tsc -b && vite build — this is the only type check / lint
```

There is no linter, formatter, or frontend test runner configured. `npm run
build` (strict TS, `noUnusedLocals`/`noUnusedParameters`) and `pytest` are the
gates.

## Architecture rules (enforced by convention, cited throughout the code)

- **Layers do one job.** Vision reports detections; state detection turns them
  into a state; the QA engine decides PASS/FAIL/UNCERTAIN/NOT_EVALUATED/
  MANUAL_REVIEW; evidence stores frames. `backend/app/api/*` is the only place
  that wires services together. Never put a PASS/FAIL decision in
  `services/vision`, `services/pose`, `services/tracking`, or
  `services/metrology`, and never run inference from `services/qa_engine`.
- **Config goes through `app/core/config.py`** (`Settings`, pydantic-settings).
  No `os.environ`/`os.getenv` elsewhere. Add a typed field with a default and a
  comment saying why the default is what it is.
- **Coordinate math lives in `app/services/pose/transforms.py`.** OpenCV camera
  space (X right, Y down, Z forward) is converted to Three.js space (Y up, Z
  backward) on the server; the frontend applies poses as given. Marker poses
  use `cv_pose_to_threejs` (`C·R·C`); model/CAD poses use
  `cv_model_pose_to_threejs` (`C·R`). Mixing them flips the model 180°.
- **Binary data never goes in Postgres.** Evidence frames → MinIO; GLB files →
  `data/models/` served at `/static/models/*`; only keys/paths are stored.
- **A manual override never overwrites the engine's result**
  (`InspectionStep.result` vs `manual_override_*`), and audit rows are written
  in the same transaction as the change (`services/audit/audit_log.record`).
- **ArUco product tags are their own module** (`services/markers/`,
  `api/marker_tags.py`, `frontend/src/features/markers/`; see
  [docs/marker-tags.md](docs/marker-tags.md)). It reuses the ArUco detector
  and segmenter through small public helpers in `api/vision.py` and plugs into
  `RegistrationOverlay` via the optional `markerPanel` prop — keep it that
  way rather than adding marker logic to the pose/tracking code.
- **Enums are shared vocabulary** (`app/models/enums.py`) — reuse them in
  models, schemas, and engines rather than string literals.

## Conventions

- Python: type hints everywhere, SQLAlchemy 2.0 `Mapped[...]` style, Pydantic
  v2. Each module opens with a docstring explaining *why* it exists and which
  `Project.md #N` section it implements; comments record measured numbers and
  the bug a line fixes. Match that style — explain reasons, not mechanics.
- Request/response models that are only used by one router live in that
  router file; shared ones live in `app/schemas/`.
- Heavy imports (`ultralytics`, `torch`) are done lazily inside the class that
  needs them, so importing a module never requires a model or network.
- Services are written to be unit-testable with fakes (e.g. `ARSession` takes
  a detector callable and a `PoseTracker`). New service logic should come with
  a test in `tests/backend/` that needs no external service.
- TypeScript: function components + hooks, no state library. All HTTP goes
  through `frontend/src/services/api.ts`; all shared types through
  `frontend/src/types/index.ts` (hand-kept in sync with the backend schemas).

## Gotchas

- **Migrations are frozen, explicit `op.*` calls.** Never import `app.models`
  or call `Base.metadata.create_all()` in one: it applies today's models, which
  is what used to make `alembic upgrade head` fail with `DuplicateColumn` on an
  empty database. `tests/backend/test_migrations.py` enforces this; set
  `MIGRATION_TEST_DATABASE_URL` to also run it against a real server (it only
  touches databases it creates). See [docs/data-model.md](docs/data-model.md#migrations).
- **`.env` is read from the working directory.** Local `uvicorn`/`alembic`
  read `backend/.env`; the repo-root `.env` is only used by docker-compose.
- **In-memory server state** in `app/api/vision.py` (`_ar_sessions`,
  `_object_trackers`, `_reference_planes`, mesh/calibration caches). It is
  lost on restart/`--reload` and is not shared across workers — run a single
  uvicorn worker.
- **No trained detector.** With `MODEL_PATH` unset the procedure-inspection
  flow uses `MockDetector` (detects nothing), so results reflect "component
  absent" only. The AR/3D features use a separate stock `yolov8n-seg.pt`
  (auto-downloaded into `backend/` on first use).
- **`data/procedures/`, `data/clips/`, `data/debug_frames/`,
  `data/part_calibration/`, `data/reference_images/*` are gitignored** runtime
  state. `data/models/*.glb` is tracked, and uploads through the API write
  new files there.
- **Model-based (CAD) tracking needs `pose_service/`** (MegaPose, CUDA, its
  own Python 3.10 env — see `pose_service/README.md`). Without it those
  endpoints return 503; everything else works.
- FBX upload needs Blender on `PATH` (or `BLENDER_PATH`); 5 tests skip without it.
- Auth is intentionally minimal: email-only JWT, open `/api/auth/register`,
  `allow_origins=["*"]`, and most endpoints are unauthenticated. Don't treat it
  as a security boundary; `JWT_SECRET_KEY` has a dev-only default.

## This machine

Two checkouts share the `tvasta-postgres` / `tvasta-minio` containers (created
from `~/Desktop/AR`, compose project `ar`):

| | `~/Desktop/AR` (other branch) | `~/Projects/AR` (this repo) |
|---|---|---|
| Database | `tvasta` (existing data) | `tvasta_master`, set in `backend/.env` |
| Usual backend port | 8000 | 8001 when 8000 is taken |

- Observed 2026-10-09 ~15:40: the containers and both volumes (`ar_pgdata`,
  `ar_miniodata`) had been recreated from *this* checkout at 15:05, `tvasta`
  was empty (no tables), `tvasta_master` did not exist, and `backend/.env`
  pointed at `tvasta`. The table above describes the earlier setup — re-check
  with `docker inspect tvasta-postgres` and `\l` before relying on it.
- The other checkout is ahead of this one and has changed the `tvasta` schema
  before (it was at Alembic `0005` on 2026-10-09 morning, `0004` later that
  day). Check `select version_num from alembic_version` before pointing this
  checkout at `tvasta`, and don't run migrations/stamps against it from here.
- `docker compose up -d postgres minio` fails here with "pull access denied
  for minio/minio": that image is no longer on Docker Hub. The other checkout
  carries a committed `docker-compose.override.yml` selecting
  `pgsty/minio:latest`; this checkout doesn't have it until it is pulled. The
  containers are normally already running — check `docker ps` first.
- Before trusting a response from `:8000`/`:5173`, check which checkout owns
  the port (`ss -ltnp`, then `readlink /proc/<pid>/cwd`). The frontend's API
  target is `VITE_API_BASE_URL` (default `http://127.0.0.1:8000`).
