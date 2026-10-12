# Project onboarding

Everything needed to run TVASTA locally, understand how its parts interact,
and change it safely. Verified against a clean setup on Linux (Python 3.11,
Node 26, Docker) on 2026-10-09; anything not verified is marked as such.

Related docs: [architecture.md](architecture.md) · [data-model.md](data-model.md)
· [pose.md](pose.md) · [coordinates.md](coordinates.md) · [3d.md](3d.md) · [marker-tags.md](marker-tags.md) ·
[qa-engine.md](qa-engine.md) · [procedure-engine.md](procedure-engine.md) ·
[vision-pipeline.md](vision-pipeline.md) · [roadmap.md](roadmap.md)

## 1. What this is

An industrial maintenance / inspection / QA platform. A procedure is not a PDF
or a video: it is a graph of `State → Action → Validation → State` steps. An
operator points a camera at a physical asset, the system observes what state
the asset is in, and a deterministic rule engine decides whether each step
passed. A second workspace shows the asset's 3D model and overlays it on the
live camera image (AR), tracking the real object's 6DoF pose.

The repo holds two largely independent feature areas that share one backend:

| Area | What the user does | Main code |
|---|---|---|
| Procedure inspection | Pick an asset + published procedure, capture a frame per step, get PASS / FAIL / UNCERTAIN with a reason | `backend/app/api/inspection.py`, `frontend/src/features/inspection/` |
| 3D & AR workspace | Upload/view GLB models, overlay them on the camera, track an object, check part presence and model fit, play authored procedures | `backend/app/api/vision.py`, `backend/app/api/models3d.py`, `frontend/src/features/viewer3d/` |

## 2. Tech stack

| Layer | Technology | Notes |
|---|---|---|
| Frontend | React 18, TypeScript 5 (strict), Vite 5, Three.js 0.169, axios | No router, no state library, no CSS framework (`App.css`) |
| API | FastAPI 0.115, Uvicorn, Pydantic v2, pydantic-settings | REST + one WebSocket |
| Persistence | PostgreSQL 16 via SQLAlchemy 2.0 + Alembic, psycopg2 | UUID primary keys, JSONB for rule config |
| Object storage | MinIO (S3 API) via `minio` client | Evidence frames only |
| Vision | OpenCV (contrib, headless) 4.10, NumPy 1.26, Ultralytics 8.3.10 (YOLOv8), supervision 0.24 (ByteTrack), Open3D 0.18 | PyTorch is an unpinned transitive dependency of Ultralytics |
| Pose service (optional) | MegaPose via HappyPose, FastSAM, DINOv2, CUDA | Separate process and Python 3.10 env, see `pose_service/README.md` |
| Auth | PyJWT (HS256) | Email-only, deliberately minimal |
| Tests | pytest 8, FastAPI `TestClient` | Backend only; 272 tests |

## 3. Repository layout

```
backend/
  app/
    main.py            FastAPI app: CORS, routers, /static/models mount, /health
    core/              config.py (all settings), database.py (engine, Base, get_db), security.py (JWT)
    api/               Routers — the ONLY layer that wires services together
      auth.py  assets.py  procedures.py  inspection.py  evidence.py
      vision.py        detection, tracking, pose, AR session (HTTP + WebSocket), clips
      models3d.py      GLB upload/registry, parts, presence calibration, fit check, QA inspect, procedures
      marker_tags.py   four-corner ArUco product tags: mapping, printable tags, scan
      video.py  metrology.py  deps.py (get_current_user, require_role)
    models/            SQLAlchemy models + enums.py (shared vocabulary)
    schemas/           Pydantic contracts shared across routers/services
    services/          One package per concern (see section 5)
    workers/           CLI entry points: seed_demo, generate_marker, calibrate_camera, replay_clip
  alembic/             Migrations 0001–0005
  requirements.txt  pytest.ini  Dockerfile
frontend/
  src/
    App.tsx            Three tabs: "Procedure inspection", "3D & AR workspace", "ArUco tags"
    services/api.ts    Every HTTP/WebSocket call
    types/index.ts     Hand-maintained mirror of backend schemas
    hooks/             useCamera (getUserMedia + frame capture), useDetectableClasses
    components/        Camera select/status, class select
    features/inspection/InspectionRunner.tsx
    features/viewer3d/ Viewer3DPage, ThreeViewer, RegistrationOverlay (AR, ~1500 lines), panels
    features/markers/  MarkerTagsPage (navbar "ArUco tags"), ProductScanPanel (ArUco marker tab)
pose_service/          Standalone MegaPose server (not part of the backend venv)
tests/backend/         pytest suite (run from backend/, see pytest.ini)
data/
  models/              GLB files served at /static/models/* (tracked; uploads land here too)
  procedures/          Seed procedure JSON (tracked) + per-model authored procedures (ignored)
  calibration/ images/ videos/ reference_images/   runtime inputs, mostly ignored
docs/                  Per-subsystem documentation
Project.md             The design brief; code comments cite it as "Project.md #N"
Phase-2.md … Phase-5.md, Phae-4.md   Phase-specific briefs
*.glb at repo root, 3D-models/       Sample/reference assets
docker-compose.yml  .env.example
```

## 4. Architecture

```
            Browser (React + Three.js)            camera via getUserMedia
                 │  REST (axios)        │  WebSocket: JPEG frames in, poses out
                 ▼                      ▼
        ┌────────────────────── FastAPI (backend/app) ──────────────────────┐
        │  api/  ── wires ──►  services/                                    │
        │    procedure_engine   vision (detect/segment)   pose   tracking   │
        │    state_detection    qa_engine   evidence   audit   model3d      │
        │    video   metrology                                              │
        └───────┬───────────────┬───────────────┬──────────────────┬────────┘
                │               │               │                  │ HTTP (optional)
          PostgreSQL          MinIO        data/models/*.glb   pose_service (MegaPose, GPU)
        (all relational)   (evidence)     (/static/models)        :8765
```

### Layer contract

Each service knows one thing and is tested in isolation. This is the main
design rule of the codebase; breaking it is the easiest way to make a change
that "works" but that a reviewer will reject.

| Service package | Answers | Must not |
|---|---|---|
| `procedure_engine` | What step follows which state; where to resume | Touch pixels or decide PASS/FAIL |
| `vision` (detector, segmentation) | What object is where | Know about states or requirements |
| `state_detection` | Which discrete state a component is in | Decide PASS/FAIL |
| `qa_engine` | PASS / FAIL / UNCERTAIN / NOT_EVALUATED / MANUAL_REVIEW from fixed thresholds | Run inference |
| `evidence` | Stores and retrieves frames | Contain QA logic |
| `pose`, `tracking` | Where the object is in 3D and how it moved | Decide PASS/FAIL |
| `metrology` | Distances between point clouds | Decide PASS/FAIL |
| `audit` | Records who did what, in the caller's transaction | Commit on its own |

### Data model

```
Asset ──< Component (self-referential tree; class_label = detector class; cad_node_id = GLB node)
Asset ──< Procedure ──< ProcedureRevision (draft|published) ──< Step ──< ValidationRule (JSONB config)
                                             Step >── State (starting, expected)   Step >── Component (target)
Asset ──< Model3D (storage_key, scale, anchor offset)  >── Component (optional)
Asset ──< MarkerTag (dictionary, marker_id, corner) — four per tagged product
InspectionRun >── ProcedureRevision
InspectionRun ──< InspectionStep (result + manual_override_*) ──< Observation (detections JSONB, frame_ref)
User (email, role)        AuditLog (user, action, entity, details JSONB)        Evidence (see risks)
```

Roles: `admin`, `engineer`, `technical_author`, `qa_inspector`, `maintainer`,
`operator`, `reviewer`.

### API surface (54 paths)

| Prefix | Purpose |
|---|---|
| `GET /health` | Liveness |
| `/api/auth` | `register`, `login` → bearer token |
| `/api/assets` | Assets and their components |
| `/api/procedures` | Create from procedure JSON, list, `/{id}/publish` (role-gated) |
| `/api/inspection` | `start`, `/{id}`, `step/{id}/observe`, `validate`, `override` (role-gated), `complete` |
| `/api/vision` | `classes`, `detect`, `track`, `state`, `segment`, `pose` (ArUco), `reference-image` + `feature-pose`, `object-registration`, `ar-session/frame`, `ar-session/stream` (WebSocket), `clips`, `debug-frame` |
| `/api/models3d` | List/register/upload/delete, settings, anchor, parts, presence calibration, per-model procedure, `fit-check`, `qa-inspect` |
| `/api/marker-tags` | Four ArUco tags per product: `config`, `suggest`, `sets`, tag/sheet PNGs, `scan` (tags → YOLO-verified product) |
| `/api/video` | `extract-candidates` (video → candidate steps for human review) |
| `/api/metrology` | `measure-gap` |
| `/api/evidence` | Presigned URL for a stored frame |
| `/static/models/*` | GLB files |

Interactive docs: `http://127.0.0.1:<port>/docs`.

## 5. Main execution flows

### 5.1 Procedure inspection

1. `App.tsx` loads assets and procedures; the user picks an asset and a
   *published* revision.
2. `InspectionRunner` calls `POST /api/inspection/start`, which creates an
   `InspectionRun` and one `InspectionStep` (`NOT_EVALUATED`) per step.
3. "Capture & Validate" grabs a JPEG from the `<video>` element (`useCamera`)
   and calls `observe`:
   `decode_frame` → `Detector.detect` → `StateEstimator.estimate` (component
   class present ⇒ starting state, absent ⇒ expected state at confidence 0.90)
   → frame uploaded to MinIO → `Observation` row → `InspectionStep.observed_state`.
4. `validate` builds a `QAInput` and calls `QAEngine.evaluate`:
   no observation ⇒ `NOT_EVALUATED`; unstable tracking/pose ⇒ `UNCERTAIN`;
   state matches and thresholds met ⇒ `PASS`; state mismatch ⇒ `FAIL` with a
   reason built from the data; matched but low confidence ⇒ `UNCERTAIN`.
   For REMOVE / DISCONNECT / OPEN actions the pass condition is the
   component's *absence* (`action_expects_presence`).
5. The UI advances to the next `NOT_EVALUATED` step; `complete` closes the run.
6. `override` (admin / QA inspector) records a human decision alongside the
   original result and writes an audit row.

### 5.2 Procedure authoring

`POST /api/procedures` takes the procedure JSON (`schemas/procedure.py`,
example in `data/procedures/pump-pcb-maintenance-demo.json`), creates
`States` and missing `Components` on the fly, and stores a **draft** revision.
`POST /api/procedures/{id}/publish` promotes the newest draft (role-gated,
audited). `seed_demo` reuses the same `create_procedure` function.

### 5.3 3D viewer and model management

`Viewer3DPage` lists `Model3D` rows for an asset and loads the GLB from
`/static/models/<storage_key>` with Three.js (`ThreeViewer`). Upload
(`POST /api/models3d/upload`) saves the file under `data/models/`, converts
FBX → GLB through headless Blender if needed, and computes `scale` from the
GLB's bounds and the real-world height the user enters (a GLB's units are
rarely meters). Delete moves the file to `data/models/.deleted/`.

### 5.4 Live AR tracking

`RegistrationOverlay` opens `WS /api/vision/ar-session/stream`, sends a JSON
configuration, then binary messages of `float64 capture time + JPEG`. The
server keeps only the newest unprocessed frame, runs it through the session's
state machine, and replies with an `ARFrameResponse` (pose in Three.js space,
intrinsics, tracking state, part checks).

```
SEARCHING ─detector finds object→ INITIALIZING ─pose + lock→ TRACKING
TRACKING ─confidence below lost threshold→ LOST ─grace frames used→ RECOVERING
RECOVERING ─re-found→ INITIALIZING      RECOVERING ─timeout→ SEARCHING (model hidden)
```

`services/tracking/ar_session.py` owns the state machine and never touches
images; it is given a detector callable and a `PoseTracker`:

| Mode | How the object is found | Tracker | Pose quality |
|---|---|---|---|
| `markerless` | YOLO class, a user-drawn box, or the CAD model | `FlowPoseTracker` (Lucas-Kanade optical flow; distance from apparent size) | Position only, approximate |
| `model` | CAD model via pose service (`/detect`), class, or box | `MegaPoseTracker` (MegaPose refine + flow between refines) | Full 6DoF; needs `pose_service` |

Accepted poses pass through `pose_filter.py` (Kalman on translation, slerp on
rotation). Other registration paths: ArUco marker (`/api/vision/pose`) and
planar feature tracking (`reference-image` + `feature-pose`).

### 5.5 Product identification by ArUco tags

A product carries four tags, one per corner (`marker_tags` table, managed on
the "ArUco tags" page). In AR tracking → ArUco marker, the product scan posts
frames to `/api/marker-tags/scan`: tags → one product with all four on their
registered corners → YOLO must see the product's detection class at the tags
→ three consecutive agreeing frames → `CONFIRMED`, and the workspace opens
that product's model. Details and limits: [marker-tags.md](marker-tags.md).

### 5.6 Part presence, fit check, QA inspect

With a tracked pose, a named GLB part is projected into the image
(`vision/part_presence.py`) and compared against a saved calibration
("cap on" vs "cap off") or against the real object's outline
(`vision/model_fit.py`, `vision/qa_engine.py`). Calibrations are JSON files in
`data/part_calibration/`.

## 6. Setup

### Prerequisites

| Tool | Version | Why |
|---|---|---|
| Python | **3.11** | The pins (`open3d==0.18.0`, `numpy==1.26.4`) have no wheels for newer interpreters — a 3.14 venv cannot install them; the Dockerfile uses 3.11 |
| Node / npm | 20+ | Vite 5 |
| Docker + Compose | any recent | PostgreSQL and MinIO |
| Webcam | — | Both workspaces need a live camera for capture |
| Blender | optional | FBX upload |
| NVIDIA GPU + CUDA | optional | `pose_service` (model-based tracking) and faster YOLO |

### Steps

```bash
# 1. Infrastructure
docker compose up -d postgres minio        # Postgres :5432, MinIO :9000 (console :9001)

# 2. Backend
cd backend
uv venv --python 3.11 --seed .venv         # or: python3.11 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt         # large: pulls PyTorch (several GB with CUDA)
```

To skip the multi-GB CUDA download when no GPU work is planned, install the
CPU build of PyTorch instead (this is what the setup was verified with):

```bash
uv pip install -r requirements.txt torch \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

```bash
# 3. Configuration — optional; every setting has a working local default
#    Put overrides in backend/.env (NOT the repo root, see below). Keys: ../.env.example

# 4. Database
alembic upgrade head                         # fresh or existing database
python -m app.workers.seed_demo              # PUMP-001 + 8-step procedure, BOTTLE-001 + bottle.glb

# 5. Run
uvicorn app.main:app --reload --reload-dir app --port 8000

# 6. Frontend (second terminal)
cd frontend
npm install
npm run dev                                  # http://localhost:5173
# Backend on another port:  VITE_API_BASE_URL=http://127.0.0.1:8001 npm run dev
```

### Configuration

`app/core/config.py` is the complete, commented list of settings. The ones
that matter first:

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | local `tvasta` database | SQLAlchemy URL |
| `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_EVIDENCE_BUCKET` | match `docker-compose.yml` | Evidence storage |
| `MODEL_PATH` | unset | YOLO weights for the inspection detector; unset ⇒ `MockDetector` |
| `SEGMENTATION_MODEL_NAME` | `yolov8n-seg.pt` | AR object finder, auto-downloaded |
| `CAMERA_CALIBRATION_PATH` | unset | Unset ⇒ approximate intrinsics, responses flagged `approximate` |
| `ARUCO_DICTIONARY`, `ARUCO_MARKER_LENGTH_M` | `DICT_4X4_50`, `0.05` | Must match the printed marker |
| `MODELS_3D_DIR` | `../data/models` | Relative to `backend/` |
| `POSE_SERVICE_URL` | `http://127.0.0.1:8765` | MegaPose service |
| `JWT_SECRET_KEY` | dev-only placeholder | Must be overridden outside local dev |
| `VITE_API_BASE_URL` (frontend) | `http://127.0.0.1:8000` | Backend base URL |

Where `.env` is read from: local `uvicorn`, `alembic`, and the workers load
`.env` from the **current directory**, i.e. `backend/.env`. The repo-root
`.env` (what `.env.example` suggests copying to) is only used by
`docker-compose.yml`'s `backend` service.

### Verifying the setup

```bash
cd backend && python -m pytest -q                 # expect 267 passed, 5 skipped (Blender)
cd frontend && npm run build                      # expect success (one chunk-size warning)
curl http://127.0.0.1:8000/health                 # {"status":"ok"}
curl http://127.0.0.1:8000/api/assets             # PUMP-001, BOTTLE-001 after seeding
curl -I http://127.0.0.1:8000/static/models/bottle.glb
```

Verified end to end on 2026-10-09 against a real Postgres and MinIO: start
run → observe (frame stored in MinIO) → validate → override (401 without a
token, 403 as operator, 200 as QA inspector, original result preserved) →
complete; ArUco pose on a synthetic marker; segmentation model load (80
classes). **Not verified:** anything in a real browser with a real camera,
FBX conversion, and `pose_service` / model-based tracking.

## 7. Known issues and risks

Found during setup or code reading. None are fixed here; each is a candidate
for a small, separate change.

| # | Issue | Impact | Suggested fix |
|---|---|---|---|
| 1 | ~~`alembic upgrade head` fails on an empty database~~ — **fixed**: `0001` is now an explicit Phase 1 baseline and `0002` creates `models_3d` explicitly, so the chain applies cleanly ([data-model.md](data-model.md#migrations)) | — | — |
| 2 | Local runs read `backend/.env`, but `.env.example` sits at the root | Overrides silently ignored | Document, or set `env_file` to an absolute path |
| 3 | `requirements.txt` does not pin `torch`/`torchvision` | Each install resolves whatever is newest; multi-GB CUDA download by default | Pin them (2.14.1 / 0.29.1 verified working with Ultralytics 8.3.10) |
| 4 | README says Python `venv` with no version; a newer system Python (3.14 here) cannot install the pins | Install fails | State 3.11 in the README |
| 5 | `validate_step` picks the "latest" `Observation` with `order_by(Observation.id.desc())`, but `id` is a random UUID and there is no timestamp column | After a retry on the same step, validation may use an older observation | Add `created_at` and order by it |
| 6 | The `Evidence` table is never written; frames are referenced only by `Observation.frame_ref` | `GET /api/evidence/{id}` can never find a row | Insert an `Evidence` row in `observe_step` |
| 7 | `forget_model_pose()` in `api/vision.py` has only a docstring | Deleting a model leaves its tracking state and pose-service mesh behind until restart | Implement or remove |
| 8 | `start_inspection` accepts draft revisions and does not check the asset exists; `validate` hard-codes `tracking_stable`/`pose_valid` to `True` | Weaker guarantees than the docs imply | Enforce published-only; wire real tracking/pose |
| 9 | Open `/api/auth/register` (any role, no password), CORS `*`, most endpoints unauthenticated, dev JWT secret default | Fine for a demo, unsafe on any shared network | Gate registration, restrict origins, require the secret |
| 10 | Tracker/session state is in process memory | Lost on reload; breaks with more than one worker | Keep a single worker, or move state out of process |
| 15 | `docker compose up -d postgres minio` fails: `minio/minio` can no longer be pulled from Docker Hub | The README's first setup command fails on a machine without the image | Pull the newer commits, which add `docker-compose.override.yml` selecting `pgsty/minio:latest` |
| 11 | `docker-compose.yml`'s `backend` service was rebuilt (Python 3.11 image, `migrate` service) and verified on 2026-10-09, see [docker.md](docker.md); the `frontend` image still uses Node 20 and `npm install` without the lockfile | Frontend container may drift from local | Use `npm ci` with the lockfile |
| 12 | `data/procedures/` is gitignored while the seed JSON inside it is tracked | New procedure files there will not show up in `git status` | Narrow the ignore rule |
| 13 | Frontend has no tests or linter; `types/index.ts` is maintained by hand | API drift is only caught at runtime | Generate types from `/openapi.json` |
| 14 | `pose_service/start.sh` and its README carry Windows/WSL paths | Needs adapting for native Linux | Parameterize |

## 8. Changing the code safely

### Workflow

1. Read the module docstring and the matching `docs/*.md` first; they record
   why things are the way they are, often with measurements.
2. Make the change in the layer that owns the concern (section 4).
3. Add or extend a test in `tests/backend/` that runs without external services.
4. Run `python -m pytest -q` (seconds) and, for any frontend or schema change,
   `npm run build`.
5. For API changes, exercise the endpoint at `/docs` against a running backend.

### Recipes

| To add… | Touch | Notes |
|---|---|---|
| A setting | `core/config.py` (+ `.env.example`) | Typed field, default, comment with the reason |
| An endpoint | `api/<router>.py`, `schemas/` if shared, `frontend/src/services/api.ts`, `frontend/src/types/index.ts` | Register new routers in `main.py`; gate with `Depends(require_role(...))` and `audit_log.record` for privileged mutations |
| A table or column | `models/*.py` (import new modules in `models/__init__.py`), new Alembic revision | `alembic revision --autogenerate -m "..."`, review it, keep explicit `op.*` calls and a real `downgrade`; test on a copy of the DB |
| A detector | Subclass `Detector` in `services/vision/detector.py`, select it in `build_detector` | Routers must not branch on the implementation |
| A QA rule | `services/qa_engine/engine.py` + `tests/backend/test_qa_engine.py` | Deterministic, threshold-driven, reason derived from inputs |
| A validation method or action type | `models/enums.py` | These are Postgres enum types: adding a value needs a migration (`ALTER TYPE … ADD VALUE`) |
| A tracker | Implement the `PoseTracker` protocol, select it in `_build_ar_session` | Return poses already in Three.js space via `transforms.py` |
| A 3D model | Upload in the UI or `POST /api/models3d/upload` | Always give the real height for GLB |

### Things to be careful with

- **Coordinate conversions.** Use the functions in `pose/transforms.py`; read
  [coordinates.md](coordinates.md) before changing anything that produces or
  consumes a pose. `tests/backend/test_transforms.py` pins the conventions.
- **`Model3D.scale` and recentering.** The frontend, `glb_inspect`, and the
  pose service must agree on the same scaled, bbox-centered frame.
- **Thresholds in `config.py`** were tuned by measurement (see the comments).
  Change them with a replay (`python -m app.workers.replay_clip`) rather than
  by feel.
- **Dependency versions.** `numpy<2`, OpenCV contrib 4.10, Ultralytics 8.3.10,
  supervision 0.24, and Open3D 0.18 are mutually constrained; upgrading one
  usually forces the others. Don't upgrade as a side effect of another change.
- **Published procedures and inspection history** are records. Create a new
  revision rather than editing steps of a published one.
- **Model deletion and uploads modify `data/models/`**, which is tracked in git.

## 9. Local environment on this machine (2026-10-09)

Two checkouts of the same repository are in use and share the Docker
containers `tvasta-postgres` and `tvasta-minio` (created 2026-10-01 from
`~/Desktop/AR`; MinIO runs the `pgsty/minio` image).

| | `~/Desktop/AR` | `~/Projects/AR` (this checkout, `master`) |
|---|---|---|
| Git | 4 commits ahead (`fa2ec1d`), includes `docker-compose.override.yml` | `4b67efb` |
| Schema | Was Alembic `0005` with a `marker_bindings` table in the morning; back to `0004` after a pull that removed marker scanning | Alembic `0004` |
| Database | `tvasta` (existing data) | `tvasta_master` (created for this checkout, seeded) |
| Config | defaults | `backend/.env` overrides `DATABASE_URL` only |
| Evidence bucket | `evidence` | `evidence` (shared) |

Because both checkouts default to ports 8000 and 5173, check which one owns a
port before drawing conclusions from a response:

```bash
ss -ltnp | grep -E ':(8000|8001|517[0-9])\b'
readlink /proc/<pid>/cwd
```

A frontend started without `VITE_API_BASE_URL` talks to whatever is on
`127.0.0.1:8000`, regardless of which checkout it was started from.
