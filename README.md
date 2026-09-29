# TVASTA — Industrial AR/MR Procedure & QA Platform

An industrial maintenance, inspection, and QA platform where a procedure is not a
PDF or a video — it is an executable **State → Action → Validation → State**
workflow, verified against a physical asset by computer vision.

See [`Project.md`](Project.md) for the full architecture brief this build follows.

## Architecture (Phase 1)

```
        Web Client (React + TS)
                 │
              FastAPI
   ┌─────────────┼─────────────┐
Procedure      Vision         QA
 Engine        Engine        Engine
   │             │             │
   └─────────────┼─────────────┘
            Evidence Engine
                 │
        PostgreSQL   +   MinIO
```

Procedure/Vision/QA/Evidence engines are independent layers (see `docs/architecture.md`).
No subsystem performs another's job — the vision engine never decides PASS/FAIL, the
QA engine never runs inference.

Unity/AR/MR, 3D↔physical registration (pose/6DoF) and depth-based metrology are
**future phases** (see `Project.md` §58, §74) — Phase 1 proves the core loop with a
plain camera:

```
CAMERA → DETECTION → STATE → QA → EVIDENCE
```

## Project layout

```
backend/     FastAPI app: procedure engine, QA engine, vision service, evidence
frontend/    React + TS web client (camera capture, inspection runner UI)
data/        Seed procedure JSON, reference images/videos
models/yolo/ Trained/placeholder YOLO weights (not committed)
docker/      Supporting docker assets
docs/        Per-subsystem documentation
tests/       Backend + vision tests
3D-models/   Reference GLB/glTF assets (Phase 4+)
```

## Running locally (development)

```bash
# 1. Infra
docker compose up -d postgres minio

# 2. Backend
cd backend
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload --port 8000

# 3. Frontend
cd frontend
npm install
npm run dev
```

## Status

Phase 1 (Camera QA MVP) and Phase 4/5 (3D viewer + marker-based physical↔3D
registration) are done. See `docs/roadmap.md` for phase-by-phase status
against `Project.md` §58.

## Testing the 3D / AR registration (Phase 4/5)

```bash
# after seeding (see below), generate a printable ArUco marker
cd backend
python -m app.workers.generate_marker --id 0 --size-px 600
# -> data/images/aruco_DICT_4X4_50_id0.png
```

Print it, measure its printed side length in meters, set `ARUCO_MARKER_LENGTH_M`
in `.env` to match, then in the app open **3D / AR Registration** → select
`BOTTLE-001` → **3D Viewer** to see the provided bottle model, or
**AR Registration** to point your camera at the printed marker and see the
model align to it. See `docs/pose.md` for the details and known limitations.
