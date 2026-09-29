# Roadmap

Phases per Project.md §58. Status as of this build:

| Phase | Scope | Status |
|---|---|---|
| 0 | Data model (Asset, Component, State, Procedure, Step, Requirement, Evidence) | ✅ Done |
| 1 | Camera QA MVP (camera input, detection, state recognition, deterministic QA) | ✅ Done — see below |
| 2 | Tracking (ByteTrack/BoT-SORT, lost/recovery) | ⬜ Not started |
| 3 | Video-to-procedure extraction (candidate steps, human approval) | ⬜ Not started |
| 4 | 3D (GLB/glTF, Three.js viewer, animation) | ⬜ Not started — see `3D-models/` |
| 5 | 3D ↔ physical registration (calibration, pose, solvePnP, 6DoF) | ⬜ Not started |
| 6 | Depth / metrology (depth camera, Open3D, gap/clearance) | ⬜ Not started |
| 7 | AR/MR (Unity, OpenXR) | ⬜ Not started |
| 8 | Enterprise (production Postgres, RBAC, audit, S1000D, PLM/MES/QMS, tools) | ⬜ Not started |

## Phase 1 — what's actually implemented

- Procedure engine (`backend/app/services/procedure_engine`) — state/action
  graph, load-time validation, resume-candidate lookup.
- Vision: `MockDetector` + `YoloDetector` behind one interface
  (`backend/app/services/vision/detector.py`); `/api/vision/detect`,
  `/api/vision/state`.
- State detection: config-driven presence/absence rule per component
  (`backend/app/services/state_detection`).
- QA engine: deterministic PASS/FAIL/UNCERTAIN/NOT_EVALUATED/MANUAL_REVIEW
  rules (`backend/app/services/qa_engine`).
- Evidence: MinIO-backed frame storage (`backend/app/services/evidence`).
- Full API: assets, procedures (author + publish), inspection runs
  (start/observe/validate/complete), vision, evidence
  (`backend/app/api/*.py`).
- Data model + Alembic migration (`backend/app/models`, `backend/alembic`).
- Demo data: `PUMP-001` asset + 6 components + the 8-step
  `PUMP-MAINT-001` procedure (`data/procedures/pump-pcb-maintenance-demo.json`,
  seeded via `python -m app.workers.seed_demo`).
- Minimal React/TS frontend: asset/procedure picker, camera capture,
  step-by-step inspection runner with PASS/FAIL/UNCERTAIN display and reason
  (`frontend/src/features/inspection/InspectionRunner.tsx`).
- Tests: procedure engine, QA engine, state engine
  (`tests/backend/*.py`, 25 passing).
- **Verified end-to-end** against a real Postgres instance via
  `TestClient`: start → observe → validate → complete, exercising both PASS
  (expected flow) and FAIL (component still present when it should be
  removed) outcomes with correct, evidence-grounded reasons.

## Known Phase 1 limitations (intentional, documented at the point they occur)

- No trained YOLO model yet — `MockDetector`/manually-supplied detections
  stand in until a custom model is trained on real assembly footage
  (Project.md §78 permits this explicitly).
- State detection is presence/absence-only, not a learned classifier — see
  `docs/vision-pipeline.md`.
- `tracking_stable`/`pose_valid` are placeholder `True` values when a step
  requires `TRACKING`/`POSE` — real tracking/pose land in Phases 2 and 5.
- MinIO evidence upload path is implemented and unit-testable, but couldn't
  be pulled/verified against a live container in this sandbox (registry
  access to `minio/minio` was blocked here); `docker-compose.yml` is
  otherwise ready — verify `docker compose up -d minio` in an environment
  with normal Docker Hub access.
- The 3D model provided (`3D-models/TEST BOTTLEglb.glb`) is a single-mesh
  placeholder (no component hierarchy) — usable to prove the Phase 4 GLB
  loader, not yet a multi-part demo assembly.
