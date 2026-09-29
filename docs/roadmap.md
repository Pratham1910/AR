# Roadmap

Phases per Project.md §58. Status as of this build:

| Phase | Scope | Status |
|---|---|---|
| 0 | Data model (Asset, Component, State, Procedure, Step, Requirement, Evidence) | ✅ Done |
| 1 | Camera QA MVP (camera input, detection, state recognition, deterministic QA) | ✅ Done — see below |
| 2 | Tracking (ByteTrack/BoT-SORT, lost/recovery) | ⬜ Not started |
| 3 | Video-to-procedure extraction (candidate steps, human approval) | ⬜ Not started |
| 4 | 3D (GLB/glTF, Three.js viewer, animation) | ✅ Done (viewer; animation not yet — see below) |
| 5 | 3D ↔ physical registration (calibration, pose, solvePnP, 6DoF) | ✅ Done — marker-based, see `docs/pose.md` |
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

## Phase 4/5 — what's actually implemented

- `Model3D` model + `Component.cad_node_id` + Alembic migration `0002`.
- GLB static serving (`/static/models/*`) + `/api/models3d` list/register API.
- Three.js viewer (`frontend/src/features/viewer3d/ThreeViewer.tsx`): load,
  orbit/zoom/pan, auto-frame, scene-node listing. No animation playback yet
  (Project.md §21/§22's timeline/animation controls are not built — the
  bottle GLB has no animations to play).
- Camera calibration (`app/services/pose/calibration.py`) with an explicit
  approximate/real distinction, plus a checkerboard calibration worker.
- Marker-based 6DoF pose (`app/services/pose/aruco_pose.py`, ArUco +
  `solvePnP`) and a dedicated coordinate-transform module
  (`app/services/pose/transforms.py`) converting OpenCV camera space to
  Three.js space server-side.
- `POST /api/vision/pose` API, plus a 2D outline (`corners_px`) so the
  detected marker is visibly drawn on the live feed, not just reported as
  found/not-found (Project.md §57).
- Object segmentation (`app/services/vision/segmentation.py`,
  `POST /api/vision/segment`, Project.md §16): stock COCO-pretrained
  `yolov8n-seg.pt` returns a real pixel-outline polygon per detected object
  (not just a bounding box), separate from the Phase 1 procedure detector.
- Markerless registration (`app/services/pose/markerless.py`,
  `POST /api/vision/object-registration`): no printed marker needed —
  detects a named object class and estimates an **approximate,
  position-only** pose from its apparent size vs. a user-provided real-world
  height. Explicitly not 6DoF and explicitly not what an industrial AR
  platform like DELMIA Augmented Experience does (feature/CAD-matched,
  sub-millimeter, full 6DoF) — see `docs/pose.md` for the gap and what
  closing it would take.
- Registration overlay (`frontend/src/features/viewer3d/RegistrationOverlay.tsx`):
  mode toggle between markerless (default) and ArUco marker, transparent
  Three.js canvas + 2D outline canvas over the live camera feed, "Detect &
  Align" / live-tracking mode, camera device selection, connection status
  badge, places the GLB using whichever pose was returned.
- Demo data: `BOTTLE-001` asset + `BOTTLE-BODY-001` component
  (`cad_node_id="Cylinder"`) + `Model3D` row for `data/models/bottle.glb`
  (copied from the provided `3D-models/TEST BOTTLEglb.glb`).
- **Verified end-to-end**: `TestClient` calls confirm `/api/models3d` returns
  the correct URL, `/static/models/bottle.glb` serves the real 84KB glTF
  file (magic bytes `glTF`), and `/api/vision/pose` correctly detects a
  synthetic ArUco marker (low reprojection error) and correctly reports
  `found: false` for a marker-free frame. Frontend builds clean (`tsc -b` +
  `vite build`) with the new Three.js dependency.
- Tests: `tests/backend/test_transforms.py` (pure coordinate-transform math)
  and `tests/backend/test_aruco_pose.py` (synthetic-marker detection,
  including "no marker" and "wrong target id" cases) — no physical camera or
  printed marker required to verify the pipeline logic.

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
  loader and Phase 5 registration, not yet a multi-part demo assembly or a
  procedure target.
- Pose estimation is marker-based only (Project.md §24 explicitly allows
  this as the initial method) and was verified against a synthetic marker
  image, not a physical printed marker + real camera in this session —
  the `generate_marker`/`calibrate_camera` workers are ready for that test
  in a normal environment.
- The AR overlay's Three.js camera FOV is a fixed guess, not derived from the
  real camera_matrix — see `docs/pose.md`'s "known limitation".
