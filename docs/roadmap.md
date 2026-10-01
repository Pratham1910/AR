# Roadmap

Phases per Project.md §58. Status as of this build:

| Phase | Scope | Status |
|---|---|---|
| 0 | Data model (Asset, Component, State, Procedure, Step, Requirement, Evidence) | ✅ Done |
| 1 | Camera QA MVP (camera input, detection, state recognition, deterministic QA) | ✅ Done — see below |
| 2 | Tracking (ByteTrack/BoT-SORT, lost/recovery) | ✅ Done — see below |
| 3 | Video-to-procedure extraction (candidate steps, human approval) | ✅ Done — see below |
| 4 | 3D (GLB/glTF, Three.js viewer, animation) | ✅ Done (viewer; animation not yet — see below) |
| 5 | 3D ↔ physical registration (calibration, pose, solvePnP, 6DoF) | ✅ Done — marker, feature-tracking, and **model-based (MegaPose)** modes; see `docs/pose.md`, `pose_service/README.md` |
| 6 | Depth / metrology (depth camera, Open3D, gap/clearance) | 🟡 Built, **unverified against real depth hardware** — see below |
| 7 | AR/MR (Unity, OpenXR) | ⬜ **Cannot be built in this environment** — no Unity Editor/toolchain available here; needs a machine with Unity installed |
| 8 | Enterprise (RBAC, audit) done; (S1000D, PLM/MES/QMS, tools) not started | 🟡 Partial — see below |

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

## Phase 5 — model-based (CAD) registration

The marker, feature-tracking and markerless modes all track a *proxy* (a
printed marker, a flat photo patch, a YOLO box) and place the model relative
to it — none of them compares the 3D model's own shape to the image, which
is why an overlay could track "something" yet not sit on the real object.
Model-based mode closes that gap:

- `pose_service/` — MegaPose (HappyPose), RGB-only, in WSL2 + CUDA. Matches
  the Model3D mesh itself (scaled by `Model3D.scale`, recentered exactly as
  the frontend renders it) and returns `T_camera_object`.
- `POST /api/vision/model-pose` — YOLO finds the object's class for the
  first lock (full search, ~2.6s), then each frame refines from the previous
  pose (~220ms on an RTX 4090). A pose score below 0.5 drops the track.
- `cv_model_pose_to_threejs` — the object frame here is the mesh's own glTF
  frame, so the conversion is `C·R`, not the marker path's `C·R·C` (which
  would render the model flipped 180°). Covered by `test_transforms.py`.
- **Verified** with a known-answer test (the cup GLB rendered at a known
  pose, 35cm away): 3.5mm / 4.5° error on first lock, ~3mm / 3° while
  tracking; score 1.0 with the cup present vs 0.21 on an empty frame.
  **Not yet verified on a live webcam** in this session (the camera was held
  by the browser). Untextured GLBs match on silhouette/shape only, and a mug
  with its handle hidden has an ambiguous spin angle.

## Live AR: detect once, then track (Phase-3.md)

Markerless and model-based AR used to be tracking-by-detection: markerless
ran YOLO on every frame and rebuilt the pose from scratch, and model-based
hid its state and re-ran YOLO on every frame after a loss.
`POST /api/vision/ar-session/frame` now runs one explicit state machine per
camera session (`app/services/tracking/ar_session.py`):

- SEARCHING: YOLO, rate-capped (`AR_DETECT_INTERVAL_MS`); found -> initial
  pose -> tracker initialized -> TRACKING, with a persistent object ID.
- TRACKING: only the tracker runs — Lucas-Kanade optical flow inside the
  detected outline for markerless (`flow_tracker.py`; scale change carries
  depth), the MegaPose refiner from the last accepted pose for model-based.
  Up to `AR_GRACE_FRAMES` low-confidence frames hold the last pose.
- LOST: model held at the last valid pose; recovery via the tracker
  (MegaPose) or the rate-capped detector; after `AR_LOST_TIMEOUT_MS` the
  model hides and the session returns to SEARCHING.
- Per-tracker confidence thresholds are configurable (`AR_FLOW_*`,
  `AR_MODEL_*`). Rendering stays continuous, smoothed toward each new pose.
- **Verified** with Phase-3.md's 12-step sequence as a unit test (detector
  call count unchanged across 150 tracking frames), and end-to-end on a real
  frame with real YOLO: detector ran once for 30 frames of motion + zoom;
  tracker median 13.6ms vs detector 71ms per frame.

## Phase 2 — Tracking

- `ObjectTracker` (`backend/app/services/tracking/tracker.py`) wraps
  ByteTrack via the `supervision` library — Project.md §17's explicitly
  named option. One instance per camera/inspection session, cached by
  `session_id` (`backend/app/api/vision.py`).
- `POST /api/vision/track` / `DELETE /api/vision/track/{session_id}`: same
  request/response shape as `/detect`, plus a stable `tracker_id` per
  physical object across frames.
- **Known real behavior, not a bug**: a brand-new track isn't returned on
  the frame it first appears — ByteTrack requires it to be matched again on
  the *next* frame before "activating" it, to suppress one-off spurious
  detections. Confirmed by directly querying the underlying `supervision`
  library (not assumed) and documented in `ObjectTracker.update()`'s
  docstring.
- Tests: `tests/backend/test_tracker.py` — same-object identity continuity,
  new-object activation delay, empty-frame handling, `reset()`.
- **Verified end-to-end**: `POST /api/vision/track` and the `DELETE` reset
  both respond correctly via `TestClient` against real Postgres (the
  detections list is empty because the default `MockDetector` never detects
  anything absent a trained model — expected, not a defect; the tracking
  *logic* itself is fully covered by the unit tests above).

## Phase 3 — Video-to-procedure extraction

- `extract_frames` (`backend/app/services/video/extraction.py`): streams a
  video file, yielding every Nth frame.
- `extract_candidates_from_frames` / `extract_candidates_from_video`
  (`backend/app/services/video/candidate_extraction.py`): runs the detector
  + the same `ComponentStateRule`/`StateEstimator` the live inspection flow
  uses, and emits one `CandidateStep` per **debounced** state transition per
  component — a state must hold for `min_consecutive_frames` sampled frames
  before it's accepted, filtering single-frame detector flicker out of the
  transition timeline.
- Every candidate's `status` is `pending_review` (Project.md §28's explicit
  requirement) — this module never writes a Procedure/Step row; a human
  reviews and an authoring workflow (not built here) turns approved
  candidates into a real `ProcedureDefinition` via the existing
  `POST /api/procedures`.
- Added `CandidateStep.componentId` (was missing) — without it, a
  multi-component video's candidates would be ambiguous about which part
  each transition refers to.
- `POST /api/video/extract-candidates` (multipart: video file + JSON rules).
- Tests: `tests/backend/test_candidate_extraction.py` (a controllable fake
  detector keyed to synthetic frame brightness — single removal, reinstall,
  flicker-suppressed-by-debounce vs. flicker-detected-without-debounce,
  independent multi-component tracking) and
  `tests/backend/test_video_extraction.py` (real frame sampling against a
  tiny synthetic MJPG/avi video file).
- **Verified end-to-end**: a real synthetic video uploaded via `TestClient`
  round-trips through the endpoint correctly (200, correct response shape);
  zero candidates is expected here too, for the same `MockDetector` reason
  as Phase 2.

## Phase 6 — Depth / metrology

**Honest status: built and unit-tested against synthetic data, but never
run against a real depth sensor — this environment has none.** Project.md
§25/§31/§53's "don't overclaim accuracy" applies directly: treat this as
correct math proven on known geometry, not as a field-validated measurement
system.

- `depth_to_points` (`backend/app/services/metrology/depth_to_points.py`):
  back-projects a depth map into a 3D point cloud, reusing the same pinhole
  `CameraCalibration` as the pose pipeline (one camera model, not two).
- `measure_min_distance` / `measure_gap`
  (`backend/app/services/metrology/gap_measurement.py`): the minimum
  distance between two point clouds via Open3D's nearest-neighbor search
  (Project.md #32's gap/clearance inspection). `is_within_tolerance` is a
  pure helper — this module never decides PASS/FAIL (Project.md §74); that
  stays the QA engine's job.
- `POST /api/metrology/measure-gap`.
- Tests: `tests/backend/test_depth_to_points.py` (back-projection against
  known depth/geometry, including a similar-triangles proportionality
  check) and `tests/backend/test_gap_measurement.py` (a known 2cm gap
  between two synthetic planes, minimum-not-average distance, tolerance
  boundary cases).
- **Verified end-to-end**: `POST /api/metrology/measure-gap` returns the
  exact expected distance and tolerance verdict via `TestClient`, and
  correctly rejects an empty point cloud (400).

## Phase 8 — Enterprise (partial: RBAC + audit)

Scoped down from the full phase (S1000D/PLM/MES/QMS/connected-tools need a
specific real external system to integrate against, which wasn't chosen) to
what's genuinely buildable without that: role-based access control and
audit logging, using the `User`/`AuditLog` models that have existed since
Phase 1 but were never wired to anything.

- **Deliberately minimal auth** (`backend/app/core/security.py`): email-only
  JWT login, no password/SSO — this exists so `require_role()` has a real
  current-user to check, not as a production identity system. `POST
  /api/auth/register` / `POST /api/auth/login`.
- `require_role(*roles)` (`backend/app/api/deps.py`): a reusable per-endpoint
  permission gate — 401 if unauthenticated, 403 if authenticated but not
  permitted.
- Wired onto two real endpoints:
  - `POST /api/procedures/{id}/publish` — Admin/Engineer/Technical Author
    only.
  - `POST /api/inspection/{id}/step/{stepId}/override` (**new** — Project.md
    §44's manual override, whose DB columns existed since Phase 1 but had no
    API until now) — Admin/QA Inspector only. **The original AI/rule result
    is never overwritten**, per §44's explicit requirement — verified
    directly: after an override, `result` still reads `NOT_EVALUATED` while
    `manual_override_result` reads the override.
- `audit_log.record()` (`backend/app/services/audit/audit_log.py`): one
  helper, written in the same DB transaction as the mutation it describes.
  Called from both endpoints above.
- Tests: `tests/backend/test_security.py` (JWT round-trip, tampered/garbage
  token rejection) and `tests/backend/test_rbac.py` (`require_role` tested
  directly against plain `User` objects, no DB/HTTP needed).
- **Verified end-to-end** against real Postgres: register two users
  (admin/operator) → publish without a token (401) → wrong role (403) →
  correct role (200) → one matching audit log row exists. Same
  401/403/200 sequence for the override endpoint, plus direct confirmation
  that the original result survives the override untouched.

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
- Fixed: the AR overlay's Three.js camera now matches the backend's
  calibration exactly (`camera_vertical_fov_deg`/`camera_aspect` returned on
  every pose response) instead of a hardcoded guess — see `docs/pose.md`'s
  "known limitations" for what's still not corrected (an off-center
  principal point, only relevant once a real calibration is in use).
