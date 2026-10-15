# Phase 8 — pose orientation, internal features, calibration: audit and report

Scope: `Phases/Phase-8.md`. Everything below was implemented in the existing
code (no estimator replaced, no API removed) and is covered by tests. Numbers
are measured unless marked otherwise.

## 1. Audit — how a frame becomes an overlay

```
BRIO (1280x720@30, auto exposure) ─ useCamera.ts ─ canvas.toBlob(JPEG 0.85)
  └─ WebSocket /api/vision/ar-session/stream (≤2 frames in flight; newest wins)
       └─ vision.process_ar_frame
            ├─ load_calibration → K (+ distortion)  [Phase 8: undistort here]
            ├─ hand mask (MediaPipe + YOLO person, async, newest)
            ├─ station lock?  → frozen pose, image-shift check
            ├─ guide view / click-to-select → session.seed(detection, pose)
            └─ ARSession.step (SEARCHING→INITIALIZING→TRACKING→LOST→RECOVERING)
                 ├─ detector: CAD detect (pose service: FastSAM + DINOv2) | YOLO | box | guide
                 └─ MegaPoseTracker
                      initialize: MegaPose full search (pose service, ~1.4–2.1 s)
                                  → resting side → 3D edge fit (5 starts)
                                  → [Phase 8] orientation decision (twins × features)
                      update:     optical flow every frame → propagate_pose
                                  background MegaPose refine (~0.25–0.34 s) → edge fit
                                  → [Phase 8] correction guard → blend | switch | reject
       ├─ component inspection (tracking_ok gate) ─ registry, projection, CAD visibility, verifiers, hysteresis
       └─ ARFrameResponse: T_camera_object → cv_model_pose_to_threejs → position + quaternion (Three.js)
  frontend: applyCameraIntrinsics (projection from fx,fy,cx,cy) + model transform; overlays on a 2D canvas
```

| Responsibility | Code |
|---|---|
| Intrinsics | `services/pose/calibration.py` (`load_calibration`, `approximate` from 82° HFOV) |
| Mirroring | none anywhere (checked: no `scaleX(-1)` / mirrored capture) |
| MegaPose init / refine | `pose_service/server.py` (`/estimate`), `services/pose/model_pose_client.py`, `MegaPoseTracker` |
| Outline pose | `services/pose/outline_pose.py` |
| Edges / correspondence | `services/pose/edge_align.py` (Canny → distance field → robust LSQ) |
| Candidate selection | silhouette IoU (pose service), resting side, **new** `services/pose/disambiguation.py` |
| Optical flow | `services/tracking/flow_tracker.py`, `propagate_pose` |
| Pose filter | `services/tracking/pose_filter.py` (via `ARSession`) |
| GLB → tracked frame | recentred on the bounding-box centre × `Model3D.scale` (pose service + `assembly_center`) |
| Three.js transforms | `transforms.cv_model_pose_to_threejs`; `RegistrationOverlay.applyCameraIntrinsics` |
| Component projection / visibility | `services/component_inspection/{projection,visibility}.py` |

**Pose representation:** `T_camera_object` 4×4 (rotation matrix + translation),
OpenCV camera frame (x right, y down, z forward), object = glTF frame
recentred, **metres**; per frame (`capture_ms` timestamp from the browser).
The frontend receives position + quaternion (x,y,z,w) in Three.js convention.

## 2. Root cause of the "circle at top-right shows at top-left" flip

Checked separately, as the phase asks:

| Possible cause | Finding |
|---|---|
| Ambiguous 6DoF pose | **Root cause.** A plain box has 4 rotations that look identical (identity + 180° about each axis). Outline, 3D edges and MegaPose on an untextured mesh fit every twin equally (test: edge-fit scores of truth and twin differ < 0.02). Nothing in the pipeline looked at the circle, so the twin returned was arbitrary — and each background correction could hand over the other twin, which was then **blended** into the shown pose (halfway poses, up to 179° off). Your real Case model has the same 4 twins (detected in 39 ms). |
| Corner ordering / face orientation | Not used for 6DoF (no PnP from corners on the model path). |
| GLB origin / scale / transform | Consistent: pose service and frontend recentre the same way (existing tests). |
| Calibration | Approximate (82° guess, no file captured). Affects accuracy (scale/tilt), **not** left↔right. Fixed (§3B). |
| Camera ↔ renderer conversion | Correct — proven to < 1e-6 px by a new end-to-end test. |
| Display mirroring | None. |
| Internal edges absent from CAD | Yes for printed marks: a printed circle has no crease in the mesh. Now added as feature rims (§3C). |
| Hand occlusion | A hand over the only distinguishing feature makes the orientation genuinely undecidable — now reported as such. |
| Optical-flow drift / corrections | Corrections to a twin were blended in. Now guarded (§3D). |
| Temporal filter smoothing a wrong pose | Same mechanism — blending across orientations. Now never done. |

## 2a. Live measurements available (pose service log of the user's session, RTX 4090 / WSL)

| Call | Measured |
|---|---|
| CAD detection (`/detect`) | 476 – 1057 ms |
| MegaPose full search (`coarse+refine`) | 1386 – 2078 ms |
| MegaPose refine (tracking correction) | 245 – 336 ms |

Not measured (needs a live session; both the pose service and PostgreSQL were
down when this report was written): end-to-end FPS, GPU utilisation, real
flip rate on camera, recovery time after occlusion on camera. The evaluation
below therefore uses rendered scenes with exact ground truth.

## 3. What changed

**B — calibration and coordinates**
- `calibration.py`: calibration files carry camera label, date, RMS, images used, version; `data/calibration/camera.json` is used automatically; a file for **another camera** (label) or another aspect is refused with the reason (`calibration_note`), never silently used; `undistort()` with cached maps.
- `process_ar_frame`: model-based frames are undistorted once on arrival (re-encoded for the pose service), so every consumer sees the same image; the residual on the raw video is reported (`distortion_removed_px`).
- In-app calibration: `api/calibration.py` (capture with corner check, compute, status, remove) + `CalibrationPanel.tsx` (AR view → *Calibrate camera*). The worker records RMS.
- New tests: calibration recovered from synthetic distorted checkerboard photos (fx within 2 %, k1 within 0.04), undistortion restores pinhole corners (median < 0.8 px), wrong camera refused, renderer chain = OpenCV to 1e-6 px, "upper-right stays upper-right" on screen.

**C — ambiguous orientations**
- `pose/symmetry.py`: detects a model's (near-)symmetric rotations from its geometry (180° about principal axes, 90° on square sections; tolerance relative to size, sampling noise discounted).
- `pose/pose_features.py`: per-model pose features in the **existing** mapping file (`poseFeatures` beside `components`): circles/holes (from a GLB node, a registered component, or a model-local position for marks in a combined mesh) and whole nodes. For a pose: projected outline, status (detected / not detected / occluded by hand / hidden by CAD / facing away / outside), edge + contrast evidence, and the actual circle found nearby (fitted ellipse) with its residual in px. Identity, not appearance: a feature is only looked for where *that* feature projects.
- `pose/disambiguation.py`: scores each twin (resting side as station evidence), **confirmed** only when one twin's features are clearly seen and clearly better (or, for near-symmetric parts, the CAD edges clearly prefer it and features don't disagree), else **ambiguous** — keeping the twin closest to the last reliable pose, never presenting a guess as fact. Feature rims are never used as *geometric* evidence (an empty corner is not proof: a missing print could be the defect).

**D — edges**: feature rims (printed circles) join the 3D edge fit so the overlay aligns on internal features too; visibility (facing + CAD depth buffer), hand-pixel exclusion and robust loss were already in place.

**F — tracking**
- At lock: edge fit → orientation decision → re-fit of the chosen twin; a `requiredForInitialization` feature not seen refuses the lock with the reason.
- Correction guard (background, never blocking frames): a correction that is a symmetric twin of the tracked orientation is turned back unless that frame's features prove a real turnaround; any other turn > 15° (same frame, so not motion) must fit the CAD edges clearly better. Accepted turns **replace** the pose — never blended across orientations. Counters in `quality`.

**H — inspection**: an ambiguous orientation on a model with pose features configured stops component verdicts ("orientation not confirmed — …"); pose uncertainty is never a defect. Models without features keep working as before (their orientation is reported as unverified).

**Frontend diagnostics**: HUD "Orientation confirmed / unconfirmed — reason"; per feature a dashed predicted outline + cross, the found circle as a dot, status and residual px (toggle *Pose features*); existing axes gizmo shows the model origin/axes.

## 4. Before / after (synthetic scenes, real tracker code) — `docs/phase8/eval_results.md`

MegaPose simulated as a random symmetric twin of the truth (±2°, ±3 mm), 6 locks per scene.

| | Before | After |
|---|---|---|
| Locks correct, 14 scenes with the circle in view (84 locks) | 30 (51 flipped) | 83 (1 flipped, flagged *ambiguous*); 79 *confirmed* |
| Locks where inspection would judge a **wrong** pose (all 16 scenes, 96 locks) | 61 | **0** |
| Hand over the circle | 2/6 wrong, reported as fine | 6/6 *ambiguous*, inspection paused |
| Plain box, no circle | 5/6 wrong, reported as fine | 6/6 *ambiguous* |
| Circle position error at a correct lock | — | 0.8 – 4.0 px (mean per scene) |
| 12 twin corrections while tracking: shown-pose flips | 8 (worst 179°) | **0** (worst 1.1°) |
| Operator turns the box round (one correction) | 126° off (blended halfway) | 0.0° off |
| Lock time p95 (CPU part, excl. MegaPose) | 365 – 725 ms | 647 – 948 ms |

Grazing view (camera 35° above the desk): 5/6 correct but only 1/6 confirmed — the circle is a thin ellipse there; the rest are honestly *ambiguous*.

## 5. Configure your Case

The Case model is symmetric for the outline. Give it at least one pose
feature in `data/component_registry/0233106a-a63d-486a-8c21-7052953b3d12.json`,
e.g. the LED cluster on the glossy end panel (position/normal/radius in the
GLB's own units and axes; read them off the model in Vishwa or the viewer):

```json
"poseFeatures": [
  {"featureId": "LEDS", "type": "circle", "position": [X, Y, Z], "normal": [0, 0, 1], "radius": R}
]
```

or a separate GLB node (`"meshName"` / `"nodeIndex"` / `"componentId"`). Then
restart the backend; the HUD shows whether each lock is *confirmed*.

## 6. Remaining limitations / next steps

1. **No real-camera numbers yet** for flip rate, FPS, GPU, occlusion recovery — run the live checklist (Phase J list) with the BRIO once the services are up; `python -m app.eval.pose_eval` covers the synthetic set.
2. Pose features must be configured per model; there is no automatic learning of printed marks yet ("teach appearance" from the earlier plan is the natural next step).
3. Feature types implemented: circles/holes and node outlines. Labels, connectors and corners as *point* landmarks (and solvePnP from multiple matched landmarks) are not implemented.
4. The video on screen is not undistorted; with a strongly distorting lens the overlay can be off near the corners by `distortion_removed_px`.
5. Twins are handled for whole-object tracking only (not when tracking one part with an offset).
6. The camera calibration has not been captured on the BRIO yet — the system still runs on the 82° assumption until you do (AR view → Calibrate camera).

## 7. Run

```
cd backend
.venv/Scripts/python -m pytest ../tests/backend -q          # 402 tests
.venv/Scripts/python -m app.eval.pose_eval                  # before/after → docs/phase8/eval_results.*
python -m uvicorn app.main:app --port 8000                  # restart the backend to load Phase 8
cd ../frontend && npm test && npm run build
```
