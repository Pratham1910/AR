# Architecture

TVASTA is built around a central digital-thread model (Project.md §80):

```
ASSET → COMPONENT → STATE → PROCEDURE → ACTION → OBSERVATION → VALIDATION → EVIDENCE → QA RECORD
```

## Layers (Project.md §3, §74)

Each layer knows only its own concern and is independently testable:

| Layer | Module | Knows | Does NOT know |
|---|---|---|---|
| Procedure Engine | `backend/app/services/procedure_engine` | states, actions, transitions | pixels, models, PASS/FAIL |
| Vision Engine | `backend/app/services/vision` | detections, bounding boxes | states, requirements |
| State Detection | `backend/app/services/state_detection` | discrete component states from detections | PASS/FAIL, tolerances |
| QA Engine | `backend/app/services/qa_engine` | requirements, thresholds, PASS/FAIL/UNCERTAIN | how to run inference |
| Evidence Engine | `backend/app/services/evidence` | frames, storage keys | QA logic |

No subsystem performs another's job. The API layer (`backend/app/api`) is the only place that wires them together.

## Request flow (Phase 1)

```
Browser (camera) --image_base64--> POST /api/inspection/{id}/step/{stepId}/observe
    -> Detector.detect()                (app/services/vision/detector.py)
    -> StateEstimator.estimate()        (app/services/state_detection/state_engine.py)
    -> persists Observation + Evidence

Browser --> POST /api/inspection/{id}/step/{stepId}/validate
    -> QAEngine.evaluate()              (app/services/qa_engine/engine.py)
    -> persists InspectionStep.result
    -> returns PASS / FAIL / UNCERTAIN / NOT_EVALUATED / MANUAL_REVIEW
```

## Why FastAPI, not Node (Project.md §10)

The vision/ML stack (OpenCV, PyTorch, Ultralytics) is Python-native. Putting the
API in the same runtime avoids an unnecessary service boundary for the MVP.

## 3D / pose (Phase 4/5 — see `docs/3d.md`, `docs/pose.md`)

```
Browser (live camera) --image_base64--> POST /api/vision/pose
    -> ArucoPoseEstimator.estimate()   (app/services/pose/aruco_pose.py)
    -> cv_pose_to_threejs()            (app/services/pose/transforms.py)
    -> {position, quaternion} in Three.js space
Browser -> sets the loaded GLB's transform directly, no coordinate math client-side
```

This is intentionally a separate pipeline from the Phase 1 procedure/QA loop
— pose/registration doesn't gate or get gated by tracking (Phase 2) or
video-extraction (Phase 3), so it could be (and was) built ahead of them.

## Deferred (see `roadmap.md`)

Tracking (ByteTrack/BoT-SORT), video-to-procedure extraction, depth
metrology, Unity/AR/MR, S1000D ingestion, and connected-tool protocols are
architected for (stable IDs, `ValidationMethod.TRACKING`/`.POSE` already in
the schema) but not built yet, per Project.md §58.
