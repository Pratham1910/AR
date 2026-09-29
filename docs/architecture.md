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

## Deferred (see `roadmap.md`)

3D/Three.js viewer, pose/6DoF registration, depth metrology, Unity/AR/MR,
S1000D ingestion, and connected-tool protocols are architected for (stable
IDs, `Model3D`/`ValidationMethod.POSE` already in the schema) but not built in
Phase 1, per Project.md §13 and §58.
