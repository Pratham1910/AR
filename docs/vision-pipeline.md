# Vision Pipeline

Phase 1 implements the first half of Project.md §14's pipeline:

```
Camera → Frame acquisition → Preprocessing → Object detection → State estimation → (QA engine, separate)
```

Segmentation, full multi-object tracking (ByteTrack/BoT-SORT), and pose are
Phase 2+ (Project.md §58) — `ValidationMethod.TRACKING` / `.POSE` already
exist in the schema so steps can declare the requirement now, but the QA
engine currently treats them as `True` placeholders when required, pending
the real implementations. See `docs/roadmap.md`.

## Detection — `backend/app/services/vision/detector.py`

`Detector` is an abstract interface with two implementations:

- **`MockDetector`** — returns injected/fixed detections. This is what makes
  the full procedure/QA/evidence loop provable *before* a custom YOLO model
  is trained on real assembly footage (Project.md §78's stated first success
  criterion: `CAMERA → DETECTION → STATE → QA → EVIDENCE`).
- **`YoloDetector`** — thin wrapper around `ultralytics.YOLO`. Class names
  come from the loaded model, not a hard-coded list (§15) — swap
  `MODEL_PATH` and the class set changes with it.

`build_detector(model_path)` is the only place that decides which
implementation to use, driven by `Settings.model_path` (`.env`'s
`MODEL_PATH`) — application code never branches on "do we have a real model".

## State detection — `backend/app/services/state_detection/state_engine.py`

Phase 1 deliberately does **not** train a state classifier. `StateEstimator`
uses a configured presence/absence rule per component
(`ComponentStateRule(component_id, class_label, present_state_id,
absent_state_id)`): if the target class is detected, report the "present"
state; otherwise the "absent" state, at a fixed confidence (0.90) since a
negative observation from one frame is inherently less certain than a direct
detection.

`action_expects_presence(action_type)` maps a step's `ActionType` to whether
its *expected* (post-action) state implies the component should still be
visible. This matters to the QA engine (see `docs/qa-engine.md`): a `REMOVE`
step's success condition is the component's *absence* — Project.md §8's own
example is `"PCB detected = false ... -> PASS"`.

This whole module is a placeholder for a trained per-component state
classifier and can be swapped without touching the QA engine or the API
layer, since both only ever see `(state_id, confidence)`.
