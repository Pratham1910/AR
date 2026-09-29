# QA Engine

`backend/app/services/qa_engine/engine.py`

**AI confidence never automatically equals QA approval** (Project.md §8, §30).
`QAEngine.evaluate()` is pure, deterministic Python over already-computed
inputs (detections, observed state, confidence) — it never calls a model.

## Inputs (`QAInput`)

- `expected_state_id` / `observed_state_id` / `state_confidence`
- `detections` — raw vision output for this step
- `required_component_class` — which detection class matters, if any
- `required_methods` — the step's declared `ValidationMethod`s
- `tracking_stable` / `pose_valid` — placeholders until Phase 2/5 land
- `expects_presence` — **the key nuance**: whether the expected state implies
  the target component should still be detectable. Derived from the step's
  `ActionType` via `state_detection.state_engine.action_expects_presence`.
  Without this, a `REMOVE` step could never PASS, because "required component
  detected" would always be false on success — Project.md §8's own example
  (`PCB detected = false → PASS`) is exactly this case.

## Decision order

1. No observed state at all → `NOT_EVALUATED` (we didn't fail to satisfy the
   requirement, we failed to observe — a different thing, per §8).
2. Required `TRACKING`/`POSE` present but reported unstable/invalid →
   `UNCERTAIN` (never guess through an unreliable signal).
3. `expects_presence` requirement met (or not applicable) **and** state
   confidence clears threshold **and** state matches expected → `PASS`.
4. State doesn't match expected → `FAIL`, with a reason built only from
   validation data (component detected/not, state observed) — **never
   hallucinated** (§43). See `_fail_reason`.
5. State matches but confidence is borderline → `UNCERTAIN` (ask a human,
   don't silently pass).

## Thresholds

`QAThresholds` (detection confidence, state confidence) come from
`Settings` (`.env`: `CONFIDENCE_THRESHOLD` and the QA-specific settings in
`app/core/config.py`) — configuration, not hard-coded constants (§62).

## Tests

`tests/backend/test_qa_engine.py` covers PASS/FAIL/UNCERTAIN/NOT_EVALUATED
for both presence-expecting and absence-expecting steps, and the
tracking-unstable → UNCERTAIN path.
