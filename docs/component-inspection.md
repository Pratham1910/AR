# Component inspection (missing parts)

Finds which components of a tracked object are missing — "MISSING: SCREW_004" —
and marks where they belong in the AR view. Runs on the model-based (CAD)
tracking stream; it reuses that session's pose and camera calibration and has
no pose estimation of its own.

```
AR frame + its tracked pose (vision.process_ar_frame)
  -> registry      which GLB parts are components, stable IDs, geometry in the object frame
  -> projection    each component's pixels at that pose (pinhole, same K as the overlay)
  -> visibility    tracking ok? in frame? big enough? hidden by other CAD parts? sharp/exposed?
  -> verifiers     present / absent / uncertain, on the component's *visible* pixels
  -> decision      per-component state machine, temporal confirmation
  -> results       with every AR frame; markers drawn by frontend discrepancy.ts
```

Code: `backend/app/services/component_inspection/`, API glue
`backend/app/api/component_inspection.py`, frontend
`frontend/src/features/viewer3d/{discrepancy.ts,ComponentInspectionPanel.tsx}`.

## Components

Without a mapping file, components are the parts named in the Parts panel and
nodes whose names say what they are (`Screw_Top_Left` → `screw`). Other nodes
are never assumed to be components. A mapping file makes it explicit —
`data/component_registry/<model_id>.json` (or `PUT /api/models3d/{id}/components`):

```json
{
  "objectId": "PUMP_001",
  "components": [
    {"id": "CAP_001", "meshName": "Protective_Cap", "type": "cap"},
    {"id": "SCREW_001", "nodeIndex": 7, "type": "screw", "required": false},
    {"idPrefix": "SCREW_", "meshName": "Screws", "split": true, "type": "screw"}
  ]
}
```

- `meshName`: glTF node name as authored or as Three.js shows it; repeated
  names are reported as ambiguous — use `nodeIndex`.
- `split`: one node holding several separate bodies (all screws exported as one
  mesh) becomes one component per body, numbered by position; `piece` picks one.
- Unresolvable entries stay listed with the reason and keep the verdict
  INCOMPLETE; they are never dropped silently.

## Telling present from absent

| Check | Needs | Notes |
|---|---|---|
| calibrated | Parts panel calibration ("cap-on"/"cap-off" frames) | two-state; doubts values unlike both states |
| reference | "Learn good state" on the AR page | one-state: flags what stops looking like the good state; stands down > 35° from learned views |
| detector | `COMPONENT_DETECTOR_WEIGHTS` + `COMPONENT_DETECTOR_CLASSES` | a custom-trained YOLO model; not finding a part is weak evidence only |

The stock COCO detector has no screw/bolt/nut/cap classes, so it isn't used.
With no check available a component stays UNCERTAIN.

## Decision rules

- Only valid observations count: confident TRACKING pose, ≥ 90% in frame, ≥ 40 px,
  ≥ 30% not hidden by other CAD parts (and that part ≥ 40 px), sharp, not too dark
  or clipped. Otherwise the frame neither confirms nor clears anything; the reason
  is shown.
- MISSING_CONFIRMED after 6 consecutive valid "absent"; PRESENT after 3 "present";
  a confirmed missing clears after 4 "present". A "present" breaks a missing streak;
  so does a viewpoint change > 30° or 5 s without a valid view.
- Each component is independent. PASS only when every required component is
  confirmed PRESENT; any confirmed missing is FAIL; anything else INCOMPLETE.
- A hand over a component: not in the CAD model, so it is caught by the checks
  ("like neither calibrated state", "differs from …" without the confirmation run
  completing while the hand moves) — see limitations.

All thresholds are settings (`INSPECTION_*` in `.env`, see `app/core/config.py`).
Each pass is logged to `data/inspection_logs/<model>/<session>-<run>.jsonl`.

## Limitations

- RGB only: occlusion by things outside the CAD model (hands, tools) is not
  measured directly. Phase 7's hand mask is the planned fix.
- Lens distortion is not modelled (as in the rest of the pose pipeline); the
  camera matrix is the approximate default unless calibrated.
- The reference check compares brightness/edge/colour statistics: it needs the
  exposure locked and the same lighting as when it learned, and can't say *why*
  something looks different.
- Very heavy meshes (> 60k triangles in view) skip CAD occlusion and say so.
