
# Task: Implement AI-Based Missing Component Detection in the Existing TVASTA AR System

## 1. Project Context

I am developing an AR-based quality inspection system in the existing `TVASTA` project.

The following functionality is already working:

- Camera-based physical object tracking.
- Pose estimation and tracing as the physical object moves.
- Loading and overlaying the GLB/glTF 3D model onto the physical object.
- Alignment between the real object captured by the camera and the virtual 3D model.

**Do not rebuild or replace the existing tracking, pose estimation, or overlay system.** Inspect the current codebase first and reuse the existing implementation.

### New requirement

Implement missing-component detection by comparing the expected components in the 3D model against visual evidence from the real physical object.

Example:

- The 3D model contains eight screws.
- The physical object has only seven screws.
- The system identifies which screw is missing.
- The expected location of the missing screw is highlighted in red in the AR camera view.
- The system displays a component-specific warning such as `MISSING: SCREW_004`.

The same architecture must eventually support caps, bolts, nuts, connectors, covers, electrical components, and other mechanical parts.

## 2. Inspect the Existing Codebase First

Before modifying any files:

1. Inspect the entire relevant project structure.
2. Identify the frontend framework and rendering engine.
3. Identify how the GLB/glTF model is loaded.
4. Identify how individual meshes and scene nodes are represented.
5. Locate the existing camera capture pipeline.
6. Locate the existing object detection, tracking, and pose estimation modules.
7. Determine how camera intrinsics and the tracked object pose are stored and updated.
8. Identify how the existing 3D overlay is positioned and rendered.
9. Identify the communication mechanism between the backend and frontend, if applicable.
10. Inspect existing dependencies, configuration files, and tests.

Produce a short summary of the existing architecture and the files that should be modified before implementation.

Do not assume that proposed filenames or technologies already exist. Adapt all implementation details to the actual repository.

## 3. Required Architecture

Implement the following independent modules, adapting the names to the existing project:

### A. Component Registry

Create a registry of inspectable components extracted from the existing GLB/glTF scene.

Each component should have:

- Stable component ID.
- Original GLB node or mesh reference.
- Component type.
- Required or optional status.
- Local transform relative to the tracked object's root.
- Expected geometry and bounding volume.
- Inspection method.
- Current inspection status.
- Detection confidence and timestamps where applicable.

Example:

```json
{
  "objectId": "PUMP_001",
  "components": [
    {
      "id": "SCREW_001",
      "meshName": "Screw_Top_Left",
      "type": "screw",
      "required": true,
      "inspectionMode": "presence"
    },
    {
      "id": "CAP_001",
      "meshName": "Protective_Cap",
      "type": "cap",
      "required": true,
      "inspectionMode": "presence"
    }
  ]
}
```

This is illustrative data. Use actual mesh names from the project.

Support an explicit mapping between registry IDs and GLB nodes. Handle duplicate names, nested transforms, missing meshes, and combined geometry gracefully. Never assume that every scene node is an inspectable component.

### B. Component Projection

Reuse the existing object pose and camera calibration.

For each inspectable component:

1. Calculate its transform relative to the tracked object.
2. Calculate its current world-space transform using the existing tracking transform.
3. Transform its geometry into camera coordinates.
4. Project its expected visible geometry into the camera image.
5. Determine its projected bounding region or component mask.
6. Estimate whether the component should be visible from the current camera viewpoint.

Use the existing camera intrinsics, distortion handling, coordinate conventions, and pose transformation wherever available.

Do not introduce a separate pose estimation pipeline merely to project the components.

Handle coordinate-system differences between the GLB renderer, camera and computer vision backend correctly.

### C. Component Presence Verification

Implement a modular inspection interface so different verification algorithms can be used for different component types.

Start with:

1. Expected component location from the 3D model.
2. Local camera crop around the projected component.
3. Component detection or classification.
4. Optional reference-image comparison.
5. Visibility and image-quality checks.
6. Confidence-based inspection decisions.

The inspection pipeline must be capable of answering:

- Is the expected component present?
- Is it missing?
- Is it partially installed or damaged?
- Is the component occluded?
- Is the observation unreliable?

Use the existing AI/computer vision stack where practical.

If YOLO is already integrated, reuse it where appropriate. Otherwise, evaluate whether a pretrained detector recognizes the required component classes before introducing dependencies.

If existing models cannot reliably recognize screws or caps, create a clean interface for a custom-trained component detector.

Do not assume a generic pretrained YOLO model can reliably recognize tiny industrial fasteners.

Do not require AI training before implementing the component registry, projection, debug visualization and inspection pipeline.

### D. Visibility and Occlusion Handling

Implement visibility checks before declaring a component missing.

A component must not be marked missing merely because it is:

- Behind another component.
- Outside the camera frame.
- Too small in the current image.
- Blurred or poorly illuminated.
- Affected by tracking loss or significant misalignment.
- Visually ambiguous due to reflections or similar surfaces.

Use CAD geometry and camera viewpoint to estimate occlusion where feasible. Since the current camera is RGB-only, do not assume reliable real-world depth is available.

When visibility or image quality is insufficient, return `UNCERTAIN` rather than `MISSING_CONFIRMED`.

### E. Inspection Decision Engine

Implement a component-level state machine with the following states:

- `NOT_INSPECTED`
- `PRESENT`
- `MISSING_CANDIDATE`
- `MISSING_CONFIRMED`
- `UNCERTAIN`
- `OCCLUDED`

Requirements:

1. Evaluate only components that are required for the current inspection.
2. Combine detection confidence, visibility, tracking quality and temporal consistency.
3. Use configurable thresholds.
4. Require repeated valid observations before confirming a missing component.
5. Ignore invalid observations during tracking loss or poor image quality.
6. Reset or re-evaluate results when the viewpoint changes significantly.
7. Avoid flickering between present and missing states.
8. Keep component results independent so one missing screw does not make the entire object fail every inspection check.
9. Record why a component received its current status.

Begin with configurable temporal confirmation settings, such as five to ten valid observations, then tune them using test results. Do not treat that range as a universal threshold.

### F. AR Discrepancy Renderer

Implement a separate discrepancy-rendering module integrated with the existing overlay.

When a component reaches `MISSING_CONFIRMED`:

1. Locate its expected GLB mesh and transform.
2. Suppress its normal opaque rendering in inspection mode, if necessary, so the virtual reference component does not misleadingly appear installed.
3. Preserve its geometry and expected transform for inspection.
4. Render a red outline, red ring, or translucent red ghost representation at the expected location.
5. Display the component ID and status.
6. Keep the marker aligned with the physical object as it moves.
7. Remove the discrepancy marker when the component is confirmed present again or the inspection is reset.

Use these visual conventions:

- Green: component verified present, if the UI supports status colors.
- Amber: missing candidate or inspection in progress.
- Red: confirmed missing component.
- Neutral: uncertain or occluded.

For tiny screws, prefer a clearly visible red ring or a slightly enlarged projected marker rather than relying only on a thin 3D outline.

Keep the normal AR overlay and inspection visualization logically separate so that toggling inspection does not break ordinary AR viewing.

### G. Inspection Results and UI

Display an inspection panel showing:

- Total required components.
- Number verified present.
- Number confirmed missing.
- Number uncertain or occluded.
- Component ID.
- Component type.
- Inspection status.
- Confidence or evidence score.
- Last valid inspection time.
- Reason for a missing or uncertain result.

Allow users to select a component in the results panel and highlight its corresponding location in the AR view.

Provide controls to start, pause and reset an inspection if compatible with the current UI architecture.

Do not mark the overall object as passing while required components remain missing or uninspected.

## 4. Rendering and Comparison Requirements

Do not compare the entire raw RGB image with a rendered 3D image using simple pixel subtraction as the sole detection method.

Lighting, reflections, material differences, shadows, camera exposure and slight registration errors can create false differences.

Use the projected component location to focus verification on the relevant image region.

Use a combination of:

- Component detection.
- Local image classification or reference comparison.
- Geometry-based projection.
- Visibility estimation.
- Temporal consistency.

Design the verification interface so more advanced methods can be added later without rewriting the tracking pipeline.

## 5. Suggested Implementation Structure

Adapt this proposed structure to the actual repository:

```text
backend/
  app/
    inspection/
      component_registry.py
      projection.py
      visibility.py
      component_detector.py
      presence_verifier.py
      decision_engine.py
      inspection_service.py
    workers/
      tracking_worker.py
      inspection_worker.py
    schemas/
      inspection_result.py

frontend/
  src/
    ar/
      ComponentRegistry.js
      DiscrepancyRenderer.js
      InspectionPanel.js
```

These filenames are suggestions, not assumptions. Reuse the existing project conventions, language, framework and folder structure.

Avoid duplicating existing tracking or camera modules.

## 6. Implementation Order

Implement the feature incrementally in the following order.

### Phase 1: GLB component inventory

- Load the existing GLB.
- Enumerate scene nodes and meshes.
- Identify individual screw and cap meshes.
- Generate a component registry.
- Display each component's stable ID and transform in a debug panel.

Acceptance criteria:

- Inspectable components can be individually identified.
- Their transforms remain correct relative to the tracked object.
- No existing overlay functionality is broken.

### Phase 2: Projected-location visualization

- Reuse the existing tracking pose and camera calibration.
- Project expected component locations into the camera frame.
- Draw temporary debug markers at the expected positions.
- Test while the physical object is moved and rotated.

Acceptance criteria:

- Each debug marker corresponds to the correct physical component location.
- Markers remain aligned while the object moves.
- Coordinate-system and calibration errors are identified before adding AI.

### Phase 3: Basic presence verification

- Implement local component crops.
- Add a modular detection/classification interface.
- Test with a screw installed and then removed.
- Record detections and inspection evidence for debugging.

Acceptance criteria:

- The system can distinguish the two cases under controlled conditions.
- Poor visibility results in `UNCERTAIN`, not a false missing alert.

### Phase 4: Missing-component state machine

- Add temporal confirmation.
- Integrate visibility and tracking-quality checks.
- Prevent status flickering.
- Record component-level results.

Acceptance criteria:

- A missing component can be confirmed reliably.
- Tracking loss and occlusion do not produce false confirmed-missing states.

### Phase 5: Red AR highlighting

- Connect confirmed missing results to the discrepancy renderer.
- Highlight the correct component position.
- Suppress the normal opaque reference mesh where necessary.
- Verify correct behavior while the object moves.

Acceptance criteria:

- A missing screw or cap is highlighted in red at its expected location.
- Present components are not falsely highlighted.
- The original AR overlay still works when inspection mode is disabled.

### Phase 6: Testing and expansion

Test:

- All required components present.
- One screw missing.
- Multiple screws missing.
- A cap missing.
- Partial installation.
- Occluded component.
- Camera movement and object rotation.
- Poor lighting.
- Blurred images.
- Tracking loss and reacquisition.
- Different camera distances.
- Small components near image-resolution limits.

Measure false-positive and false-negative rates, not only overall accuracy.

## 7. Performance Requirements

- Reuse the existing camera stream.
- Avoid starting duplicate camera capture loops.
- Keep tracking and rendering responsive while inspection runs.
- Run expensive AI inference asynchronously or at a configurable inspection rate.
- Reuse the latest valid tracking pose.
- Cache component metadata and reference geometry.
- Avoid reconstructing the entire GLB scene for every frame.
- Avoid unnecessary GPU-to-CPU image transfers.
- Ensure that inspection results are synchronized with the correct camera frame and pose.
- Use explicit cleanup for temporary rendering objects and resources.

Prioritize correctness and stability before performance optimization.

## 8. Testing and Debugging

Add automated tests for:

- GLB component enumeration.
- Component registry mapping.
- Transform calculations.
- Camera projection.
- Component visibility decisions.
- State transitions.
- Temporal confirmation.
- Discrepancy-rendering lifecycle.

Add a debug mode that shows:

- Camera frame.
- Current tracked pose.
- Expected component positions.
- Projected component bounding boxes.
- Detector results.
- Visibility decisions.
- Component status and confidence.
- Red discrepancy markers.

Log why a component was marked missing or uncertain.

Do not hide errors silently or report successful detection without supporting evidence.

## 9. Engineering Constraints

- Do not replace the current tracking system.
- Do not rewrite the application from scratch.
- Do not introduce Unity or another rendering engine unless the existing architecture genuinely requires it.
- Do not require a depth camera for the initial MVP.
- Do not assume every GLB mesh is an independently inspectable component.
- Do not hardcode component pixel coordinates.
- Do not declare components missing solely because a detector returned no result.
- Do not treat occlusion as a confirmed defect.
- Do not introduce large dependencies without explaining why they are necessary.
- Preserve existing configuration and environment settings.
- Follow the project's current coding conventions.
- Keep changes modular and backward-compatible.

## 10. Required Deliverables

After implementation, provide:

1. A concise summary of the existing architecture.
2. The files created and modified.
3. An explanation of how the existing tracking pose is reused.
4. The component registry format and how to map GLB meshes.
5. The component projection and verification logic.
6. The missing-component decision rules.
7. The red highlighting implementation.
8. Any new dependencies and their licences.
9. Setup and run instructions.
10. Test results and known limitations.
11. The exact remaining steps, if model training or manually labelled data is required.

Implement the feature in the actual existing repository. Do not stop at a theoretical plan or pseudocode.

First inspect the codebase, then make the smallest coherent changes in stages. Run the relevant tests after each stage and report any blockers honestly.

**Primary objective:** preserve the working AR tracking and overlay while adding reliable, component-level missing-part detection and red highlighting based on the existing GLB reference model.
