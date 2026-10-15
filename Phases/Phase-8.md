
# TVASTA — Improve 6DoF Pose Accuracy, Internal Feature Alignment, and AR Overlay Stability

## Role

Act as a senior computer-vision engineer specializing in industrial augmented reality, CAD-based 6DoF pose estimation, model-based tracking, OpenCV, GPU inference, and real-time quality inspection systems.

Your task is to inspect and improve my existing TVASTA AR application. This is an existing, partially working implementation. Do not create a separate demo, replace the architecture wholesale, or break existing features.

First understand the repository and its actual implementation. Then identify the causes of incorrect orientation, left-right flips, internal-feature misalignment, tracking drift, and unstable overlays. Implement improvements incrementally, preserving working behavior.

---

## 1. Existing project and architecture

TVASTA already detects physical objects and overlays a corresponding 3D GLB/glTF model onto the live camera feed.

### Camera and frontend

- Logitech BRIO webcam.
- Browser camera capture through React, using `useCamera.ts`.
- Resolution: 1280×720 at 30 FPS, with fallback to 15 FPS and then 640×480.
- JPEG frames sent through WebSocket, quality 0.85, with at most two frames in flight.
- React + Vite + Three.js.
- Existing X-Ray, Solid, Wireframe, and crease-edge visualization modes.
- Existing component highlighting and inspection interface.

### Backend

- FastAPI with Python 3.11.
- OpenCV, NumPy, and SciPy.
- A separate GPU pose service running in WSL Ubuntu, currently using port 8765.
- RTX 4090 GPU available for inference.
- PostgreSQL for model assets, procedures, and inspections.

### Existing object initialization methods

The application already has some or all of the following methods:

- Guide view: align the object with the model's outline.
- Click-to-select: choose an object from the camera image.
- FastSAM + DINOv2: generate candidate regions and compare them with rendered CAD views.
- YOLOv8n-seg: detect known object classes and provide masks.
- Manual rectangle selection.
- MegaPose (`megapose-1.0-RGB` through HappyPose) for CAD-based 6DoF pose estimation.
- Outline-based pose initialization for difficult objects.
- Distance gating, approximately 0.15–2 metres.
- Configurable resting-side assumptions.

### Existing pose refinement

- Mesh simplification, targeting at most 10,000 faces.
- OpenCV Canny edge extraction.
- SciPy least-squares fitting for 3D edge alignment.
- Alignment of external outlines and internal crease edges.
- Refinement when initializing a pose and after MegaPose corrections.

### Existing tracking

- OpenCV optical flow between frames.
- Periodic MegaPose corrections in the background.
- Blended corrections to reduce sudden jumps.
- Pose smoothing with a Kalman-style filter.
- Hold pose while the object is stationary.
- Tracking state machine: Searching → Initializing → Tracking → Lost → Recovering.
- Station lock for fixed-camera inspection stations using image-shift detection.
- MediaPipe hand landmarks and YOLO person masks.
- Occlusion hold of up to approximately 1.5 seconds.

### Existing component inspection

- Component registry derived from GLB model objects and a mapping file.
- CAD projection and depth-buffer visibility checks.
- Component presence verifiers, including calibrated thresholds, reference appearances, and detectors.
- Decision hysteresis: six absent frames to confirm missing, three present frames to confirm present, and four frames to clear a missing verdict.
- Inspection reports.

**Important:** Verify each item against the actual repository before changing anything. These are the intended/current architecture details, not permission to assume every feature is implemented correctly.

---

## 2. Primary problem

The outer shape of a simple rectangular object aligns reasonably well, but internal features are not consistently aligned.

Example:

- The real object is a rectangular box.
- A circular mark or hole is physically located near the upper-right corner.
- The corresponding 3D feature is also located at the upper-right corner in the GLB.
- Sometimes the overlay is correct.
- Sometimes the pose flips, rotates, or mirrors, making the circle appear on the upper-left.
- Similar problems may occur with internal edges, screws, connectors, holes, labels, and separate assembly components.

The system appears to rely too heavily on the external silhouette or accepts ambiguous pose estimates without sufficient verification.

The goal is to make the overlay correspond to the physical object's actual orientation and visible internal geometry.

Do not assume this is exclusively a pose-estimation problem. Investigate these possible causes separately:

1. Incorrect or ambiguous 6DoF pose.
2. Incorrect corner ordering or face orientation.
3. GLB coordinate-system, origin, scale, or transform errors.
4. Camera intrinsic calibration inaccuracies.
5. Incorrect camera-to-renderer coordinate conversion.
6. Mirroring in the video texture or CSS canvas.
7. Internal edges that are absent from or inconsistent with the CAD mesh.
8. Occlusion by hands or other objects.
9. Optical-flow drift or an incorrect background pose correction.
10. Temporal filtering that smooths an incorrect pose instead of rejecting it.

---

## 3. Non-negotiable engineering requirements

1. Audit the existing code before editing.
2. Reuse the existing MegaPose service, edge alignment, optical flow, tracking state machine, hand masking, and GLB registry wherever appropriate.
3. Do not replace MegaPose with another pose estimator unless a measured benchmark demonstrates that this is necessary.
4. Do not run a costly pose search on every frame.
5. Do not use YOLO bounding boxes as the final source of precise 6DoF orientation.
6. Do not assume that the external rectangle uniquely determines orientation.
7. Do not blindly average two competing poses if they represent different orientations.
8. Do not accept a pose solely because it has a low silhouette error.
9. Do not mark a component missing merely because its feature is occluded or the pose confidence is low.
10. Preserve existing APIs, data formats, UI modes, and inspection workflows wherever possible.
11. Keep all existing working features operational.
12. Avoid hard-coded feature coordinates or object-specific rules in general-purpose tracking code. Use per-model configuration.
13. Every change must have tests and measurable before/after results.
14. Never silently substitute a guessed pose when the correct pose cannot be established.

---

## 4. Phase A — Repository audit and baseline measurements

Before implementing improvements:

1. Map the frontend, backend, GPU pose service, tracking workers, component registry, and configuration files.
2. Identify the actual code responsible for:
   - Camera intrinsic calculation.
   - Image orientation and mirroring.
   - MegaPose initialization and refinement.
   - Outline-based pose search.
   - Edge extraction and correspondence.
   - Pose scoring and candidate selection.
   - Optical-flow tracking.
   - Pose filtering.
   - GLB coordinate conversion.
   - Three.js camera and model transforms.
   - Internal component projection and visibility.
3. Trace one camera frame from browser capture to backend inference, pose estimation, tracking, and rendering.
4. Record how a pose is represented: rotation matrix, quaternion, translation vector, coordinate frame, units, and timestamp.
5. Establish a reproducible test set using the existing box and GLB.

Create a baseline report with:

- Pose initialization success rate.
- Number of incorrect orientation flips.
- Reprojection error, where applicable.
- Visible internal-feature alignment error in pixels.
- Overlay drift during object movement.
- Tracking recovery time after occlusion.
- Average and 95th-percentile inference latency.
- Frame rate and GPU usage.
- False missing-component decisions during hand occlusion.

Do not claim improvements without comparable measurements.

---

## 5. Phase B — Camera calibration and coordinate-system correctness

### 5.1 Replace approximate calibration when calibration data is available

The current camera model approximates intrinsics using an assumed 82-degree horizontal field of view. This is acceptable for initial development but can introduce pose and alignment errors.

Implement or verify a repeatable checkerboard-based calibration workflow using OpenCV.

Store the actual camera calibration parameters:

- Intrinsic matrix `K`.
- Distortion coefficients `D`.
- Calibration image dimensions.
- Camera identifier.
- Calibration date and version.

Use the calibrated values for pose estimation and projection. Scale intrinsics consistently if the capture resolution changes.

Do not silently use calibration values for a different camera or resolution.

If calibration is unavailable, retain a clearly labelled approximate-calibration fallback and expose its confidence or status.

### 5.2 Verify transforms end to end

Document and validate:

- Camera coordinate frame.
- OpenCV camera axes.
- MegaPose output convention.
- GLB local coordinates and units.
- World or station coordinates.
- Three.js camera and model transforms.
- Any image mirroring or axis inversion.

Add explicit conversion functions with unit tests.

Verify whether the webcam image is mirrored for display while the processing image is not. Do not let display mirroring contaminate pose estimation.

Add a diagnostic mode showing:

- Camera axes.
- Model axes.
- Model origin.
- Projected corners.
- Coordinate-frame names and units.

Fix coordinate-conversion errors before tuning the pose estimator.

---

## 6. Phase C — Resolve ambiguous orientations

### 6.1 Maintain multiple candidate poses when ambiguity exists

For simple, nearly symmetric objects, the external outline may match several orientations.

During initialization, preserve plausible candidate poses instead of immediately accepting the first silhouette match.

Use the existing MegaPose and outline-based initialization paths. Add alternative candidates only where the existing methods are ambiguous.

Score candidates using a combination of:

1. External silhouette agreement.
2. Visible 3D edge alignment.
3. Internal feature correspondence.
4. Camera projection consistency.
5. Configured resting-side constraints, when applicable.
6. Temporal consistency with the last reliable pose.
7. Valid camera distance and physically plausible geometry.

Treat resting-side constraints as station-specific evidence, not a universal assumption.

### 6.2 Add internal-feature-based pose disambiguation

Use the GLB component registry and per-model configuration to define distinctive reference features.

Support, where applicable:

- Circular holes and marks.
- Screws and fasteners.
- Connectors.
- Notches and cut-outs.
- Internal creases and edges.
- Labels or high-contrast markings.
- Distinctive corners.
- Known component centers and geometric landmarks.

A feature definition should be tied to the relevant GLB object or model-local coordinate, not to a hard-coded image location.

Example configuration concept:

```json
{
  "model_id": "rectangular_box",
  "pose_features": [
    {
      "feature_id": "upper_right_circle",
      "type": "circle",
      "glb_node": "TopRightCircle",
      "enabled": true,
      "required_for_initialization": false
    }
  ]
}
```

This is a conceptual schema. Adapt it to the existing configuration format rather than introducing an incompatible new registry.

Support features that are part of a combined mesh as well as separate GLB nodes. Do not require every feature to be a separate object.

### 6.3 Match feature identity, not just feature appearance

The system must distinguish between the physical upper-right circle and a visually similar feature elsewhere.

Use known model-local positions, projected geometry, feature descriptors, detector outputs, and multi-feature geometric consistency.

A detected circle alone is insufficient to establish pose. It must be associated with a particular CAD feature, and the association must be geometrically plausible.

When possible, use multiple independent correspondences to disambiguate orientation.

If the object is visually symmetric and the distinctive feature is occluded or undetectable, keep the pose ambiguous rather than inventing certainty.

---

## 7. Phase D — Improve 3D edge alignment

The existing OpenCV Canny + SciPy least-squares refinement should be improved, not discarded.

### 7.1 Separate edge types

Where the CAD geometry supports them, distinguish:

- External silhouette edges.
- Internal crease edges.
- Component boundaries.
- Circular-hole contours.
- Other known geometric features.

Avoid treating every edge in the camera image as a valid model edge. Background texture, hand boundaries, shadows, and reflections can produce misleading correspondences.

### 7.2 Use visibility-aware edge matching

For each candidate pose:

1. Project the relevant model edges into the image.
2. Determine which edges should be visible.
3. Exclude hidden CAD edges using depth or visibility information.
4. Compare projected edges with image edges.
5. Calculate robust alignment residuals.
6. Downweight unreliable or occluded regions.
7. Reject candidates that fit the silhouette but misplace distinctive internal geometry.

Investigate a distance-transform or distance-field-based edge loss, which can be more stable than relying exclusively on nearest-edge least squares.

Use robust loss functions to reduce the influence of outliers.

### 7.3 Feature-specific refinement

When a reliable circle, hole, corner, or connector correspondence is available, include its geometric residual in the pose-refinement objective.

For example:

- Circle center versus projected CAD circle center.
- Detected hole contour versus projected hole contour.
- Corner locations versus projected model corners.
- Component boundaries versus rendered CAD boundaries.

Use appropriate weighting based on feature visibility and detection confidence.

Do not allow an unreliable detector to override strong, consistent geometric evidence.

---

## 8. Phase E — Improve camera-to-model projection

Implement a consistent projection pipeline.

For each tracked frame:

1. Obtain the current camera intrinsics and distortion parameters.
2. Obtain the accepted model pose.
3. Project the relevant 3D landmarks and geometry into the original camera image.
4. Compare their locations against the detected image features.
5. Report residuals and confidence.
6. Render the GLB using a mathematically consistent camera and model transform.

Ensure that the image used for feature extraction and the image used for pose validation share the same orientation, dimensions, and distortion treatment.

Avoid mixing undistorted coordinates with distorted images without explicitly converting between them.

Add a diagnostic overlay showing predicted versus observed feature locations, with different symbols for each.

---

## 9. Phase F — Improve tracking and prevent pose flips

Retain the existing optical-flow tracker, periodic MegaPose corrections, pose filter, stationary hold, and tracking state machine.

Improve their interactions.

### 9.1 Pose plausibility checks

For each candidate update, evaluate:

- Translation change relative to elapsed time.
- Rotation change relative to elapsed time.
- Internal-feature reprojection error.
- Edge alignment error.
- Tracking quality and feature visibility.
- Agreement with the previous accepted pose.

Use adaptive thresholds based on frame interval, camera movement, object motion, and uncertainty. Do not use one rigid angular threshold for every scenario.

A large rotation is not automatically wrong: the operator may legitimately rotate the object. It should trigger stronger verification rather than unconditional rejection.

### 9.2 Correction handling

When MegaPose produces a new candidate:

1. Compare its score with the currently tracked pose.
2. Verify its internal geometry and coordinate conventions.
3. Check whether the update represents plausible motion or an unexplained orientation flip.
4. Accept, reject, or enter recovery based on evidence.
5. If ambiguous, preserve the last reliable pose and indicate reduced confidence.

Never blend two poses representing different discrete orientation hypotheses. Select a valid hypothesis first, then smooth the continuous motion within that hypothesis.

### 9.3 Optical-flow tracking

Track reliable features associated with the object, not arbitrary background points.

Reject tracks inconsistent with the object's motion and geometry.

Use the existing hand mask to prevent hand pixels from becoming object tracking features.

Reinitialize when tracking quality drops below the configured threshold.

### 9.4 Occlusion and recovery

During hand occlusion:

- Hold the last reliable pose for the configured duration.
- Mark pose confidence as reduced.
- Avoid generating a new orientation from hand edges.
- Reacquire using internal features and CAD geometry after the hand moves away.
- Resume component inspection only when visibility and pose confidence are sufficient.

---

## 10. Phase G — Improve hand masking and internal-feature reliability

The existing MediaPipe hand landmarks and YOLO person masks should continue to be used.

Investigate whether the current hand mask is sufficiently expanded to remove fingers, motion blur, and boundary pixels from feature detection.

Do not simply mask a large region of the object, because that would hide genuine features.

Where useful, distinguish:

- Visible and reliably detected.
- Occluded.
- Outside the camera view.
- Expected to be visible but not detected.
- Uncertain due to pose or image quality.

This distinction is critical for quality assurance. A covered screw must not be reported as missing.

---

## 11. Phase H — Component registry and inspection integration

Reuse the existing GLB-derived component registry and mapping files.

Extend them only as necessary to support:

- Pose-disambiguation landmarks.
- Feature types and expected model-local locations.
- Component visibility and occlusion status.
- Feature-detection confidence.
- Expected geometric relationships between components.

Keep pose estimation and component-presence verification separate.

A pose failure must not be interpreted as a missing-component defect.

Before making an absence decision, verify:

1. The component should be visible from the current camera pose.
2. The expected region is not occluded.
3. The pose is reliable enough for projection.
4. Image quality is sufficient.
5. The component verifier has adequate evidence.

Preserve the existing temporal decision rules unless test evidence demonstrates a need to adjust them.

---

## 12. Phase I — Performance requirements

Preserve the current real-time tracking architecture.

Use a staged computation strategy:

- Every frame: lightweight tracking, visibility checks, and pose-quality monitoring.
- On significant motion or declining tracking confidence: additional geometric refinement.
- Periodically or when ambiguity appears: background MegaPose correction.
- On tracking loss: full initialization and candidate search.

Avoid running expensive multi-hypothesis searches continuously.

Measure actual latency and throughput on the existing RTX 4090 + WSL deployment.

Keep GPU-heavy pose inference isolated in the existing pose service unless the repository reveals a compelling reason to change this design.

Do not block the frontend waiting for every background correction.

---

## 13. Phase J — Testing and evaluation

Build a repeatable evaluation dataset from the existing rectangular box and GLB.

Include:

1. Circle in the upper-right position.
2. Object rotated 90°, 180°, and 270°.
3. Object viewed from different camera angles.
4. Different distances and scales.
5. Object translated across the frame.
6. Object tilted and rotated in 3D.
7. Hand covering the circle.
8. Partial occlusion of external corners.
9. Low-contrast or poorly lit conditions.
10. A plain box with no distinctive internal feature.
11. Repeated stationary frames.
12. Sudden but legitimate object rotations.

For each test, record:

- Correct pose or incorrect pose.
- Orientation-flip count.
- Internal-feature alignment error in pixels.
- Reprojection error.
- Pose confidence.
- Recovery time.
- Average and 95th-percentile processing latency.
- False component-missing decisions.

Use labelled reference images or measured ground-truth poses where practical.

Define acceptance criteria based on the baseline. Do not claim perfect alignment or guaranteed accuracy.

An orientation that cannot be uniquely inferred from the available image must be marked ambiguous.

---

## 14. Implementation workflow

Follow this sequence.

### Step 1: Inspect

Read the relevant source files and document the actual data flow and pose conventions.

### Step 2: Reproduce

Create a deterministic test that reproduces the right-to-left circle flip.

### Step 3: Diagnose

Determine whether the failure originates in pose initialization, coordinate conversion, feature association, edge refinement, tracking, or rendering.

### Step 4: Fix fundamental issues

Address calibration, coordinate frames, image mirroring, corner ordering, and GLB transforms before tuning complex algorithms.

### Step 5: Improve geometric pose validation

Use the existing CAD edges and internal-feature correspondences to rank candidate poses.

### Step 6: Improve tracking

Reject implausible corrections, handle occlusion explicitly, and prevent discrete pose flips from being blended into the displayed pose.

### Step 7: Integrate

Update the existing component registry and frontend diagnostics without breaking existing workflows.

### Step 8: Test

Run automated tests, existing tests, and the repeatable box evaluation dataset.

### Step 9: Report

Provide a concise implementation report with changed files, root cause, algorithmic changes, before/after metrics, remaining limitations, and exact run commands.

---

## 15. Required deliverables

Produce the following in the existing repository:

1. A repository audit and root-cause report.
2. A corrected and documented camera/model coordinate transformation pipeline.
3. Camera calibration support or a documented calibration validation result.
4. Internal-feature-aware pose validation.
5. Improved CAD edge alignment with visibility and outlier handling.
6. Tracking safeguards against false pose flips.
7. Diagnostic visualization for observed and projected internal features.
8. Tests for coordinate transforms, candidate scoring, orientation ambiguity, tracking corrections, and occlusion.
9. Before/after evaluation metrics.
10. A list of any remaining problems and proposed next steps.

### Final instructions

Do not stop after writing a plan. After inspecting the repository, implement the highest-priority safe changes, run the relevant tests, and report what actually works.

If a required dependency, dataset, or source file is unavailable, state that explicitly and continue with the portions that can be implemented and tested.

Never claim that a feature works unless it has been implemented and verified.

The end goal is a stable industrial AR digital twin in which the physical object and the GLB remain geometrically aligned, internal features appear on the correct side, object motion is tracked accurately, and uncertain observations are handled safely rather than producing misleading inspection results.

My Disscusion >

The issue you are facing is a 3D pose estimation and orientation ambiguity problem, not just an object detection problem.

Your rectangular box is detected, and the 3D model overlays correctly in simple cases. But when you add a feature such as a circle on the top-right side, the overlay sometimes places it on the top-left. This means the system can identify the box's outer boundaries but cannot always determine its exact orientation or match its internal features to the correct locations.

For your AR quality-inspection project, the solution is to make tracking feature-aware and geometry-aware, rather than relying only on the outer shape.

## 1. Why your current overlay flips

[Closed cardboard small box isolated](https://images.openai.com/static-rsc-4/e9W1lSflWHa9Fe4XtKRLeC9YnBfdhe4lSHMn4jpu1mqWQAWrORGY_DAXmDnoo1yB0hVwLx4Vaq7V56cOvHzVOCqp78P4BiGjq6wmS5WI9InuKuBejlf3q53IfC3alV8NzGPb7XJeg6KPRbhRJ205crJ6qZsMZPxw2euOTEbWR5J-vHN1GtkZQ-GMwhsxL2f-?purpose=inline)

Case A — Correct overlay

The system estimates the box's position and orientation correctly, so the circle appears on the correct side.

[Cardboard box isolated](https://images.openai.com/static-rsc-4/HNytYwmYHJ_2QrqyhLDvEmWnydVQd7gP3-ZKHe6XaY-dnHBbynJ4DYTDhUv_MCw5KB7aFm5xa-6XaPlzge0IH-yS6mzN-xYYgf-oSciNDeAPqSxj9lTd-3leHFnrM9BTfcIOs_5JpRU5Sb8LT_R14zx5wHlXWi9K09w2diaZq2s_frZIDSkB0-d-dlDlv64m?purpose=inline)

Case B — Incorrect orientation

The system detects the box but chooses a different orientation, or the pose estimate is unstable. Your 3D model is consequently mirrored or rotated relative to the physical box.

A plain rectangle is especially difficult because its edges alone do not always uniquely identify which side is which. With a nearly symmetric shape, multiple poses can produce similar camera images.

Important distinction: a 2D circle detected in the image is useful evidence, but merely detecting it is not enough. The system must know that this circle corresponds to a specific feature at a known position on the 3D model.

## 2. The solution: combine four techniques

[numpy - Find 7 vertices of a box using openCV - Stack Overflow](https://images.openai.com/static-rsc-4/R_Ed1-F3WjDssLPj_w-pbPPXDAYkozgSs_5priMQz82NovBXZvLxp_xsPPD_zSfjslsVzzmSYyrDKX7Wqh4w6ZAZDRfItAY2iDWFalmpNCiERlSGoKbfX3CEadkwjWMavf143ZAmnFUDplXI7mordgnsL87i9W1f7_JqKtYjcYY?purpose=inline)

1\. Outer-boundary detection

Keep existing

Continue using your current detector to locate the physical object. YOLO or your existing tracking system can provide the initial region of interest.

[Feature Matching + Homography to find a known object](https://images.openai.com/static-rsc-4/A8pbGG_zd5IstfRQjxgCjNnU40CIeaoy7kgFRCsVqV6P7Cgep6GMkglujN6sckJFnIJXXaoFfTrOvpk2TJRJTh18vR8E-1wYXUv8NcKI1xek3KxrfmnML8uxQIoDphTthJZC0JlwVtS1qvEGGERIesseSJ5sXQBzlUUJvQdPFLc?purpose=inline)

2\. Internal feature detection

Detect circles, holes, screws, connectors, labels, corners and other distinctive features. Match them to known locations on the 3D model.

[CV | pzrobotics](https://images.openai.com/static-rsc-4/r4zVRUhNvxItDY0q4lLpY4I3gUpw4HMP9CShCSIqR9YUExFUcfOmdXv4A5bne5yOzL5suD_toN5QN5YCkZSndG40BUiq7yqYoT6Gd_6F42aPzKFT-lnEXbrDTwIY7z2d3QGuc5hC5VEGxreFhSlQE22sGWSvt2a5cf2Dw6R1Uw8?purpose=inline)

3\. Six-degree-of-freedom (6DoF) pose estimation

Estimate the object's translation and rotation: X, Y, Z, roll, pitch and yaw. Use the internal features and known 3D geometry to resolve orientation ambiguity.

[Generate State-Based Model Targets - Vuforia Engine Library](https://images.openai.com/static-rsc-4/4XNJR0F7pe_BYa1GLXksC3w5rOYCZfEfYw1EhQQzAjAiodlp6uFo6rWp2GkNWfe9VMVt0-fVLRdmYvflYLYmfNBJl_DZ7FYldqGFm1Pn4r7jCHX1uj5aCUMq4aI-r1yoi3vll3-v01IwuOp8vqDHQLy0w-VK4rQexb54qdFp9ho?purpose=inline)

4\. Temporal tracking and pose validation

Keep the overlay attached as the object moves. Reject implausible sudden flips and verify that the projected internal features match the camera image before accepting a new pose.

## 3. How to solve your circle-on-the-wrong-side problem

Suppose your 3D model has the following features:

| Feature             | Known 3D location | Expected image evidence          |
| ------------------- | ----------------- | -------------------------------- |
| Top-left corner     | \\(P_1\\)         | Image point \\(p_1\\)            |
| Top-right corner    | \\(P_2\\)         | Image point \\(p_2\\)            |
| Circular hole       | \\(P_3\\)         | Detected circle center \\(p_3\\) |
| Bottom-right corner | \\(P_4\\)         | Image point \\(p_4\\)            |
| Bottom-left corner  | \\(P_5\\)         | Image point \\(p_5\\)            |

The 3D locations come from your GLB/glTF model or a measured reference. The image points come from the camera.

The system estimates the pose using these correspondences, then projects the model back into the image. OpenCV's `solvePnP` methods support this 3D-to-2D pose calculation.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1

Pose validation logic

Detect box corners + internal features

Estimate candidate 3D poses

Compare projected circles, holes and corners against the camera image

Good match

Accept pose and update overlay.

Poor match

Reject pose, keep tracking or reacquire.

The key is that the system must evaluate the correct pose against the actual image, rather than accepting whichever orientation the outer rectangle happens to produce.

For your circle example, if the candidate pose predicts the circle on the left but the camera clearly shows it on the right, that candidate should be rejected.

One caution: if the box is almost perfectly symmetric, the circle is occluded, or the camera sees too little of the object, the pose may remain ambiguous. In that situation, report low confidence rather than pretending the overlay is correct.

## 4. Which algorithms should you use?

| Technique                                | Purpose                                        | Priority      |
| ---------------------------------------- | ---------------------------------------------- | ------------- |
| OpenCV `solvePnP` / `solvePnPRansac` | Estimate pose from matched 3D and 2D points    | First         |
| OpenCV feature detection and matching    | Find distinctive internal visual features      | First         |
| YOLO detection or segmentation           | Locate the object, holes, screws or components | Keep existing |
| Optical flow / pose filtering            | Reduce frame-to-frame jitter and sudden flips  | First         |
| FoundationPose                           | Model-based 6DoF pose estimation and tracking  | Evaluate next |

FoundationPose is particularly relevant because it supports object pose estimation using a known CAD model and includes pose tracking. Its official repository is available at [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose).&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

I would not immediately replace your working detector with FoundationPose. First establish whether your current problem is caused by incorrect corner ordering, an ambiguous pose, or inaccurate camera calibration. Otherwise, a more advanced model may not fix the underlying issue.

## 5. A practical development plan for your existing TVASTA project

1. Fix orientation and coordinate conventions. Verify the GLB's origin, axes, scale, rotation, face winding and camera coordinate system. Check whether the incorrect result is a genuine pose flip or a left-right mirroring error in rendering.
2. Add internal feature correspondence. Start with the circle's center and several box corners. If the circle is a hole, use a circle detector or a trained detector; for more complex parts, use keypoints or segmentation.
3. Estimate pose from multiple features. Use camera calibration and known 3D coordinates. Add RANSAC and reprojection-error checks so unreliable matches are rejected.
4. Prevent sudden pose flips. Compare candidate poses with the previous accepted pose, use temporal filtering, and reacquire when confidence falls too low. Do not blindly lock the overlay to an incorrect pose.
5. Test with difficult cases. Rotate the box, move it closer and farther away, partially cover the circle with your hand, change lighting, and test from different camera angles.

For future industrial assemblies, the same approach extends to matching screws, connectors, board edges, missing caps and other components against the digital twin.

## 6. One important question about your current implementation

To recommend the right fix without rebuilding your working system, I need to know how you currently calculate the overlay's pose.

A. What currently determines the object's position and rotation?

ArUco marker / printed marker

Detected outer corners or bounding box

YOLO detection or segmentation

FoundationPose or another 6DoF pose model

Other / not sure

B. What exactly happens when it fails?

The entire 3D model rotates or flips 180°

The model stays in place but internal features are mirrored left-to-right

The overlay gradually drifts away from the physical object

The overlay suddenly jumps between different positions

More than one of these / not sure

C. Does your GLB have separate meshes for the internal components?

Yes, the circles/holes/components are separate meshes or objects

No, it is one combined mesh

Not sure

Recommend the specific fix
