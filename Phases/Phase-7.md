
# Improve TVASTA AR Tracking During Hand Occlusion

Inspect the existing TVASTA repository before modifying anything. The current system already detects a physical cup and renders a 3D GLB/glTF overlay with a green tracking outline and tracking confidence. Preserve this working functionality.

## Main problem

When a user picks up or holds the cup, the hand enters the camera view and interferes with object detection and pose estimation. The overlay becomes unstable or stops aligning correctly.

## Requirements

### 1. Inspect the existing implementation

- Identify the current detector, pose tracker, camera pipeline, overlay renderer and confidence calculation.
- Determine whether the tracker uses an ArUco marker, image features, contours, a learned detector, or another pose-estimation method.
- Identify where hand pixels are currently influencing detection or pose estimation.
- Do not replace the existing tracker without first demonstrating why it fails.

### 2. Separate the cup and hand

- Add hand detection or segmentation as a separate signal.
- Where necessary, add cup segmentation to estimate which visible pixels belong to the cup.
- Use the hand mask to exclude hand pixels from cup-specific feature extraction and pose refinement.
- Do not treat the hand silhouette as part of the cup contour.
- Handle hands that cover the cup's handle, rim or other important features.

### 3. Stabilize pose tracking

- Reuse the previous valid cup pose when the hand briefly occludes the object.
- Apply appropriate temporal filtering to position and rotation.
- Use short-term pose prediction only while tracking confidence remains acceptable.
- Reacquire the cup from visual evidence when it becomes visible again.
- Never update the cup pose using hand features.
- If tracking is lost, freeze or fade the overlay and display a tracking-lost status instead of allowing it to jump to the hand or background.

### 4. Improve the overlay

- Investigate the visible offset between the cyan 3D overlay and the real cup.
- Validate camera intrinsics, coordinate systems, model scale, projection and pose transforms.
- Smooth contour visualization without changing the underlying geometry.
- Ensure the overlay remains attached to the cup as it is lifted, rotated and moved.
- Keep tracking and rendering synchronized with the appropriate camera frame.

### 5. Improve confidence and status

Use explicit states such as:

- TRACKING
- TRACKING_OCCLUDED
- REACQUIRING
- TRACKING_LOST

Base confidence on available visual evidence, pose consistency and temporal stability. Do not interpret a detector's raw confidence as complete pose confidence.

### 6. Protect future QA inspection

When component inspection is enabled:

- A hand covering a component must produce OCCLUDED or UNCERTAIN, not MISSING.
- Verify component visibility before declaring a defect.
- Confirm suspected defects across multiple valid observations.
- Keep component inspection separate from object pose tracking.

### 7. Performance and implementation constraints

- Preserve the existing project architecture and working tracking behavior.
- Prefer lightweight models and reuse existing dependencies where practical.
- Do not add Unity or require a depth camera for this improvement.
- Avoid running expensive detection or segmentation unnecessarily on every frame if a lower-rate update is sufficient.
- Add debug views showing the original frame, cup mask, hand mask, pose status and overlay.
- Make new thresholds configurable.

## Acceptance tests

1. Cup stationary with no hand: stable tracking and aligned overlay.
2. Hand approaches the cup without touching it: no false pose change.
3. User picks up the cup: overlay follows without jumping to the hand.
4. Hand covers a large portion of the cup: system reports occlusion and maintains a plausible pose briefly.
5. Cup rotates while held: tracker recovers and corrects its pose.
6. Cup leaves the field of view: system reports tracking lost.
7. Cup re-enters the field of view: tracker reacquires it.
8. A component is covered by the hand: QA does not falsely flag it as missing.

First provide a brief analysis of the existing implementation and the proposed minimal changes. Then implement the changes incrementally, run available tests, and report the files changed, dependencies added and test results. Do not claim a test passed unless it was actually run.
