Use this prompt with your coding agent. It is specifically written to **adapt your current project using FoundationPose++ as the reference**, without blindly replacing your existing detection/rendering system.

```text
I want you to modify my CURRENT AR object tracking project.

IMPORTANT:
Do NOT rebuild the application from scratch.
Do NOT remove the existing working object detection, segmentation, 3D model loading, or rendering unless necessary.

Use this GitHub repository as the PRIMARY REFERENCE for the pose-tracking architecture:

https://github.com/teal024/FoundationPose-plus-plus

Study its approach to:
- initial 6D pose estimation
- continuous tracking
- 2D tracking
- depth-based Z estimation
- Kalman filtering
- pose refinement
- tracking/recovery

The goal is to adapt the useful architecture and ideas to my existing application.

============================================================
CURRENT APPLICATION
============================================================

The application currently does the following:

Camera
  ↓
Object detection / segmentation
  ↓
Detect physical object
  ↓
Estimate pose
  ↓
Render 3D model over physical object

The 3D overlay is now working.

The current problem is that detection is still being performed repeatedly.

For example:

Frame 1:
    detect cup
    calculate pose
    render 3D cup

Frame 2:
    detect cup AGAIN
    calculate pose AGAIN
    render

Frame 3:
    detect cup AGAIN
    calculate pose AGAIN
    render

This is NOT the desired architecture.

============================================================
DESIRED BEHAVIOR
============================================================

I want:

FIRST DETECTION:

Camera
  ↓
AI detection
  ↓
Object found
  ↓
Initial 6DoF pose estimation
  ↓
Initialize tracker
  ↓
Attach 3D model
  ↓
STOP RUNNING DETECTOR

Then:

EVERY FOLLOWING FRAME:

Camera
  ↓
Tracker
  ↓
Update object pose
  ↓
Update 3D model transform
  ↓
Render

The expensive detector should NOT run again while tracking is successful.

Only run detection again when tracking is lost or confidence becomes too low.

============================================================
REFERENCE ARCHITECTURE
============================================================

Use FoundationPose++ as the conceptual reference.

The desired architecture is:

                     CAMERA
                       │
                       ▼
                OBJECT DETECTOR
                       │
                       │ FIRST DETECTION ONLY
                       ▼
                INITIAL POSE
                 ESTIMATION
                       │
                       ▼
                 6DoF POSE
                 X Y Z Rx Ry Rz
                       │
                       ▼
                INITIALIZE TRACKER
                       │
                       ▼
                 3D DIGITAL TWIN
                       │
                       ▼
              ┌──────────────────┐
              │ CONTINUOUS LOOP  │
              │                  │
              │ 2D TRACKING      │
              │ + DEPTH          │
              │ + POSE FILTER    │
              │ + POSE REFINEMENT│
              └────────┬─────────┘
                       │
                       ▼
                  UPDATED POSE
                       │
                       ▼
                  3D MODEL
                  TRANSFORM
                       │
                       ▼
                    RENDER

If tracking fails:

                  TRACKING
                     │
                     ▼
                LOST / LOW CONF
                     │
                     ▼
                  DETECTOR
                     │
                     ▼
               RE-INITIALIZE
                     │
                     ▼
                  TRACKING
```

============================================================
STATE MACHINE
=============

Implement an explicit state machine.

States:

```
SEARCHING
INITIALIZING
TRACKING
LOST
RECOVERING
```

Initial state:

```
SEARCHING
```

---

## SEARCHING

Run the existing object detector.

Example:

```
detector.detect(frame)
```

If object is found:

```
calculate initial pose
initialize tracker
initialize tracking state
state = TRACKING
```

Once successfully initialized:

```
STOP detector
```

---

## TRACKING

While tracking:

```
DO NOT call the detector.
```

Instead use the tracking system.

Conceptually:

```
frame
  ↓
2D tracker
  ↓
tracked object location
  ↓
depth estimation
  ↓
pose update
  ↓
filtering
  ↓
optional pose refinement
  ↓
3D model transform
  ↓
render
```

This must run continuously.

The tracker should follow the SAME physical object that was originally
detected.

Do NOT create a new detection on every frame.

---

## LOST

If tracking confidence drops below a configurable threshold:

```
state = LOST
```

Do not immediately run expensive detection on every frame.

Use a short grace period.

For example:

```
frame 1 lost
    keep last valid pose

frame 2 lost
    keep last valid pose

frame 3 lost
    attempt recovery
```

If recovery fails:

```
state = RECOVERING
```

---

## RECOVERING

Run the detector at a controlled interval.

For example:

```
detector every N frames
OR
detector every 100-300 ms
```

When the object is found:

```
calculate/recover pose
reinitialize tracker
state = TRACKING
```

Once tracking resumes:

```
STOP detector again.
```

============================================================
FOUNDATIONPOSE++ INSPIRED TRACKING
==================================

Use the FoundationPose++ architecture as a reference.

Where applicable:

```
2D tracker
    ↓
X/Y movement

Depth
    ↓
Z movement

Kalman filter
    ↓
smooth rotation / pose

FoundationPose-style pose refinement
    ↓
improve 6DoF registration
```

The exact implementation should depend on what is already available
in my project.

Do NOT blindly copy code from FoundationPose++.

First inspect the repository and understand the existing project.

============================================================
IMPORTANT: CAMERA INPUT
=======================

Determine whether the current camera provides:

```
RGB only
```

or:

```
RGB + depth
```

If RGB + depth is available:

```
use depth for Z estimation.
```

If RGB only:

```
do NOT pretend that true depth is available.
```

Instead determine whether depth can be estimated from:

```
known object dimensions
camera intrinsics
pose estimation
3D model geometry
```

The application currently knows the real object height.

Example:

```
real_height = 0.2 meters
```

Use physical dimensions wherever possible.

============================================================
CAMERA CALIBRATION
==================

Inspect whether camera calibration already exists.

We need:

```
fx
fy
cx
cy
distortion coefficients
```

The virtual rendering camera and pose estimation must use the same
camera model.

Do NOT use arbitrary screen coordinates.

Do NOT use arbitrary FOV values.

============================================================
3D MODEL
========

The 3D model is loaded successfully already.

Do NOT reload the 3D model every frame.

The model should be loaded once.

After initialization, only update:

```
position
rotation
scale
visibility
```

Example:

```
model = load_model()
```

Then:

```
model.transform = tracked_pose
```

NOT:

```
every frame:
    load_model()
    calculate_model()
    render_model()
```

============================================================
OBJECT REPLACEMENT
==================

The 3D model is intended to behave like the digital twin of the
physical object.

This is NOT just a floating AR object.

The model should occupy the physical object's exact location.

For the cup example:

```
physical cup
      ↓
detect once
      ↓
register 3D cup
      ↓
digital cup occupies same pose
      ↓
physical cup moves
      ↓
digital cup follows
```

The model must match:

```
X
Y
Z
Roll
Pitch
Yaw
Scale
Perspective
```

============================================================
MOVEMENT TEST
=============

After initial detection, test:

TEST 1:
Cup stationary.

Expected:
3D model remains locked to cup.

TEST 2:
Move cup left.

Expected:
3D model follows left.

TEST 3:
Move cup right.

Expected:
3D model follows right.

TEST 4:
Move cup up/down.

Expected:
3D model follows.

TEST 5:
Move cup toward camera.

Expected:
3D model gets larger and remains registered.

TEST 6:
Move cup away from camera.

Expected:
3D model gets smaller.

TEST 7:
Rotate cup.

Expected:
3D model rotates with cup.

TEST 8:
Move camera while cup stays stationary.

Expected:
3D model remains registered to the cup.

TEST 9:
Move both camera and cup.

Expected:
3D model remains registered.

============================================================
TRACKING SMOOTHING
==================

Tracking should not produce jitter.

Implement configurable smoothing.

Start with:

```
exponential smoothing
```

If appropriate, implement:

```
Kalman filter
```

Use separate handling for:

```
translation
rotation
```

Do not introduce excessive latency.

The objective is:

```
stable + responsive
```

rather than:

```
heavily smoothed + delayed
```

============================================================
POSE REPRESENTATION
===================

Use a consistent pose representation.

For example:

```
R = rotation matrix / quaternion
t = translation vector
```

or:

```
x
y
z
rx
ry
rz
```

Internally use the representation best suited to the existing
renderer.

Avoid repeatedly converting between incompatible coordinate systems.

Document:

```
camera coordinate system
object coordinate system
renderer coordinate system
```

Especially document:

```
+X
+Y
+Z
```

and their relationship.

============================================================
TRACKING CONFIDENCE
===================

Expose:

```
tracking_confidence
```

Use configurable thresholds.

Example:

```
confidence > HIGH_THRESHOLD
    TRACKING

confidence between thresholds
    TRACKING_WITH_WARNING

confidence < LOW_THRESHOLD
    LOST
```

Do not hard-code these thresholds into multiple files.

Put them in configuration.

============================================================
PERSISTENT OBJECT ID
====================

After detection:

```
object_id = 1
```

Maintain that identity while tracking.

Store:

```
object_id
object_class
last_pose
last_position
last_rotation
last_bbox
tracking_confidence
last_seen_frame
velocity if available
```

Do not recreate the object identity every frame.

============================================================
RENDERING
=========

Rendering must be independent from detection.

This is critical.

The renderer should receive:

```
current_pose
```

and render the already-loaded 3D model.

Architecture:

```
Detector
   ↓
Tracker
   ↓
Pose
   ↓
Renderer
```

NOT:

```
Detector
   ↓
Renderer
   ↓
Detector
   ↓
Renderer
```

Rendering should happen every frame.

Detection should NOT.

============================================================
PERFORMANCE
===========

Measure separately:

```
Detection time
Tracking time
Pose refinement time
Rendering time
Total frame time
```

Display:

```
Detection FPS
Tracking FPS
Rendering FPS
```

The expected architecture is:

```
DETECTION:
    expensive
    infrequent

TRACKING:
    lightweight
    continuous

RENDERING:
    continuous
```

============================================================
UI
==

The current UI contains:

```
"3D model loaded (hidden until a pose is found)"
"Detecting..."
"Live tracking (as fast as detection responds, smoothed)"
"Solid"
"Wireframe"
"X-Ray"
```

Change the UI so that it reflects the actual pipeline.

SEARCHING:

```
🔵 Searching for object...
```

INITIALIZING:

```
🟡 Initializing pose...
```

TRACKING:

```
🟢 Tracking: cup
Confidence: 94%
```

LOST:

```
🟠 Tracking lost
```

RECOVERING:

```
🔵 Reacquiring object...
```

Also show useful debug values:

```
Object ID
Tracking confidence
Tracking FPS
Detection count
Frames since detection
X
Y
Z
Rx
Ry
Rz
```

============================================================
IMPORTANT PERFORMANCE REQUIREMENT
=================================

Add a counter:

```
detection_count
```

When the cup is initially detected:

```
detection_count = 1
```

Then move the cup around.

The counter MUST remain:

```
detection_count = 1
```

as long as tracking remains successful.

If tracking is lost and the object is reacquired:

```
detection_count = 2
```

This gives us a simple way to verify that detection is no longer
running continuously.

Also add:

```
tracking_frame_count
```

which should continuously increase while tracking.

============================================================
DEBUG LOGGING
=============

Use logs such as:

```
[SEARCHING]
Running detector...

[DETECTION]
Cup detected
Detection count = 1

[POSE]
Initial pose calculated

[TRACKER]
Tracker initialized
Object ID = 1

[TRACKER]
Tracking
confidence = 0.94

[POSE]
Updated pose:
X=...
Y=...
Z=...
Rx=...
Ry=...
Rz=...

[TRACKER]
Tracking lost

[RECOVERY]
Running detector...
```

Do NOT print:

```
[DETECTION]
```

on every frame while tracking.

============================================================
CODE ARCHITECTURE
=================

Before changing anything, inspect the existing project.

Identify:

1. detector implementation
2. segmentation implementation
3. current pose estimation
4. current tracker, if any
5. 3D model loader
6. renderer
7. camera handling
8. camera calibration
9. current frame loop
10. current live-tracking implementation
11. where detection is called
12. where the model transform is calculated

Find the exact reason why detection is currently called repeatedly.

Do not fix this by simply adding:

```
sleep()
delay()
detection_interval
```

unless that is part of the recovery system.

The actual solution must separate:

```
DETECTION
```

from:

```
TRACKING
```

and:

```
RENDERING.
```

============================================================
IMPLEMENTATION ORDER
====================

Do this incrementally.

PHASE 1:

Refactor the current loop into:

```
SEARCHING
TRACKING
LOST
RECOVERING
```

without changing the existing detector or renderer.

Verify:

```
detection once
tracking continuously
redetection only after loss
```

PHASE 2:

Improve tracking using a suitable 2D tracker.

PHASE 3:

Add depth/Z tracking if depth data is available.

PHASE 4:

Add Kalman filtering / pose smoothing.

PHASE 5:

Integrate FoundationPose-style 6DoF pose refinement.

PHASE 6:

Optimize GPU/CPU performance.

Do not implement all phases at once.

============================================================
SUCCESS CRITERIA
================

The implementation is successful when:

1. Camera starts.
2. Detector searches.
3. Cup is detected.
4. Initial pose is calculated.
5. 3D cup appears exactly over the physical cup.
6. Detector stops.
7. Tracker takes over.
8. Cup moves.
9. 3D model follows the cup.
10. Cup rotates.
11. 3D model rotates with it.
12. Camera moves.
13. 3D model remains registered.
14. Detector is NOT called again while tracking is healthy.
15. If tracking is genuinely lost, detector is activated again.
16. Cup is reacquired.
17. Tracker is initialized again.
18. Detector stops again.
19. 3D model continues following the cup.

============================================================
FINAL GOAL
==========

The final system should behave like:

```
DETECT
   ↓
REGISTER
   ↓
LOCK
   ↓
TRACK
   ↓
TRACK
   ↓
TRACK
   ↓
TRACK
   ↓
TRACK
   ↓
TRACK LOST?
   │
   ├── NO → continue tracking
   │
   └── YES
          ↓
       RE-DETECT
          ↓
       RE-REGISTER
          ↓
       TRACK
```

The 3D model is a persistent digital twin of the detected physical
object.

Use FoundationPose++ as the technical reference for improving the
6DoF tracking and pose estimation, but preserve the existing
application architecture wherever possible.

FIRST inspect the current code and provide:

1. Current detection flow
2. Current tracking flow
3. Current pose flow
4. Current rendering flow
5. Exact reason detection is repeated
6. Which parts of FoundationPose++ are applicable
7. Proposed minimal code changes

Then implement the changes incrementally.

Do not make a large rewrite before explaining the existing architecture.
[github.com/teal024/FoundationPose-plus-plus](https://github.com/teal024/FoundationPose-plus-plus)

```

**The most important instruction in this prompt is:** don't just reduce detector frequency. The agent needs to **separate detection, tracking, pose estimation, and rendering into independent stages**. FoundationPose++ should then be used to improve the pose/tracking stage rather than being bolted onto the current frame loop.
```
