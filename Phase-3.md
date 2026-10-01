
I need to change the live AR pipeline from "detect every frame" to
"detect once → initialize tracker → continuously track → detect again
only when tracking is lost."

CURRENT PROBLEM:

The 3D overlay is now working.

However, the current application appears to perform object detection
continuously.

The pipeline is effectively:

    FRAME
      ↓
    DETECTION
      ↓
    POSE
      ↓
    RENDER
      ↓
    FRAME
      ↓
    DETECTION AGAIN
      ↓
    POSE AGAIN
      ↓
    RENDER

This causes unnecessary detection and causes the 3D rendering/pose
registration to be unstable.

THIS IS NOT WHAT I WANT.

==================================================
DESIRED ARCHITECTURE
====================

Use a TWO-STAGE pipeline:

    DETECTION
         ↓
    INITIALIZATION
         ↓
    TRACKING
         ↓
    3D RENDERING

Detection should NOT run continuously once the object has been
successfully detected and registered.

The object should be detected once, then tracked continuously.

==================================================
STATE MACHINE
=============

Implement an explicit tracking state machine.

States:

    SEARCHING
    TRACKING
    LOST

Initial state:

    SEARCHING

---

STATE 1: SEARCHING
------------------

Run the existing AI object detector.

Example:

    camera frame
        ↓
    YOLO / segmentation
        ↓
    cup detected
        ↓
    calculate initial pose
        ↓
    initialize tracker
        ↓
    save tracking state
        ↓
    state = TRACKING

Once the cup has been successfully detected and the initial 3D pose
has been calculated, DO NOT immediately run the detector again.

---

STATE 2: TRACKING
-----------------

This is the important state.

While TRACKING:

DO NOT run the object detector.

Instead:

    camera frame
        ↓
    tracker
        ↓
    updated object position
        ↓
    updated object pose
        ↓
    3D model transform
        ↓
    render

The tracker should follow the already-detected object.

The 3D model should remain attached to the physical object.

If the physical cup moves:

    physical cup moves
          ↓
    tracker detects movement
          ↓
    update pose
          ↓
    3D model moves with cup

If the physical cup rotates:

    physical cup rotates
          ↓
    tracker estimates new pose
          ↓
    3D model rotates with cup

If the physical cup moves closer:

    physical cup gets larger
          ↓
    tracker updates depth/scale
          ↓
    3D model gets larger

If the physical cup moves farther:

    physical cup gets smaller
          ↓
    tracker updates pose
          ↓
    3D model gets smaller

==================================================
STATE 3: LOST
=============

If tracking confidence becomes too low:

    TRACKING
       ↓
    tracker confidence < threshold
       ↓
    LOST

Only then should the AI detector be activated again.

Example:

    LOST
      ↓
    run detector
      ↓
    cup found?
       ↓
      YES
       ↓
    calculate/reinitialize pose
       ↓
    initialize tracker again
       ↓
    TRACKING

If cup is not found:

    remain in LOST / SEARCHING
    continue detection at a controlled interval

==================================================
VERY IMPORTANT
==============

Do NOT do:

    detect()
    track()
    detect()
    track()
    detect()
    track()

every frame.

Instead do:

    detect()
    initialize_tracker()

    track()
    track()
    track()
    track()
    track()
    track()
    ...

Only:

    tracking lost
        ↓
    detect()
        ↓
    reinitialize_tracker()

==================================================
TRACKING OPTIONS
================

Inspect the existing project first and determine what tracking
technology is currently available.

Prefer a tracker that works with the existing segmentation/bounding
box/keypoints.

Possible approaches:

1. OpenCV object tracker
2. Optical-flow tracking
3. Feature tracking
4. YOLO tracking / ByteTrack / BoT-SORT
5. Keypoint tracking
6. Pose tracking

Choose the approach that best fits the existing architecture.

Do not blindly add a new framework if the current project already has
a suitable tracker.

==================================================
IMPORTANT DIFFERENCE:
DETECTION VS TRACKING
=====================

Detection answers:

    "Where is the cup?"

Tracking answers:

    "Where did the cup that I already found move?"

We need the second behavior after initialization.

==================================================
PERSIST THE OBJECT IDENTITY
===========================

Once the cup is detected, assign it a persistent tracking identity.

For example:

    object_id = 1
    class = cup

Store:

    object_id
    class
    last_position
    last_pose
    last_bbox
    last_mask
    tracking_confidence
    last_seen_frame
    velocity if available

Do not recreate the object from scratch every frame.

==================================================
3D MODEL
========

The 3D model should be created/loaded ONCE.

Do not reload the model whenever a new frame arrives.

Current desired behavior:

    FIRST DETECTION
         ↓
    load 3D model
         ↓
    initialize pose
         ↓
    show model
         ↓

    TRACKING:
         ↓
    update transform ONLY
         ↓
    render existing model

The renderer should not recreate the 3D model every frame.

Only update:

    position
    rotation
    scale
    visibility

as required.

==================================================
POSE
====

The initial detection should establish the initial pose:

    R_initial
    T_initial

Then tracking should maintain/update that pose.

Do NOT recalculate the entire pose from scratch using AI detection
every frame.

Conceptually:

    INITIAL:

    Detection
       ↓
    Pose estimation
       ↓
    R0, T0
       ↓
    Tracker initialized

    TRACKING:

    Frame N
       ↓
    Tracker
       ↓
    ΔR, ΔT
       ↓
    Update pose
       ↓
    Rn, Tn
       ↓
    Render

==================================================
TEMPORAL SMOOTHING
==================

The tracker should have smoothing.

Do not directly apply noisy tracking values to the 3D model.

Use something like:

    measured_pose
          ↓
    smoothing/filter
          ↓
    rendered_pose

Possible methods:

- exponential smoothing
- Kalman filter
- One Euro filter

Start with simple exponential smoothing if necessary.

==================================================
DETECTION INTERVAL AFTER LOSS
=============================

Do not run the expensive detector on every frame after tracking is
lost either.

Use a controlled recovery interval.

For example:

    detector every N frames

or:

    detector every 100-200 ms

until the object is found.

Once found:

    STOP DETECTION
    RETURN TO TRACKING

==================================================
TRACKING CONFIDENCE
===================

Add:

    tracking_confidence

For example:

    > 0.7
        TRACKING

    0.4 - 0.7
        TRACKING but monitor

    < 0.4
        LOST

Use values appropriate for the selected tracker.

Do not hard-code these values without making them configurable.

==================================================
DO NOT LOSE THE 3D MODEL IMMEDIATELY
====================================

If tracking fails for one frame, do NOT immediately hide the model.

Use a short grace period.

Example:

    tracker lost for 1 frame
        ↓
    keep last valid pose

    tracker lost for several consecutive frames
        ↓
    attempt recovery

    recovery timeout
        ↓
    hide 3D model
        ↓
    SEARCHING

This prevents flickering.

==================================================
EXPECTED BEHAVIOR
=================

Test this exact sequence:

1. Start camera.

Expected:

    "Searching..."

2. Put cup in front of camera.

Expected:

    detector runs

3. Cup detected.

Expected:

    detector stops

    3D model appears

    tracker starts

4. Keep cup stationary.

Expected:

    detector does NOT run again

    3D model remains registered

5. Move cup left.

Expected:

    tracker follows cup

    3D model moves left

6. Move cup right.

Expected:

    tracker follows cup

    3D model moves right

7. Rotate cup.

Expected:

    tracker updates pose

    3D model rotates

8. Move cup toward camera.

Expected:

    tracker updates depth

    3D model scales appropriately

9. Move cup away.

Expected:

    tracker updates depth

    3D model scales appropriately

10. Temporarily hide cup.

Expected:

    tracker loses confidence

    model remains briefly at last valid pose

11. Keep cup hidden long enough.

Expected:

    model hides

    state changes to SEARCHING / LOST

12. Bring cup back.

Expected:

    detector runs

    cup is detected

    tracker initializes again

    3D model reappears

    detector stops again

==================================================
UI
==

The current UI says:

    "Detecting..."

Change it to reflect the actual state.

For example:

    SEARCHING:
        "Searching for object..."

    TRACKING:
        "Tracking cup"

    LOST:
        "Object lost — searching..."

Also show:

    Tracking confidence: XX%

Optional:

    Detection count: 1
    Tracking frames: 1245
    Object ID: 1

This will allow us to confirm that detection is actually happening
only when necessary.

==================================================
DEBUG LOGGING
=============

Add logs such as:

    [DETECTION] Searching for object...
    [DETECTION] Cup found
    [POSE] Initial pose calculated
    [TRACKER] Initialized object ID=1
    [TRACKER] Tracking object ID=1
    [TRACKER] Confidence=0.92
    [TRACKER] Confidence=0.85
    [TRACKER] Object moved
    [TRACKER] Object lost
    [RECOVERY] Running detector
    [DETECTION] Cup reacquired
    [TRACKER] Reinitialized

Do NOT print "[DETECTION]" every frame while tracking.

==================================================
PERFORMANCE
===========

Measure separately:

    detection FPS
    tracking FPS
    rendering FPS

The goal is:

    Detection = expensive and infrequent
    Tracking = lightweight and continuous
    Rendering = continuous

For example:

    Detection:
        5-10 times/sec only when searching/recovering

    Tracking:
        every camera frame

    Rendering:
        every camera frame

Do not enforce these exact numbers if the hardware requires different
values. Make them configurable.

==================================================
ARCHITECTURE
============

The final architecture should be:

             CAMERA FRAME
                   │
                   ▼
          ┌─────────────────┐
          │ STATE MACHINE   │
          └────────┬────────┘
                   │
        ┌──────────┴───────────┐
        │                      │
   SEARCHING                TRACKING
        │                      │
        ▼                      ▼
    DETECTOR               TRACKER
        │                      │
        ▼                      ▼
   INITIAL POSE            UPDATE POSE
        │                      │
        └──────────┬───────────┘
                   │
                   ▼
             3D MODEL
             TRANSFORM
                   │
                   ▼
              RENDERING
                   │
                   ▼
                CAMERA

If tracker fails:

             TRACKING
                 │
                 ▼
             LOST
                 │
                 ▼
             DETECTOR
                 │
          ┌──────┴──────┐
          │             │
       FOUND          NOT FOUND
          │             │
          ▼             ▼
     REINITIALIZE    SEARCHING
       TRACKER

==================================================
MOST IMPORTANT REQUIREMENT
==========================

Once an object has been detected:

    DO NOT DETECT IT AGAIN ON EVERY FRAME.

Instead:

    DETECT ONCE
       ↓
    LOCK TRACKING
       ↓
    TRACK MOVEMENT
       ↓
    UPDATE 3D MODEL
       ↓
    KEEP FOLLOWING OBJECT
       ↓
    ONLY REDETECT IF TRACKING IS LOST

The 3D model must therefore behave like a digital twin attached to the
physical object, not like a new AR overlay generated independently
from every camera frame.

Before modifying code, inspect the current detection, rendering, and
tracking implementation and identify exactly why detection is being
called repeatedly. Then modify the architecture rather than simply
adding delays to the detector.
