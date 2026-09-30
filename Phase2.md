
### Use this prompt

```text
I need to fix the AR 3D model overlay in my current object-tracking application.

CURRENT STATE:
- The camera successfully detects/segments the real physical object.
- In the current example, the object class is "cup".
- The segmentation contour/boundary is correctly drawn around the physical cup.
- The real cup height is provided as 0.2 meters.
- However, the 3D cup model is currently rendered above the physical cup instead of being aligned with it.
- I do NOT want the solution to simply place the 3D model at the center of the bounding box.

GOAL:
The 3D model must be spatially registered with the real physical object.

For the detected real cup:
1. Detect/segment the physical cup.
2. Estimate its 3D pose relative to the camera:
   - X position
   - Y position
   - Z/depth
   - rotation around X
   - rotation around Y
   - rotation around Z
3. Project the 3D model into the camera image using the actual camera intrinsic parameters.
4. Render the 3D model at that estimated pose.
5. The 3D model should visually sit directly ON TOP OF the physical cup.
6. The 3D model must have the same apparent height/width as the real cup.
7. If the real cup moves, the 3D model must follow it continuously.
8. If the camera moves, the 3D model must remain registered to the real cup.
9. The 3D model should not remain fixed at an arbitrary screen coordinate.

IMPORTANT:
Do not solve this by simply doing:

    model_x = bbox_center_x
    model_y = bbox_center_y

and scaling the model according to the bounding box.

That only gives 2D placement and does not solve the AR registration problem.

IMPLEMENTATION REQUIREMENTS:

A. CAMERA CALIBRATION
- Determine whether the current project has camera intrinsic calibration.
- If not, add camera calibration support.
- Obtain:
    fx
    fy
    cx
    cy
    distortion coefficients
- Store the calibration parameters in a configuration file.
- Use cv2.calibrateCamera / equivalent OpenCV calibration where appropriate.

B. OBJECT POSE
Implement an actual 6DoF object-pose pipeline.

Prefer this order:

1. If a reliable set of known 3D object keypoints/features is available:
   - establish 3D model points
   - establish corresponding 2D image points
   - use cv2.solvePnP / solvePnPRansac
   - obtain rotation vector and translation vector

2. If the current detector only provides segmentation:
   - do NOT pretend that segmentation alone gives accurate 6DoF pose.
   - create an intermediate pose-estimation method appropriate for the known 3D object.
   - Use the known physical dimensions of the object where possible.
   - Clearly separate approximate pose estimation from accurate pose estimation.

3. For the first working prototype, support an ArUco-marker-based pose mode as a reliable reference implementation.
   - The marker can be attached near/on the object.
   - Estimate camera-to-object pose using the marker.
   - Transform the 3D model from marker coordinates into camera coordinates.
   - This should provide a reliable baseline for validating the rendering pipeline.

C. 3D MODEL COORDINATE SYSTEM
Define a clear coordinate system for the 3D cup.

For example:

    origin = center of cup bottom
    +Y = cup height
    +X = cup width direction
    +Z = depth direction

The model dimensions must correspond to real-world units.

If real cup height = 0.2 m:

    model_height = 0.2 meters

Do NOT use arbitrary Unity/Three.js/PyOpenGL units without documenting the conversion.

D. CAMERA PROJECTION

Use the estimated object pose:

    X_camera = R * X_model + t

Then project the 3D vertices using the calibrated camera matrix.

The rendered model must use the same camera intrinsics/FOV as the physical camera.

Do not use an unrelated virtual camera FOV.

E. ALIGNMENT

The following points should align:

- 3D model bottom ↔ real cup bottom
- 3D model top ↔ real cup top
- 3D model left/right boundaries ↔ real cup boundaries
- cup handle ↔ real cup handle
- model orientation ↔ real cup orientation

The model should appear as if it physically occupies the same location as the real cup.

F. DEPTH

Use the estimated translation vector to determine real-world depth.

Do not use an arbitrary fixed Z value such as:

    z = 1
    z = 2
    z = 5

unless that value is explicitly derived from camera/object geometry.

The provided real object height of 0.2 m should be used to estimate scale/depth when appropriate.

G. TRACKING

After initial pose estimation:

- continuously track the object
- update pose every frame or through a tracking pipeline
- use temporal smoothing to reduce jitter
- recover pose if tracking is temporarily lost
- do not reset the 3D model to the screen center

Implement something like:

    detection
       ↓
    segmentation
       ↓
    object/keypoint/marker correspondence
       ↓
    pose estimation
       ↓
    R + t
       ↓
    smoothing/filtering
       ↓
    3D model transformation
       ↓
    camera projection/rendering

H. OCCLUSION

Eventually the real cup should be able to occlude the 3D model correctly.

For the first implementation, however, prioritize:

1. correct position
2. correct scale
3. correct rotation
4. correct depth
5. stable tracking

Then add depth/occlusion handling.

I. DEBUG MODE

Add a debug mode that displays:

- segmentation contour
- bounding box
- object center
- detected keypoints/marker corners
- camera coordinate axes
- projected 3D model points
- estimated X/Y/Z
- estimated rotation
- estimated object dimensions

Draw XYZ axes on the detected object so I can visually verify whether the pose is correct.

For example:

    X = red
    Y = green
    Z = blue

Also display:

    Position:
    X = ...
    Y = ...
    Z = ...

    Rotation:
    Rx = ...
    Ry = ...
    Rz = ...

J. VERY IMPORTANT DIAGNOSTIC

Before changing the rendering code, inspect the existing project and identify:

1. How the cup is currently detected.
2. What detector/model is being used.
3. Whether the current output is:
   - bounding box
   - segmentation mask
   - contour
   - keypoints
   - depth
4. How the 3D model is currently loaded.
5. How the virtual camera is configured.
6. How the model position/rotation/scale is currently calculated.
7. Whether camera calibration already exists.
8. Whether the project already contains ArUco functionality.
9. Whether the current coordinate systems are consistent.

Then explain the current cause of the misalignment before modifying the code.

EXPECTED RESULT:

Given this scene:

    REAL CUP
        ↓
    camera detects cup
        ↓
    estimate cup pose
        ↓
    3D cup receives same pose
        ↓

The result should visually look like:

        REAL + 3D CUP
          ┌───────┐
          │       │
          │  3D   │
          │ MODEL │  ← aligned with physical cup
          │       │
          └───────┘
             ↑
        same position,
        scale and rotation

NOT:

             ┌───────┐
             │  3D   │
             │ MODEL │
             └───────┘

        ┌───────────────┐
        │  REAL CUP     │
        └───────────────┘

The second result is what the current application is doing and must be fixed.

FIRST IMPLEMENTATION TARGET:

Do not attempt to solve every object type immediately.

Make the pipeline work accurately for ONE known cup model first.

Use:

    known cup 3D model
    +
    calibrated camera
    +
    real cup dimensions
    +
    reliable pose estimation
    +
    continuous tracking

Once the cup registration is working, design the architecture so other objects/models can use the same pipeline.
```

### One important point

Your current screenshot tells us something important:

**Segmentation is not the same thing as 3D pose.**

You already have:

```text
Camera
   ↓
Object detector
   ↓
Cup segmentation
   ↓
Blue contour
```

You need to turn it into:

```text
Camera
   ↓
Object detection / segmentation
   ↓
Object geometry / keypoints
   ↓
Camera calibration
   ↓
6DoF pose estimation
   ↓
R + T
   ↓
3D model transformation
   ↓
Camera projection
   ↓
AR overlay
```

The biggest issue is likely **pose/depth**, not the segmentation.

For your **first prototype**, I would actually make the system support **two modes**:

```text
MODE 1 — ArUco
Real cup + small ArUco marker
        ↓
Very reliable pose
        ↓
3D cup overlay
```

Then:

```text
MODE 2 — Markerless
Real cup
   ↓
YOLO/segmentation + keypoints/features
   ↓
Pose estimation
   ↓
3D cup overlay
```

That lets you prove that your **camera → pose → 3D rendering pipeline is correct** before spending time solving the much harder markerless 6DoF problem.

And in your screenshot, the fact that the cup contour is already quite good means you're at a useful point to start working on the **pose-registration layer**, rather than replacing the detector.
