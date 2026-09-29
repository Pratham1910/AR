# Pose / Physical ↔ 3D Registration (Phase 5)

Implements Project.md §23-§26: camera sees the physical object → estimate its
6DoF pose → apply that transform to the 3D model → the model overlays the
physical object.

## Three registration methods

| | Marker | Feature tracking | Markerless (bbox) |
|---|---|---|---|
| Orientation? | Yes (real 6DoF) | Yes (real 6DoF) | No (identity only) |
| Setup | Print + measure a marker | Register one reference photo | Type an object's real height |
| Requires | A marker in the scene | The object to have real texture (label/logo) | Nothing extra |
| Accuracy | Best (sub-pixel corner detection) | Good if texture is rich; degrades off the reference viewing angle | Coarse (position only) |

### 1. Marker-based (`/api/vision/pose`) — accurate 6DoF, needs a printed marker

**AprilTag/ArUco, explicitly not the final product requirement (Project.md
§24)** — it proves the coordinate-system architecture before investing in
markerless/CAD-based registration. Generate a printable marker with:

```bash
python -m app.workers.generate_marker --id 0 --size-px 600
```

Print it, **measure the printed square's actual side length in meters**, and
set `ARUCO_MARKER_LENGTH_M` in `.env` to match — pose accuracy depends
entirely on that measurement, not on anything the code can infer.

### 2. Feature tracking (`/api/vision/reference-image` + `/api/vision/feature-pose`) — no marker, real 6DoF

`app/services/pose/feature_tracker.py`. The closest of the three to DELMIA's
actual technique (see the comparison below), while staying feasible on a
plain RGB webcam:

1. **Register** a reference photo of the object's labeled/textured surface,
   filling the frame as closely as possible, plus its real physical width
   and height in meters:
   `POST /api/vision/reference-image { asset_id, image_base64, label_width_m, label_height_m }`.
   This runs ORB keypoint detection on the photo and assigns each keypoint a
   3D coordinate on a flat plane of that size, persisted to
   `data/reference_images/<asset_id>.{npz,json}`. The response's
   `feature_count`/`quality` tells you immediately whether the surface has
   enough texture to track (`"too_few"` under ~30 features, `"good"` at 100+)
   — a plain glossy/matte surface will not work, same as any real
   feature-based tracker (Vuforia Image Targets included).
2. **Track**: `POST /api/vision/feature-pose { asset_id, image_base64 }` runs
   ORB on the live frame, matches descriptors against the reference (ratio
   test + `cv2.BFMatcher`), and solves the resulting 2D↔3D correspondences
   with `cv2.solvePnPRansac` — real position **and** orientation, with
   RANSAC outlier rejection. Requires at least 12 inlier matches to report a
   pose; otherwise `found: false`.

**Honest limits**: the reference is a single flat plane, not the object's
full 3D geometry — accuracy degrades as the viewing angle departs from the
one the reference photo was taken at (a curved label on a cylindrical
bottle is only approximately flat). It is still not matching against the
GLB's actual mesh from arbitrary angles.

### 3. Markerless / bounding-box (`/api/vision/object-registration`) — no marker, approximate, position-only

`app/services/pose/markerless.py`. No printed marker needed: segmentation
(below) finds `target_class_label` (e.g. `"bottle"`, a stock COCO class), and
`estimate_object_placement()` derives an approximate camera-space position
from the detected box's apparent pixel height vs. a `real_world_height_m` you
provide (similar-triangles depth-from-size — the same principle a rangefinder
uses: `Z = f_y * real_height / pixel_height`).

**This is explicitly not 6DoF pose** — only position is estimated;
orientation is always identity (a single 2D box carries no rotation
information), and `ObjectRegistrationResponse.approximate` is always `true`.
Accuracy depends entirely on (a) the calibration in use and (b) how correctly
you measured `real_world_height_m`.

**Reference point — how close is any of this to DELMIA?** Dassault Systèmes'
DELMIA Augmented Experience (formerly Diota) does markerless tracking that
matches live camera features against the object's real **CAD geometry**,
locking in at sub-millimeter accuracy and staying registered as the operator
moves around the object. Feature tracking (method 2, above) is a real step in
that direction — genuine keypoint matching + PnP, real 6DoF — but it matches
against one photographed *flat plane*, not the object's full 3D mesh from
every angle, and it isn't sub-millimeter or drift-corrected as you move.
Closing that remaining gap means either (a) matching against the GLB's actual
3D geometry from arbitrary viewpoints (multi-view feature reconstruction, or
a modern generalizable 6D pose network like FoundationPose), or (b) a depth
sensor doing real-time ICP against the mesh (Project.md's own planned Phase 6,
via Open3D) — both meaningfully larger builds than this step.
[DELMIA Augmented Experience](https://www.3ds.com/products/delmia/augmented-experience) ·
[maintenance use case](https://www.3ds.com/products/delmia/augmented-experience/maintenance)

## Object outline / segmentation — `app/services/vision/segmentation.py`

`POST /api/vision/segment` (Project.md §16): returns each detected object's
class, confidence, bounding box, and pixel-space **polygon outline** — a real
contour, not just a rectangle. Uses a stock COCO-pretrained
`yolov8n-seg.pt` by default (auto-downloaded by `ultralytics` on first use,
configurable via `SEGMENTATION_MODEL_NAME`), since COCO already includes a
"bottle" class — no custom training needed for this debug/demo view. This is
explicitly *not* the same detector used by the Phase 1 procedure/QA pipeline
(`app/services/vision/detector.py`), which must stay custom-trainable and
config-driven per §15; this one exists purely to show "what does the vision
layer actually see" (§57) and to drive markerless registration above.

## Camera calibration (Project.md §25)

`app/services/pose/calibration.py`: `CameraCalibration` is loaded once from a
JSON file (`camera_matrix`, `dist_coeffs`, `image_width/height`) and reused
for every pose estimate — "store calibration, do not repeatedly assume an
arbitrary camera matrix."

Without a real calibration, `CameraCalibration.approximate()` builds a coarse
pinhole model from an assumed 70° horizontal FOV. Every pose response says so
explicitly via `calibration_is_approximate` / `calibration_source` — this is
for proving the pipeline, not a measurement claim (§31, §53).

To calibrate for real: photograph a printed checkerboard ~15-20 times with
the target camera (see `app/workers/calibrate_camera.py`'s docstring), then:

```bash
python -m app.workers.calibrate_camera
# writes data/calibration/camera.json
```

Set `CAMERA_CALIBRATION_PATH=data/calibration/camera.json` in `.env`.

## Pose estimation — `app/services/pose/aruco_pose.py`

`ArucoPoseEstimator.estimate(frame, calibration, target_marker_id=None)`
detects markers via `cv2.aruco.ArucoDetector`, then solves the pose with
`cv2.solvePnP` against the marker's known 3D corner geometry directly
(not the deprecated `cv2.aruco.estimatePoseSingleMarkers`, for compatibility
across opencv-contrib versions). Returns `found`, `marker_id`, `rvec`/`tvec`,
and a reprojection error in pixels — a sanity check on solution quality that
the API also returns.

## Coordinate transforms — `app/services/pose/transforms.py`

A dedicated module (Project.md §26), because raw XYZ/rotation math must never
be scattered across the codebase:

- `rvec_tvec_to_matrix` / `invert_transform` / `compose_transforms` — generic
  4x4 rigid-transform utilities (`T_camera_object`, `T_object_camera`, etc.).
- `cv_pose_to_threejs` — the **one** conversion point between OpenCV camera
  convention (X right, Y down, Z forward) and Three.js convention (X right,
  Y up, Z backward). Converting server-side means the frontend never touches
  this math and its Three.js camera can stay at the identity transform.

## APIs

- `POST /api/vision/pose` — `{ image_base64, target_marker_id? }` →
  `{ found, marker_id, position, quaternion, reprojection_error_px,
  corners_px, calibration_is_approximate, calibration_source }`.
  `corners_px` is the marker's 4 detected image-space corners, purely for
  drawing an outline — the pose math never uses it.
- `POST /api/vision/segment` — `{ image_base64, confidence_threshold? }` →
  `{ objects: [{ class_label, confidence, bbox, polygon }], model_version,
  inference_ms }`.
- `POST /api/vision/object-registration` — `{ image_base64,
  target_class_label, real_world_height_m, confidence_threshold? }` →
  `{ found, class_label, confidence, bbox, polygon, position, quaternion,
  approximate: true, calibration_is_approximate, calibration_source }`.
  Combines segmentation + `markerless.py` placement in one call (one frame in,
  both the outline and the approximate pose out).
- `POST /api/vision/reference-image` — `{ asset_id, image_base64,
  label_width_m, label_height_m }` → `{ feature_count, quality }`.
- `POST /api/vision/feature-pose` — `{ asset_id, image_base64 }` →
  `{ found, position, quaternion, num_matches, num_inliers, inlier_points_px,
  calibration_is_approximate, calibration_source }`.

`position`/`quaternion` in all pose-bearing responses are already in
Three.js space.

## Frontend — `frontend/src/features/viewer3d/RegistrationOverlay.tsx`

A transparent Three.js canvas sits over the live `<video>` feed, plus a
separate 2D outline canvas. A mode toggle switches between:

- **Markerless** (default): calls `/api/vision/object-registration` with the
  configured object class + real-world height, draws the detected polygon in
  cyan, and places the model at the (approximate, position-only) pose.
- **ArUco Marker**: calls `/api/vision/pose`, draws the marker's 4 corners in
  green, places the model at the accurate 6DoF pose.
- **Feature Tracking**: a "Register Reference Image" step captures the
  object's labeled surface and reports feature-extraction quality, then
  **Detect & Align** calls `/api/vision/feature-pose`, draws matched inlier
  points in yellow, and places the model at the recovered real 6DoF pose.

Either way, the model's `position`/`quaternion` are set directly from the
response — no coordinate math on the frontend. A "live tracking" checkbox
repeats detection on an interval; **Detect & Align** does one shot.

**Known limitations**: the overlay's Three.js camera FOV is a fixed guess,
not derived from the backend's actual `camera_matrix`, so the overlay's
*perspective* won't exactly match the real video's. Markerless mode has no
orientation estimate at all; feature-tracking mode needs real object texture
and only models a flat reference plane — see "Three registration methods"
above for exactly what each does and doesn't estimate, and what closing the
remaining gap to DELMIA-grade registration would actually require.

## Tests

`tests/backend/test_transforms.py` — pure transform math (identity/rotation
round-trips, the OpenCV→Three.js axis flip), no camera needed.
`tests/backend/test_aruco_pose.py` — renders a synthetic marker into a frame
and verifies detect→solvePnP→pose end-to-end, including the "no marker" and
"wrong target id" cases, without requiring a physical printed marker.
`tests/backend/test_markerless.py` — pure depth-from-apparent-size math
(centered/off-center boxes, near-vs-far, degenerate zero-height box), no
camera or real model download needed.
`tests/backend/test_feature_tracker.py` — a rigorously constructed synthetic
scene (a textured plane at a *known* ground-truth 6DoF pose, projected and
warped into a synthesized frame via real pinhole-camera math) verifies the
full ORB-detect → match → RANSAC-solvePnP pipeline recovers a pose close to
that ground truth, plus the "no match on an unrelated frame" and
save/load round-trip cases — no physical camera or printed label needed.
