# Pose / Physical ↔ 3D Registration (Phase 5)

Implements Project.md §23-§26: camera sees the physical object → estimate its
6DoF pose → apply that transform to the 3D model → the model overlays the
physical object.

## Registration method (Project.md §24)

**Marker-based (AprilTag/ArUco), explicitly not the final product
requirement** — it proves the coordinate-system architecture before investing
in markerless/CAD-based registration. Generate a printable marker with:

```bash
python -m app.workers.generate_marker --id 0 --size-px 600
```

Print it, **measure the printed square's actual side length in meters**, and
set `ARUCO_MARKER_LENGTH_M` in `.env` to match — pose accuracy depends
entirely on that measurement, not on anything the code can infer.

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

## API — `POST /api/vision/pose`

Request: `{ image_base64, target_marker_id? }`.
Response: `{ found, marker_id, position, quaternion, reprojection_error_px,
calibration_is_approximate, calibration_source }` — `position`/`quaternion`
are already in Three.js space.

## Frontend — `frontend/src/features/viewer3d/RegistrationOverlay.tsx`

A transparent Three.js canvas sits over the live `<video>` feed. **Detect &
Align** captures a frame, calls `/api/vision/pose`, and — if found — sets the
loaded GLB's `position`/`quaternion` directly from the response (no
coordinate math on the frontend) and shows it; otherwise hides it. A "live
tracking" checkbox repeats this on an interval.

**Known limitation**: the overlay's Three.js camera FOV is a fixed guess, not
derived from the backend's actual `camera_matrix` — so the overlay's
*perspective* won't exactly match the real video's, only the marker-relative
pose is accurate (and only as accurate as the calibration in use).

## Tests

`tests/backend/test_transforms.py` — pure transform math (identity/rotation
round-trips, the OpenCV→Three.js axis flip), no camera needed.
`tests/backend/test_aruco_pose.py` — renders a synthetic marker into a frame
and verifies detect→solvePnP→pose end-to-end, including the "no marker" and
"wrong target id" cases, without requiring a physical printed marker.
