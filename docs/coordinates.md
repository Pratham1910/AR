# Coordinate systems

Three frames are involved in placing a 3D model on a real object. Every
conversion between them happens in one place, `backend/app/services/pose/transforms.py`.

| Frame | Used by | +X | +Y | +Z | Units |
|---|---|---|---|---|---|
| **Camera (OpenCV)** | pose estimation: solvePnP, MegaPose, intrinsics `K` | right in the image | **down** in the image | **forward**, out of the lens | meters |
| **Object (glTF)** | the 3D model's own mesh | the model's authored right | the model's authored **up** | the model's authored front | meters, after `Model3D.scale` |
| **Renderer (Three.js)** | the AR overlay | right | **up** | **toward the viewer**; the camera looks down **−Z** | meters |

## Relationships

- **Camera -> renderer:** `C = diag(1, -1, -1)`, a 180° rotation about X. A point
  `p_cv` in camera space is `C @ p_cv` in renderer space. The render camera
  sits at the renderer origin with identity rotation, so "the camera" is the
  same physical lens in both frames.
- **Object pose:** pose estimators return `T_camera_object = [R | t]`, mapping
  object points into camera space: `p_cv = R @ p_obj + t`.
  - Model-based (MegaPose): the object frame *is* the glTF frame, so the
    renderer pose is `R_gl = C @ R`, `t_gl = C @ t`
    (`cv_model_pose_to_threejs`). Using `C @ R @ C` here would draw the
    model flipped 180° about X.
  - Marker / feature tracking: the object frame is the marker's or reference
    plane's own OpenCV-style frame, so `R_gl = C @ R @ C`
    (`cv_pose_to_threejs`); the anchor offset then maps it to the model.
  - Markerless: position only, from apparent size; identity rotation.
- **Model origin:** the overlay recenters the loaded GLB on its bounding-box
  center (after scaling to meters), and the pose service registers the mesh
  recentered the same way, so "the pose" refers to the same point in both.

## Camera model

One pinhole model is used for pose and rendering: `fx, fy, cx, cy` from
`CameraCalibration` (`backend/app/services/pose/calibration.py`), scaled to
the frame size. The AR overlay builds its projection matrix directly from
those intrinsics (`applyCameraIntrinsics` in `RegistrationOverlay.tsx`), so
an off-center principal point is honored, not approximated by a field of view.

- Without a calibration file the intrinsics are an **assumed** 70° horizontal
  field of view with a centered principal point and no distortion; the UI says
  so. Run `python -m app.workers.calibrate_camera` with a printed checkerboard
  and set `CAMERA_CALIBRATION_PATH` for a real one.
- Lens distortion is not undistorted before rendering. With an approximate
  calibration it's zero by definition; with a real one, strongly distorting
  lenses will show some edge-of-frame mismatch.

## Depth (Z)

The webcam is **RGB only**; there is no depth image, and none is faked.
Z comes from known geometry instead:
- markerless: `Z = fy * real_height / box_height_px`;
- model-based: MegaPose matches the mesh at its real (metric) size, so Z is
  whatever distance makes it project to the observed size.

Either way, a wrong real-world size gives a proportionally wrong Z. An RGB-D
camera would allow FoundationPose++-style "Z from the depth at (x, y)".

## Pose display

Rx/Ry/Rz in the UI and logs are XYZ Euler angles in the **renderer** frame,
computed from the filtered quaternion (`euler_xyz_deg`), for reading only.
All pose math stays in matrices/quaternions.
