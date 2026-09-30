"""
Coordinate transform utilities (Project.md #26): a dedicated module so raw
XYZ/rotation math is never scattered across the codebase.

Two coordinate systems meet here:
  - OpenCV camera space: X right, Y down, Z forward (into the scene).
  - Three.js space: X right, Y up, Z backward (camera looks down -Z).

`cv_pose_to_threejs` is the single conversion point between them — everything
downstream (the frontend AR overlay) works only in Three.js space.
"""

import math
from dataclasses import dataclass

import numpy as np

# Flips Y and Z to go from OpenCV camera convention to Three.js convention.
_CV_TO_GL = np.diag([1.0, -1.0, -1.0])


@dataclass
class Pose6DoF:
    """A rigid transform: translation (meters) + rotation, Three.js convention."""

    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]  # (x, y, z, w)


def rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """OpenCV rotation/translation vectors -> a 4x4 homogeneous transform (T_camera_object)."""
    import cv2

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = tvec.reshape(3)
    return matrix


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    """Inverts a 4x4 rigid transform (e.g. T_camera_object -> T_object_camera)."""
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverted = np.eye(4, dtype=np.float64)
    inverted[:3, :3] = rotation.T
    inverted[:3, 3] = -rotation.T @ translation
    return inverted


def compose_transforms(t_a_b: np.ndarray, t_b_c: np.ndarray) -> np.ndarray:
    """T_a_c = T_a_b @ T_b_c — e.g. T_world_object = T_world_camera @ T_camera_object."""
    return t_a_b @ t_b_c


def _rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Standard matrix->quaternion (x, y, z, w), no scipy dependency."""
    m = rotation_matrix
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def cv_pose_to_threejs(rvec: np.ndarray, tvec: np.ndarray) -> Pose6DoF:
    """
    Converts an OpenCV object pose (rvec/tvec, object relative to camera) into
    a Three.js-space position + quaternion, so the frontend AR overlay's
    camera can stay at the identity transform and just place the model.
    """
    cv_matrix = rvec_tvec_to_matrix(rvec, tvec)
    gl_rotation = _CV_TO_GL @ cv_matrix[:3, :3] @ _CV_TO_GL
    gl_position = _CV_TO_GL @ cv_matrix[:3, 3]

    quaternion = _rotation_matrix_to_quaternion(gl_rotation)
    return Pose6DoF(position=tuple(float(v) for v in gl_position), quaternion=quaternion)


def cv_model_pose_to_threejs(t_camera_object: np.ndarray) -> Pose6DoF:
    """
    Like cv_pose_to_threejs, but for a pose whose object frame IS the 3D
    model's own (glTF) frame — e.g. MegaPose, which matches the mesh itself.

    Only the camera side changes convention here: x_gl = C @ (R @ m + t), so
    the rotation is C @ R, not C @ R @ C. cv_pose_to_threejs's extra right-hand
    C re-interprets the object's own axes as OpenCV-style, which is right for
    an ArUco marker's frame but would render a mesh-matched model flipped
    180 degrees about X.
    """
    rotation = t_camera_object[:3, :3]
    translation = t_camera_object[:3, 3]
    gl_rotation = _CV_TO_GL @ rotation
    gl_position = _CV_TO_GL @ translation
    quaternion = _rotation_matrix_to_quaternion(gl_rotation)
    return Pose6DoF(position=tuple(float(v) for v in gl_position), quaternion=quaternion)


def euler_angles_deg_from_rotation_matrix(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
    """
    Tait-Bryan (XYZ order) Euler angles in degrees, purely for human-readable
    debug display — never used for the actual pose/placement math, which
    stays in matrix/quaternion form throughout specifically to avoid gimbal
    lock and Euler-order ambiguity. This exists only so a debug UI can show
    "Rx=.., Ry=.., Rz=.." next to the position, per the platform's own
    debug-mode requirement (show enough to visually sanity-check a pose).
    """
    r = rotation_matrix
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        rx = math.atan2(r[2, 1], r[2, 2])
        ry = math.atan2(-r[2, 0], sy)
        rz = math.atan2(r[1, 0], r[0, 0])
    else:
        rx = math.atan2(-r[1, 2], r[1, 1])
        ry = math.atan2(-r[2, 0], sy)
        rz = 0.0
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def euler_angles_deg(rvec: np.ndarray) -> tuple[float, float, float]:
    import cv2

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    return euler_angles_deg_from_rotation_matrix(rotation_matrix)


def project_pose_axes(
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length_m: float,
) -> dict[str, tuple[float, float]]:
    """
    Projects the pose's own origin and X/Y/Z axis endpoints into image pixel
    space (OpenCV camera convention, the same space `corners_px`/
    `inlier_points_px` already use), for drawing a debug "this is the pose I
    estimated" gizmo directly on the live feed — the standard X=red, Y=green,
    Z=blue convention.
    """
    import cv2

    object_points = np.array(
        [[0.0, 0.0, 0.0], [axis_length_m, 0.0, 0.0], [0.0, axis_length_m, 0.0], [0.0, 0.0, axis_length_m]],
        dtype=np.float64,
    )
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    pts = projected.reshape(4, 2)
    return {
        "origin": (float(pts[0][0]), float(pts[0][1])),
        "x_axis": (float(pts[1][0]), float(pts[1][1])),
        "y_axis": (float(pts[2][0]), float(pts[2][1])),
        "z_axis": (float(pts[3][0]), float(pts[3][1])),
    }
