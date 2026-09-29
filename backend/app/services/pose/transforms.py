"""
Coordinate transform utilities (Project.md #26): a dedicated module so raw
XYZ/rotation math is never scattered across the codebase.

Two coordinate systems meet here:
  - OpenCV camera space: X right, Y down, Z forward (into the scene).
  - Three.js space: X right, Y up, Z backward (camera looks down -Z).

`cv_pose_to_threejs` is the single conversion point between them — everything
downstream (the frontend AR overlay) works only in Three.js space.
"""

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
