"""
Rendered test scenes for pose evaluation (tests and app/eval/pose_eval.py):
a box, optionally with a printed circle near one corner of its top face —
the "circle at top-right" case — on a light desk, with optional hand,
low contrast and noise. Ground truth is exact: the pose the scene is drawn at.

Not photo-realistic: flat-shaded faces and a filled disc. It reproduces the
geometry of the problem (a symmetric outline, one asymmetric mark), which is
what the orientation logic has to get right; real-camera behaviour still has
to be measured on real frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
SHAPE = (720, 1280)


def box_mesh(size=(0.19, 0.035, 0.114)) -> tuple[np.ndarray, np.ndarray]:
    """A box centred on the origin, +Y = top (x long, z deep), outward-wound faces."""
    h = np.asarray(size, float) / 2
    v = np.array([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    f = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
        [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
        [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ])
    # Make every face wind outward (normal away from the centre).
    for i, (a, b, c) in enumerate(f):
        n = np.cross(v[b] - v[a], v[c] - v[a])
        if n @ (v[a] + v[b] + v[c]) < 0:
            f[i] = [a, c, b]
    return v, f


@dataclass
class Mark:
    """A printed disc on the box: centre and outward normal (object frame), radius."""

    center: np.ndarray
    normal: np.ndarray
    radius: float
    shade: int = 215  # light mark on a dark box


def corner_circle(size=(0.19, 0.035, 0.114), radius=0.012) -> Mark:
    """The circle near the top face's +x/-z corner ("upper right" seen from the front, above)."""
    hx, hy, hz = np.asarray(size, float) / 2
    return Mark(np.array([hx - 0.03, hy, -hz + 0.028]), np.array([0.0, 1.0, 0.0]), radius)


def look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """T_camera_object for a camera at `eye` (object frame) looking at `target`."""
    eye, target, up = (np.asarray(x, float) for x in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    t = np.eye(4)
    t[:3, :3] = np.stack([x, y, z])
    t[:3, 3] = -t[:3, :3] @ eye
    return t


def turned(t_co: np.ndarray, axis, degrees: float) -> np.ndarray:
    """The object turned about its own axis (object frame) — e.g. the box rotated on the desk."""
    out = np.array(t_co, float)
    out[:3, :3] = out[:3, :3] @ cv2.Rodrigues(np.asarray(axis, float) * np.radians(degrees))[0]
    return out


def project(points: np.ndarray, t_co: np.ndarray, k: np.ndarray = K) -> np.ndarray:
    cam = np.asarray(points) @ t_co[:3, :3].T + t_co[:3, 3]
    px = cam @ k.T
    return px[:, :2] / px[:, 2:3]


def render(
    vertices: np.ndarray,
    faces: np.ndarray,
    t_co: np.ndarray,
    marks: list[Mark] = (),
    k: np.ndarray = K,
    shape=SHAPE,
    background: int = 232,
    contrast: float = 1.0,
    noise: int = 3,
    hand: tuple[int, int, int, int] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """(gray image, hand mask or None). `contrast` < 1 flattens the lighting
    (dim, low-contrast scene); `hand`: an x1, y1, x2, y2 skin-tone block drawn
    over everything, returned as the hand mask."""
    img = np.full(shape, background, np.float32)
    cam = vertices @ t_co[:3, :3].T + t_co[:3, 3]
    px = cam @ k.T
    px = px[:, :2] / px[:, 2:3]
    light = np.array([0.3, -0.8, -0.5])
    light /= np.linalg.norm(light)
    order = np.argsort(-cam[faces].mean(axis=1)[:, 2])
    visible_normals = []
    for fi in order:
        a, b, c = cam[faces[fi]]
        n = np.cross(b - a, c - a)
        n /= np.linalg.norm(n) or 1.0
        if n @ (-(a + b + c) / 3) <= 0:
            continue
        visible_normals.append(n)
        shade = 30 + 110 * max(0.0, float(n @ -light))
        cv2.fillConvexPoly(img, np.round(px[faces[fi]]).astype(np.int32), float(shade))
    for mark in marks:
        n_cam = t_co[:3, :3] @ mark.normal
        c_cam = t_co[:3, :3] @ mark.center + t_co[:3, 3]
        if n_cam @ (-c_cam) <= 0:
            continue  # on a face turned away
        helper = np.array([1.0, 0, 0]) if abs(mark.normal[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(mark.normal, helper)
        u /= np.linalg.norm(u)
        w = np.cross(mark.normal, u)
        ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        rim = mark.center + mark.radius * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w)
        cv2.fillPoly(img, [np.round(project(rim, t_co, k)).astype(np.int32)], float(mark.shade))
    if contrast != 1.0:
        img = background + (img - background) * contrast
    mask = None
    if hand is not None:
        x1, y1, x2, y2 = hand
        img[y1:y2, x1:x2] = 160.0
        mask = np.zeros(shape, bool)
        mask[y1:y2, x1:x2] = True
    rng = np.random.default_rng(seed)
    img = img + rng.integers(-noise, noise + 1, shape)
    return np.clip(img, 0, 255).astype(np.uint8), mask
