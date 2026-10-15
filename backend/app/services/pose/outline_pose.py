"""
Initial 6DoF pose from an object's outline alone: the rotation (and the
position along the viewing ray that makes its size right) at which the
model's projected silhouette best covers a given image region.

Why: MegaPose's coarse search matches rendered appearance, and on a plain,
untextured object (a matte black case) it can settle on a wrong viewpoint —
on the user's upright case it posed the model edge-on (silhouette 55%) even
given the right region. The outline of a box-like part pins down most of its
pose geometrically. The result seeds MegaPose's refiner and the 3D edge
alignment, which settle the rest.

Silhouettes are drawn as the convex hull of the model's projected hull
vertices (exact for convex parts, a close outline for box-like assemblies),
on a coarse grid around the region — about 1-2 s for a few thousand
rotations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import ConvexHull


@dataclass
class OutlinePose:
    t_co: np.ndarray  # 4x4, model (recentred) -> OpenCV camera
    iou: float  # silhouette vs the region's outline


def _random_rotations(count: int, seed: int) -> np.ndarray:
    """Uniformly distributed rotation matrices (from uniform unit quaternions)."""
    q = np.random.default_rng(seed).normal(size=(count, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1),
    ], axis=1)


def _small_rotations(count: int, max_deg: float, rng: np.random.Generator) -> np.ndarray:
    vecs = rng.normal(size=(count, 3))
    vecs *= (np.radians(max_deg) * rng.random(count) ** (1 / 3) / np.linalg.norm(vecs, axis=1))[:, None]
    return np.stack([cv2.Rodrigues(v)[0] for v in vecs])


class _Scorer:
    def __init__(self, hull: np.ndarray, outline: np.ndarray, k: np.ndarray, grid: int):
        self.hull, self.k = hull, k
        x1, y1 = outline.min(axis=0)
        x2, y2 = outline.max(axis=0)
        pad = 0.35 * max(x2 - x1, y2 - y1)
        self.origin = np.array([x1 - pad, y1 - pad])
        self.scale = grid / (max(x2 - x1, y2 - y1) + 2 * pad)
        size = int(np.ceil((max(x2 - x1, y2 - y1) + 2 * pad) * self.scale)) + 1
        self.target = np.zeros((size, size), np.uint8)
        cv2.fillPoly(self.target, [np.round((outline - self.origin) * self.scale).astype(np.int32)], 1)
        self.target_area = max(int(self.target.sum()), 1)
        self.area_px = float(cv2.contourArea(outline.astype(np.float32)))
        self.center = outline.mean(axis=0) if self.area_px <= 0 else self._centroid(outline)

    @staticmethod
    def _centroid(poly: np.ndarray) -> np.ndarray:
        m = cv2.moments(poly.astype(np.float32))
        return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]) if m["m00"] else poly.mean(axis=0)

    def _project(self, rotation: np.ndarray, t: np.ndarray) -> np.ndarray | None:
        cam = self.hull @ rotation.T + t
        if (cam[:, 2] <= 1e-3).any():
            return None
        px = cam @ self.k.T
        return px[:, :2] / px[:, 2:3]

    def place(self, rotation: np.ndarray) -> np.ndarray | None:
        """Translation putting the model's silhouette at the region's centre, at the region's size."""
        k_inv = np.linalg.inv(self.k)
        ray = k_inv @ np.array([*self.center, 1.0])
        t = ray * 0.5
        for _ in range(2):
            px = self._project(rotation, t)
            if px is None:
                return None
            hull = cv2.convexHull(px.astype(np.float32))
            area = float(cv2.contourArea(hull))
            if area <= 0 or self.area_px <= 0:
                return None
            t = t * np.sqrt(area / self.area_px)  # area ~ 1 / distance^2
            px = self._project(rotation, t)
            if px is None:
                return None
            shift = self.center - self._centroid(cv2.convexHull(px.astype(np.float32)).reshape(-1, 2))
            t = t + np.array([shift[0] / self.k[0, 0], shift[1] / self.k[1, 1], 0.0]) * t[2]
        return t

    def iou(self, rotation: np.ndarray, t: np.ndarray) -> float:
        px = self._project(rotation, t)
        if px is None:
            return 0.0
        drawn = np.zeros_like(self.target)
        hull = cv2.convexHull(np.round((px - self.origin) * self.scale).astype(np.int32))
        cv2.fillConvexPoly(drawn, hull, 1)
        inter = int((drawn & self.target).sum())
        union = int(drawn.sum()) + self.target_area - inter
        return inter / union if union else 0.0


def outline_pose(
    vertices: np.ndarray,
    outline_px: np.ndarray,
    camera_matrix: np.ndarray,
    allowed: Callable[[np.ndarray], bool] | None = None,
    rotations: int = 3000,
    grid: int = 96,
    seed: int = 0,
) -> OutlinePose | None:
    """The pose whose silhouette best covers the region `outline_px`
    ((N, 2) image pixels). `vertices`: the model, recentred, metres.
    `allowed(t_co)`: rejects poses (the station's resting side)."""
    outline = np.asarray(outline_px, float)
    if len(outline) < 3:
        return None
    pts = np.asarray(vertices, float)
    try:
        hull = pts[ConvexHull(pts).vertices]
    except Exception:  # flat / degenerate model: every vertex
        hull = pts
    k = np.asarray(camera_matrix, float)
    scorer = _Scorer(hull, outline, k, grid)
    if scorer.area_px <= 0:
        return None

    def pose(rotation, t):
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = rotation, t
        return out

    def evaluate(candidates):
        scored = []
        for rotation in candidates:
            t = scorer.place(rotation)
            if t is None:
                continue
            if allowed is not None and not allowed(pose(rotation, t)):
                continue
            scored.append((scorer.iou(rotation, t), rotation, t))
        return sorted(scored, key=lambda s: -s[0])

    ranked = evaluate(_random_rotations(rotations, seed))
    if not ranked:
        return None
    # Local search around the best few: coarse, then fine perturbations.
    rng = np.random.default_rng(seed + 1)
    best = ranked[:5]
    for max_deg in (12.0, 4.0):
        candidates = [b[1] for b in best]
        for _, rotation, _ in best:
            candidates += list(_small_rotations(40, max_deg, rng) @ rotation)
        best = evaluate(candidates)[:5]
    iou, rotation, t = best[0]
    return OutlinePose(pose(rotation, t), float(iou))
