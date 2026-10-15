"""
Which rotations of a model look the same as the model itself.

A plain rectangular box turned 180 degrees about any of its axes has the same
outline, the same 3D edges and the same MegaPose fit as before: geometry
alone cannot say which way round it is. Those rotations are the model's
(approximate) symmetries. Every pose found from geometry has a "twin" for
each of them — the same image fit, the object turned — and only something
that breaks the symmetry (a printed circle, a hole near one corner, a label)
can choose between the twins (pose/disambiguation.py).

Candidates tested: 180 degrees about each of the model's three principal
axes, and 90/270 degrees about an axis whose two cross extents match (a
square section). A candidate counts when the rotated surface lies on the
original one: mean two-way nearest-surface distance below `tolerance` of
the model's size. Approximate on purpose — a box with a small engraved
circle is still "symmetric" for the outline, which is exactly the case to
catch.

Rotations are about the model's origin (the tracked frame is recentred on the
bounding-box centre, so a box's centre). All matrices are 3x3, object frame:
a twin of T_camera_object is T_camera_object @ [S, 0; 0, 1].
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree


def surface_samples(vertices: np.ndarray, faces: np.ndarray, count: int = 4000, seed: int = 0) -> np.ndarray:
    """Points spread over the mesh surface by area (deterministic)."""
    v = np.asarray(vertices, float)
    f = np.asarray(faces, np.int64)
    if len(f) == 0:
        return v.copy()
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = float(areas.sum())
    if total <= 0:
        return v.copy()
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(f), size=count, p=areas / total)
    r1, r2 = rng.random(count), rng.random(count)
    s = np.sqrt(r1)
    return (1 - s)[:, None] * a[pick] + (s * (1 - r2))[:, None] * b[pick] + (s * r2)[:, None] * c[pick]


def _axis_sets(points: np.ndarray) -> list[np.ndarray]:
    """Axis triples (rows) to try turns about: the model's own axes (CAD parts
    are usually modelled axis-aligned), and the point cloud's principal
    directions when they differ (a part modelled at an angle). A cube's
    principal directions are arbitrary — its own axes still catch it."""
    sets = [np.eye(3)]
    centred = points - points.mean(axis=0)
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    distinct = singular.min() > 0 and np.min(np.diff(np.sort(singular))) > 0.02 * singular.max()
    if distinct and not np.all(np.abs(vt).max(axis=1) > 0.98):
        sets.append(vt)
    return sets


def _rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(axis, float) * np.radians(degrees))[0]


def find_symmetries(
    vertices: np.ndarray, faces: np.ndarray, tolerance: float = 0.01, samples: int = 4000
) -> list[np.ndarray]:
    """The model's rotational symmetries (3x3, object frame), identity first.
    `tolerance`: allowed mean surface mismatch as a share of the model's diagonal."""
    pts = surface_samples(vertices, faces, samples)
    if len(pts) < 10:
        return [np.eye(3)]
    diagonal = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) or 1.0
    tree = cKDTree(pts)
    candidates = []
    for axes in _axis_sets(pts):
        extent = np.ptp(pts @ axes.T, axis=0)
        candidates += [_rotation(axis, 180.0) for axis in axes]
        for i, axis in enumerate(axes):
            others = [extent[j] for j in range(3) if j != i]
            if abs(others[0] - others[1]) <= 0.02 * max(others):  # square section: quarter turns too
                candidates += [_rotation(axis, 90.0), _rotation(axis, 270.0)]

    # The same surface sampled twice is already this far apart (sampling
    # density): only a mismatch beyond that counts against a candidate.
    other = surface_samples(vertices, faces, samples, seed=1)
    other_tree = cKDTree(other)
    baseline = max(other_tree.query(pts)[0].mean(), tree.query(other)[0].mean())

    found = [np.eye(3)]
    for rotation in candidates:
        moved = pts @ rotation.T
        there = other_tree.query(moved)[0].mean()
        back = cKDTree(moved).query(other)[0].mean()
        if max(there, back) <= baseline + tolerance * diagonal:
            found.append(rotation)
    return _closed(found)


def _closed(rotations: list[np.ndarray]) -> list[np.ndarray]:
    """Add products (two 180-degree turns make the third), without duplicates."""
    out: list[np.ndarray] = []

    def add(r):
        if not any(np.allclose(r, o, atol=1e-6) for o in out):
            out.append(r)

    for r in rotations:
        add(r)
    changed = True
    while changed:
        changed = False
        for a in list(out):
            for b in list(out):
                p = a @ b
                if not any(np.allclose(p, o, atol=1e-6) for o in out):
                    out.append(p)
                    changed = True
    return out


def twins(t_co: np.ndarray, symmetries: list[np.ndarray]) -> list[np.ndarray]:
    """The pose and its symmetric twins: same image fit, object turned."""
    out = []
    for s in symmetries:
        twin = np.array(t_co, dtype=float)
        twin[:3, :3] = twin[:3, :3] @ s
        out.append(twin)
    return out


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    r = np.asarray(a)[:3, :3] @ np.asarray(b)[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def closest_twin(t_co: np.ndarray, reference: np.ndarray, symmetries: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """Of `t_co`'s twins, the one turned least from `reference` (and its index)."""
    candidates = twins(t_co, symmetries)
    angles = [rotation_angle_deg(c, reference) for c in candidates]
    best = int(np.argmin(angles))
    return candidates[best], best
