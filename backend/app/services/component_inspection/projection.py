"""
Where each component is in the camera image, at the AR session's tracked pose.

Frames (docs/coordinates.md): component geometry is in the tracked object's
frame (registry.py); `t_co` maps it into the OpenCV camera frame (x right,
y down, z forward, metres) — the same T_camera_object the live session
renders the overlay from and the part checks use. Projection is pinhole with
the session's camera matrix; lens distortion is not modelled, matching the
rest of the pose pipeline (the overlay is drawn without it too).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.services.component_inspection.registry import ComponentGeometry, SceneGeometry

NEAR_M = 1e-3  # geometry closer than this to the camera (or behind it) isn't drawn


@dataclass
class Projection:
    """One component at one pose. Pixel coordinates are the frame's own."""

    behind_camera: bool  # some of it is behind/at the camera: not judged
    center_px: tuple[float, float] | None  # its 3D bounding-box centre, projected
    depth_m: float  # distance along the optical axis to that centre
    in_frame: float  # fraction of its projected vertices inside the image
    roi: tuple[int, int, int, int] | None  # x1, y1, x2, y2 (exclusive) of the in-frame mask
    mask: np.ndarray | None  # bool, roi-sized: the pixels the component covers
    area_px: int  # in-frame pixels it covers (ignoring occlusion)
    facing: float  # fraction of its surface area facing the camera (by triangle winding)
    view_dir: np.ndarray  # unit vector from the component towards the camera, object frame

    @property
    def outline(self) -> list[tuple[float, float]]:
        """The covered pixels' outer outline in frame coordinates, for drawing."""
        if self.mask is None or self.roi is None:
            return []
        contours, _ = cv2.findContours(self.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        outline = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True)
        x0, y0 = self.roi[0], self.roi[1]
        return [(float(p[0][0] + x0), float(p[0][1] + y0)) for p in outline]


def to_camera(vertices: np.ndarray, t_co: np.ndarray) -> np.ndarray:
    return vertices @ t_co[:3, :3].T + t_co[:3, 3]


def to_pixels(cam: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    px = cam @ np.asarray(camera_matrix, float).T
    return px[:, :2] / px[:, 2:3]


def project_component(
    geometry: ComponentGeometry, t_co: np.ndarray, camera_matrix: np.ndarray, shape: tuple[int, int]
) -> Projection:
    height, width = shape
    cam = to_camera(geometry.vertices, t_co)
    lo, hi = geometry.bounds
    center_cam = to_camera(((lo + hi) / 2)[None], t_co)[0]
    # Camera position in the object frame, for the viewing direction.
    camera_in_object = -t_co[:3, :3].T @ t_co[:3, 3]
    view = camera_in_object - (lo + hi) / 2
    view_dir = view / (np.linalg.norm(view) or 1.0)

    behind = bool((cam[:, 2] <= NEAR_M).any())
    if behind:
        return Projection(True, None, float(center_cam[2]), 0.0, None, None, 0, 0.0, view_dir)

    px = to_pixels(cam, camera_matrix)
    inside = (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)
    center = to_pixels(center_cam[None], camera_matrix)[0]

    # Facing: triangles whose front (counter-clockwise, glTF's convention)
    # points at the camera, weighted by area.
    a, b, c = cam[geometry.faces[:, 0]], cam[geometry.faces[:, 1]], cam[geometry.faces[:, 2]]
    normals = np.cross(b - a, c - a)
    areas = np.linalg.norm(normals, axis=1)
    towards = (normals * -((a + b + c) / 3)).sum(axis=1) > 0
    facing = float(areas[towards].sum() / areas.sum()) if areas.sum() > 0 else 0.0

    x1 = int(np.clip(np.floor(px[:, 0].min()), 0, width))
    y1 = int(np.clip(np.floor(px[:, 1].min()), 0, height))
    x2 = int(np.clip(np.ceil(px[:, 0].max()) + 1, 0, width))
    y2 = int(np.clip(np.ceil(px[:, 1].max()) + 1, 0, height))
    if x2 <= x1 or y2 <= y1:
        return Projection(False, (float(center[0]), float(center[1])), float(center_cam[2]), 0.0, None, None, 0, facing, view_dir)

    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    local = np.round(px - (x1, y1)).astype(np.int32)
    # One triangle at a time: fillPoly over all of them uses an even-odd rule,
    # so overlapping front/back faces would cancel into holes.
    for triangle in local[geometry.faces]:
        cv2.fillConvexPoly(mask, triangle, 1)
    mask_bool = mask.astype(bool)
    rows, cols = np.flatnonzero(mask_bool.any(axis=1)), np.flatnonzero(mask_bool.any(axis=0))
    if len(rows) == 0:
        return Projection(False, (float(center[0]), float(center[1])), float(center_cam[2]), float(inside.mean()), None, None, 0, facing, view_dir)
    # Tighten the roi to the pixels actually covered.
    roi = (x1 + int(cols[0]), y1 + int(rows[0]), x1 + int(cols[-1]) + 1, y1 + int(rows[-1]) + 1)
    mask_bool = mask_bool[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    return Projection(
        behind_camera=False,
        center_px=(float(center[0]), float(center[1])),
        depth_m=float(center_cam[2]),
        in_frame=float(inside.mean()),
        roi=roi,
        mask=mask_bool,
        area_px=int(mask_bool.sum()),
        facing=facing,
        view_dir=view_dir,
    )


@dataclass
class CadVisibility:
    """Which CAD surface is nearest the camera around the inspected
    components, from a small software depth buffer (cad_visibility)."""

    origin: tuple[int, int]  # frame pixel at grid (0, 0)
    scale: float  # grid cells per frame pixel
    owner: np.ndarray  # (grid_h, grid_w): component index nearest there, -1 other geometry, -2 nothing
    fractions: dict[int, float]  # per target: share of its own pixels where it is the nearest surface

    def visible_mask(self, index: int, projection: Projection) -> np.ndarray | None:
        """The component's own pixels (projection.mask) where it is the
        nearest surface, at full resolution — what can actually be seen of it."""
        if projection.roi is None or projection.mask is None:
            return None
        x1, y1, x2, y2 = projection.roi
        gx = np.clip(((np.arange(x1, x2) + 0.5 - self.origin[0]) * self.scale).astype(int), 0, self.owner.shape[1] - 1)
        gy = np.clip(((np.arange(y1, y2) + 0.5 - self.origin[1]) * self.scale).astype(int), 0, self.owner.shape[0] - 1)
        return projection.mask & (self.owner[np.ix_(gy, gx)] == index)


def _rasterize(
    xs: np.ndarray, ys: np.ndarray, boxes: np.ndarray, side: int, grid_w: int, grid_h: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pixel candidates of a batch of triangles whose boxes fit in side x side
    cells: (face row in batch, pixel x, pixel y, w0, w1) for every covered
    pixel centre — the vectorised inner loop of a rasteriser."""
    offsets = np.arange(side)
    px = boxes[:, 0, None, None] + offsets[None, None, :]  # (F, 1, side)
    py = boxes[:, 1, None, None] + offsets[None, :, None]  # (F, side, 1)
    px, py = np.broadcast_arrays(px, py)
    cx, cy = px + 0.5, py + 0.5
    x0, x1, x2 = (xs[:, k, None, None] for k in range(3))
    y0, y1, y2 = (ys[:, k, None, None] for k in range(3))
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    with np.errstate(divide="ignore", invalid="ignore"):
        w0 = ((x1 - cx) * (y2 - cy) - (x2 - cx) * (y1 - cy)) / area
        w1 = ((x2 - cx) * (y0 - cy) - (x0 - cx) * (y2 - cy)) / area
    w2 = 1.0 - w0 - w1
    inside = (
        (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9) & (np.abs(area) > 1e-12)
        & (px < boxes[:, 2, None, None]) & (py < boxes[:, 3, None, None]) & (px < grid_w) & (py < grid_h)
    )
    face, row, col = np.nonzero(inside)
    return face, px[face, row, col], py[face, row, col], w0[face, row, col], w1[face, row, col]


def cad_visibility(
    scene: SceneGeometry,
    targets: dict[int, tuple[int, int, int, int]],
    t_co: np.ndarray,
    camera_matrix: np.ndarray,
    shape: tuple[int, int],
    depth_bias_m: float = 5e-4,
    max_side_px: int = 200,
    max_faces: int = 60000,
) -> CadVisibility | None:
    """For each target component (index into the registry -> its roi): where,
    and on what share of its own pixels, it is the nearest surface of the
    whole CAD assembly. 1 = in plain view; less = partly hidden behind other
    parts (or, for a ring or an open cap, its own far side behind the part
    it sits on).

    A small software depth buffer over just the targets' region, rasterised
    in vectorised batches. Component faces are pulled `depth_bias_m` towards
    the camera so a part sitting flush in a surface (a countersunk screw
    head) isn't hidden by that surface through depth ties. Returns None when
    the assembly is too heavy to rasterise here (more than `max_faces`
    triangles in the region): the caller then can't use CAD occlusion and
    must say so."""
    height, width = shape
    if not targets or len(scene.faces) == 0:
        return CadVisibility((0, 0), 1.0, np.full((1, 1), -2, np.int64), {})
    rx1 = max(0, min(r[0] for r in targets.values()) - 2)
    ry1 = max(0, min(r[1] for r in targets.values()) - 2)
    rx2 = min(width, max(r[2] for r in targets.values()) + 2)
    ry2 = min(height, max(r[3] for r in targets.values()) + 2)
    if rx2 <= rx1 or ry2 <= ry1:
        return CadVisibility((rx1, ry1), 1.0, np.full((1, 1), -2, np.int64), {i: 0.0 for i in targets})
    # Sample finely enough that the smallest target still spans some cells
    # (by its longer side: a thin ring is wide even when it is few pixels tall).
    smallest = max(1, min(max(r[2] - r[0], r[3] - r[1]) for r in targets.values()))
    scale = min(1.0, max(max_side_px / max(rx2 - rx1, ry2 - ry1), 16.0 / smallest))
    grid_w, grid_h = max(1, int(np.ceil((rx2 - rx1) * scale))), max(1, int(np.ceil((ry2 - ry1) * scale)))

    cam = to_camera(scene.vertices, t_co)
    faces = scene.faces
    usable = (cam[faces, 2] > NEAR_M).all(axis=1)
    px = np.zeros((len(cam), 2))
    in_front = cam[:, 2] > NEAR_M
    px[in_front] = to_pixels(cam[in_front], camera_matrix)
    gx = (px[:, 0] - rx1) * scale
    gy = (px[:, 1] - ry1) * scale
    tri_x, tri_y = gx[faces], gy[faces]
    boxes = np.stack([
        np.floor(tri_x.min(axis=1)), np.floor(tri_y.min(axis=1)),
        np.ceil(tri_x.max(axis=1)) + 1, np.ceil(tri_y.max(axis=1)) + 1,
    ], axis=1)
    boxes[:, 0:2] = np.maximum(boxes[:, 0:2], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], grid_w)
    boxes[:, 3] = np.minimum(boxes[:, 3], grid_h)
    candidates = np.flatnonzero(usable & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
    if len(candidates) > max_faces:
        return None
    boxes = boxes.astype(np.int64)

    inv_z = 1.0 / np.maximum(cam[:, 2], NEAR_M)
    pixel_sets, depth_sets, label_sets = [], [], []
    extent = np.maximum(boxes[candidates, 2] - boxes[candidates, 0], boxes[candidates, 3] - boxes[candidates, 1])
    # Batches by box size, so a few big triangles don't make every small one pay for their area.
    for low, high in ((0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 4096)):
        in_group = (extent > low) & (extent <= high)
        group, group_extent = candidates[in_group], extent[in_group]
        chunk = max(1, 400_000 // (high * high))
        for start in range(0, len(group), chunk):
            batch = group[start : start + chunk]
            side = int(group_extent[start : start + chunk].max())  # this batch's own size, not the bucket's
            face, x, y, w0, w1 = _rasterize(tri_x[batch], tri_y[batch], boxes[batch], side, grid_w, grid_h)
            if len(face) == 0:
                continue
            f = batch[face]
            v = faces[f]
            surface = w0 * inv_z[v[:, 0]] + w1 * inv_z[v[:, 1]] + (1.0 - w0 - w1) * inv_z[v[:, 2]]
            labels = scene.labels[f]
            if depth_bias_m > 0:
                biased = 1.0 / np.maximum(1.0 / np.maximum(surface, 1e-9) - depth_bias_m, NEAR_M)
                surface = np.where(labels >= 0, biased, surface)
            pixel_sets.append(y * grid_w + x)
            depth_sets.append(surface)
            label_sets.append(labels)

    owner = np.full(grid_h * grid_w, -2, np.int64)
    fractions = {i: 0.0 for i in targets}
    if pixel_sets:
        pixels, depths, labels = np.concatenate(pixel_sets), np.concatenate(depth_sets), np.concatenate(label_sets)
        # Nearest surface per pixel: sort by pixel, then by 1/z; the last of each pixel wins.
        order = np.lexsort((depths, pixels))
        pixels, labels = pixels[order], labels[order]
        last = np.r_[pixels[1:] != pixels[:-1], True]
        owner[pixels[last]] = labels[last]
        for i in targets:
            own = np.unique(pixels[labels == i])
            if len(own):
                fractions[i] = float((owner[own] == i).mean())
    return CadVisibility((rx1, ry1), scale, owner.reshape(grid_h, grid_w), fractions)
