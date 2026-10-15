"""
Model-based 3D edge alignment: refine a 6DoF pose so the model's own 3D
edges land on the edges in the camera image.

Why: the pose search picks among candidates by outline (silhouette), and the
tracker moves the pose by 2D image motion between corrections. Neither sees
tilt well — a flat box leaning 15 degrees has nearly the same outline, and a
truncated outline (object at the image border, a hand over one end) is
matched best by tilting. The model's INTERNAL edges — the border of the top
face, a lid seam, a step — do slide off their real counterparts when the pose
leans, outline or not. That is what this aligns (the approach of model-based
edge trackers such as RAPID / Drummond-Cipolla and industrial "model target"
tracking).

    EdgeModel (once per model): the mesh's sharp crease edges (adjacent faces
        meeting at more than `crease_deg`) and every edge's two face normals,
        sampled into 3D points.
    edge_image (per frame): Canny edges of the camera image, hand pixels
        removed, distance transform -> "how far to the nearest edge".
    align (per refinement): visible model edges at the current pose (crease
        edges with a front-facing side; silhouette edges between a front- and
        a back-facing face), projected; the 6DoF pose is optimised so they sit
        on image edges (robust least squares, outliers ignored).

Frames: points are in the tracked object's frame (recentred, metres) and
`t_co` maps them into the OpenCV camera — the same T_camera_object the
overlay is drawn from.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass
class EdgeModel:
    points: np.ndarray  # (P, 3) sample points along edges, object frame (m)
    normals_a: np.ndarray  # (P, 3) normal of one face at the point's edge
    normals_b: np.ndarray  # (P, 3) the other face's (== normals_a on an open boundary)
    crease: np.ndarray  # (P,) bool: a sharp edge, visible whenever one side faces the camera
    vertices: np.ndarray | None = None  # the mesh, for hidden-edge removal (None: facing test only)
    faces: np.ndarray | None = None

    @staticmethod
    def from_mesh(
        vertices: np.ndarray, faces: np.ndarray, crease_deg: float = 30.0, max_points: int = 6000
    ) -> EdgeModel:
        vertices = np.asarray(vertices, float)
        faces = np.asarray(faces, np.int64)
        # Weld vertices at the same position: exporters split them along seams,
        # which would make every seam look like an open boundary.
        extent = float(np.ptp(vertices, axis=0).max()) or 1.0
        keys = np.round(vertices / (extent * 1e-6)).astype(np.int64)
        _, welded = np.unique(keys, axis=0, return_inverse=True)
        welded = welded.reshape(-1)
        f = welded[faces]
        a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        normals = np.cross(b - a, c - a)
        lengths = np.linalg.norm(normals, axis=1)
        good = lengths > 1e-15
        f, faces, normals = f[good], faces[good], normals[good] / lengths[good, None]

        # Every edge (welded vertex pair) with the faces on each side.
        pairs = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        corners = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        owner = np.tile(np.arange(len(f)), 3)
        pairs_sorted = np.sort(pairs, axis=1)
        _, edge_id, counts = np.unique(pairs_sorted, axis=0, return_inverse=True, return_counts=True)
        edge_id = edge_id.reshape(-1)
        order = np.argsort(edge_id, kind="stable")
        first = np.r_[0, np.cumsum(counts)[:-1]]
        face_a = owner[order[first]]
        face_b = np.where(counts >= 2, owner[order[np.minimum(first + 1, len(order) - 1)]], face_a)
        ends = corners[order[first]]
        na, nb = normals[face_a], normals[face_b]
        cos = np.clip((na * nb).sum(axis=1), -1.0, 1.0)
        crease = (np.degrees(np.arccos(cos)) > crease_deg) | (counts == 1)

        # Keep crease edges, and smooth-surface edges only as silhouette
        # candidates (their two normals differ a little): skip coplanar ones,
        # which can never be an edge in the image.
        keep = crease | (cos < np.cos(np.radians(5.0)))
        p0, p1 = vertices[ends[keep, 0]], vertices[ends[keep, 1]]
        na, nb, crease = na[keep], nb[keep], crease[keep]
        seg = np.linalg.norm(p1 - p0, axis=1)
        if len(seg) == 0:
            return EdgeModel(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, bool), vertices, faces)
        # Evenly by length along all edges laid end to end: a long outline gets
        # its share, and a cluster of tiny detail edges (fasteners, internals)
        # only theirs — not one point each.
        ends_at = np.cumsum(seg)
        total = float(ends_at[-1])
        n = int(min(max_points, max(1, len(seg) * 50)))
        at = (np.arange(n) + 0.5) * (total / n)
        idx = np.minimum(np.searchsorted(ends_at, at), len(seg) - 1)
        t = ((at - (ends_at[idx] - seg[idx])) / np.maximum(seg[idx], 1e-12))[:, None]
        points = p0[idx] + np.clip(t, 0.0, 1.0) * (p1[idx] - p0[idx])
        return EdgeModel(points, na[idx], nb[idx], crease[idx], vertices, faces)


@dataclass
class EdgeImage:
    distance: np.ndarray  # (H, W) float32: px to the nearest image edge (capped), edges under hands removed
    unknown: np.ndarray | None  # (H, W) bool: under (around) a hand — model edges there can't be judged

    @property
    def shape(self) -> tuple[int, int]:
        return self.distance.shape


def edge_image(
    gray: np.ndarray, exclude: np.ndarray | None = None, cap_px: float = 80.0, low: float = 20.0, high: float = 40.0
) -> EdgeImage:
    """The camera image's edges as a distance field: distance (px, capped)
    from every pixel to the nearest image edge, with edges on `exclude`
    (hand pixels) removed and that area marked unknown.

    Fixed, low gradient thresholds (not ones scaled by the image brightness):
    the edges that pin down tilt — a dark case's top-face border, a lid seam —
    are faint steps of ~10 grey levels between two dark faces, and a light desk
    would push brightness-scaled thresholds far above them. Clutter edges this
    lets through are handled by the robust fit."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low, high)
    grown = None
    if exclude is not None and exclude.any():
        # Grown a little: the hand's own outline lands just outside its mask.
        grown = cv2.dilate(exclude.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        edges[grown] = 0
    if not edges.any():
        distance = np.full(gray.shape, cap_px, np.float32)
    else:
        distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
        distance = np.minimum(distance, cap_px)
        # Slightly smoothed so the optimiser sees a slope towards each edge.
        distance = cv2.GaussianBlur(distance, (5, 5), 0).astype(np.float32)
    # Under a hand nothing is known: model edges there neither count for nor
    # against a pose (they're left out), instead of all looking like misses.
    # The distance values there stay smooth, so the area doesn't push edges
    # away from it either.
    return EdgeImage(distance, grown)


def _sample(field: np.ndarray, px: np.ndarray, outside: float) -> np.ndarray:
    """Bilinear lookup of `field` at (N, 2) pixel positions; `outside` off the image."""
    h, w = field.shape
    x, y = px[:, 0], px[:, 1]
    inside = (x >= 0) & (y >= 0) & (x < w - 1) & (y < h - 1)
    out = np.full(len(px), outside, np.float32)
    if inside.any():
        xi, yi = x[inside], y[inside]
        x0, y0 = np.floor(xi).astype(int), np.floor(yi).astype(int)
        fx, fy = xi - x0, yi - y0
        top = field[y0, x0] * (1 - fx) + field[y0, x0 + 1] * fx
        bottom = field[y0 + 1, x0] * (1 - fx) + field[y0 + 1, x0 + 1] * fx
        out[inside] = top * (1 - fy) + bottom * fy
    return out


def _depth_buffer(model: EdgeModel, t_co: np.ndarray, k: np.ndarray, region: np.ndarray, cells: int):
    """Nearest mesh depth over the pixel box `region` (x1, y1, x2, y2) on a
    grid of about `cells` along its longer side: (origin, scale, depth)."""
    from app.services.component_inspection.projection import _rasterize  # the shared vectorised rasteriser

    x1, y1, x2, y2 = (float(v) for v in region)
    scale = cells / max(x2 - x1, y2 - y1, 1.0)
    grid_w, grid_h = int(np.ceil((x2 - x1) * scale)) + 1, int(np.ceil((y2 - y1) * scale)) + 1
    depth = np.full((grid_h, grid_w), np.inf)
    cam = model.vertices @ t_co[:3, :3].T + t_co[:3, 3]
    faces = model.faces[(cam[model.faces, 2] > 1e-3).all(axis=1)]
    # Only faces turned towards the camera: on a closed part a face turned
    # away is never the nearest surface (halves the work).
    a, b, c = cam[faces[:, 0]], cam[faces[:, 1]], cam[faces[:, 2]]
    faces = faces[(np.cross(b - a, c - a) * -(a + b + c)).sum(axis=1) > 0]
    px = cam @ k.T
    px = px[:, :2] / np.maximum(px[:, 2:3], 1e-6)
    gx, gy = (px[:, 0] - x1) * scale, (px[:, 1] - y1) * scale
    tri_x, tri_y = gx[faces], gy[faces]
    boxes = np.stack([
        np.floor(tri_x.min(axis=1)), np.floor(tri_y.min(axis=1)),
        np.ceil(tri_x.max(axis=1)) + 1, np.ceil(tri_y.max(axis=1)) + 1,
    ], axis=1)
    boxes[:, 0:2] = np.maximum(boxes[:, 0:2], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], grid_w)
    boxes[:, 3] = np.minimum(boxes[:, 3], grid_h)
    ok = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    faces, tri_x, tri_y, boxes = faces[ok], tri_x[ok], tri_y[ok], boxes[ok].astype(np.int64)
    inv_z = 1.0 / np.maximum(cam[:, 2], 1e-3)
    extent = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    # Batches by box size (powers of two), so a few big triangles don't make
    # every small one pay for their area, and big ones still go many at a time.
    bounds = [0, 4]
    while bounds[-1] < max(grid_w, grid_h) + 2:
        bounds.append(bounds[-1] * 2)
    for low, high in zip(bounds[:-1], bounds[1:]):
        group = np.flatnonzero((extent > low) & (extent <= high))
        chunk = max(1, 400_000 // (high * high))
        for start in range(0, len(group), chunk):
            batch = group[start : start + chunk]
            side = int(extent[batch].max())
            face, x, y, w0, w1 = _rasterize(tri_x[batch], tri_y[batch], boxes[batch], side, grid_w, grid_h)
            if len(face) == 0:
                continue
            tri = faces[batch][face]
            # Perspective-correct depth: 1/z is linear in screen space.
            iz = w0 * inv_z[tri[:, 0]] + w1 * inv_z[tri[:, 1]] + (1 - w0 - w1) * inv_z[tri[:, 2]]
            np.minimum.at(depth, (y, x), 1.0 / np.maximum(iz, 1e-9))
    return (x1, y1), scale, depth


def _thin(mask: np.ndarray, budget: int) -> np.ndarray:
    """At most `budget` of the True entries, spread evenly (by sample order = along the edges)."""
    where = np.flatnonzero(mask)
    if len(where) <= budget:
        return mask
    out = np.zeros_like(mask)
    out[where[np.linspace(0, len(where) - 1, budget).astype(int)]] = True
    return out


def _facing(model: EdgeModel, t_co: np.ndarray) -> np.ndarray:
    """Sample points that are image edges by orientation alone: crease edges
    with a side towards the camera, silhouette edges (one side each way)."""
    r, t = t_co[:3, :3], t_co[:3, 3]
    to_camera = -(model.points @ r.T + t)
    fa = ((model.normals_a @ r.T) * to_camera).sum(axis=1) > 0
    fb = ((model.normals_b @ r.T) * to_camera).sum(axis=1) > 0
    return (to_camera[:, 2] < -1e-3) & ((model.crease & (fa | fb)) | (fa != fb))


def visible_edge_points(
    model: EdgeModel, t_co: np.ndarray, camera_matrix: np.ndarray | None = None, cells: int = 120,
    budget: int | None = None,
) -> np.ndarray:
    """Which sample points are edges in the image at this pose: crease edges
    with at least one side facing the camera, and silhouette edges (one side
    facing, one facing away) — minus, given the camera and the mesh, those
    hidden behind other parts of the object (a lid edge behind a raised
    block, the far rim of a hole)."""
    cam = model.points @ t_co[:3, :3].T + t_co[:3, 3]
    visible = _facing(model, t_co)
    if camera_matrix is None or model.vertices is None or len(model.faces) == 0 or not visible.any():
        return visible
    k = np.asarray(camera_matrix, float)
    px = cam[visible] @ k.T
    px = px[:, :2] / px[:, 2:3]
    region = np.array([px[:, 0].min() - 4, px[:, 1].min() - 4, px[:, 0].max() + 4, px[:, 1].max() + 4])
    (ox, oy), scale, depth = _depth_buffer(model, t_co, k, region, cells)
    # An edge point is seen if some surface right next to it is at its depth
    # or behind it (its own faces end there); a hidden point has nearer
    # surfaces all around it. Max over 3x3 cells = the farthest neighbour.
    far = cv2.dilate(np.where(np.isfinite(depth), depth, 1e9).astype(np.float32), np.ones((3, 3), np.uint8))
    gx = np.clip(((px[:, 0] - ox) * scale).astype(int), 0, depth.shape[1] - 1)
    gy = np.clip(((px[:, 1] - oy) * scale).astype(int), 0, depth.shape[0] - 1)
    z = cam[visible, 2]
    seen = z <= far[gy, gx] + np.maximum(0.002, 0.005 * z)
    out = visible.copy()
    out[np.flatnonzero(visible)[~seen]] = False
    return out if budget is None else _thin(out, budget)


@dataclass
class EdgeFit:
    t_co: np.ndarray
    score: float  # fit to the image edges, 0..1 (both ways; see _fit_score)
    support: float  # share of visible model edge points within `near_px` of an image edge
    coverage: float  # share of image edges on the object that a model edge explains
    mean_px: float  # mean distance of visible model edge points to image edges (misses count as 30)
    points: int  # visible model edge points
    improved: bool  # the refined pose fits the edges clearly better than the start did


SCORE_CAP_PX = 30.0
FIT_POINTS = 1500  # visible edge points used per pose (thinned evenly beyond this)


def _project(model: EdgeModel, t_co: np.ndarray, k: np.ndarray, mask: np.ndarray) -> np.ndarray:
    cam = model.points[mask] @ t_co[:3, :3].T + t_co[:3, 3]
    px = cam @ k.T
    return px[:, :2] / np.maximum(px[:, 2:3], 1e-6)


def _unknown_at(edges: EdgeImage, px: np.ndarray) -> np.ndarray:
    if edges.unknown is None:
        return np.zeros(len(px), bool)
    h, w = edges.unknown.shape
    x = np.clip(np.round(px[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(px[:, 1]).astype(int), 0, h - 1)
    return edges.unknown[y, x]


def _fit_score(model, t_co, k, edges: EdgeImage, near_px):
    """(score, support, coverage, mean_px, visible points) of a pose.

    Both ways, so a pose can't score well by turning edges out of view:
    support = the model's visible edges lying on image edges; coverage = the
    image edges on the object's area lying near some visible model edge
    (an edge the model can't explain counts against it)."""
    mask = visible_edge_points(model, t_co, k, budget=FIT_POINTS)
    n = int(mask.sum())
    if n == 0:
        return 0.0, 0.0, 0.0, SCORE_CAP_PX, 0
    distance = edges.distance
    px = _project(model, t_co, k, mask)
    known = ~_unknown_at(edges, px)
    if known.sum() < 0.25 * n:  # mostly under a hand: can't judge
        return 0.0, 0.0, 0.0, SCORE_CAP_PX, 0
    d = np.minimum(_sample(distance, px[known], outside=SCORE_CAP_PX), SCORE_CAP_PX)
    support = float((d <= near_px).mean())
    # The object's area in the image: the hull of the projected mesh.
    cam = model.vertices @ t_co[:3, :3].T + t_co[:3, 3] if model.vertices is not None else None
    if cam is None or (cam[:, 2] <= 1e-3).any():
        return support, support, support, float(d.mean()), n
    hull_px = cam @ k.T
    hull_px = hull_px[:, :2] / hull_px[:, 2:3]
    h, w = distance.shape
    x1, y1 = np.floor(np.maximum(hull_px.min(axis=0) - 8, 0)).astype(int)
    x2, y2 = np.ceil(np.minimum(hull_px.max(axis=0) + 8, [w, h])).astype(int)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return support, support, 0.0, float(d.mean()), n
    area = np.zeros((y2 - y1, x2 - x1), np.uint8)
    cv2.fillConvexPoly(area, cv2.convexHull(np.round(hull_px - [x1, y1]).astype(np.int32)), 1)
    area = cv2.erode(area, np.ones((5, 5), np.uint8))  # not the clutter just outside the outline
    image_edges = (distance[y1:y2, x1:x2] <= 0.75) & (area > 0)
    if edges.unknown is not None:
        image_edges &= ~edges.unknown[y1:y2, x1:x2]
    if image_edges.sum() < 20:
        coverage = 0.0
    else:
        drawn = np.zeros_like(area)
        for x, y in np.round(px - [x1, y1]).astype(int):
            if 0 <= x < drawn.shape[1] and 0 <= y < drawn.shape[0]:
                drawn[y, x] = 1
        reach = int(2 * near_px + 1)
        drawn = cv2.dilate(drawn, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1)))
        coverage = float((drawn[image_edges] > 0).mean())
    return 0.5 * (support + coverage), support, coverage, float(d.mean()), n


def _refine(model, t0, k, edges: EdgeImage, near_px, min_points, coarse_rounds, fine_rounds, max_nfev):
    """Local least-squares refinement from t0: the pose and its parameters
    (rotation vector, translation in units of the distance)."""
    field = edges.distance
    cap = float(field.max())
    # Slope of the distance field, for the Jacobian.
    grad_x = cv2.Sobel(field, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    grad_y = cv2.Sobel(field, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    fx, fy = k[0, 0], k[1, 1]
    depth = float(t0[2, 3])

    def pose_of(x):
        rot, _ = cv2.Rodrigues(x[:3])
        out = t0.copy()
        out[:3, :3] = rot @ t0[:3, :3]
        out[:3, 3] = t0[:3, 3] + x[3:] * depth  # translation in units of the distance: well scaled
        return out

    # Coarse to fine: first with a wide basin (distances up to `cap`, a loose
    # robust scale) so edges tens of pixels off are still pulled in, then
    # tight so the result sits on the edges themselves. Which edges are
    # visible depends on the pose: re-checked after every stage, and fixed
    # within one (so a stage can't shed the edges it fits badly).
    x = np.zeros(6)
    seen = visible_edge_points(model, t0, k)
    hidden = _facing(model, t0) & ~seen  # behind other geometry: stays so over the small moves here
    visible = _thin(seen, FIT_POINTS)
    if visible.sum() < min_points:
        return t0, x
    stages = [(cap, max(near_px, cap / 4))] * coarse_rounds + [(min(cap, 4 * near_px), near_px)] * fine_rounds
    for stage_cap, scale in stages:
        pts = model.points[visible]
        # Points under a hand at the start of the stage sit it out (a constant
        # residual would still pull nothing, but they'd blur the robust scale).
        pts = pts[~_unknown_at(edges, _project(model, pose_of(x), k, visible))]
        if len(pts) < min_points:
            break

        def residuals(v, pts=pts, stage_cap=stage_cap):
            pose = pose_of(v)
            cam = pts @ pose[:3, :3].T + pose[:3, 3]
            px = cam @ k.T
            px = px[:, :2] / np.maximum(px[:, 2:3], 1e-6)
            return np.minimum(_sample(field, px, outside=stage_cap), stage_cap)

        def jacobian(v, pts=pts, stage_cap=stage_cap):
            pose = pose_of(v)
            rotated = pts @ pose[:3, :3].T  # about the object's origin
            cam = rotated + pose[:3, 3]
            z = np.maximum(cam[:, 2], 1e-6)
            u = fx * cam[:, 0] / z + k[0, 2]
            w = fy * cam[:, 1] / z + k[1, 2]
            px = np.stack([u, w], axis=1)
            d = _sample(field, px, outside=stage_cap)
            live = d < stage_cap  # capped or off-image: flat, no pull
            gu = _sample(grad_x, px, outside=0.0) * live
            gv = _sample(grad_y, px, outside=0.0) * live
            # d residual / d camera point, through the projection.
            dc = np.stack([gu * fx / z, gv * fy / z, -(gu * fx * cam[:, 0] + gv * fy * cam[:, 1]) / z**2], axis=1)
            # Rotation (small-angle, applied on the left): d cam = -[rotated]x dw; translation: depth * ds.
            d_rot = np.cross(rotated, dc)
            return np.hstack([d_rot, dc * depth])

        x = least_squares(residuals, x, jac=jacobian, loss="soft_l1", f_scale=scale, max_nfev=max_nfev).x
        next_visible = _thin(_facing(model, pose_of(x)) & ~hidden, FIT_POINTS)
        if next_visible.sum() >= min_points:
            visible = next_visible
    return pose_of(x), x


def _turned(t_co: np.ndarray, axis: int, degrees: float) -> np.ndarray:
    """t_co turned about a camera axis through the object's origin."""
    rotvec = np.zeros(3)
    rotvec[axis] = np.radians(degrees)
    rot, _ = cv2.Rodrigues(rotvec)
    out = t_co.copy()
    out[:3, :3] = rot @ t_co[:3, :3]
    return out


def align(
    model: EdgeModel,
    t_co: np.ndarray,
    camera_matrix: np.ndarray,
    edges: EdgeImage,
    near_px: float = 3.0,
    min_points: int = 30,
    max_rotation_deg: float = 25.0,
    max_translation_frac: float = 0.25,
    tilt_starts_deg: float = 8.0,
    min_gain: float = 0.03,
    coarse_rounds: int = 2,
    fine_rounds: int = 2,
    max_nfev: int = 40,
) -> EdgeFit | None:
    """Refine `t_co` so the model's visible 3D edges land on image edges.

    Tilt is what the outline can't pin down, so besides the given pose the
    refinement also starts from it tilted by +-`tilt_starts_deg` about the
    camera's x and y axes, and keeps the best result by _fit_score. None when
    too few edges are visible to judge. The result is only `improved` when it
    scores at least `min_gain` better than the start and stays within the
    given rotation / translation (a fraction of the distance) of it — a
    refinement must not jump to some other set of edges."""
    k = np.asarray(camera_matrix, float)
    t0 = np.asarray(t_co, float)
    start_score, start_support, start_coverage, start_mean, start_n = _fit_score(model, t0, k, edges, near_px)
    if start_n < min_points:
        return None
    starts = [t0]
    if tilt_starts_deg > 0:
        starts += [_turned(t0, axis, sign * tilt_starts_deg) for axis in (0, 1) for sign in (1, -1)]
    best = None
    for start in starts:
        refined, _ = _refine(model, start, k, edges, near_px, min_points, coarse_rounds, fine_rounds, max_nfev)
        rel = refined[:3, :3] @ t0[:3, :3].T
        rotation_deg = float(np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1))))
        translation_frac = float(np.linalg.norm(refined[:3, 3] - t0[:3, 3]) / max(abs(t0[2, 3]), 1e-6))
        if rotation_deg > max_rotation_deg or translation_frac > max_translation_frac:
            continue
        result = _fit_score(model, refined, k, edges, near_px)
        if result[4] >= min_points and (best is None or result[0] > best[1][0]):
            best = (refined, result)
    if best is None or best[1][0] < start_score + min_gain:
        return EdgeFit(t0, start_score, start_support, start_coverage, start_mean, start_n, False)
    refined, (score, support, coverage, mean, n) = best
    return EdgeFit(refined, score, support, coverage, mean, n, True)
