"""
Distinctive features of a model — a printed circle, a hole near one corner,
a connector — that tell its symmetric orientations apart, and how strongly
the camera image shows each one where a given pose puts it.

Defined per model in the component mapping file
(data/component_registry/<model_id>.json), next to "components":

    "poseFeatures": [
      {"featureId": "UR_CIRCLE", "type": "circle", "nodeIndex": 4},
      {"featureId": "UR_MARK", "type": "circle",
       "position": [0.04, 0.035, 0.07], "normal": [0, 1, 0], "radius": 0.008},
      {"featureId": "PORT", "type": "node", "componentId": "CONN_001"}
    ]

  type "circle": a round mark or hole. From a GLB node (its bounding box: centre,
      thinnest axis as the normal, radius from the other two), or given by
      `position` / `normal` / `radius` in the GLB's own units and axes — for
      marks inside one combined mesh, or printed on a plain face
      (`positionM` / `radiusM`: the same in the tracked frame, metres).
  type "node": any separate part (screw, connector, label patch): its outline
      at the pose.
  Optional: "enabled" (default true), "requiredForInitialization" (default
  false: a lock with this feature unconfirmed is refused), "weight" (1).

Identity, not appearance: a feature is only ever looked for where the pose
puts THIS feature (model-local position projected), so a look-alike circle
elsewhere doesn't count, and under a twin pose the same feature lands
somewhere else — which is what separates the twins.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from app.services.pose.edge_align import EdgeImage, EdgeModel, _depth_buffer

FeatureType = Literal["circle", "node"]
FeatureStatus = Literal["detected", "not_detected", "occluded", "hidden", "facing_away", "outside"]
MEASURABLE: tuple[FeatureStatus, ...] = ("detected", "not_detected")


@dataclass
class PoseFeature:
    feature_id: str
    type: FeatureType
    center: np.ndarray  # (3,) tracked frame, metres
    normal: np.ndarray | None = None  # (3,) unit, outward (circle)
    radius: float = 0.0  # metres (circle)
    vertices: np.ndarray | None = None  # (N, 3) node geometry (node), tracked frame
    enabled: bool = True
    required_for_initialization: bool = False
    weight: float = 1.0

    def contour(self, samples: int = 48) -> np.ndarray:
        """3D points along the feature's outline (circle: its rim)."""
        if self.type == "circle" and self.normal is not None and self.radius > 0:
            n = self.normal / np.linalg.norm(self.normal)
            helper = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
            u = np.cross(n, helper)
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
            angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
            return self.center + self.radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
        if self.vertices is not None and len(self.vertices):
            return self.vertices
        return self.center[None, :]


@dataclass
class FeatureObservation:
    feature_id: str
    status: FeatureStatus
    evidence: float = 0.0  # 0..1: how strongly the image shows it where the pose puts it
    support: float = 0.0  # share of its projected outline on image edges
    contrast: float = 0.0  # inside vs around it, in local noise units
    predicted_center_px: tuple[float, float] | None = None
    predicted_contour_px: list[tuple[float, float]] = field(default_factory=list)
    observed_center_px: tuple[float, float] | None = None  # the circle actually found nearby
    residual_px: float | None = None  # predicted vs observed centre


# ── configuration ────────────────────────────────────────────────────────


def _circle_from_vertices(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre, outward normal (thinnest axis), radius of a disc-like part."""
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    center = (lo + hi) / 2
    extent = hi - lo
    thin = int(np.argmin(extent))
    normal = np.zeros(3)
    normal[thin] = 1.0
    if center @ normal < 0:  # outward: away from the model's centre
        normal = -normal
    others = [extent[i] for i in range(3) if i != thin]
    return center, normal, float(np.mean(others) / 2)


def load_pose_features(mapping_path: Path | None, glb: bytes | None, scale: float) -> tuple[list[PoseFeature], list[str]]:
    """(features, warnings) from the mapping file's "poseFeatures"; none when absent."""
    if mapping_path is None or not Path(mapping_path).exists():
        return [], []
    data = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    entries = data.get("poseFeatures") or []
    if not isinstance(entries, list):
        raise ValueError(f"{mapping_path}: 'poseFeatures' must be a list")
    if not entries:
        return [], []

    nodes, meshes, center = [], {}, np.zeros(3)
    if glb is not None:
        from app.services.component_inspection.registry import assembly_center, sanitize_node_name
        from app.services.model3d.glb_inspect import glb_nodes, glb_part_meshes

        nodes = glb_nodes(glb)
        meshes = glb_part_meshes(glb)
        center = assembly_center(meshes) if meshes else np.zeros(3)
    components = {c.get("id"): c for c in data.get("components") or [] if isinstance(c, dict)}

    def node_vertices(entry: dict) -> np.ndarray | None:
        ref = dict(entry)
        if "componentId" in entry:
            comp = components.get(entry["componentId"])
            if comp is None:
                return None
            ref = comp
        index = ref.get("nodeIndex")
        name = ref.get("meshName")
        if index is None and name:
            hits = [n for n in nodes if name in (n.name, sanitize_node_name(n.name))]
            if len(hits) != 1:
                return None
            index = hits[0].node_index
        if index is None or index not in meshes:
            return None
        return (meshes[index].vertices - center) * scale

    features, warnings = [], []
    for entry in entries:
        fid = entry.get("featureId") or entry.get("feature_id") or entry.get("id")
        kind = entry.get("type", "circle")
        if not fid or kind not in ("circle", "node"):
            warnings.append(f"Pose feature {fid!r}: needs a featureId and type 'circle' or 'node'")
            continue
        common = dict(
            enabled=bool(entry.get("enabled", True)),
            required_for_initialization=bool(entry.get("requiredForInitialization", False)),
            weight=float(entry.get("weight", 1.0)),
        )
        if any(k in entry for k in ("position", "positionM")):
            if "positionM" in entry:
                c = np.asarray(entry["positionM"], float)
                r = float(entry.get("radiusM", 0.0))
            else:
                c = (np.asarray(entry["position"], float) - center) * scale
                r = float(entry.get("radius", 0.0)) * scale
            n = np.asarray(entry.get("normal", entry.get("normalM", [0, 1, 0])), float)
            n = n / (np.linalg.norm(n) or 1.0)
            features.append(PoseFeature(fid, "circle", c, n, r, **common))
            continue
        verts = node_vertices(entry)
        if verts is None:
            warnings.append(f"Pose feature {fid!r}: its GLB node / component could not be found")
            continue
        if kind == "circle":
            c, n, r = _circle_from_vertices(verts)
            features.append(PoseFeature(fid, "circle", c, n, r, **common))
        else:
            lo, hi = verts.min(axis=0), verts.max(axis=0)
            features.append(PoseFeature(fid, "node", (lo + hi) / 2, vertices=verts, **common))
    return features, warnings


def feature_edge_points(features: list[PoseFeature]) -> tuple[np.ndarray, np.ndarray]:
    """(points, normals) on enabled circle rims, to add to the edge model: a
    printed circle has no CAD crease, but its rim is an edge in the image."""
    pts, nrm = [], []
    for f in features:
        if f.enabled and f.type == "circle" and f.normal is not None and f.radius > 0:
            rim = f.contour(64)
            pts.append(rim)
            nrm.append(np.repeat(f.normal[None, :], len(rim), axis=0))
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(pts), np.concatenate(nrm)


def with_feature_edges(model: EdgeModel, features: list[PoseFeature]) -> EdgeModel:
    """The edge model plus the features' rims (visible when their face is)."""
    pts, nrm = feature_edge_points(features)
    if not len(pts):
        return model
    return EdgeModel(
        np.concatenate([model.points, pts]),
        np.concatenate([model.normals_a, nrm]),
        np.concatenate([model.normals_b, nrm]),
        np.concatenate([model.crease, np.ones(len(pts), bool)]),
        model.vertices,
        model.faces,
    )


# ── evidence in an image ─────────────────────────────────────────────────


def _project(points: np.ndarray, t_co: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cam = points @ t_co[:3, :3].T + t_co[:3, 3]
    px = cam @ k.T
    return px[:, :2] / np.maximum(px[:, 2:3], 1e-6), cam


def _observe_circle(gray: np.ndarray, contour_px: np.ndarray, exclude: np.ndarray | None) -> tuple[tuple[float, float], float] | None:
    """The ellipse actually in the image near the predicted one: (centre, its
    size relative to the prediction), or None."""
    h, w = gray.shape
    (x1, y1), (x2, y2) = contour_px.min(axis=0), contour_px.max(axis=0)
    size = max(x2 - x1, y2 - y1, 4.0)
    pad = size
    rx1, ry1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
    rx2, ry2 = int(min(w, x2 + pad)), int(min(h, y2 + pad))
    if rx2 - rx1 < 8 or ry2 - ry1 < 8:
        return None
    roi = cv2.GaussianBlur(gray[ry1:ry2, rx1:rx2], (3, 3), 0)
    edges = cv2.Canny(roi, 20, 40)
    if exclude is not None:
        edges[exclude[ry1:ry2, rx1:rx2]] = 0
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    predicted_center = contour_px.mean(axis=0) - [rx1, ry1]
    best, best_d = None, None
    for c in contours:
        if len(c) < 10:
            continue
        (cx, cy), (a, b), _ = cv2.fitEllipse(c)
        major = max(a, b)
        if not 0.6 * size <= major <= 1.6 * size or min(a, b) < 0.25 * major:
            continue
        d = float(np.hypot(cx - predicted_center[0], cy - predicted_center[1]))
        if d <= 0.75 * size and (best_d is None or d < best_d):
            best, best_d = ((cx + rx1, cy + ry1), major / size), d
    return best


def observe_features(
    features: list[PoseFeature],
    t_co: np.ndarray,
    camera_matrix: np.ndarray,
    gray: np.ndarray,
    edges: EdgeImage,
    model: EdgeModel | None = None,
    near_px: float = 2.5,
) -> list[FeatureObservation]:
    """How each enabled feature shows in `gray` where pose `t_co` puts it."""
    k = np.asarray(camera_matrix, float)
    h, w = gray.shape
    out: list[FeatureObservation] = []
    depth = None
    for f in features:
        if not f.enabled:
            continue
        contour3 = f.contour()
        px, cam = _project(contour3, t_co, k)
        center_px, center_cam = _project(f.center[None, :], t_co, k)
        obs = FeatureObservation(
            f.feature_id, "outside",
            predicted_center_px=(float(center_px[0, 0]), float(center_px[0, 1])),
            predicted_contour_px=[(float(x), float(y)) for x, y in px],
        )
        out.append(obs)
        if (cam[:, 2] <= 1e-3).any():
            continue
        if f.type == "circle" and f.normal is not None:
            n_cam = t_co[:3, :3] @ f.normal
            if n_cam @ (-center_cam[0]) <= 0:
                obs.status = "facing_away"
                continue
        if f.type == "node":  # its outline in the image: the hull of its projected geometry
            px = cv2.convexHull(px.astype(np.float32)).reshape(-1, 2)
            obs.predicted_contour_px = [(float(x), float(y)) for x, y in px]
        inside = (px[:, 0] >= 0) & (px[:, 0] < w - 1) & (px[:, 1] >= 0) & (px[:, 1] < h - 1)
        if inside.mean() < 0.7:
            continue  # "outside"
        # Hidden behind other parts of the CAD model (its centre isn't the nearest surface)?
        if model is not None and model.vertices is not None and len(model.faces):
            if depth is None:
                region = np.array([0.0, 0.0, float(w), float(h)])
                depth = _depth_buffer(model, t_co, k, region, 160)
            (ox, oy), scale, buf = depth
            gx = int(np.clip((center_px[0, 0] - ox) * scale, 0, buf.shape[1] - 1))
            gy = int(np.clip((center_px[0, 1] - oy) * scale, 0, buf.shape[0] - 1))
            near = cv2.erode(np.where(np.isfinite(buf), buf, 1e9).astype(np.float32), np.ones((3, 3), np.uint8))[gy, gx]
            if center_cam[0, 2] > near + max(0.004, 0.01 * center_cam[0, 2]):
                obs.status = "hidden"
                continue
        # Under a hand: can't be judged.
        if edges.unknown is not None:
            xi = np.clip(np.round(px[:, 0]).astype(int), 0, w - 1)
            yi = np.clip(np.round(px[:, 1]).astype(int), 0, h - 1)
            ci = (int(np.clip(round(center_px[0, 0]), 0, w - 1)), int(np.clip(round(center_px[0, 1]), 0, h - 1)))
            if edges.unknown[yi, xi].mean() > 0.4 or edges.unknown[ci[1], ci[0]]:
                obs.status = "occluded"
                continue
        # Edge support along its outline...
        xi = np.clip(np.round(px[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(px[:, 1]).astype(int), 0, h - 1)
        obs.support = float((edges.distance[yi, xi] <= near_px).mean())
        # ...and contrast between inside and a ring around it.
        poly = np.round(px).astype(np.int32)
        inner = np.zeros((h, w), np.uint8)
        cv2.fillPoly(inner, [cv2.convexHull(poly)], 1)
        size = max(np.ptp(px[:, 0]), np.ptp(px[:, 1]), 2.0)
        ring_px = max(2, int(0.35 * size))
        ring = cv2.dilate(inner, np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8)) - inner
        core = cv2.erode(inner, np.ones((3, 3), np.uint8))
        if core.sum() >= 3 and ring.sum() >= 3:
            a = gray[core > 0].astype(float)
            b = gray[ring > 0].astype(float)
            noise = 0.5 * (a.std() + b.std()) + 4.0
            obs.contrast = float(abs(a.mean() - b.mean()) / noise)
        obs.evidence = float(0.6 * obs.support + 0.4 * min(1.0, obs.contrast / 2.0))
        obs.status = "detected" if obs.evidence >= 0.45 else "not_detected"
        if f.type == "circle":
            found = _observe_circle(gray, px, edges.unknown)
            if found is not None:
                (ox_, oy_), _ = found
                obs.observed_center_px = (float(ox_), float(oy_))
                obs.residual_px = float(np.hypot(ox_ - center_px[0, 0], oy_ - center_px[0, 1]))
    return out
