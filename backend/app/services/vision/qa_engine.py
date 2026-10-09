"""
Phase 5 QA engine: are all named assembly parts present on the real object?

Uses the tracked 6DoF pose to project each named part into the camera view
and compares it against the segmentation mask of the real object.
No depth sensor required — visibility is inferred from face geometry.

Status classifications:
  present  — projected region is mostly covered by the real object mask
  missing  — part should be visible but real mask doesn't cover it
  partial  — partly covered (damaged, partially removed)
  occluded — facing away from camera; cannot verify from this viewpoint
  unknown  — too few projected pixels to judge (tiny/off-frame part)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from app.services.model3d.glb_inspect import GlbPartMesh
from app.services.vision.part_presence import MIN_PART_PIXELS, project_part_mask

# Coverage thresholds (projected area that overlaps the real object mask)
PRESENT_THRESHOLD = 0.55   # ≥ this → present
PARTIAL_THRESHOLD = 0.20   # ≥ this → partial  (else missing)
VISIBILITY_THRESHOLD = 0.15  # fraction of visible face area below this → occluded

# BGR colors per status (for OpenCV drawing)
_COLORS: dict[str, tuple[int, int, int]] = {
    "present":  (0, 200, 0),     # green
    "missing":  (30, 30, 210),   # red
    "partial":  (0, 200, 220),   # yellow
    "occluded": (0, 140, 255),   # orange
    "unknown":  (150, 150, 150),
}


@dataclass
class PartInspection:
    name: str
    node_index: int
    status: Literal["present", "missing", "partial", "occluded", "unknown"]
    coverage: float       # (projected ∩ real) / projected pixels
    visibility: float     # fraction of this part's face area facing the camera
    projected_area: int   # projected pixel count
    outline: list[list[float]]  # [[x,y], …] outer contour for the frontend


def _face_visibility(
    mesh: GlbPartMesh,
    assembly_center: np.ndarray,
    scale: float,
    t_co: np.ndarray,
) -> float:
    """
    Fraction of the part's face area that faces the camera (centroid.z > 0).
    Uses 3D face area as weight so many tiny back-faces don't dominate a large
    front-facing surface.  Independent of normal winding order.
    """
    if len(mesh.faces) == 0:
        return 0.0
    verts = (mesh.vertices - assembly_center) * scale
    cam = verts @ t_co[:3, :3].T + t_co[:3, 3]
    A = cam[mesh.faces[:, 0]]
    B = cam[mesh.faces[:, 1]]
    C = cam[mesh.faces[:, 2]]
    centroid_z = ((A + B + C) / 3)[:, 2]
    face_area = np.linalg.norm(np.cross(B - A, C - A), axis=1) * 0.5
    total = face_area.sum()
    if total < 1e-12:
        return 0.0
    return float(face_area[centroid_z > 0].sum() / total)


def _outline(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    best = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(best, 1.5, True)
    return [[float(p[0][0]), float(p[0][1])] for p in approx]


def inspect_parts(
    real_mask: np.ndarray,        # bool H×W — where the real object is
    meshes: dict[int, GlbPartMesh],
    part_names: dict[int, str],   # node_index → display name
    assembly_center: np.ndarray,
    scale: float,
    t_co: np.ndarray,             # 4×4 OpenCV object-in-camera transform
    camera_matrix: np.ndarray,
) -> list[PartInspection]:
    h, w = real_mask.shape
    results: list[PartInspection] = []
    for node_idx, mesh in meshes.items():
        name = part_names.get(node_idx, f"node {node_idx}")
        vis = _face_visibility(mesh, assembly_center, scale, t_co)
        proj = project_part_mask(mesh, assembly_center, scale, t_co, camera_matrix, (h, w))
        area = int(proj.sum())
        outline = _outline(proj)

        if area < MIN_PART_PIXELS:
            status: Literal["present", "missing", "partial", "occluded", "unknown"] = "unknown"
            coverage = 0.0
        elif vis < VISIBILITY_THRESHOLD:
            status = "occluded"
            coverage = 0.0
        else:
            coverage = float((proj & real_mask).sum() / max(area, 1))
            if coverage >= PRESENT_THRESHOLD:
                status = "present"
            elif coverage >= PARTIAL_THRESHOLD:
                status = "partial"
            else:
                status = "missing"

        results.append(PartInspection(
            name=name,
            node_index=node_idx,
            status=status,
            coverage=coverage,
            visibility=vis,
            projected_area=area,
            outline=outline,
        ))
    return results


def qa_overlay(frame_bgr: np.ndarray, parts: list[PartInspection]) -> bytes:
    """
    JPEG of the camera frame with each part's projected outline drawn in its
    status color.  Missing/partial parts get a translucent fill so the
    operator can immediately see where the component should be.
    """
    vis = frame_bgr.copy()
    fill = frame_bgr.copy()

    for part in parts:
        if not part.outline:
            continue
        color = _COLORS.get(part.status, (150, 150, 150))
        pts = np.array(part.outline, np.int32).reshape(-1, 1, 2)

        if part.status in ("missing", "partial"):
            cv2.fillPoly(fill, [pts], color)

        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

        # Label missing / partial parts at their centroid
        if part.status in ("missing", "partial"):
            cx = int(np.mean([p[0] for p in part.outline]))
            cy = int(np.mean([p[1] for p in part.outline]))
            label = f"{part.name}: {part.status.upper()}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Black background behind the label so it reads on any background
            cv2.rectangle(vis, (cx - 2, cy - th - 4), (cx + tw + 2, cy + 2), (0, 0, 0), -1)
            cv2.putText(vis, label, (cx, cy - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Blend: 30% tint fill over the annotated frame
    cv2.addWeighted(fill, 0.30, vis, 0.70, 0, vis)

    ok, data = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return data.tobytes() if ok else b""
