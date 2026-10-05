"""
Is a part (e.g. a bottle's cap) still on the object?

Two ways to find the part's pixels:

- "pose" (model-based tracking): the part's own triangles, drawn at the
  tracked 6DoF pose — exactly where the part is (or would be) in the image,
  however the object is turned.
- "box" (markerless, no 3D pose): a fixed slice of the object's image box.
  Approximate: it assumes an upright object and a box that hugs it, and it
  breaks while tracking (the box keeps its cap-on size and drifts, so after
  removal the slice can sample whatever is behind where the cap was).

Either way the region's look is compared with what "present" and "absent"
looked like in labelled calibration frames for this exact part.

Calibration picks the brightness statistic that best separates the labelled
frames. On the user's flask (pose method, 15 frames, 3 lightings) that was
the spread: the matte black cap reads 21-27, the threaded steel neck 35-45,
while mean brightness barely separated in a dim room. Keeping that choice in
data, not code, lets another part (a light lid on a dark box) calibrate
whichever way it differs.

Answers are three-way: present / absent / uncertain (too close to call),
matching the QA engine's PASS / FAIL / UNCERTAIN discipline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from app.schemas.vision import BoundingBox
from app.services.model3d.glb_inspect import GlbPart, GlbPartMesh

# Brightness statistics of a part's pixels that calibration picks from: a
# shiny metal neck shows as bright highlights (p90), a dark opening as dark
# pixels (p10), a part swapped for something textured as spread (std).
FEATURES = ("mean", "p10", "p90", "std")
MIN_PART_PIXELS = 50  # fewer visible pixels than this: too small / off-frame to judge


@dataclass
class PartRegion:
    """Where a part sits on its object, as fractions of the object's image box
    (top = 0, bottom = 1; left = 0, right = 1). Assumes the object is roughly
    upright in the image, as when tracking a bottle on a table."""

    top: float
    bottom: float
    left: float
    right: float

    @staticmethod
    def from_parts(part: GlbPart, all_parts: list[GlbPart]) -> PartRegion:
        lo = np.min([p.bounds.min for p in all_parts], axis=0)
        hi = np.max([p.bounds.max for p in all_parts], axis=0)
        height, width = hi[1] - lo[1], hi[0] - lo[0]
        b = part.bounds
        top, bottom = (hi[1] - b.max[1]) / height, (hi[1] - b.min[1]) / height
        left, right = (b.min[0] - lo[0]) / width, (b.max[0] - lo[0]) / width
        # Stay inside the part: trim 10% of its band at each edge so a slightly
        # off box doesn't sample what's next to it (e.g. a cap's orange ring).
        dy, dx = (bottom - top) * 0.1, (right - left) * 0.1
        return PartRegion(top=top + dy, bottom=bottom - dy, left=left + dx, right=right - dx)

    def pixels(self, box: BoundingBox) -> tuple[int, int, int, int]:
        w, h = box.x2 - box.x1, box.y2 - box.y1
        return (
            int(round(box.x1 + w * self.left)),
            int(round(box.y1 + h * self.top)),
            int(round(box.x1 + w * self.right)),
            int(round(box.y1 + h * self.bottom)),
        )


def region_brightness(frame_bgr: np.ndarray, region: PartRegion, box: BoundingBox) -> float | None:
    """Mean HSV value (0-255) of the part's region, or None if it's off-frame / too small."""
    x1, y1, x2, y2 = region.pixels(box)
    height, width = frame_bgr.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    hsv = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    return float(hsv[..., 2].mean())


def project_part_mask(
    mesh: GlbPartMesh,
    assembly_center: np.ndarray,
    scale: float,
    t_camera_object: np.ndarray,
    camera_matrix: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Pixels the part covers with the assembly at `t_camera_object` (OpenCV
    camera frame, meters; the assembly recentered on its bounding-box center,
    as the pose service registers it). `mesh` is in GLB units, assembly frame."""
    vertices = (mesh.vertices - assembly_center) * scale
    cam = vertices @ t_camera_object[:3, :3].T + t_camera_object[:3, 3]
    mask = np.zeros(shape, np.uint8)
    faces = mesh.faces[(cam[mesh.faces, 2] > 1e-3).all(axis=1)]  # drop triangles behind the camera
    px = cam @ np.asarray(camera_matrix, float).T
    px = np.round(px[:, :2] / px[:, 2:3]).astype(np.int32)
    # One triangle at a time: fillPoly with all of them at once uses an
    # even-odd rule, so overlapping front/back faces would cancel into holes.
    for triangle in px[faces]:
        cv2.fillConvexPoly(mask, triangle, 1)
    return mask.astype(bool)


def mask_features(frame_bgr: np.ndarray, mask: np.ndarray) -> dict[str, float] | None:
    """Brightness (grey level, 0-255) statistics of the masked pixels, or None if too few are visible.
    Grey level rather than HSV value: V rates a saturated red as bright as white,
    so a cap's red ring looked like the bare steel neck's highlights."""
    if np.count_nonzero(mask) < MIN_PART_PIXELS:
        return None
    value = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)[mask].astype(float)
    return {
        "mean": float(value.mean()),
        "p10": float(np.percentile(value, 10)),
        "p90": float(np.percentile(value, 90)),
        "std": float(value.std()),
    }


def mask_outline(mask: np.ndarray) -> list[tuple[float, float]]:
    """The mask's outer outline, for drawing what was measured."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    outline = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.5, True)
    return [(float(p[0][0]), float(p[0][1])) for p in outline]


def best_feature(present: list[dict[str, float]], absent: list[dict[str, float]]) -> tuple[str, dict[str, float]]:
    """The feature that best separates the labelled frames, and its stats
    (present/absent mean and spread). Ranked by the worst-case gap between the
    two groups relative to their spread, so one outlier frame counts."""
    best: tuple[float, str, dict[str, float]] | None = None
    for name in FEATURES:
        on = np.array([f[name] for f in present])
        off = np.array([f[name] for f in absent])
        gap = off.min() - on.max() if off.mean() >= on.mean() else on.min() - off.max()
        spread = max(on.std() + off.std(), 1.0)
        rank = gap / spread
        stats = {
            "present_mean": float(on.mean()),
            "absent_mean": float(off.mean()),
            "present_std": float(on.std()),
            "absent_std": float(off.std()),
        }
        if best is None or rank > best[0]:
            best = (rank, name, stats)
    assert best is not None
    return best[1], best[2]


@dataclass
class PresenceCalibration:
    region: PartRegion
    present_mean: float
    absent_mean: float
    present_samples: int
    absent_samples: int
    present_std: float = 0.0
    absent_std: float = 0.0
    node_index: int = -1  # which assembly part this calibrates
    part_name: str = ""
    # How the part's pixels are found ("pose" / "box", see module docstring)
    # and which of their FEATURES the means above are of. Files saved before
    # these existed are box-slice mean brightness.
    method: str = "box"
    feature: str = "mean"

    # A measurement within this fraction of the present-absent gap around the
    # midpoint is "uncertain" rather than forced into present/absent.
    UNCERTAIN_BAND = 0.25

    def classify(self, brightness: float) -> tuple[str, float]:
        """('present' | 'absent' | 'uncertain', confidence 0..1)."""
        gap = self.absent_mean - self.present_mean
        if abs(gap) < 1e-6:
            return "uncertain", 0.0
        # 0 at the present mean, 1 at the absent mean (whichever way round they are).
        t = (brightness - self.present_mean) / gap
        distance = abs(t - 0.5)  # 0 at the midpoint, 0.5 at either calibrated mean
        confidence = float(min(1.0, distance / 0.5))
        if distance < self.UNCERTAIN_BAND / 2:
            return "uncertain", confidence
        return ("absent" if t > 0.5 else "present"), confidence

    @property
    def separation(self) -> float:
        """Gap between the two states in units of their spread; >3 is clearly separable."""
        spread = max(self.present_std + self.absent_std, 1.0)
        return abs(self.absent_mean - self.present_mean) / spread

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path: Path) -> PresenceCalibration | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        data["region"] = PartRegion(**data["region"])
        return PresenceCalibration(**data)
