"""
Is a part (e.g. a bottle's cap) still on the object?

The tracker already knows where the object is; the assembly GLB says where
the part sits on it (as fractions of the object's box). This module measures
how that region looks and compares it with what "present" and "absent"
looked like in labelled calibration frames for this exact part.

Measured on the user's flask (6 raw frames): the cap region's mean
brightness (HSV V) was ~79 with the black cap on and ~156-161 with the
bright steel neck exposed. Calibrating per part keeps that knowledge in
data, not code: a different part (a light cap on a dark body) just
calibrates the other way round.

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
from app.services.model3d.glb_inspect import GlbPart


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
