"""
Can this component be judged in this frame?

A component is only ever reported missing from frames where it could have
been seen: inside the image, big enough, not hidden behind other parts of
the CAD model, the image sharp and exposed well enough, and the object
tracked confidently (a pose that's off puts the "expected" pixels on
something else). Anything else makes the observation invalid: it neither
confirms nor clears a missing component, and the reason is kept.

RGB only: occlusion is judged from the CAD geometry at the tracked pose. A
hand or tool in front of a component is not in the CAD model; the verifiers
see it as "doesn't look like either state" (see verifiers.py) rather than
as missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from app.services.component_inspection.projection import Projection

VisibilityStatus = Literal[
    "visible", "tracking", "behind_camera", "out_of_frame", "too_small", "occluded", "facing_away", "poor_image"
]


@dataclass
class VisibilityConfig:
    min_tracking_confidence: float = 0.7  # below: the pose may be off by more than a small part's size
    min_in_frame: float = 0.9  # fraction of the component that must be inside the image
    min_area_px: int = 40  # visible pixels needed to judge it at all
    min_side_px: int = 6  # ... and on its shorter side
    # Share of it that must be the nearest CAD surface. Low on purpose: a
    # ring or open cap seen from above hides its own far half behind the
    # neck it sits on, and the visible half is plenty to judge; evidence is
    # only ever taken from the visible pixels (verifiers.Observation).
    min_cad_visible: float = 0.3
    min_facing: float = 0.15  # used only when CAD occlusion couldn't be computed
    min_sharpness: float = 8.0  # variance of the Laplacian around it; below = motion/focus blur
    dark_level: float = 20.0  # mean grey level below this: too dark to judge
    clipped_fraction: float = 0.5  # more of its pixels than this at 0 or 255: under/over-exposed
    context: float = 0.5  # quality is measured on the component's box grown by this much each side
    max_hand_cover: float = 0.15  # a hand over more of its visible pixels than this: occluded


@dataclass
class Visibility:
    observable: bool
    status: VisibilityStatus
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def image_quality(gray: np.ndarray, projection: Projection, context: float) -> dict[str, float]:
    """Sharpness / exposure of the component's own pixels and their surroundings."""
    assert projection.roi is not None and projection.mask is not None
    x1, y1, x2, y2 = projection.roi
    h, w = gray.shape
    gx, gy = int((x2 - x1) * context) + 2, int((y2 - y1) * context) + 2
    cx1, cy1, cx2, cy2 = max(0, x1 - gx), max(0, y1 - gy), min(w, x2 + gx), min(h, y2 + gy)
    around = gray[cy1:cy2, cx1:cx2]
    sharpness = float(cv2.Laplacian(around, cv2.CV_64F).var()) if around.size else 0.0
    pixels = gray[y1:y2, x1:x2][projection.mask]
    return {
        "sharpness": sharpness,
        "brightness": float(pixels.mean()) if pixels.size else 0.0,
        "clipped": float(((pixels <= 3) | (pixels >= 252)).mean()) if pixels.size else 0.0,
    }


def assess(
    projection: Projection,
    gray: np.ndarray,
    tracking_ok: bool,
    tracking_reason: str | None,
    cad_visible: float | None,
    config: VisibilityConfig,
    visible_area_px: int | None = None,
    hand_cover: float | None = None,
) -> Visibility:
    """Every failed check is listed; `status` is the first, in the order
    tracking -> geometry -> image (the most fundamental reason)."""
    reasons: list[tuple[VisibilityStatus, str]] = []
    metrics: dict[str, float] = {"in_frame": projection.in_frame, "area_px": float(projection.area_px),
                                 "facing": projection.facing, "depth_m": projection.depth_m}
    if not tracking_ok:
        reasons.append(("tracking", tracking_reason or "object not tracked confidently"))
    if projection.behind_camera:
        reasons.append(("behind_camera", "part of it is behind the camera"))
        return Visibility(False, reasons[0][0], [r for _, r in reasons], metrics)
    if projection.in_frame < config.min_in_frame:
        reasons.append(("out_of_frame", f"only {projection.in_frame:.0%} of it is inside the image"))
    if projection.roi is None or projection.mask is None:
        reasons.append(("out_of_frame", "not in the image"))
        return Visibility(False, reasons[0][0], [r for _, r in reasons], metrics)

    x1, y1, x2, y2 = projection.roi
    if projection.area_px < config.min_area_px or min(x2 - x1, y2 - y1) < config.min_side_px:
        reasons.append(("too_small", f"{projection.area_px} px ({x2 - x1}x{y2 - y1}) — too small at this distance"))
    if cad_visible is not None:
        metrics["cad_visible"] = cad_visible
        if cad_visible < config.min_cad_visible:
            reasons.append(("occluded", f"{1 - cad_visible:.0%} hidden behind other parts from this viewpoint"))
        elif visible_area_px is not None and visible_area_px < config.min_area_px <= projection.area_px:
            reasons.append(("occluded", f"only {visible_area_px} px of it in view, the rest behind other parts"))
    if hand_cover is not None:
        metrics["hand_cover"] = hand_cover
        if hand_cover > config.max_hand_cover:
            reasons.append(("occluded", f"{hand_cover:.0%} covered by a hand"))
    if visible_area_px is not None:
        metrics["visible_px"] = float(visible_area_px)
    elif projection.facing < config.min_facing:
        reasons.append(("facing_away", f"only {projection.facing:.0%} of its surface faces the camera"))

    quality = image_quality(gray, projection, config.context)
    metrics.update(quality)
    if quality["sharpness"] < config.min_sharpness:
        reasons.append(("poor_image", f"blurred (sharpness {quality['sharpness']:.1f} < {config.min_sharpness})"))
    if quality["brightness"] < config.dark_level:
        reasons.append(("poor_image", f"too dark (grey level {quality['brightness']:.0f})"))
    if quality["clipped"] > config.clipped_fraction:
        reasons.append(("poor_image", f"{quality['clipped']:.0%} of its pixels are clipped black/white"))

    if reasons:
        return Visibility(False, reasons[0][0], [r for _, r in reasons], metrics)
    return Visibility(True, "visible", [], metrics)
