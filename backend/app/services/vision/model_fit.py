"""
Does the 3D model have the same shape as the real object?

Tracking can only overlay the model as well as its shape matches. Given the
real object's outline in a camera frame and the model drawn at its solved
pose (per part), this compares the two along the object's long axis and says
what to fix in the GLB — e.g. on the user's flask: "Cap: model 0.9 cm
narrower (-17%)", "Model 0.5 cm taller". Measured, not guessed: widths are in
cm at the object's distance.

The real outline comes from the detector, so anything it wrongly includes (a
strap, a hand) shows up as "model too narrow" at those heights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class FitBand:
    from_top_cm: float  # center of the band, measured from the real object's top
    real_cm: float
    model_cm: float
    part: str | None  # the part forming the model's silhouette at this height

    @property
    def diff_cm(self) -> float:
        return self.model_cm - self.real_cm


@dataclass
class PartFit:
    part: str
    from_top_cm: tuple[float, float]  # the height range it forms the silhouette over
    real_cm: float  # mean width there
    model_cm: float
    diff_cm: float
    diff_rel: float  # diff / real width


@dataclass
class FitReport:
    iou: float
    real_height_cm: float
    model_height_cm: float
    top_offset_cm: float  # model top above (+) / below (-) the real top
    bottom_offset_cm: float  # model bottom below (+) / above (-) the real bottom
    bands: list[FitBand] = field(default_factory=list)
    parts: list[PartFit] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    good_fit: bool = False


def _upright(masks: list[np.ndarray], reference: np.ndarray) -> list[np.ndarray]:
    """Rotate masks so `reference`'s long axis is vertical (top stays the side nearer image-up)."""
    ys, xs = np.nonzero(reference)
    points = np.stack([xs, ys], axis=1).astype(float)
    center = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((points - center).T))
    axis = vectors[:, -1]  # largest variance
    if axis[1] > 0:
        axis = -axis  # point "up" (negative y)
    tilt_deg = math.degrees(math.atan2(axis[0], -axis[1]))  # 0 = vertical; > 0 = top leaning right
    if abs(tilt_deg) < 0.5:
        return masks
    # OpenCV's angle is counter-clockwise on screen: leaning right is undone by turning it back left.
    rotation = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), tilt_deg, 1.0)
    h, w = reference.shape
    return [cv2.warpAffine(m.astype(np.uint8), rotation, (w, h), flags=cv2.INTER_NEAREST).astype(bool) for m in masks]


def _rows(mask: np.ndarray) -> tuple[int, int]:
    rows = np.flatnonzero(mask.any(axis=1))
    return int(rows[0]), int(rows[-1])


def fit_report(
    real_mask: np.ndarray,
    part_masks: dict[str, np.ndarray],
    cm_per_px: float,
    bands: int = 24,
    tolerance_cm: float = 0.3,
    tolerance_rel: float = 0.05,
) -> FitReport:
    """Compare the real outline with the model's (the union of `part_masks`), in cm."""
    names = list(part_masks)
    model_mask = np.zeros_like(real_mask, dtype=bool)
    for m in part_masks.values():
        model_mask |= m
    iou = float((real_mask & model_mask).sum() / max((real_mask | model_mask).sum(), 1))

    upright = _upright([real_mask, model_mask, *[part_masks[n] for n in names]], real_mask)
    real, model, parts = upright[0], upright[1], dict(zip(names, upright[2:]))
    real_top, real_bottom = _rows(real)
    model_top, model_bottom = _rows(model)
    height = real_bottom - real_top + 1

    sampled: list[FitBand] = []
    edges = np.linspace(real_top, real_bottom + 1, bands + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows = slice(int(lo), max(int(hi), int(lo) + 1))
        real_width = real[rows].sum(axis=1).mean()
        model_width = model[rows].sum(axis=1).mean()
        widths = {n: parts[n][rows].sum(axis=1).mean() for n in names}
        widest = max(widths, key=widths.get) if widths and max(widths.values()) > 0 else None
        sampled.append(
            FitBand(
                from_top_cm=float(((lo + hi) / 2 - real_top) * cm_per_px),
                real_cm=float(real_width * cm_per_px),
                model_cm=float(model_width * cm_per_px),
                part=widest,
            )
        )

    part_fits: list[PartFit] = []
    for name in names:
        mine = [b for b in sampled if b.part == name and b.real_cm > 0]
        if not mine:
            continue
        real_cm = float(np.mean([b.real_cm for b in mine]))
        model_cm = float(np.mean([b.model_cm for b in mine]))
        part_fits.append(
            PartFit(
                part=name,
                from_top_cm=(min(b.from_top_cm for b in mine), max(b.from_top_cm for b in mine)),
                real_cm=real_cm,
                model_cm=model_cm,
                diff_cm=model_cm - real_cm,
                diff_rel=(model_cm - real_cm) / real_cm,
            )
        )

    report = FitReport(
        iou=iou,
        real_height_cm=height * cm_per_px,
        model_height_cm=(model_bottom - model_top + 1) * cm_per_px,
        top_offset_cm=(real_top - model_top) * cm_per_px,
        bottom_offset_cm=(model_bottom - real_bottom) * cm_per_px,
        bands=sampled,
        parts=part_fits,
    )

    findings: list[tuple[float, str]] = []
    for p in part_fits:
        if abs(p.diff_cm) > max(tolerance_cm, tolerance_rel * p.real_cm):
            word = "wider" if p.diff_cm > 0 else "narrower"
            findings.append(
                (
                    abs(p.diff_cm),
                    f"{p.part}: model {abs(p.diff_cm):.1f} cm {word} ({p.diff_rel:+.0%}) "
                    f"— real {p.real_cm:.1f} cm vs model {p.model_cm:.1f} cm, "
                    f"{p.from_top_cm[0]:.0f}-{p.from_top_cm[1]:.0f} cm from the top",
                )
            )
    height_diff = report.model_height_cm - report.real_height_cm
    if abs(height_diff) > max(tolerance_cm, tolerance_rel * report.real_height_cm / 4):
        where = []
        if abs(report.top_offset_cm) > tolerance_cm / 2:
            where.append(f"{abs(report.top_offset_cm):.1f} cm {'above' if report.top_offset_cm > 0 else 'below'} the top")
        if abs(report.bottom_offset_cm) > tolerance_cm / 2:
            where.append(
                f"{abs(report.bottom_offset_cm):.1f} cm {'below' if report.bottom_offset_cm > 0 else 'above'} the bottom"
            )
        findings.append(
            (
                abs(height_diff),
                f"Model {abs(height_diff):.1f} cm {'taller' if height_diff > 0 else 'shorter'} than the object"
                + (f" (ends {' and '.join(where)})" if where else ""),
            )
        )
    report.findings = [text for _, text in sorted(findings, reverse=True)]
    report.good_fit = iou >= 0.95 and not report.findings
    if report.good_fit:
        report.findings = [f"Good fit: the model covers the object's outline at {iou:.0%}."]
    return report


def fit_overlay(frame_bgr: np.ndarray, real_mask: np.ndarray, model_mask: np.ndarray, zoom: float = 2.0) -> bytes:
    """JPEG of the object area: real outline red, model outline green."""
    vis = frame_bgr.copy()
    for mask, color in ((real_mask, (0, 0, 255)), (model_mask, (0, 255, 0))):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, contours, -1, color, 1)
    ys, xs = np.nonzero(real_mask | model_mask)
    pad = 20
    crop = vis[max(0, ys.min() - pad) : ys.max() + pad, max(0, xs.min() - pad) : xs.max() + pad]
    crop = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    ok, data = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return data.tobytes() if ok else b""
