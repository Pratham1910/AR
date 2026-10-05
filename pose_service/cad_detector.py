"""
Finds a registered object in a camera image from its CAD model alone — no
object class, no training (the CNOS approach, Nguyen et al. 2023):

1. Templates: the mesh rendered from ~42 viewpoints (render_templates.py),
   each turned into a DINOv2 descriptor — once per object, cached on disk.
2. Proposals: FastSAM splits the camera image into candidate object masks.
3. Matching: each proposal gets a DINOv2 descriptor too; its score is the mean
   cosine similarity to its best-matching templates. The best proposal is the
   object, if it scores above a threshold.

Proposals and templates are prepared identically (background blacked out,
cropped to the mask, padded square), so the descriptors compare shape and
shading rather than the surroundings.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("uvicorn.error")

WEIGHTS_DIR = Path(os.getenv("TVASTA_POSE_WEIGHTS_DIR", Path.home() / "tvasta-pose" / "weights"))
# The small variants (24 MB + 88 MB). CNOS reports the large ones (FastSAM-x,
# dinov2_vitl14: 138 MB + 1.2 GB) as somewhat more accurate; switch with these
# variables once their weights are in WEIGHTS_DIR.
FASTSAM_WEIGHTS = os.getenv("TVASTA_FASTSAM_WEIGHTS", "FastSAM-s.pt")
DINO_MODEL = os.getenv("TVASTA_DINO_MODEL", "dinov2_vits14")
# DINOv2's code (github.com/facebookresearch/dinov2, unzipped) and its
# <model>_pretrain.pth checkpoint, both local: nothing is fetched at runtime.
DINO_REPO = WEIGHTS_DIR / "dinov2"
CROP_SIZE = 224
TOP_K_TEMPLATES = 5  # a proposal's score: mean similarity to its best K templates
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass
class Candidate:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in image pixels
    score: float  # mean cosine similarity to the best-matching templates (~0.3 unrelated .. ~0.8+ same object)
    area_fraction: float
    polygon: list[tuple[float, float]]  # outline of the proposal's mask


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """x1, y1, x2, y2 (exclusive) of a bool mask. Row/column any() rather than
    nonzero(): ~10x faster on full-resolution masks, and there are ~100 per frame."""
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def masked_square_crop(rgb: np.ndarray, mask: np.ndarray, size: int = CROP_SIZE) -> np.ndarray | None:
    """The masked region only (background black), cropped to it and padded to a centred square."""
    box = mask_box(mask)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    crop = rgb[y1:y2, x1:x2] * mask[y1:y2, x1:x2, None]
    h, w = crop.shape[:2]
    side = max(h, w)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    square[(side - h) // 2 : (side - h) // 2 + h, (side - w) // 2 : (side - w) // 2 + w] = crop
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


def mask_polygon(mask: np.ndarray) -> list[tuple[float, float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    outline = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 2.0, True)
    return [(float(p[0][0]), float(p[0][1])) for p in outline]


class CadDetector:
    def __init__(self, device: torch.device):
        self.device = device
        self._dino = None
        self._sam = None

    # Loaded on first use: ~1.5 GB of weights, needless for pose-only use.
    def _dino_model(self):
        if self._dino is None:
            started = time.perf_counter()
            model = torch.hub.load(str(DINO_REPO), DINO_MODEL, source="local", pretrained=False)
            state = torch.load(WEIGHTS_DIR / f"{DINO_MODEL}_pretrain.pth", map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            self._dino = model.to(self.device).eval()
            log.info("cad-detect: loaded %s in %.1fs", DINO_MODEL, time.perf_counter() - started)
        return self._dino

    def _sam_model(self):
        if self._sam is None:
            from ultralytics import FastSAM  # only the detector needs ultralytics

            self._sam = FastSAM(str(WEIGHTS_DIR / FASTSAM_WEIGHTS))
        return self._sam

    @torch.no_grad()
    def descriptors(self, crops: list[np.ndarray]) -> torch.Tensor:
        """L2-normalised DINOv2 class-token descriptors, one row per crop."""
        batch = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().div(255.0)
        batch = ((batch - _MEAN) / _STD).to(self.device)
        out = []
        for chunk in batch.split(64):
            out.append(self._dino_model()(chunk))
        return F.normalize(torch.cat(out), dim=-1)

    def template_descriptors(self, rgb: np.ndarray, mask: np.ndarray) -> torch.Tensor:
        crops = [masked_square_crop(r, m) for r, m in zip(rgb, mask)]
        return self.descriptors([c for c in crops if c is not None])

    @torch.no_grad()
    def proposals(self, rgb: np.ndarray, min_area: float, max_area: float) -> list[np.ndarray]:
        """Candidate object masks (full-resolution bool arrays) in the image."""
        results = self._sam_model()(
            np.ascontiguousarray(rgb[..., ::-1]),  # ultralytics takes numpy images as BGR
            device=self.device,
            retina_masks=True,
            imgsz=1024,
            conf=0.25,
            iou=0.7,
            verbose=False,
        )
        if not results or results[0].masks is None:
            return []
        masks = results[0].masks.data.bool().cpu().numpy()
        total = rgb.shape[0] * rgb.shape[1]
        return [m for m in masks if min_area <= m.sum() / total <= max_area]

    def detect(
        self,
        rgb: np.ndarray,
        templates: torch.Tensor,
        min_area: float = 0.002,
        max_area: float = 0.8,
        top: int = 5,
    ) -> tuple[list[Candidate], dict[str, float]]:
        """Proposals ranked by how well they match the templates (best first), and step timings in ms."""
        timings: dict[str, float] = {}
        started = time.perf_counter()
        masks = self.proposals(rgb, min_area, max_area)
        timings["proposals"] = (time.perf_counter() - started) * 1000.0
        if not masks:
            return [], timings

        started = time.perf_counter()
        crops = [masked_square_crop(rgb, m) for m in masks]
        keep = [i for i, c in enumerate(crops) if c is not None]
        similarity = self.descriptors([crops[i] for i in keep]) @ templates.T  # (proposals, templates)
        k = min(TOP_K_TEMPLATES, similarity.shape[1])
        scores = similarity.topk(k, dim=1).values.mean(dim=1).cpu().numpy()
        timings["matching"] = (time.perf_counter() - started) * 1000.0

        total = rgb.shape[0] * rgb.shape[1]
        ranked = []
        for idx in np.argsort(-scores)[:top]:
            mask = masks[keep[idx]]
            ranked.append(
                Candidate(
                    bbox=tuple(float(v) for v in mask_box(mask)),
                    score=float(scores[idx]),
                    area_fraction=float(mask.sum() / total),
                    polygon=mask_polygon(mask),
                )
            )
        return ranked, timings
