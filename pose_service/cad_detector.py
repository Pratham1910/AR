"""
Finds a registered object in a camera image from its CAD model alone — no
object class, no training (the CNOS approach, Nguyen et al. 2023):

1. Templates: the mesh rendered from ~42 viewpoints (render_templates.py),
   each turned into a DINOv2 descriptor — once per object, cached on disk.
2. Proposals: FastSAM splits the camera image into candidate object masks.
3. Matching: each proposal gets a DINOv2 descriptor too; its appearance
   score is the mean cosine similarity to its best-matching templates.
4. Shape (SAM-6D's geometric check, Lin et al. 2024): the proposal's outline
   is compared with the outlines of those same templates. Appearance alone
   can't tell the object from the object plus whatever touches it — a hand
   holding a bottle comes back from FastSAM as one bottle+hand+arm region,
   and the bottle dominates its descriptor (0.53-0.66 vs ~0.39 for the next
   region, logged 2026-10-13). Its outline is nothing like a bottle's, so
   ranking by appearance x shape puts the bottle-only region first.

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
# Objects made of visibly different panels (a case's glossy end cap and its
# matte body) come out of FastSAM as separate segments, and one panel alone
# then gets matched — the model squeezed into a strip of it. So unions of
# touching segments are scored too: pairs that include one of the best
# MERGE_SEEDS single segments, at most MERGE_MAX unions.
MERGE_SEEDS = 8
MERGE_MAX = 48
# Outlines are compared at this size: plenty to tell a bottle from a bottle
# with an arm attached, and ~100 proposals x 42 templates stays a single
# small matrix product.
SHAPE_SIZE = 64
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass
class Templates:
    descriptors: torch.Tensor  # (T, D) L2-normalised DINOv2 descriptors
    shapes: torch.Tensor  # (T, SHAPE_SIZE**2) outlines, cropped and padded like the proposals


@dataclass
class Candidate:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in image pixels
    score: float  # mean cosine similarity to the best-matching templates (~0.3 unrelated .. ~0.8+ same object)
    shape: float  # best outline IoU with those same templates (~0.8 the object alone, much lower with a hand attached)
    area_fraction: float
    polygon: list[tuple[float, float]]  # outline of the proposal's mask


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """x1, y1, x2, y2 (exclusive) of a bool mask. Row/column any() rather than
    nonzero(): ~10x faster on full-resolution masks, and there are ~100 per frame."""
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _padded_square(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    side = max(h, w)
    square = np.zeros((side, side) + crop.shape[2:], dtype=crop.dtype)
    square[(side - h) // 2 : (side - h) // 2 + h, (side - w) // 2 : (side - w) // 2 + w] = crop
    return square


def masked_square_crop(rgb: np.ndarray, mask: np.ndarray, size: int = CROP_SIZE) -> np.ndarray | None:
    """The masked region only (background black), cropped to it and padded to a centred square."""
    box = mask_box(mask)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    square = _padded_square(rgb[y1:y2, x1:x2] * mask[y1:y2, x1:x2, None])
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


def square_shape(mask: np.ndarray, size: int = SHAPE_SIZE) -> np.ndarray | None:
    """The mask cropped and padded exactly as masked_square_crop does, as a
    flat 0/1 vector: outlines compare independent of position and scale."""
    box = mask_box(mask)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    square = _padded_square(mask[y1:y2, x1:x2].astype(np.uint8))
    return (cv2.resize(square, (size, size), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32).ravel()


def prefer_whole(masks: list[np.ndarray], scores: np.ndarray, shape_iou: np.ndarray, margin: float = 0.9) -> np.ndarray:
    """Ranking by appearance x shape, except that a candidate containing the
    leader (>= 90% of it) moves ahead of it when it looks more like the object
    and scores within `margin` of it: the whole object, not one of its panels
    (a case's end cap alone has the outline of the case seen edge-on)."""
    combined = scores * shape_iou
    order = list(np.argsort(-combined))
    if len(order) < 2:
        return np.asarray(order)
    lead = order[0]
    lead_mask = masks[lead][::4, ::4]
    lead_area = max(int(lead_mask.sum()), 1)
    best = None
    for idx in order[1:]:
        if combined[idx] < margin * combined[lead] or scores[idx] <= scores[lead]:
            continue
        if (masks[idx][::4, ::4] & lead_mask).sum() >= 0.9 * lead_area and masks[idx].sum() > masks[lead].sum():
            best = idx
            break
    if best is not None:
        order.remove(best)
        order.insert(0, best)
    return np.asarray(order)


def touching_unions(masks: list[np.ndarray], seeds, max_area: float, limit: int) -> list[np.ndarray]:
    """Unions of a seed segment with each segment touching it (within a few
    pixels, at quarter resolution) — skipping pairs where one lies inside the other; at most
    `limit`, none above `max_area` of the image."""
    if len(masks) < 2:
        return []
    total = masks[0].size
    small = [m[::4, ::4] for m in masks]
    grown = [cv2.dilate(s.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0 for s in small]
    out, seen = [], set()
    by_size = np.argsort([-s.sum() for s in small])  # a body before the specks around it
    per_seed = max(1, limit // max(1, len(seeds)))
    for i in seeds:
        i = int(i)
        made = 0
        for j in by_size:
            if made >= per_seed:
                break
            j = int(j)
            pair = (min(i, j), max(i, j))
            if j == i or pair in seen:
                continue
            seen.add(pair)
            if not (grown[i] & small[j]).any():
                continue
            # Not one inside the other: that union is just the bigger segment.
            inter = (small[i] & small[j]).sum()
            if inter >= 0.9 * min(small[i].sum(), small[j].sum()):
                continue
            union = masks[i] | masks[j]
            if union.sum() / total > max_area:
                continue
            out.append(union)
            made += 1
            if len(out) >= limit:
                return out
    return out


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

    def templates(self, rgb: np.ndarray, mask: np.ndarray) -> Templates:
        keep = [i for i, m in enumerate(mask) if m.any()]
        return Templates(
            descriptors=self.descriptors([masked_square_crop(rgb[i], mask[i]) for i in keep]),
            shapes=torch.from_numpy(np.stack([square_shape(mask[i]) for i in keep])).to(self.device),
        )

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

    def _match(self, rgb: np.ndarray, masks: list[np.ndarray], templates: Templates):
        """(index into masks of each kept proposal, appearance score, outline IoU)."""
        crops = [masked_square_crop(rgb, m) for m in masks]
        keep = [i for i, c in enumerate(crops) if c is not None]
        similarity = self.descriptors([crops[i] for i in keep]) @ templates.descriptors.T  # (proposals, templates)
        k = min(TOP_K_TEMPLATES, similarity.shape[1])
        best = similarity.topk(k, dim=1)
        scores = best.values.mean(dim=1)

        # Outline IoU with each of the K templates that matched best on
        # appearance (not with all of them: from some viewpoint a forearm alone
        # has a bottle's outline). Best of the K, as the right view is among them.
        shapes = torch.from_numpy(np.stack([square_shape(masks[i]) for i in keep])).to(self.device)
        overlap = shapes @ templates.shapes.T
        union = shapes.sum(dim=1, keepdim=True) + templates.shapes.sum(dim=1) - overlap
        shape_iou = (overlap / union.clamp(min=1)).gather(1, best.indices).max(dim=1).values
        return keep, scores.cpu().numpy(), shape_iou.cpu().numpy()

    def segments_at(self, rgb: np.ndarray, x: float, y: float, max_area: float = 0.6) -> list[np.ndarray]:
        """FastSAM's segments containing the pixel (x, y) (within a few px),
        smallest first: usually nested — a panel, the object, a background region."""
        h, w = rgb.shape[:2]
        xi, yi = int(np.clip(round(x), 0, w - 1)), int(np.clip(round(y), 0, h - 1))
        y1, y2, x1, x2 = max(0, yi - 3), yi + 4, max(0, xi - 3), xi + 4
        hits = [m for m in self.proposals(rgb, 0.0005, max_area) if m[y1:y2, x1:x2].any()]
        return sorted(hits, key=lambda m: int(m.sum()))

    def detect(
        self,
        rgb: np.ndarray,
        templates: Templates,
        min_area: float = 0.002,
        max_area: float = 0.8,
        top: int = 5,
    ) -> tuple[list[Candidate], dict[str, float]]:
        """Proposals ranked by appearance x shape match to the templates (best first), and step timings in ms."""
        timings: dict[str, float] = {}
        started = time.perf_counter()
        masks = self.proposals(rgb, min_area, max_area)
        timings["proposals"] = (time.perf_counter() - started) * 1000.0
        if not masks:
            return [], timings

        started = time.perf_counter()
        keep, scores, shape_iou = self._match(rgb, masks, templates)
        masks = [masks[i] for i in keep]
        merged = touching_unions(masks, np.argsort(-(scores * shape_iou))[:MERGE_SEEDS], max_area, MERGE_MAX)
        if merged:
            merged_keep, merged_scores, merged_shape = self._match(rgb, merged, templates)
            masks += [merged[i] for i in merged_keep]
            scores = np.concatenate([scores, merged_scores])
            shape_iou = np.concatenate([shape_iou, merged_shape])
        keep = list(range(len(masks)))
        timings["matching"] = (time.perf_counter() - started) * 1000.0

        total = rgb.shape[0] * rgb.shape[1]
        order = prefer_whole(masks, scores, shape_iou)
        ranked = []
        for idx in order[:top]:
            mask = masks[keep[idx]]
            ranked.append(
                Candidate(
                    bbox=tuple(float(v) for v in mask_box(mask)),
                    score=float(scores[idx]),
                    shape=float(shape_iou[idx]),
                    area_fraction=float(mask.sum() / total),
                    polygon=mask_polygon(mask),
                )
            )
        return ranked, timings
