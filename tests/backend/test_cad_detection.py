"""Finding an object by its 3D model (no class): pose_service/cad_detector.py and the backend's use of /detect."""

import json
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
import torch

from app.services.pose.model_pose_client import ModelPoseClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pose_service"))
from cad_detector import CadDetector, mask_polygon, masked_square_crop  # noqa: E402


# ── pose service: matching ───────────────────────────────────────────────


def test_crop_keeps_only_the_masked_region_centred_on_a_square():
    rgb = np.full((100, 200, 3), 255, np.uint8)
    mask = np.zeros((100, 200), bool)
    mask[10:90, 50:90] = True  # 80 tall x 40 wide
    crop = masked_square_crop(rgb, mask, size=80)
    assert crop.shape == (80, 80, 3)
    # Padded left/right to a square: the 40-wide region sits in the middle half.
    assert crop[:, 25:55].min() == 255
    assert crop[:, :15].max() == 0 and crop[:, -15:].max() == 0


def test_empty_mask_gives_no_crop():
    assert masked_square_crop(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), bool)) is None


def test_polygon_outlines_the_mask():
    mask = np.zeros((50, 50), bool)
    mask[10:40, 20:30] = True
    xs, ys = zip(*mask_polygon(mask))
    assert (min(xs), max(xs), min(ys), max(ys)) == (20, 29, 10, 39)


class _StubDetector(CadDetector):
    """FastSAM / DINOv2 replaced: proposals are given masks, a crop's descriptor is its mean colour."""

    def __init__(self, masks):
        super().__init__(torch.device("cpu"))
        self._masks = masks

    def proposals(self, rgb, min_area, max_area):
        return self._masks

    def descriptors(self, crops):
        return torch.nn.functional.normalize(torch.tensor(np.array([c.reshape(-1, 3).mean(0) for c in crops]), dtype=torch.float32), dim=-1)


def test_the_region_that_looks_like_the_templates_ranks_first():
    rgb = np.zeros((100, 100, 3), np.uint8)
    rgb[10:40, 10:40] = (200, 30, 30)  # red thing
    rgb[60:90, 60:90] = (30, 30, 200)  # blue thing = what the templates look like
    red, blue = np.zeros((100, 100), bool), np.zeros((100, 100), bool)
    red[10:40, 10:40] = True
    blue[60:90, 60:90] = True
    detector = _StubDetector([red, blue])
    templates = torch.nn.functional.normalize(torch.tensor([[0.15, 0.15, 1.0], [0.1, 0.2, 1.0]]), dim=-1)

    ranked, timings = detector.detect(rgb, templates)

    assert [c.bbox for c in ranked] == [(60.0, 60.0, 90.0, 90.0), (10.0, 10.0, 40.0, 40.0)]
    assert ranked[0].score > 0.95 > ranked[1].score
    assert ranked[0].area_fraction == pytest.approx(0.09)
    assert {"proposals", "matching"} <= timings.keys()


def test_no_proposals_means_nothing_found():
    ranked, _ = _StubDetector([]).detect(np.zeros((10, 10, 3), np.uint8), torch.ones(1, 3))
    assert ranked == []


# ── backend: /detect client ──────────────────────────────────────────────


def _client(response: dict, seen: list) -> ModelPoseClient:
    def service(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=response)

    return ModelPoseClient("http://pose", 5.0, transport=httpx.MockTransport(service))


def test_detect_returns_the_best_region_and_sends_the_threshold():
    seen: list = []
    response = {
        "found": True,
        "best": {"bbox": [10, 20, 110, 220], "score": 0.71, "area_fraction": 0.1, "polygon": [[10, 20], [110, 20], [60, 220]]},
        "candidates": [
            {"bbox": [10, 20, 110, 220], "score": 0.71, "area_fraction": 0.1, "polygon": []},
            {"bbox": [0, 0, 5, 5], "score": 0.42, "area_fraction": 0.01, "polygon": []},
        ],
        "elapsed_ms": 120.0,
        "timings_ms": {},
    }
    found = _client(response, seen).detect("m1", b"jpeg", 0.5)
    assert seen[0][0] == "/detect" and seen[0][1]["label"] == "m1" and seen[0][1]["min_score"] == 0.5
    assert found.bbox == (10.0, 20.0, 110.0, 220.0)
    assert (found.score, found.runner_up_score) == (0.71, 0.42)
    assert found.polygon[2] == (60.0, 220.0)


def test_detect_returns_none_when_nothing_matches_well_enough():
    response = {"found": False, "best": None, "candidates": [], "elapsed_ms": 90.0, "timings_ms": {}}
    assert _client(response, []).detect("m1", b"jpeg", 0.5) is None


def test_find_object_by_model_is_a_detection_named_after_the_model(monkeypatch):
    from app.api import vision
    from app.services.pose.model_pose_client import CadDetection

    monkeypatch.setattr(
        vision._model_pose_client,
        "detect",
        lambda label, jpeg, min_score: CadDetection((1, 2, 3, 4), 0.8, [(1, 2), (3, 2), (3, 4)], None, 50.0),
    )
    obj = vision.find_object_by_model(b"jpeg", "m1", "Junction box")
    assert obj.class_label == "Junction box" and obj.confidence == 0.8
    assert (obj.bbox.x1, obj.bbox.y2) == (1, 4) and len(obj.polygon) == 3
