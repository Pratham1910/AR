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
from cad_detector import CadDetector, Templates, mask_polygon, masked_square_crop, square_shape, prefer_whole, touching_unions  # noqa: E402


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


def _templates(descriptors, outline_masks):
    return Templates(
        descriptors=torch.nn.functional.normalize(torch.tensor(descriptors, dtype=torch.float32), dim=-1),
        shapes=torch.from_numpy(np.stack([square_shape(m) for m in outline_masks])),
    )


def _box_mask(y1, y2, x1, x2, size=100):
    mask = np.zeros((size, size), bool)
    mask[y1:y2, x1:x2] = True
    return mask


def test_the_region_that_looks_like_the_templates_ranks_first():
    rgb = np.zeros((100, 100, 3), np.uint8)
    rgb[10:40, 10:40] = (200, 30, 30)  # red thing
    rgb[60:90, 60:90] = (30, 30, 200)  # blue thing = what the templates look like
    red, blue = np.zeros((100, 100), bool), np.zeros((100, 100), bool)
    red[10:40, 10:40] = True
    blue[60:90, 60:90] = True
    detector = _StubDetector([red, blue])
    square = _box_mask(0, 50, 0, 50)  # both proposals have the templates' outline: only appearance differs
    templates = _templates([[0.15, 0.15, 1.0], [0.1, 0.2, 1.0]], [square, square])

    ranked, timings = detector.detect(rgb, templates)

    assert [c.bbox for c in ranked] == [(60.0, 60.0, 90.0, 90.0), (10.0, 10.0, 40.0, 40.0)]
    assert ranked[0].score > 0.95 > ranked[1].score
    assert ranked[0].shape == pytest.approx(1.0)
    assert ranked[0].area_fraction == pytest.approx(0.09)
    assert {"proposals", "matching"} <= timings.keys()


def test_the_object_alone_beats_the_object_with_a_hand_attached():
    """A hand holding the bottle: FastSAM proposes the bottle alone and the
    bottle+hand+arm as one region. Both look like the bottle (same colour
    here); only the outline tells them apart."""
    rgb = np.zeros((100, 100, 3), np.uint8)
    bottle = _box_mask(20, 90, 30, 50)  # tall and narrow, like the templates
    with_arm = bottle | _box_mask(50, 70, 50, 100)  # plus an arm reaching in from the right
    rgb[with_arm] = (30, 30, 200)
    detector = _StubDetector([with_arm, bottle])
    templates = _templates([[0.15, 0.15, 1.0]], [_box_mask(10, 80, 40, 60)])

    ranked, _ = detector.detect(rgb, templates)

    assert ranked[0].bbox == (30.0, 20.0, 50.0, 90.0)
    assert ranked[0].score == pytest.approx(ranked[1].score, abs=0.01)  # appearance alone can't decide
    assert ranked[0].shape > 0.9 and ranked[1].shape < 0.5


def test_no_proposals_means_nothing_found():
    ranked, _ = _StubDetector([]).detect(np.zeros((10, 10, 3), np.uint8), _templates([[1.0, 1.0, 1.0]], [_box_mask(0, 5, 0, 5, size=10)]))
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
        lambda label, jpeg, min_score, min_combined=0.0: CadDetection((1, 2, 3, 4), 0.8, [(1, 2), (3, 2), (3, 4)], None, 50.0),
    )
    obj = vision.find_object_by_model(b"jpeg", "m1", "Junction box")
    assert obj.class_label == "Junction box" and obj.confidence == 0.8
    assert (obj.bbox.x1, obj.bbox.y2) == (1, 4) and len(obj.polygon) == 3


def test_touching_segments_are_also_offered_as_one_object():
    """A case whose glossy end cap and matte body come out as two segments."""
    body = np.zeros((200, 300), bool)
    body[60:180, 40:200] = True
    cap = np.zeros_like(body)
    cap[60:180, 200:240] = True  # right against the body
    far = np.zeros_like(body)
    far[10:40, 260:290] = True  # elsewhere
    inside = np.zeros_like(body)
    inside[100:120, 80:100] = True  # a label on the body
    unions = touching_unions([body, cap, far, inside], seeds=[1], max_area=0.8, limit=10)
    assert len(unions) == 1 and (unions[0] == (body | cap)).all()
    assert touching_unions([body, cap], seeds=[0], max_area=0.3, limit=10) == []  # too big a share of the image


def test_the_whole_object_wins_a_near_tie_with_one_of_its_panels():
    whole = np.zeros((200, 300), bool)
    whole[60:180, 40:240] = True
    panel = np.zeros_like(whole)
    panel[60:90, 40:240] = True  # the end cap: a strip, the case's outline seen edge-on
    other = np.zeros_like(whole)
    other[0:20, 0:20] = True
    order = prefer_whole([panel, whole, other], np.array([0.43, 0.56, 0.3]), np.array([0.78, 0.59, 0.5]))
    assert order[0] == 1
    # A clearly worse whole doesn't take over; nor does a bigger region that looks less like it.
    assert prefer_whole([panel, whole], np.array([0.43, 0.56]), np.array([0.78, 0.4]))[0] == 0
    assert prefer_whole([panel, whole], np.array([0.43, 0.40]), np.array([0.78, 0.90]))[0] == 1  # simply better
    assert prefer_whole([panel, whole], np.array([0.45, 0.40]), np.array([0.78, 0.80]))[0] == 0
