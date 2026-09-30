import pytest
from fastapi import HTTPException

from app.api import vision
from app.schemas.vision import BoundingBox, SegmentedObject
from app.services.vision.segmentation import MockSegmenter


@pytest.fixture
def known_classes(monkeypatch):
    objects = [
        SegmentedObject(class_label=label, confidence=0.9, bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1), polygon=[])
        for label in ("cup", "bottle", "teddy bear")
    ]
    monkeypatch.setattr(vision, "_segmenter", MockSegmenter(objects))


def test_detectable_classes_come_from_the_segmenter(known_classes):
    assert sorted(vision.detectable_classes()) == ["bottle", "cup", "teddy bear"]


@pytest.mark.parametrize("typed, stored", [("cup", "cup"), ("Bottle", "bottle"), ("  TEDDY BEAR ", "teddy bear")])
def test_class_label_is_matched_case_insensitively(known_classes, typed, stored):
    assert vision.normalize_class_label(typed) == stored


@pytest.mark.parametrize("typed", ["bot", "Ani", "rt", ""])
def test_unknown_class_is_rejected_with_the_valid_choices(known_classes, typed):
    with pytest.raises(HTTPException) as exc:
        vision.normalize_class_label(typed)
    assert exc.value.status_code == 400
    assert "bottle, cup, teddy bear" in exc.value.detail
