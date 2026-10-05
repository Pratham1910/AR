"""Model fit report: does the GLB have the real object's shape?"""

import cv2
import numpy as np
import pytest

from app.services.vision.model_fit import fit_report

H, W = 400, 300


def _bottle(body_w: int, cap_w: int, top: int = 50, cap_h: int = 40, bottom: int = 350, cx: int = 150) -> dict[str, np.ndarray]:
    body = np.zeros((H, W), bool)
    body[top + cap_h : bottom, cx - body_w // 2 : cx + body_w // 2] = True
    cap = np.zeros((H, W), bool)
    cap[top : top + cap_h, cx - cap_w // 2 : cx + cap_w // 2] = True
    return {"Body": body, "Cap": cap}


def _union(parts: dict[str, np.ndarray]) -> np.ndarray:
    out = np.zeros((H, W), bool)
    for m in parts.values():
        out |= m
    return out


def test_identical_shapes_are_a_good_fit():
    parts = _bottle(100, 70)
    report = fit_report(_union(parts), parts, cm_per_px=0.1)
    assert report.good_fit and report.iou == pytest.approx(1.0)
    assert report.findings[0].startswith("Good fit")


def test_a_narrower_model_cap_is_named_with_its_size_difference():
    real = _union(_bottle(100, 70))  # real cap 7.0 cm wide at 0.1 cm/px
    model = _bottle(100, 56)  # model cap 5.6 cm
    report = fit_report(real, model, cm_per_px=0.1)
    cap = next(p for p in report.parts if p.part == "Cap")
    assert cap.real_cm == pytest.approx(7.0, abs=0.1) and cap.model_cm == pytest.approx(5.6, abs=0.1)
    assert cap.diff_rel == pytest.approx(-0.2, abs=0.02)
    assert report.findings[0].startswith("Cap: model 1.4 cm narrower (-20%)")
    body = next(p for p in report.parts if p.part == "Body")
    assert abs(body.diff_cm) < 0.05  # the body is fine and not reported
    assert not any(f.startswith("Body") for f in report.findings)


def test_a_taller_model_says_where_it_sticks_out():
    real = _union(_bottle(100, 70, bottom=350))
    model = _bottle(100, 70, bottom=360)  # 1 cm longer at the bottom
    report = fit_report(real, model, cm_per_px=0.1)
    assert report.model_height_cm - report.real_height_cm == pytest.approx(1.0, abs=0.05)
    assert any("1.0 cm taller" in f and "below the bottom" in f for f in report.findings)


def test_a_tilted_object_is_measured_along_its_own_axis():
    real_parts, model_parts = _bottle(100, 70), _bottle(100, 56)
    rotation = cv2.getRotationMatrix2D((150, 200), 20, 1.0)

    def tilt(m):
        return cv2.warpAffine(m.astype(np.uint8), rotation, (W, H), flags=cv2.INTER_NEAREST).astype(bool)

    report = fit_report(tilt(_union(real_parts)), {k: tilt(v) for k, v in model_parts.items()}, cm_per_px=0.1)
    cap = next(p for p in report.parts if p.part == "Cap")
    assert cap.real_cm == pytest.approx(7.0, abs=0.3) and cap.model_cm == pytest.approx(5.6, abs=0.3)
