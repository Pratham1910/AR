import numpy as np
import pytest

from app.services.metrology.gap_measurement import is_within_tolerance, measure_gap, measure_min_distance


def test_measures_known_gap_between_two_parallel_planes():
    """Two 5x5 grids of points, 2cm apart along Z — a known, exact ground truth."""
    xs, ys = np.meshgrid(np.linspace(-0.05, 0.05, 5), np.linspace(-0.05, 0.05, 5))
    plane_a = np.stack([xs.ravel(), ys.ravel(), np.zeros(25)], axis=-1)
    plane_b = np.stack([xs.ravel(), ys.ravel(), np.full(25, 0.02)], axis=-1)

    distance = measure_min_distance(plane_a, plane_b)
    assert distance == pytest.approx(0.02, abs=1e-6)


def test_zero_gap_for_touching_clouds():
    points_a = np.array([[0.0, 0.0, 0.0]])
    points_b = np.array([[0.0, 0.0, 0.0]])
    assert measure_min_distance(points_a, points_b) == pytest.approx(0.0, abs=1e-9)


def test_distance_is_the_minimum_not_average():
    points_a = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])  # one far outlier
    points_b = np.array([[0.0, 0.0, 0.01]])  # 1cm from the close point
    assert measure_min_distance(points_a, points_b) == pytest.approx(0.01, abs=1e-6)


def test_empty_cloud_raises():
    with pytest.raises(ValueError):
        measure_min_distance(np.empty((0, 3)), np.array([[0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        measure_min_distance(np.array([[0.0, 0.0, 0.0]]), np.empty((0, 3)))


def test_is_within_tolerance():
    assert is_within_tolerance(5.2, required_m=5.0, tolerance_m=0.5) is True
    assert is_within_tolerance(6.1, required_m=5.0, tolerance_m=0.5) is False
    assert is_within_tolerance(5.5, required_m=5.0, tolerance_m=0.5) is True  # exactly at the boundary


def test_measure_gap_reports_within_tolerance():
    plane_a = np.array([[0.0, 0.0, 0.0]])
    plane_b = np.array([[0.0, 0.0, 0.02]])
    result = measure_gap(plane_a, plane_b, required_m=0.02, tolerance_m=0.002)

    assert result.distance_m == pytest.approx(0.02, abs=1e-6)
    assert result.within_tolerance is True


def test_measure_gap_reports_out_of_tolerance():
    plane_a = np.array([[0.0, 0.0, 0.0]])
    plane_b = np.array([[0.0, 0.0, 0.05]])  # way outside the required 2cm +/- 0.2mm
    result = measure_gap(plane_a, plane_b, required_m=0.02, tolerance_m=0.0002)

    assert result.within_tolerance is False


def test_measure_gap_without_requirement_returns_none_for_within_tolerance():
    plane_a = np.array([[0.0, 0.0, 0.0]])
    plane_b = np.array([[0.0, 0.0, 0.02]])
    result = measure_gap(plane_a, plane_b)

    assert result.within_tolerance is None
