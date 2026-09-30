import numpy as np
import pytest

from app.services.metrology.depth_to_points import depth_to_points
from app.services.pose.calibration import CameraCalibration

WIDTH, HEIGHT = 64, 48
CALIBRATION = CameraCalibration.approximate(WIDTH, HEIGHT)


def test_back_projects_a_flat_plane_at_known_depth():
    depth = np.full((HEIGHT, WIDTH), 0.5, dtype=np.float32)
    points = depth_to_points(depth, CALIBRATION)

    assert points.shape == (HEIGHT * WIDTH, 3)
    assert np.allclose(points[:, 2], 0.5)


def test_mask_restricts_to_region():
    depth = np.full((HEIGHT, WIDTH), 0.5, dtype=np.float32)
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[10:20, 10:20] = True  # 10x10 region

    points = depth_to_points(depth, CALIBRATION, mask=mask)
    assert points.shape == (100, 3)


def test_zero_depth_pixels_are_excluded():
    depth = np.full((HEIGHT, WIDTH), 0.5, dtype=np.float32)
    depth[0, 0] = 0.0
    points = depth_to_points(depth, CALIBRATION)
    assert points.shape[0] == HEIGHT * WIDTH - 1


def test_center_pixel_projects_near_the_optical_axis():
    """The approximate calibration's principal point is exactly the image
    center, so a single depth sample there should back-project to (~0, ~0, z)."""
    depth = np.full((HEIGHT, WIDTH), 1.0, dtype=np.float32)
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    center_row, center_col = HEIGHT // 2, WIDTH // 2
    mask[center_row, center_col] = True

    points = depth_to_points(depth, CALIBRATION, mask=mask)

    assert points.shape == (1, 3)
    assert abs(points[0, 0]) < 0.03
    assert abs(points[0, 1]) < 0.03
    assert points[0, 2] == 1.0


def test_off_center_pixel_has_nonzero_xy_proportional_to_depth():
    """Doubling depth at a fixed pixel should double the back-projected X/Y (pinhole similar-triangles)."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[5, 50] = True

    depth_near = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    depth_near[5, 50] = 1.0
    depth_far = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    depth_far[5, 50] = 2.0

    points_near = depth_to_points(depth_near, CALIBRATION, mask=mask)
    points_far = depth_to_points(depth_far, CALIBRATION, mask=mask)

    assert points_far[0, 0] == pytest.approx(points_near[0, 0] * 2, abs=1e-6)
    assert points_far[0, 1] == pytest.approx(points_near[0, 1] * 2, abs=1e-6)
