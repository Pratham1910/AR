import numpy as np
import pytest

from app.schemas.vision import BoundingBox
from app.services.pose.calibration import CameraCalibration
from app.services.pose.markerless import estimate_object_placement

# fx = fy = 800, principal point at image center (400, 300) for an 800x600 frame.
CALIBRATION = CameraCalibration(
    camera_matrix=np.array([[800.0, 0, 400.0], [0, 800.0, 300.0], [0, 0, 1]]),
    dist_coeffs=np.zeros(5),
    image_width=800,
    image_height=600,
)


def test_centered_box_maps_to_centered_position():
    """A box centered on the principal point should map to X=0 in front of the camera."""
    bbox = BoundingBox(x1=350, y1=200, x2=450, y2=400)  # centered at (400, 300), height 200px
    pose = estimate_object_placement(bbox, CALIBRATION, real_world_height_m=0.2)

    # z = fy * H / pixel_height = 800 * 0.2 / 200 = 0.8m in front -> Three.js Z = -0.8
    assert pose.position[2] == pytest.approx(-0.8, abs=1e-6)
    assert pose.position[0] == pytest.approx(0.0, abs=1e-6)
    assert pose.quaternion == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-9)  # no orientation estimate


def test_smaller_apparent_size_means_further_away():
    near_bbox = BoundingBox(x1=350, y1=200, x2=450, y2=400)  # 200px tall
    far_bbox = BoundingBox(x1=375, y1=250, x2=425, y2=350)  # 100px tall, same center

    near_pose = estimate_object_placement(near_bbox, CALIBRATION, real_world_height_m=0.2)
    far_pose = estimate_object_placement(far_bbox, CALIBRATION, real_world_height_m=0.2)

    assert abs(far_pose.position[2]) > abs(near_pose.position[2])


def test_off_center_box_produces_nonzero_lateral_offset():
    bbox = BoundingBox(x1=550, y1=200, x2=650, y2=400)  # shifted right of principal point
    pose = estimate_object_placement(bbox, CALIBRATION, real_world_height_m=0.2)
    assert pose.position[0] != pytest.approx(0.0, abs=1e-6)


def test_zero_height_box_raises():
    bbox = BoundingBox(x1=350, y1=300, x2=450, y2=300)  # y1 == y2
    with pytest.raises(ValueError):
        estimate_object_placement(bbox, CALIBRATION, real_world_height_m=0.2)
