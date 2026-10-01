import math

import numpy as np
import pytest

from app.services.tracking.pose_filter import PoseFilter, PoseFilterConfig

DT = 0.05  # 20 Hz


def _yaw(deg: float) -> tuple[float, float, float, float]:
    half = math.radians(deg) / 2
    return (0.0, math.sin(half), 0.0, math.cos(half))


def test_reduces_jitter_on_a_still_object():
    rng = np.random.default_rng(0)
    f = PoseFilter(PoseFilterConfig())
    f.reset((0, 0, -0.4), (0, 0, 0, 1), 0.0)
    xs = [f.update((rng.normal(0, 0.004), 0, -0.4), (0, 0, 0, 1), k * DT)[0][0] for k in range(1, 101)]
    assert np.std(xs[50:]) < 0.004 * 0.7


def test_follows_steady_motion_without_lagging_behind():
    rng = np.random.default_rng(1)
    f = PoseFilter(PoseFilterConfig())
    f.reset((0, 0, -0.4), (0, 0, 0, 1), 0.0)
    for k in range(1, 41):
        true_x = 0.2 * k * DT
        x = f.update((true_x + rng.normal(0, 0.004), 0, -0.4), (0, 0, 0, 1), k * DT)[0][0]
    assert abs(true_x - x) < 0.006  # an exponential average at this rate would trail by ~3 cm


def test_follows_a_rotation_within_a_few_frames():
    f = PoseFilter(PoseFilterConfig())
    f.reset((0, 0, -0.4), (0, 0, 0, 1), 0.0)
    for k in range(1, 5):  # 200 ms
        q = f.update((0, 0, -0.4), _yaw(30), k * DT)[1]
    assert 2 * math.degrees(math.asin(q[1])) > 27


def test_quaternion_sign_flip_is_the_same_rotation_not_a_spin():
    f = PoseFilter(PoseFilterConfig())
    f.reset((0, 0, -0.4), _yaw(10), 0.0)
    flipped = tuple(-v for v in _yaw(10))
    q = f.update((0, 0, -0.4), flipped, DT)[1]
    assert abs(abs(np.dot(q, _yaw(10))) - 1.0) < 1e-6


def test_disabled_filter_passes_measurements_through():
    f = PoseFilter(PoseFilterConfig(enabled=False))
    position, quaternion = f.update((0.1, 0.2, -0.3), _yaw(20), 1.0)
    assert position == pytest.approx((0.1, 0.2, -0.3))
    assert quaternion == pytest.approx(_yaw(20))
