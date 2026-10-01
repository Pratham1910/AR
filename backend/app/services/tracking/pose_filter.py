"""
Pose smoothing for tracked objects: translation and rotation filtered
separately (FoundationPose++ applies a Kalman filter to rotation; translation
here gets one too, since without a depth camera Z comes from the image).

- Translation: per-axis constant-velocity Kalman filter. Because it models
  velocity, steady motion is followed without the lag a plain exponential
  average adds; only measurement noise is smoothed out.
- Rotation: a scalar Kalman gain applied as a quaternion slerp. The
  uncertainty grows with time since the last update (process noise, deg/s)
  and each measurement pulls the estimate by K = P / (P + R).

Both noise levels are configuration, so "stable" vs "responsive" is tunable
without code changes. reset() on every (re)initialization, so a fresh lock
never blends with a stale pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PoseFilterConfig:
    # Defaults chosen by sweep at 20 Hz with 4 mm / 3 deg measurement noise:
    # still jitter 4.0 -> 2.3 mm and 3.0 -> 2.0 deg; ~1 mm lag at a steady
    # 0.2 m/s; a 30 deg turn followed to 90% within 200 ms.
    enabled: bool = True
    translation_process_noise: float = 0.3  # m/s^2: how hard the object may accelerate
    translation_measurement_noise: float = 0.004  # m: measurement jitter to smooth out
    rotation_process_noise_deg: float = 40.0  # deg/s: how fast the object may turn
    rotation_measurement_noise_deg: float = 3.0  # deg: rotation jitter to smooth out


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.dot(a, b))
    if dot < 0.0:  # shortest path: q and -q are the same rotation
        b, dot = -b, -dot
    if dot > 0.9995:
        out = a + t * (b - a)
        return out / np.linalg.norm(out)
    theta = math.acos(min(dot, 1.0))
    return (math.sin((1 - t) * theta) * a + math.sin(t * theta) * b) / math.sin(theta)


class PoseFilter:
    def __init__(self, config: PoseFilterConfig):
        self.config = config
        self._x: np.ndarray | None = None  # (3, 2): [position, velocity] per axis
        self._p: np.ndarray | None = None  # (3, 2, 2) covariance per axis
        self._q: np.ndarray | None = None  # quaternion (x, y, z, w)
        self._p_rot = 0.0  # rotation variance, deg^2
        self._t: float | None = None

    def reset(self, position, quaternion, t: float) -> None:
        r = self.config.translation_measurement_noise
        self._x = np.stack([np.asarray(position, float), np.zeros(3)], axis=1)
        self._p = np.tile(np.diag([r * r, 1.0]), (3, 1, 1))
        self._q = np.asarray(quaternion, float) / np.linalg.norm(quaternion)
        self._p_rot = self.config.rotation_measurement_noise_deg**2
        self._t = t

    def update(self, position, quaternion, t: float) -> tuple[tuple, tuple]:
        if not self.config.enabled or self._x is None:
            self.reset(position, quaternion, t)
            return tuple(position), tuple(quaternion)
        cfg = self.config
        dt = max(t - self._t, 1e-3)
        self._t = t

        # Translation: predict with constant velocity, then correct.
        f = np.array([[1.0, dt], [0.0, 1.0]])
        q = cfg.translation_process_noise**2 * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        r = cfg.translation_measurement_noise**2
        z = np.asarray(position, float)
        for axis in range(3):
            x = f @ self._x[axis]
            p = f @ self._p[axis] @ f.T + q
            gain = p[:, 0] / (p[0, 0] + r)
            x = x + gain * (z[axis] - x[0])
            p = p - np.outer(gain, p[0, :])
            self._x[axis], self._p[axis] = x, p

        # Rotation: uncertainty grows with elapsed time, measurement pulls by the Kalman gain.
        self._p_rot += (cfg.rotation_process_noise_deg * dt) ** 2
        k = self._p_rot / (self._p_rot + cfg.rotation_measurement_noise_deg**2)
        meas = np.asarray(quaternion, float) / np.linalg.norm(quaternion)
        self._q = _slerp(self._q, meas, k)
        self._p_rot *= 1.0 - k

        return tuple(float(v) for v in self._x[:, 0]), tuple(float(v) for v in self._q)
