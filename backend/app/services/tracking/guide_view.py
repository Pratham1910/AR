"""
Guide view (Vuforia "Model Target" style initialisation): nothing is detected.
The model's outline is shown at a known pose; the operator lines the real
object up with it; once the model's 3D edges sit on the image's edges there,
tracking starts from that pose (refined onto the edges).

Why: finding a plain object by appearance (FastSAM regions vs renders) is
unreliable — on the user's matte black case the chosen region swung between
the case, its glossy end panel, wall patches and "case + half the table".
A known starting pose turns detection into a yes/no check of one pose, which
3D edges answer well.

    default_guide_pose: a sensible guide when none was saved — the object's
        longer side across the image, seen from `elevation_deg` above, at the
        distance where it fills about `fill` of the image.
    GuideMatcher: checks frames in the background (an edge alignment from the
        guide pose takes ~0.3-1 s) and reports a lock once `confirmations`
        checks in a row agree.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from app.services.pose.edge_align import EdgeModel, align, edge_image

_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="guide-view")


def _look_at(eye: np.ndarray, up: np.ndarray) -> np.ndarray:
    """T_camera_object for a camera at `eye` (object frame) looking at the
    object's origin, image-up along `up` (OpenCV camera: x right, y down, z forward)."""
    z = -eye / np.linalg.norm(eye)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.stack([x, y, z])
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = -rotation @ eye
    return t


def default_guide_pose(
    vertices: np.ndarray, camera_matrix: np.ndarray, image_size: tuple[int, int], resting: str = "any",
    fill: float = 0.55,
) -> np.ndarray:
    """The guide pose when none was saved for the model. The model's +Y is
    its top: seen from 45 deg above for a part lying on the bench ("camera",
    "flat", "any"), 15 deg for one standing ("upright"); the camera faces the
    model's narrower side so its longer side runs across the image."""
    pts = np.asarray(vertices, float)
    extent = pts.max(axis=0) - pts.min(axis=0)
    elevation = np.radians(15.0 if resting == "upright" else 45.0)
    # Look along the shorter horizontal axis: the longer one then spans the image.
    across_x = extent[0] >= extent[2]
    horizontal = np.array([0.0, 0.0, 1.0]) if across_x else np.array([1.0, 0.0, 0.0])
    direction = np.cos(elevation) * horizontal + np.sin(elevation) * np.array([0.0, 1.0, 0.0])
    width, height = image_size
    focal = float(camera_matrix[1, 1])
    radius = float(np.linalg.norm(extent)) / 2.0 or 0.05
    distance = focal * 2.0 * radius / (fill * min(width, height))
    return _look_at(direction * distance, np.array([0.0, 1.0, 0.0]))


@dataclass
class GuideCheck:
    t_co: np.ndarray  # the refined pose (assembly frame)
    score: float  # edge fit, 0..1 (edge_align._fit_score)
    support: float
    matched: bool  # good enough to lock


class GuideMatcher:
    """Background checks of whether the object is lined up with the guide."""

    def __init__(self, min_score: float = 0.58, min_support: float = 0.55, confirmations: int = 2,
                 agree_m: float = 0.02, agree_deg: float = 6.0):
        self.min_score = min_score
        self.min_support = min_support
        self.confirmations = confirmations
        self.agree_m = agree_m
        self.agree_deg = agree_deg
        self._job: Future | None = None
        self._streak: list[GuideCheck] = []
        self.last: GuideCheck | None = None

    def reset(self) -> None:
        self._job = None
        self._streak = []
        self.last = None

    @staticmethod
    def check(model: EdgeModel, guide: np.ndarray, gray: np.ndarray, hand_mask, camera_matrix: np.ndarray,
              min_score: float, min_support: float) -> GuideCheck:
        """One frame: refine from the guide pose onto the edges and judge the fit."""
        fit = align(
            model, guide, camera_matrix, edge_image(gray, exclude=hand_mask),
            tilt_starts_deg=0.0, max_rotation_deg=25.0, max_translation_frac=0.3,
        )
        if fit is None:
            return GuideCheck(guide, 0.0, 0.0, False)
        return GuideCheck(fit.t_co, fit.score, fit.support, fit.score >= min_score and fit.support >= min_support)

    def _agrees(self, a: GuideCheck, b: GuideCheck) -> bool:
        r = a.t_co[:3, :3] @ b.t_co[:3, :3].T
        angle = float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))
        return float(np.linalg.norm(a.t_co[:3, 3] - b.t_co[:3, 3])) <= self.agree_m and angle <= self.agree_deg

    def poll(self, model: EdgeModel, guide: np.ndarray, gray: np.ndarray, hand_mask, camera_matrix: np.ndarray,
             run_inline: bool = False) -> GuideCheck | None:
        """Start a check on this frame if none is running; when one finishes,
        return the pose to lock onto once `confirmations` checks in a row
        matched and agree (an object held in place, not a passing look-alike)."""
        if self._job is None:
            args = (model, guide, gray.copy(), hand_mask, camera_matrix, self.min_score, self.min_support)
            if run_inline:
                self._job = Future()
                self._job.set_result(self.check(*args))
            else:
                self._job = _pool.submit(self.check, *args)
        if not self._job.done():
            return None
        result: GuideCheck = self._job.result()
        self._job = None
        self.last = result
        if not result.matched:
            self._streak = []
            return None
        if self._streak and not self._agrees(self._streak[-1], result):
            self._streak = []
        self._streak.append(result)
        if len(self._streak) >= self.confirmations:
            self._streak = []
            return result
        return None
