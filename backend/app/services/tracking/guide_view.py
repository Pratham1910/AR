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

from app.services.pose.edge_align import EdgeModel, _fit_score, _project, _sample, _unknown_at, align, edge_image, visible_edge_points

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
    t_co: np.ndarray  # the pose checked (refined onto the edges when it was close)
    score: float  # edge fit, 0..1 (edge_align._fit_score)
    support: float
    matched: bool  # good enough to lock
    # Operator feedback at the GUIDE pose (what the outline shows): visible
    # model edge points as (x, y, on_an_image_edge) — green / red on screen.
    feedback: np.ndarray | None = None
    quick: bool = False  # a re-score of an earlier match, not a full refinement


FEEDBACK_POINTS = 500
FEEDBACK_NEAR_PX = 5.0


LOOSE_NEAR_PX = 12.0  # "roughly lined up": worth a full refinement


def _feedback(model: EdgeModel, guide: np.ndarray, camera_matrix: np.ndarray, edges) -> tuple[np.ndarray | None, float]:
    """(points [x, y, on an edge] at the guide pose, share of them within
    LOOSE_NEAR_PX of some image edge — the cheap "is it roughly there" test)."""
    mask = visible_edge_points(model, guide, camera_matrix, budget=FEEDBACK_POINTS)
    if not mask.any():
        return None, 0.0
    px = _project(model, guide, camera_matrix, mask)
    d = _sample(edges.distance, px, outside=99.0)
    known = ~_unknown_at(edges, px)
    on_edge = (d <= FEEDBACK_NEAR_PX) & known
    loose = float(((d <= LOOSE_NEAR_PX) & known).mean())
    return np.column_stack([px, on_edge.astype(float)]), loose


class GuideMatcher:
    """Background checks of whether the object is lined up with the guide.

    Cost per check: far off (under `min_loose` of the outline within 12 px of
    any image edge) — one score (~0.1 s), no refinement; close — a full edge refinement from the guide
    pose (~0.3-0.7 s); confirming an earlier match — the matched pose
    re-scored on a new frame (~0.1 s). So a lock takes one refinement plus
    one re-score instead of two refinements."""

    def __init__(self, min_score: float = 0.58, min_support: float = 0.55, confirmations: int = 2,
                 agree_m: float = 0.02, agree_deg: float = 6.0, min_loose: float = 0.35):
        self.min_score = min_score
        self.min_support = min_support
        self.confirmations = confirmations
        self.agree_m = agree_m
        self.agree_deg = agree_deg
        self.min_loose = min_loose
        self._job: Future | None = None
        self._streak: list[GuideCheck] = []
        self.last: GuideCheck | None = None

    def reset(self) -> None:
        self._job = None
        self._streak = []
        self.last = None

    @staticmethod
    def check(model: EdgeModel, guide: np.ndarray, gray: np.ndarray, hand_mask, camera_matrix: np.ndarray,
              min_score: float, min_support: float, verify: np.ndarray | None = None,
              min_loose: float = 0.35) -> GuideCheck:
        """One frame: how well the object sits on the guide, refined onto the
        edges when it's close; or (`verify`) whether an earlier match still holds."""
        edges = edge_image(gray, exclude=hand_mask)
        feedback, loose = _feedback(model, guide, camera_matrix, edges)
        if verify is not None:
            score, support = _fit_score(model, verify, camera_matrix, edges, 3.0)[:2]
            return GuideCheck(verify, score, support, score >= min_score and support >= min_support, feedback, quick=True)
        if loose < min_loose:  # most of the outline isn't near any edge: nothing lined up yet
            pre_score, pre_support = _fit_score(model, guide, camera_matrix, edges, 3.0)[:2]
            return GuideCheck(guide, pre_score, pre_support, False, feedback)
        fit = align(
            model, guide, camera_matrix, edges,
            tilt_starts_deg=0.0, max_rotation_deg=25.0, max_translation_frac=0.3,
        )
        if fit is None:
            return GuideCheck(guide, 0.0, 0.0, False, feedback)
        return GuideCheck(fit.t_co, fit.score, fit.support, fit.score >= min_score and fit.support >= min_support, feedback)

    def _agrees(self, a: GuideCheck, b: GuideCheck) -> bool:
        r = a.t_co[:3, :3] @ b.t_co[:3, :3].T
        angle = float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))
        return float(np.linalg.norm(a.t_co[:3, 3] - b.t_co[:3, 3])) <= self.agree_m and angle <= self.agree_deg

    def poll(self, model: EdgeModel, guide: np.ndarray, gray: np.ndarray, hand_mask, camera_matrix: np.ndarray,
             run_inline: bool = False) -> GuideCheck | None:
        """Start a check on this frame if none is running; when one finishes,
        return the pose to lock onto once `confirmations` checks in a row
        matched and agree (an object held in place, not a passing look-alike).
        After the first match, the next checks re-score that pose on new frames."""
        if self._job is None:
            verify = self._streak[-1].t_co if self._streak else None
            args = (model, guide, gray.copy(), hand_mask, camera_matrix, self.min_score, self.min_support,
                    verify, self.min_loose)
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
