"""
Choosing between a pose's symmetric twins (pose/symmetry.py) by the model's
distinctive features (pose/pose_features.py).

Every geometric method — the outline, 3D edges, MegaPose on an untextured
mesh — fits a symmetric box equally well turned 180 degrees, so the twin it
returns is arbitrary: the "circle at top-right shows at top-left" flip. Here
each twin is scored by whether the image shows the model's features where
that twin puts them:

    confirmed  — one twin's features are clearly seen and clearly better than
                 every other twin's (`min_evidence`, `min_margin`); or, for a
                 NEAR-symmetric part (a step, a chamfer at one end), one twin
                 fits the 3D edges clearly better (`min_edge_margin`) and the
                 features don't say otherwise;
    unique     — the model has no symmetry: the pose needs no choosing;
    ambiguous  — the evidence doesn't separate the twins (no features
                 configured, features covered / out of view / not visible).
                 Never a guess presented as fact: the caller keeps the twin
                 closest to the last reliable pose (or the given one) and
                 reports the orientation as unconfirmed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from app.services.pose.edge_align import EdgeImage, EdgeModel, _fit_score
from app.services.pose.pose_features import MEASURABLE, FeatureObservation, PoseFeature, observe_features
from app.services.pose.symmetry import closest_twin, rotation_angle_deg, twins

OrientationStatus = Literal["unique", "confirmed", "ambiguous"]


@dataclass
class OrientationDecision:
    t_co: np.ndarray
    status: OrientationStatus
    chosen: int  # which twin (0 = the pose given)
    scores: list[float | None]  # feature evidence per twin (None: excluded, e.g. wrong resting side)
    margin: float  # best minus second-best score
    observations: list[FeatureObservation] = field(default_factory=list)  # at the chosen pose
    reason: str = ""
    required_unconfirmed: list[str] = field(default_factory=list)  # required features not seen at the chosen pose

    @property
    def flipped(self) -> bool:
        return self.chosen != 0


def _score(observations: list[FeatureObservation], features: dict[str, PoseFeature]) -> tuple[float, int]:
    """(weighted mean evidence over the features this pose lets the camera
    judge, how many there were). A feature this pose puts out of sight
    counts as 0: it can't vouch for the pose."""
    total = weight = 0.0
    measurable = 0
    for o in observations:
        w = features[o.feature_id].weight
        weight += w
        if o.status in MEASURABLE:
            total += w * o.evidence
            measurable += 1
    return (total / weight if weight else 0.0), measurable


def choose_orientation(
    t_co: np.ndarray,
    symmetries: list[np.ndarray],
    features: list[PoseFeature],
    camera_matrix: np.ndarray,
    gray: np.ndarray,
    edges: EdgeImage,
    model: EdgeModel | None = None,
    reference: np.ndarray | None = None,
    allowed: Callable[[np.ndarray], bool] | None = None,
    min_evidence: float = 0.35,
    min_margin: float = 0.15,
    min_edge_margin: float = 0.06,
) -> OrientationDecision:
    """Pick the twin of `t_co` the image supports (see module doc).
    `reference`: the last reliable pose, for keeping the orientation while
    the evidence is inconclusive. `allowed`: station constraints (resting side)."""
    enabled = [f for f in features if f.enabled]
    by_id = {f.feature_id: f for f in enabled}
    candidates = twins(t_co, symmetries)
    usable = [i for i, c in enumerate(candidates) if allowed is None or allowed(c) or i == 0]

    def required_missing(obs: list[FeatureObservation]) -> list[str]:
        return [o.feature_id for o in obs if by_id[o.feature_id].required_for_initialization and o.status != "detected"]

    if len(symmetries) <= 1:
        obs = observe_features(enabled, t_co, camera_matrix, gray, edges, model) if enabled else []
        return OrientationDecision(np.array(t_co, float), "unique", 0, [None], 0.0, obs,
                                   "the model has no symmetric twin", required_missing(obs))

    def fallback(reason: str, scores: list[float | None], margin: float) -> OrientationDecision:
        if reference is not None:
            pose, j = closest_twin(t_co, reference, [symmetries[i] for i in usable])
            chosen = usable[j]
        else:
            pose, chosen = np.array(t_co, float), 0
        obs = observe_features(enabled, pose, camera_matrix, gray, edges, model) if enabled else []
        return OrientationDecision(pose, "ambiguous", chosen, scores, margin, obs, reason, required_missing(obs))

    # Geometry: a near-symmetric part's twins don't fit the 3D edges equally.
    geometry_choice = None
    if model is not None and len(usable) > 1:
        fits = {i: _fit_score(model, candidates[i], camera_matrix, edges, 3.0)[0] for i in usable}
        order = sorted(usable, key=lambda i: -fits[i])
        if fits[order[0]] - fits[order[1]] >= min_edge_margin:
            geometry_choice = order[0]

    if not enabled:
        if geometry_choice is not None:
            return OrientationDecision(candidates[geometry_choice], "confirmed", geometry_choice, [None] * len(candidates),
                                       0.0, [], "the part's 3D edges fit this orientation clearly better")
        return fallback("symmetric model and no pose features configured", [None] * len(candidates), 0.0)

    scores: list[float | None] = [None] * len(candidates)
    observed: dict[int, list[FeatureObservation]] = {}
    for i in usable:
        obs = observe_features(enabled, candidates[i], camera_matrix, gray, edges, model)
        observed[i] = obs
        scores[i] = _score(obs, by_id)[0]
    ranked = sorted(usable, key=lambda i: -(scores[i] or 0.0))
    best = ranked[0]
    second = scores[ranked[1]] if len(ranked) > 1 else 0.0
    margin = float((scores[best] or 0.0) - (second or 0.0))
    if len(ranked) == 1:  # the station's resting side already rules the other twins out
        return OrientationDecision(candidates[best], "confirmed", best, scores, margin, observed[best],
                                   "only one twin fits the station's resting side", required_missing(observed[best]))
    features_decide = (scores[best] or 0.0) >= min_evidence and margin >= min_margin
    if features_decide and geometry_choice not in (None, best):
        return fallback("features and 3D edges point to different orientations", scores, margin)
    if features_decide:
        return OrientationDecision(candidates[best], "confirmed", best, scores, margin, observed[best],
                                   "features seen where this orientation puts them", required_missing(observed[best]))
    if geometry_choice is not None:
        return OrientationDecision(candidates[geometry_choice], "confirmed", geometry_choice, scores, margin,
                                   observed[geometry_choice], "the part's 3D edges fit this orientation clearly better",
                                   required_missing(observed[geometry_choice]))
    if (scores[best] or 0.0) < min_evidence:
        reason = "features not seen (covered, out of view, or too faint)"
    else:
        reason = "features fit more than one orientation"
    return fallback(reason, scores, margin)


def is_flip(a: np.ndarray, b: np.ndarray, symmetries: list[np.ndarray], tolerance_deg: float = 20.0) -> bool:
    """True when `b` is (close to) a non-identity twin of `a`: the same fit,
    the object turned — a discrete orientation change, not motion."""
    if len(symmetries) <= 1:
        return False
    _, index = closest_twin(b, a, symmetries)
    return index != 0 and rotation_angle_deg(twins(b, symmetries)[index], a) < tolerance_deg
