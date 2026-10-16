"""
The two PoseTracker implementations used by ARSession (ar_session.py):

- FlowPoseTracker (markerless): optical flow follows the detected outline;
  position from its apparent size (approximate, no orientation).
- MegaPoseTracker (model-based): the initial pose comes from a full MegaPose
  search inside the detected box, then each frame refines from the previous
  pose — real 6DoF tracking with no detector involved.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from functools import cached_property, partial

import cv2
import numpy as np

from app.schemas.vision import BoundingBox, SegmentedObject, Vector2
from app.services.pose.calibration import CameraCalibration
from app.services.pose.disambiguation import OrientationDecision, choose_orientation
from app.services.pose.edge_align import EdgeModel, _fit_score, align, edge_image
from app.services.pose.pose_features import PoseFeature
from app.services.pose.symmetry import closest_twin, rotation_angle_deg
from app.services.pose.markerless import estimate_object_placement
from app.services.pose.model_pose_client import ModelPoseClient, PoseServiceResult
from app.services.pose.transforms import cv_model_pose_to_threejs
from app.services.tracking.ar_session import Measurement
from app.services.tracking.flow_tracker import FlowBoxTracker


class Frame:
    """One camera frame, with the conversions trackers need computed once."""

    def __init__(
        self,
        bgr: np.ndarray,
        calibration: CameraCalibration,
        encoded: bytes | None = None,
        hand_mask: np.ndarray | None = None,
    ):
        self.bgr = bgr
        self.calibration = calibration
        # Hand / arm pixels (app/services/vision/hand_mask.py), when known:
        # kept out of the object's features, outline and pose refinement.
        self.hand_mask = hand_mask
        # The JPEG/PNG the frame arrived as, if any: sent to the pose service
        # as-is instead of re-encoding the decoded pixels (~17 ms at 1280x720).
        self._encoded = encoded

    @cached_property
    def gray(self) -> np.ndarray:
        return cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)

    @cached_property
    def jpeg(self) -> bytes:
        if self._encoded is not None:
            return self._encoded
        ok, data = cv2.imencode(".jpg", self.bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("Could not encode frame as JPEG")
        return data.tobytes()

    def jpeg_without_hands(self) -> bytes:
        """The frame with hand pixels painted neutral grey, for the pose
        refiner: it compares the rendered model with the image and would
        otherwise fit the model to the hand's edges too."""
        if self.hand_mask is None or not self.hand_mask.any():
            return self.jpeg
        painted = self.bgr.copy()
        painted[self.hand_mask] = 128
        ok, data = cv2.imencode(".jpg", painted, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("Could not encode frame as JPEG")
        return data.tobytes()


def without_hand(detection: SegmentedObject, hand_mask: np.ndarray | None, min_kept: float = 0.25) -> tuple[SegmentedObject, float]:
    """The detection's outline with hand pixels removed — a hand holding the
    object must not become part of its contour (Phase 7). Returns the
    cleaned detection and the share of its area that was hand. Kept as is if
    there's no hand on it, or if less than `min_kept` of it would remain
    (the "object" was mostly hand: better judged by the pose check)."""
    if hand_mask is None or not detection.polygon or len(detection.polygon) < 3:
        return detection, 0.0
    h, w = hand_mask.shape
    region = np.zeros((h, w), np.uint8)
    cv2.fillPoly(region, [np.round([[p.x, p.y] for p in detection.polygon]).astype(np.int32)], 1)
    area = int(region.sum())
    covered = int((region.astype(bool) & hand_mask).sum())
    if area == 0 or covered == 0:
        return detection, 0.0
    region[hand_mask] = 0
    if (area - covered) < min_kept * area:
        return detection, covered / area
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    outline = cv2.approxPolyDP(largest, 1.5, True).reshape(-1, 2)
    x, y, bw, bh = cv2.boundingRect(largest)
    cleaned = SegmentedObject(
        class_label=detection.class_label,
        confidence=detection.confidence,
        bbox=BoundingBox(x1=float(x), y1=float(y), x2=float(x + bw), y2=float(y + bh)),
        polygon=[Vector2(x=float(px), y=float(py)) for px, py in outline],
    )
    return cleaned, covered / area


class FlowPoseTracker:
    can_recover = False  # once its points are gone it needs the detector again

    def __init__(self, real_world_height_m: float):
        self.real_world_height_m = real_world_height_m
        self._flow = FlowBoxTracker()

    def _placement(self, frame: Frame, bbox) -> tuple:
        pose = estimate_object_placement(bbox, frame.calibration, self.real_world_height_m)
        return pose.position, pose.quaternion

    def initialize(self, frame: Frame, detection: SegmentedObject) -> Measurement:
        detection, _ = without_hand(detection, frame.hand_mask)
        if not self._flow.initialize(frame.gray, detection.bbox, detection.polygon, frame.hand_mask):
            return Measurement(confidence=0.0)  # too little texture to follow
        position, quaternion = self._placement(frame, detection.bbox)
        return Measurement(
            confidence=detection.confidence,
            position=position,
            quaternion=quaternion,
            bbox=detection.bbox,
            polygon=detection.polygon,
        )

    def update(self, frame: Frame) -> Measurement:
        result = self._flow.update(frame.gray, frame.hand_mask)
        if not result.ok:
            return Measurement(confidence=result.confidence, extra={"occlusion": result.occlusion})
        position, quaternion = self._placement(frame, result.bbox)
        return Measurement(
            confidence=result.confidence,
            position=position,
            quaternion=quaternion,
            bbox=result.bbox,
            polygon=result.polygon,
            extra={"points": result.points, "occlusion": result.occlusion},
        )

    def commit(self, measurement: Measurement) -> None:
        pass  # the flow tracker already advanced to the new frame


def box_iou(a: BoundingBox, b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a.x2, b[2]) - max(a.x1, b[0]))
    iy = max(0.0, min(a.y2, b[3]) - max(a.y1, b[1]))
    inter = ix * iy
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def propagate_pose(t_camera_object: np.ndarray, motion: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    """
    A 6DoF pose (OpenCV camera frame) moved by the object's 2D image motion —
    a similarity transform (3x3: uniform scale s, in-plane rotation theta,
    translation) from optical flow:
      - the object's projected center moves with the image motion;
      - it appears s times bigger, so it is s times closer (z / s);
      - an in-plane turn is a rotation about the camera's viewing axis.
    Exact for those motions; out-of-plane turns are left to MegaPose corrections.
    """
    m = np.asarray(motion, dtype=float)
    scale = float(np.hypot(m[0, 0], m[1, 0]))
    theta = float(np.arctan2(m[1, 0], m[0, 0]))
    k = np.asarray(camera_matrix, dtype=float)
    t = np.asarray(t_camera_object, dtype=float)[:3, 3]
    center = k @ t
    center = center[:2] / center[2]
    moved = m[:2, :2] @ center + m[:2, 2]
    depth = t[2] / scale if scale > 1e-6 else t[2]
    ray = np.linalg.inv(k) @ np.array([moved[0], moved[1], 1.0])  # z = 1
    c, s = np.cos(theta), np.sin(theta)
    out = np.array(t_camera_object, dtype=float)
    out[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ out[:3, :3]
    out[:3, 3] = ray * depth
    return out


# Turns a pose 180 degrees about the model's own X axis: top <-> bottom,
# front <-> back. For an object symmetric between its top and bottom (a flat
# box) the silhouette is the same, which is exactly why the pose search can't
# tell them apart on its own.
_FLIP_X = np.diag([1.0, -1.0, -1.0, 1.0])


def resting_side_ok(t_camera_mesh: np.ndarray, resting: str) -> bool:
    """Does the pose show the model's top (+Y, glTF's up) where the station
    says it is? "camera": the camera looks down on the object, its top faces
    the camera. "upright": the camera looks from the side, its top is up in
    the image. "any": no constraint."""
    if resting == "any":
        return True
    up = np.asarray(t_camera_mesh, float)[:3, 1]  # the model's +Y in the camera frame
    if resting in ("camera", "flat"):
        to_camera = -np.asarray(t_camera_mesh, float)[:3, 3]
        return float(up @ to_camera) > 0.0
    return float(up[1]) < 0.0  # "upright": image y grows downwards, so up is -y


def level_pose(t_camera_mesh: np.ndarray) -> np.ndarray:
    """The pose turned (about its own centre) so the model's top points
    straight along the camera's viewing axis, back at the camera: an object
    lying flat under a camera looking straight down. Only the turn in the
    table plane (and position) is kept from the estimate — the tilt a plain
    flat box can't pin down from its outline is removed."""
    out = np.array(t_camera_mesh, dtype=float)
    up = out[:3, 1] / (np.linalg.norm(out[:3, 1]) or 1.0)
    target = np.array([0.0, 0.0, -1.0])
    axis = np.cross(up, target)
    sin, cos = float(np.linalg.norm(axis)), float(up @ target)
    if sin < 1e-9:
        if cos > 0:
            return out
        axis, angle = out[:3, 0] / (np.linalg.norm(out[:3, 0]) or 1.0), np.pi  # exactly upside down
    else:
        axis, angle = axis / sin, float(np.arctan2(sin, cos))
    turn, _ = cv2.Rodrigues(axis * angle)
    out[:3, :3] = turn @ out[:3, :3]
    return out


def blend_pose(a: np.ndarray, b: np.ndarray, translation_gain: float, rotation_gain: float) -> np.ndarray:
    """Pose `a` moved toward `b`: translation by `translation_gain`, rotation
    along the shortest arc by `rotation_gain` (0 = keep a, 1 = take b)."""
    out = np.array(a, dtype=float)
    out[:3, 3] = a[:3, 3] + translation_gain * (b[:3, 3] - a[:3, 3])
    relative, _ = cv2.Rodrigues(b[:3, :3] @ a[:3, :3].T)
    step, _ = cv2.Rodrigues(relative * rotation_gain)
    out[:3, :3] = step @ a[:3, :3]
    return out


# Background MegaPose refinements (one in flight per tracker; the pose service
# serializes GPU work anyway).
_refine_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="megapose-refine")


class MegaPoseTracker:
    """
    Model-based 6DoF tracking in the FoundationPose++ split:

    - every frame, optical flow (~40 ms) moves the last pose by the object's
      2D motion: X/Y from where it went, Z from how much bigger it got, the
      in-plane turn from how it rotated (propagate_pose);
    - MegaPose refines in the background (~250 ms each) and corrects the pose;
      a correction is carried forward by the motion that happened while it ran;
    - while the object is still, the pose is held and MegaPose isn't called:
      each refine starts from the last and is off by a few degrees, so
      refining a still object made its rotation random-walk (measured on a
      still flask: ~4 deg per frame, up to 60 deg). After motion stops, one
      last "settle" refine corrects what flow can't see (out-of-plane turns).

    Before this, MegaPose blocked every frame (~500 ms/frame while moving).

    Confidence is geometric, not MegaPose's appearance score: how well the
    model drawn at its pose (the service's projected_bbox) overlaps the real
    object (the detection at the first lock, then the flow box). Measured on a
    real frame of the user's flask with an untextured model: a correct upright
    pose had score 0.11 but overlap 0.96; a wrong sideways pose score 0.41 but
    overlap 0.19. An object with too little texture for optical flow falls
    back to refining every frame, judged by the appearance score.
    """

    can_recover = True  # refining from the last good pose can re-lock after a brief occlusion

    def __init__(
        self,
        client: ModelPoseClient,
        label: str,
        refine_iterations: int,
        part_offset: np.ndarray | None = None,
        still_motion_px: float = 1.5,
        min_overlap: float = 0.5,
        async_refine: bool = True,
        correction_gain_translation: float = 0.5,
        correction_gain_rotation: float = 0.3,
        occlusion_min: float = 0.2,
        occlusion_hold_ms: float = 1500.0,
        refine_max_occlusion: float = 0.3,
        flow_support_full: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
        min_distance_m: float = 0.0,
        max_distance_m: float = float("inf"),
        resting: str = "any",
        edge_model: EdgeModel | None = None,
        edge_tilt_starts_deg: float = 8.0,
        edge_gain_translation: float = 0.8,
        edge_gain_rotation: float = 0.8,
        symmetries: list[np.ndarray] | None = None,
        pose_features: list[PoseFeature] | None = None,
        orientation_min_evidence: float = 0.35,
        orientation_min_margin: float = 0.15,
        correction_verify_deg: float = 15.0,
        geometry_model: EdgeModel | None = None,
    ):
        self.client = client
        # The CAD geometry's own edges, for judging orientation from geometry.
        # `edge_model` may also hold pose-feature rims (printed circles) for
        # alignment; those must not count as geometric evidence — "no circle
        # at that corner" would otherwise decide the orientation when the real
        # circle is covered, and a missing print could be the very defect.
        self.geometry_model = geometry_model if geometry_model is not None else edge_model
        # Orientation (Phase 8). `symmetries`: the model's rotations that look
        # like itself (pose/symmetry.py) — every geometric pose has those
        # twins. `pose_features`: what tells them apart (pose/pose_features.py).
        # At every lock the twin the image supports is chosen
        # (pose/disambiguation.py); an unconfirmed orientation is reported as
        # such, never guessed. Twins are about the whole assembly's origin, so
        # this applies when tracking the whole object (no part offset).
        whole_object = part_offset is None or not np.any(np.asarray(part_offset, float))
        self.symmetries = list(symmetries) if symmetries and len(symmetries) > 1 and whole_object else [np.eye(3)]
        self.pose_features = [f for f in (pose_features or []) if f.enabled] if whole_object else []
        self.orientation_min_evidence = orientation_min_evidence
        self.orientation_min_margin = orientation_min_margin
        # A background correction turning the pose more than this from the
        # tracked one (at the same frame — it's no motion) must be backed by
        # the 3D edges, and is then adopted outright, never blended halfway.
        self.correction_verify_deg = correction_verify_deg
        self._orientation: dict | None = None  # the current hypothesis: status, features seen
        self._reliable: np.ndarray | None = None  # last pose (assembly) whose orientation was settled
        self.guard_counts = {"twin_flips_suppressed": 0, "corrections_rejected": 0, "orientation_switches": 0}
        self._last_guard: dict | None = None
        # 3D edge alignment (app/services/pose/edge_align.py): the model's own
        # edges (assembly frame) fitted onto the image's edges — at every lock
        # (from several starting tilts) and on every background correction.
        # It pins down the tilt that outline matching and 2D flow can't see.
        # A correction the edges confirm is adopted with the stronger edge
        # gains (it's measured, not one more few-degrees-off MegaPose guess).
        self.edge_model = edge_model if edge_model is not None and len(edge_model.points) else None
        self.edge_tilt_starts_deg = edge_tilt_starts_deg
        self.edge_gain_translation = edge_gain_translation
        self.edge_gain_rotation = edge_gain_rotation
        self._edge_fit: dict | None = None  # the last edge alignment, for the quality readout
        # Which side of the model faces where (see resting_side_ok): settles
        # the top/bottom ambiguity of symmetric objects at every lock.
        self.resting = resting
        # A lock putting the object outside this distance range is rejected.
        self.min_distance_m = min_distance_m
        self.max_distance_m = max_distance_m
        # Hand occlusion (Phase 7). When flow loses the object while a hand
        # covers at least `occlusion_min` of its outline, the last pose is
        # held ("occluded") for up to `occlusion_hold_ms` instead of declaring
        # it lost — and flow resumes from the object's visible pixels as soon
        # as enough show. MegaPose isn't asked to refine while the hand covers
        # more than `refine_max_occlusion` (its crop would be mostly hand).
        self.occlusion_min = occlusion_min
        self.occlusion_hold_ms = occlusion_hold_ms
        self.refine_max_occlusion = refine_max_occlusion
        # Confidence counts visual support: below this share of agreeing flow
        # points, it is scaled down (few visible points = weak evidence).
        self.flow_support_full = flow_support_full
        self.clock = clock
        self._occluded_since: float | None = None
        self.label = label
        self.refine_iterations = refine_iterations
        # When tracking one part of an assembly (e.g. a bottle's body, which
        # looks the same with or without its cap), MegaPose matches only that
        # part's mesh, recentered on itself. part_offset = that center relative
        # to the assembly's center; it converts the part's pose into the pose
        # of the whole assembly, which is what gets rendered.
        self.part_offset = np.zeros(3) if part_offset is None else np.asarray(part_offset, dtype=float)
        # The object counts as still while its tracked points have moved less
        # than this (net, px) since it last moved. 0 = never hold.
        self.still_motion_px = still_motion_px
        # A background correction overlapping the object less than this is not
        # adopted (two in a row and confidence drops, so the session re-detects).
        self.min_overlap = min_overlap
        self.async_refine = async_refine  # False: corrections run inline (tests, debugging)
        # How far a MegaPose correction pulls the flow-tracked pose. Flow is
        # precise frame to frame but can't see out-of-plane turns and slowly
        # drifts; each refine is off by a few degrees (on the flask, single
        # corrections jumped up to 14 deg of tilt). Pulling part-way averages
        # those errors over successive corrections (~4/s) instead of showing
        # each one — the Kalman-style fusion FoundationPose++ uses.
        self.correction_gain_translation = correction_gain_translation
        self.correction_gain_rotation = correction_gain_rotation

        self._flow = FlowBoxTracker()
        self._flow_ready = False  # following the object right now
        self._flow_usable = False  # the object had enough texture to follow at all
        self._reference: np.ndarray | None = None  # mesh pose to refine from (no-flow path), part frame

        # Flow path state. Mesh poses (part frame), OpenCV camera convention.
        self._base: np.ndarray | None = None  # last corrected pose; flow's motion matrix is relative to it
        self._output: np.ndarray | None = None  # what's shown now (held while still)
        self._overlap = 0.0  # last correction's overlap with the object
        self._projected: tuple | None = None  # last correction's projected box
        self._bad_corrections = 0
        self._needs_settle = False
        # (refine running in the background, flow motion when it was asked for, object box then)
        self._pending: tuple[Future, np.ndarray, BoundingBox | None] | None = None
        self._last_refine_ms = 0.0

    def _assembly_pose(self, t_camera_mesh: np.ndarray) -> np.ndarray:
        # x_cam = R (p_asm - offset) + t  =>  assembly pose is [R | t - R offset].
        t = np.array(t_camera_mesh, dtype=float)
        t[:3, 3] = t[:3, 3] - t[:3, :3] @ self.part_offset
        return t

    def _mesh_pose(self, t_camera_assembly: np.ndarray) -> np.ndarray:
        t = np.array(t_camera_assembly, dtype=float)
        t[:3, 3] = t[:3, 3] + t[:3, :3] @ self.part_offset
        return t

    def _edge_align(
        self, gray: np.ndarray, hand_mask: np.ndarray | None, camera_matrix: np.ndarray, t_mesh: np.ndarray,
        tilt_starts_deg: float,
    ) -> tuple[np.ndarray, dict | None]:
        """t_mesh refined onto the image's edges (unchanged if they don't
        clearly fit better, or would put the object on its wrong side), and a
        summary of the fit (None: too few edges in view to judge)."""
        if self.edge_model is None:
            return t_mesh, None
        edges = edge_image(gray, exclude=hand_mask)
        fit = align(self.edge_model, self._assembly_pose(t_mesh), camera_matrix, edges, tilt_starts_deg=tilt_starts_deg)
        if fit is None:
            return t_mesh, None
        refined = self._mesh_pose(fit.t_co) if fit.improved else t_mesh
        adopted = fit.improved and resting_side_ok(refined, self.resting)
        summary = {
            "score": fit.score, "support": fit.support, "coverage": fit.coverage, "mean_px": fit.mean_px,
            "points": fit.points, "adjusted": adopted,
        }
        return (refined if adopted else t_mesh), summary

    @property
    def _has_orientation_logic(self) -> bool:
        return len(self.symmetries) > 1 or bool(self.pose_features)

    def _decide_orientation(
        self, gray: np.ndarray, hand_mask, camera_matrix: np.ndarray, t_mesh: np.ndarray, reference: np.ndarray | None
    ) -> tuple[np.ndarray, OrientationDecision | None]:
        """The twin of `t_mesh` the image supports (pose/disambiguation.py)."""
        if not self._has_orientation_logic:
            return t_mesh, None
        edges = edge_image(gray, exclude=hand_mask)
        decision = choose_orientation(
            self._assembly_pose(t_mesh), self.symmetries, self.pose_features, camera_matrix, gray, edges,
            model=self.geometry_model, reference=reference,
            allowed=lambda t: resting_side_ok(self._mesh_pose(t), self.resting),
            min_evidence=self.orientation_min_evidence, min_margin=self.orientation_min_margin,
        )
        return self._mesh_pose(decision.t_co), decision

    @staticmethod
    def _orientation_summary(decision: OrientationDecision) -> dict:
        return {
            "status": decision.status,
            "reason": decision.reason,
            "margin": decision.margin,
            "flipped": decision.flipped,
            "scores": decision.scores,
            "required_unconfirmed": decision.required_unconfirmed,
            "features": [
                {
                    "feature_id": o.feature_id, "status": o.status, "evidence": o.evidence,
                    "predicted_center_px": o.predicted_center_px,
                    "predicted_contour_px": o.predicted_contour_px[:: max(1, len(o.predicted_contour_px) // 24)],
                    "observed_center_px": o.observed_center_px, "residual_px": o.residual_px,
                }
                for o in decision.observations
            ],
        }

    def _verify_correction(
        self, gray: np.ndarray, hand_mask, camera_matrix: np.ndarray, reference: np.ndarray, corrected: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """A background correction checked against the pose tracked when it
        was asked for (same frame, so any difference is disagreement, not
        motion): ("blend" | "switch" | "reject", why) and the pose to use.

        - A symmetric twin of the tracked orientation (MegaPose flipping a
          symmetric box) is turned back — unless this frame's features
          confirm the object really is the other way round ("switch").
        - Any other turn beyond `correction_verify_deg` must fit the 3D edges
          clearly better than the tracked pose to be adopted ("switch"),
          else it is rejected. A switch replaces the pose; it is never
          blended with the old one (half of each orientation is neither)."""
        pose = np.array(corrected, float)
        guard = {"action": "blend", "reason": "", "angle_deg": rotation_angle_deg(pose, reference)}
        if len(self.symmetries) > 1:
            twin, index = closest_twin(self._assembly_pose(pose), self._assembly_pose(reference), self.symmetries)
            if index != 0:
                decision = None
                if self.pose_features or self.edge_model is not None:
                    edges = edge_image(gray, exclude=hand_mask)
                    decision = choose_orientation(
                        self._assembly_pose(pose), self.symmetries, self.pose_features, camera_matrix, gray, edges,
                        model=self.geometry_model, allowed=lambda t: resting_side_ok(self._mesh_pose(t), self.resting),
                        min_evidence=self.orientation_min_evidence, min_margin=self.orientation_min_margin,
                    )
                if decision is not None and decision.status == "confirmed" and rotation_angle_deg(decision.t_co, twin) > 20.0:
                    guard.update(action="switch", reason="the features show the object turned round",
                                 orientation=self._orientation_summary(decision))
                    return self._mesh_pose(decision.t_co), guard
                pose = self._mesh_pose(twin)
                guard.update(reason="symmetric twin of the tracked orientation, turned back", twin_flip=True)
        angle = rotation_angle_deg(pose, reference)
        guard["angle_deg"] = angle
        if angle > self.correction_verify_deg:
            if self.edge_model is None:
                guard["reason"] = (guard["reason"] + "; " if guard["reason"] else "") + "large turn, no 3D edges to verify it"
                return pose, guard
            edges = edge_image(gray, exclude=hand_mask)
            tracked_fit = _fit_score(self.edge_model, self._assembly_pose(reference), camera_matrix, edges, 3.0)[0]
            new_fit = _fit_score(self.edge_model, self._assembly_pose(pose), camera_matrix, edges, 3.0)[0]
            guard.update(tracked_fit=tracked_fit, correction_fit=new_fit)
            if new_fit >= tracked_fit + 0.03:
                guard.update(action="switch", reason=f"{angle:.0f} deg correction confirmed by the 3D edges")
            else:
                guard.update(action="reject", reason=f"{angle:.0f} deg correction not supported by the 3D edges")
        return pose, guard

    def _measurement(self, result: PoseServiceResult, object_box: BoundingBox | None, polygon=None) -> Measurement:
        extra = {"pose_service_ms": result.elapsed_ms, "pose_score": result.score, "projected_bbox": result.projected_bbox}
        if result.t_camera_object is None:
            return Measurement(confidence=0.0, extra=extra)
        if object_box is not None and result.projected_bbox is not None:
            confidence = box_iou(object_box, result.projected_bbox)
            extra["confidence_source"] = "overlap"
        else:
            confidence = result.score
            extra["confidence_source"] = "pose score"
        return self._pose_measurement(result.t_camera_object, confidence, object_box, polygon, extra)

    def _pose_measurement(self, t_mesh: np.ndarray, confidence: float, object_box, polygon, extra: dict) -> Measurement:
        if self.resting == "flat":
            t_mesh = level_pose(t_mesh)
        t_assembly = self._assembly_pose(t_mesh)
        pose = cv_model_pose_to_threejs(t_assembly)
        extra["t_camera_object"] = t_assembly  # what's rendered (and the debug gizmo, part checks)
        extra["t_camera_mesh"] = t_mesh  # what MegaPose refines from
        return Measurement(
            confidence=confidence,
            position=pose.position,
            quaternion=pose.quaternion,
            bbox=object_box,
            polygon=polygon,
            extra=extra,
        )

    def initialize(
        self, frame: Frame, detection: SegmentedObject, initial_pose: np.ndarray | None = None,
        pose_source: str | None = None,
    ) -> Measurement:
        """Lock onto `detection`: MegaPose's full search, or — given
        `initial_pose` (assembly frame, e.g. from the outline of a region the
        operator clicked) — that pose, put on the image's edges below."""
        self._discard_pending()
        self._occluded_since = None
        detection, hand_share = without_hand(detection, frame.hand_mask)
        self._flow_ready = self._flow_usable = self._flow.initialize(
            frame.gray, detection.bbox, detection.polygon, frame.hand_mask
        )
        bbox = [detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2]
        outline = [(p.x, p.y) for p in detection.polygon] if detection.polygon else None
        if initial_pose is not None:
            result = PoseServiceResult(
                self._mesh_pose(initial_pose), detection.confidence, "outline", 0.0, tuple(bbox), detection.confidence
            )
        else:
            result = self.client.full_search(
                self.label, frame.jpeg_without_hands(), frame.calibration.camera_matrix, bbox, outline
            )
        flipped = False
        if result.t_camera_object is not None and not resting_side_ok(result.t_camera_object, self.resting):
            flipped_pose = np.asarray(result.t_camera_object, float) @ _FLIP_X
            if resting_side_ok(flipped_pose, self.resting):
                result = replace(result, t_camera_object=flipped_pose)
                flipped = True
        edge_fit = None
        if result.t_camera_object is not None and self.edge_model is not None:
            t_edges, edge_fit = self._edge_align(
                frame.gray, frame.hand_mask, frame.calibration.camera_matrix, np.asarray(result.t_camera_object, float),
                self.edge_tilt_starts_deg,
            )
            result = replace(result, t_camera_object=t_edges)
        decision = None
        if result.t_camera_object is not None and self._has_orientation_logic:
            k = frame.calibration.camera_matrix
            from_guide = pose_source == "guide" and initial_pose is not None
            t_chosen, decision = self._decide_orientation(
                frame.gray, frame.hand_mask, k, np.asarray(result.t_camera_object, float),
                np.asarray(initial_pose, float) if from_guide else self._reliable,
            )
            if decision is not None and from_guide and decision.status == "ambiguous":
                # The operator lined the object up with an outline that shows
                # which end is which: that settles the twin the features
                # couldn't (features that clearly disagree would have won).
                decision.status = "confirmed"
                decision.reason = "lined up by the operator with the guide view"
            if decision is not None and decision.flipped and self.edge_model is not None:
                # Near-symmetric parts: the chosen twin is fitted onto the edges itself.
                t_chosen, refit = self._edge_align(frame.gray, frame.hand_mask, k, t_chosen, 0.0)
                edge_fit = refit or edge_fit
            result = replace(result, t_camera_object=t_chosen)
        self._edge_fit = edge_fit
        self._orientation = self._orientation_summary(decision) if decision is not None else None
        measurement = self._measurement(result, detection.bbox, detection.polygon)
        measurement.extra["edge_fit"] = edge_fit
        measurement.extra["orientation"] = self._orientation
        measurement.extra["flipped_to_resting_side"] = flipped
        measurement.extra["hand_removed"] = hand_share
        if result.silhouette_iou is not None and result.t_camera_object is not None:
            # The first lock is judged by how well the posed model's silhouette
            # covers the detected outline: box overlap can't reject a bottle
            # posed upside down (same box), silhouette overlap can.
            measurement.confidence = result.silhouette_iou
            measurement.extra["confidence_source"] = "silhouette"
            measurement.extra["silhouette_iou"] = result.silhouette_iou
        measurement.extra["lock"] = True
        if decision is not None and decision.required_unconfirmed:
            measurement.confidence = 0.0
            measurement.extra["reject_reason"] = (
                "orientation not confirmed: required feature(s) "
                f"{', '.join(decision.required_unconfirmed)} not seen — show them to the camera"
            )
        t_co = measurement.extra.get("t_camera_object")
        if t_co is not None:
            distance = float(t_co[2, 3])
            measurement.extra["distance_m"] = distance
            if not self.min_distance_m <= distance <= self.max_distance_m:
                measurement.confidence = 0.0
                measurement.extra["reject_reason"] = (
                    f"the object would be {distance:.2f} m away, outside the working range "
                    f"{self.min_distance_m:.2f}-{self.max_distance_m:.2f} m (wrong region or wrong model size)"
                )
        return measurement

    def update(self, frame: Frame) -> Measurement:
        if self._flow_usable and self._base is not None:
            return self._update_with_flow(frame)
        return self._update_refine_every_frame(frame)

    # -- flow every frame + background corrections --------------------------

    def _update_with_flow(self, frame: Frame) -> Measurement:
        hand = frame.hand_mask
        if not self._flow_ready and self._occluded_since is not None:
            # Held through an occlusion that hid the object completely: pick
            # flow up again as soon as any of it shows (seeded off the hand).
            self._flow_ready = self._flow.resume(frame.gray, hand)
        flow = self._flow.update(frame.gray, hand) if self._flow_ready else None
        if flow is None or not flow.ok:
            occlusion = self._flow.occlusion(hand)
            if hand is not None and occlusion >= self.occlusion_min:
                return self._hold_occluded(frame, occlusion)
            # Flow lost the object (out of view, too fast): no evidence the pose
            # still fits, and the appearance score can't be trusted to say so.
            self._flow_ready = False
            self._discard_pending()
            self._occluded_since = None
            return Measurement(confidence=0.0, extra={"occlusion": occlusion})
        self._occluded_since = None
        object_box, polygon = flow.bbox, flow.polygon
        k = frame.calibration.camera_matrix
        assert self._base is not None and self._output is not None

        corrected = self._adopt_finished_refine(k)
        moving = self._flow.motion_since_anchor >= self.still_motion_px
        if moving:
            self._flow.set_anchor()  # measure the next motion from here
            self._needs_settle = True
        if moving or corrected or self.still_motion_px <= 0:
            self._output = propagate_pose(self._base, self._flow.motion_matrix, k)

        refine_ok = flow.occlusion <= self.refine_max_occlusion
        if self._pending is None and (moving or self._needs_settle) and refine_ok:
            self._submit_refine(frame, object_box)
            if not moving:
                self._needs_settle = False
            if not self.async_refine and self._adopt_finished_refine(k):
                self._output = propagate_pose(self._base, self._flow.motion_matrix, k)

        # Confidence: geometric fit (overlap) scaled by how much visual
        # support flow still has — never the detector's score.
        support = min(1.0, flow.confidence / self.flow_support_full) if self.flow_support_full > 0 else 1.0
        extra = {
            "held": not moving,
            "pose_service_ms": self._last_refine_ms,
            "refine_pending": self._pending is not None,
            "refine_skipped_for_hand": not refine_ok,
            "projected_bbox": self._projected,
            "confidence_source": "overlap x flow support",
            "overlap": self._overlap,
            "flow_support": flow.confidence,
            "occlusion": flow.occlusion,
            "hand_points": flow.hand_points,
            "edge_fit": self._edge_fit,
            "orientation": self._orientation,
            "correction_guard": {**self.guard_counts, "last": self._last_guard},
        }
        return self._pose_measurement(self._output, self._overlap * support, object_box, polygon, extra)

    def _hold_occluded(self, frame: Frame, occlusion: float) -> Measurement:
        """A hand covers the object and flow can't follow it: keep the last
        pose for a while (no update from any hand pixel), and pick flow up
        again from the object's visible pixels as soon as there are enough."""
        now = self.clock()
        if self._occluded_since is None:
            self._occluded_since = now
        held_ms = (now - self._occluded_since) * 1000.0
        self._discard_pending()
        if held_ms > self.occlusion_hold_ms:
            self._flow_ready = False
            return Measurement(confidence=0.0, extra={"occlusion": occlusion, "occluded_for_ms": held_ms})
        # Resume from what's visible: the next frame follows those points.
        self._flow_ready = self._flow.resume(frame.gray, frame.hand_mask)
        polygon = self._flow.polygon
        outline = [Vector2(x=float(x), y=float(y)) for x, y in polygon] if polygon is not None else None
        box = None
        if polygon is not None:
            (x1, y1), (x2, y2) = polygon.min(axis=0), polygon.max(axis=0)
            box = BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
        extra = {
            "occluded": True,
            "occlusion": occlusion,
            "occluded_for_ms": held_ms,
            "projected_bbox": self._projected,
            "confidence_source": "held (occluded)",
            "overlap": self._overlap,
        }
        assert self._output is not None
        return self._pose_measurement(self._output, self._overlap * (1.0 - occlusion), box, outline, extra)

    def _submit_refine(self, frame: Frame, object_box: BoundingBox) -> None:
        reference = self._output.copy()
        refine = partial(
            self.client.refine, self.label, frame.jpeg_without_hands(), frame.calibration.camera_matrix, reference,
            self.refine_iterations,
        )
        if self.edge_model is None and not self._has_orientation_logic:
            call = lambda: (refine(), None, None)  # noqa: E731
        else:
            gray, hand, k = frame.gray.copy(), frame.hand_mask, frame.calibration.camera_matrix

            def call():
                result = refine()
                if result.t_camera_object is None:
                    return result, None, None
                # Single start: MegaPose's answer is close; the lock already searched tilts.
                t_edges, fit = self._edge_align(gray, hand, k, np.asarray(result.t_camera_object, float), 0.0)
                t_checked, guard = self._verify_correction(gray, hand, k, reference, t_edges)
                return replace(result, t_camera_object=t_checked), fit, guard
        if self.async_refine:
            future = _refine_pool.submit(call)
        else:
            future = Future()
            try:
                future.set_result(call())
            except Exception as exc:  # surfaced when adopted, like the async path
                future.set_exception(exc)
        self._pending = (future, self._flow.motion_matrix, object_box)

    def _adopt_finished_refine(self, camera_matrix: np.ndarray) -> bool:
        """Apply a finished background refine, carried forward by the motion since it was asked for."""
        if self._pending is None or not self._pending[0].done():
            return False
        future, motion_at_submit, box_at_submit = self._pending
        self._pending = None
        result, edge_fit, guard = future.result()  # PoseServiceUnavailable propagates to the session
        result: PoseServiceResult
        self._last_refine_ms = result.elapsed_ms
        if edge_fit is not None:
            self._edge_fit = edge_fit
        if guard is not None:
            self._last_guard = guard
            if guard.get("twin_flip"):
                self.guard_counts["twin_flips_suppressed"] += 1
            if guard["action"] == "reject":
                self.guard_counts["corrections_rejected"] += 1
                return False  # the tracked pose stays; not counted against the object being there
        if result.t_camera_object is None:
            return False
        overlap = (
            box_iou(box_at_submit, result.projected_bbox)
            if box_at_submit is not None and result.projected_bbox is not None
            else result.score
        )
        if overlap < self.min_overlap:
            self._bad_corrections += 1
            if self._bad_corrections >= 2:
                self._overlap = overlap  # persistently off: let the session declare it lost
            return False
        self._bad_corrections = 0
        since_submit = self._flow.motion_matrix @ np.linalg.inv(motion_at_submit)
        corrected = propagate_pose(result.t_camera_object, since_submit, camera_matrix)
        predicted = propagate_pose(self._base, self._flow.motion_matrix, camera_matrix)
        if guard is not None and guard["action"] == "switch":
            # A different orientation hypothesis: replace, never blend across.
            self._base = corrected
            self.guard_counts["orientation_switches"] += 1
            if "orientation" in guard:
                self._orientation = guard["orientation"]
        else:
            confirmed = edge_fit is not None and edge_fit["adjusted"]
            self._base = blend_pose(
                predicted,
                corrected,
                self.edge_gain_translation if confirmed else self.correction_gain_translation,
                self.edge_gain_rotation if confirmed else self.correction_gain_rotation,
            )
        self._flow.reset_motion()
        self._overlap = overlap
        self._projected = result.projected_bbox
        return True

    def _discard_pending(self) -> None:
        self._pending = None  # a running refine finishes in the background and is ignored

    # -- no usable optical flow: refine every frame -------------------------

    def _update_refine_every_frame(self, frame: Frame) -> Measurement:
        if self._reference is None:
            return Measurement(confidence=0.0)
        object_box = polygon = None
        if self._flow_ready:
            flow = self._flow.update(frame.gray)
            if flow.ok:
                object_box, polygon = flow.bbox, flow.polygon
            else:
                self._flow_ready = False
        if self._flow_usable and object_box is None:
            return Measurement(confidence=0.0)
        result = self.client.refine(
            self.label, frame.jpeg, frame.calibration.camera_matrix, self._reference, self.refine_iterations
        )
        return self._measurement(result, object_box, polygon)

    def commit(self, measurement: Measurement) -> None:
        t_mesh = measurement.extra.get("t_camera_mesh")
        if t_mesh is None:
            return
        self._reference = t_mesh
        if (self._orientation or {}).get("status") in ("unique", "confirmed") and not measurement.extra.get("occluded"):
            # The orientation a later re-lock keeps when its evidence is inconclusive.
            self._reliable = np.array(measurement.extra["t_camera_object"], float)
        if measurement.extra.get("lock"):
            # A fresh lock the session accepted: the flow path starts from it.
            self._base, self._output = t_mesh.copy(), t_mesh.copy()
            self._overlap = measurement.confidence
            self._projected = measurement.extra.get("projected_bbox")
            self._bad_corrections = 0
            self._needs_settle = False
            self._flow.reset_motion()
            self._flow.set_anchor()
