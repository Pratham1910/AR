"""
The two PoseTracker implementations used by ARSession (ar_session.py):

- FlowPoseTracker (markerless): optical flow follows the detected outline;
  position from its apparent size (approximate, no orientation).
- MegaPoseTracker (model-based): the initial pose comes from a full MegaPose
  search inside the detected box, then each frame refines from the previous
  pose — real 6DoF tracking with no detector involved.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from functools import cached_property, partial

import cv2
import numpy as np

from app.schemas.vision import BoundingBox, SegmentedObject
from app.services.pose.calibration import CameraCalibration
from app.services.pose.markerless import estimate_object_placement
from app.services.pose.model_pose_client import ModelPoseClient, PoseServiceResult
from app.services.pose.transforms import cv_model_pose_to_threejs
from app.services.tracking.ar_session import Measurement
from app.services.tracking.flow_tracker import FlowBoxTracker


class Frame:
    """One camera frame, with the conversions trackers need computed once."""

    def __init__(self, bgr: np.ndarray, calibration: CameraCalibration, encoded: bytes | None = None):
        self.bgr = bgr
        self.calibration = calibration
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


class FlowPoseTracker:
    can_recover = False  # once its points are gone it needs the detector again

    def __init__(self, real_world_height_m: float):
        self.real_world_height_m = real_world_height_m
        self._flow = FlowBoxTracker()

    def _placement(self, frame: Frame, bbox) -> tuple:
        pose = estimate_object_placement(bbox, frame.calibration, self.real_world_height_m)
        return pose.position, pose.quaternion

    def initialize(self, frame: Frame, detection: SegmentedObject) -> Measurement:
        if not self._flow.initialize(frame.gray, detection.bbox, detection.polygon):
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
        result = self._flow.update(frame.gray)
        if not result.ok:
            return Measurement(confidence=result.confidence)
        position, quaternion = self._placement(frame, result.bbox)
        return Measurement(
            confidence=result.confidence,
            position=position,
            quaternion=quaternion,
            bbox=result.bbox,
            polygon=result.polygon,
            extra={"points": result.points},
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
    ):
        self.client = client
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

    def initialize(self, frame: Frame, detection: SegmentedObject) -> Measurement:
        self._discard_pending()
        self._flow_ready = self._flow_usable = self._flow.initialize(frame.gray, detection.bbox, detection.polygon)
        bbox = [detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2]
        outline = [(p.x, p.y) for p in detection.polygon] if detection.polygon else None
        result = self.client.full_search(self.label, frame.jpeg, frame.calibration.camera_matrix, bbox, outline)
        measurement = self._measurement(result, detection.bbox, detection.polygon)
        if result.silhouette_iou is not None and result.t_camera_object is not None:
            # The first lock is judged by how well the posed model's silhouette
            # covers the detected outline: box overlap can't reject a bottle
            # posed upside down (same box), silhouette overlap can.
            measurement.confidence = result.silhouette_iou
            measurement.extra["confidence_source"] = "silhouette"
            measurement.extra["silhouette_iou"] = result.silhouette_iou
        measurement.extra["lock"] = True
        return measurement

    def update(self, frame: Frame) -> Measurement:
        if self._flow_usable and self._base is not None:
            return self._update_with_flow(frame)
        return self._update_refine_every_frame(frame)

    # -- flow every frame + background corrections --------------------------

    def _update_with_flow(self, frame: Frame) -> Measurement:
        flow = self._flow.update(frame.gray) if self._flow_ready else None
        if flow is None or not flow.ok:
            # Flow lost the object (occluded, out of view): no evidence the pose
            # still fits, and the appearance score can't be trusted to say so.
            self._flow_ready = False
            self._discard_pending()
            return Measurement(confidence=0.0)
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

        if self._pending is None and (moving or self._needs_settle):
            self._submit_refine(frame, object_box)
            if not moving:
                self._needs_settle = False
            if not self.async_refine and self._adopt_finished_refine(k):
                self._output = propagate_pose(self._base, self._flow.motion_matrix, k)

        extra = {
            "held": not moving,
            "pose_service_ms": self._last_refine_ms,
            "refine_pending": self._pending is not None,
            "projected_bbox": self._projected,
            "confidence_source": "overlap",
        }
        return self._pose_measurement(self._output, self._overlap, object_box, polygon, extra)

    def _submit_refine(self, frame: Frame, object_box: BoundingBox) -> None:
        reference = self._output.copy()
        call = partial(
            self.client.refine, self.label, frame.jpeg, frame.calibration.camera_matrix, reference, self.refine_iterations
        )
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
        result: PoseServiceResult = future.result()  # PoseServiceUnavailable propagates to the session
        self._last_refine_ms = result.elapsed_ms
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
        self._base = blend_pose(
            predicted, corrected, self.correction_gain_translation, self.correction_gain_rotation
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
        if measurement.extra.get("lock"):
            # A fresh lock the session accepted: the flow path starts from it.
            self._base, self._output = t_mesh.copy(), t_mesh.copy()
            self._overlap = measurement.confidence
            self._projected = measurement.extra.get("projected_bbox")
            self._bad_corrections = 0
            self._needs_settle = False
            self._flow.reset_motion()
            self._flow.set_anchor()
