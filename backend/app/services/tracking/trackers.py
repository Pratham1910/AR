"""
The two PoseTracker implementations used by ARSession (ar_session.py):

- FlowPoseTracker (markerless): optical flow follows the detected outline;
  position from its apparent size (approximate, no orientation).
- MegaPoseTracker (model-based): the initial pose comes from a full MegaPose
  search inside the detected box, then each frame refines from the previous
  pose — real 6DoF tracking with no detector involved.
"""

from __future__ import annotations

from functools import cached_property

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

    def __init__(self, bgr: np.ndarray, calibration: CameraCalibration):
        self.bgr = bgr
        self.calibration = calibration

    @cached_property
    def gray(self) -> np.ndarray:
        return cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)

    @cached_property
    def jpeg(self) -> bytes:
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


class MegaPoseTracker:
    """
    Confidence is geometric, not MegaPose's appearance score: how well the
    model drawn at its pose (the service's projected_bbox) overlaps the real
    object — YOLO's box at the first lock, then an optical-flow box that
    follows the object's outline every frame with no detector. Measured on a
    real frame of the user's flask with an untextured model: a correct
    upright pose had score 0.11 but overlap 0.96; a wrong sideways pose had
    score 0.41 but overlap 0.19. If the object has too little texture for
    optical flow, the appearance score is the fallback.
    """

    can_recover = True  # refining from the last good pose can re-lock after a brief occlusion

    def __init__(
        self,
        client: ModelPoseClient,
        label: str,
        refine_iterations: int,
        part_offset: np.ndarray | None = None,
        still_motion_px: float = 1.5,
    ):
        self.client = client
        # While the object's tracked points have moved less than this since the
        # last accepted MegaPose pose, that pose is held (no refine). Each
        # refine starts from the previous pose and is off by a few degrees, so
        # refining a still object made its rotation random-walk (measured on a
        # still flask: ~4 deg per frame, up to 60 deg). 0 disables holding.
        self.still_motion_px = still_motion_px
        self._held: Measurement | None = None  # the last accepted refined measurement
        self.label = label
        self.refine_iterations = refine_iterations
        # When tracking one part of an assembly (e.g. a bottle's body, which
        # looks the same with or without its cap), MegaPose matches only that
        # part's mesh, recentered on itself. part_offset = that center relative
        # to the assembly's center; it converts the part's pose into the pose
        # of the whole assembly, which is what gets rendered.
        self.part_offset = np.zeros(3) if part_offset is None else np.asarray(part_offset, dtype=float)
        self._reference: np.ndarray | None = None  # last accepted pose of the matched mesh (part frame)
        self._flow = FlowBoxTracker()
        self._flow_ready = False  # following the object right now
        self._flow_usable = False  # the object had enough texture to follow at all

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
        t_assembly = self._assembly_pose(result.t_camera_object)
        pose = cv_model_pose_to_threejs(t_assembly)
        extra["t_camera_object"] = t_assembly  # what's rendered (and the debug gizmo)
        extra["t_camera_mesh"] = result.t_camera_object  # what MegaPose refines from next frame
        return Measurement(
            confidence=confidence,
            position=pose.position,
            quaternion=pose.quaternion,
            bbox=object_box,
            polygon=polygon,
            extra=extra,
        )

    def initialize(self, frame: Frame, detection: SegmentedObject) -> Measurement:
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
        return measurement

    def update(self, frame: Frame) -> Measurement:
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
            # Flow was following the object and lost it (occluded, out of
            # view): no evidence the pose still fits, and the appearance score
            # alone can't be trusted to say so. Wait for the detector.
            return Measurement(confidence=0.0)
        if (
            self._held is not None
            and object_box is not None
            and self._flow.motion_since_anchor < self.still_motion_px
        ):
            return self._hold(object_box, polygon)
        result = self.client.refine(
            self.label, frame.jpeg, frame.calibration.camera_matrix, self._reference, self.refine_iterations
        )
        return self._measurement(result, object_box, polygon)

    def _hold(self, object_box: BoundingBox, polygon) -> Measurement:
        """The object hasn't moved: the last accepted pose, re-judged against where flow sees it now."""
        held = self._held
        assert held is not None
        projected = held.extra.get("projected_bbox")
        confidence = box_iou(object_box, projected) if projected is not None else held.confidence
        return Measurement(
            confidence=confidence,
            position=held.position,
            quaternion=held.quaternion,
            bbox=object_box,
            polygon=polygon,
            extra={**held.extra, "held": True, "pose_service_ms": 0.0},
        )

    def commit(self, measurement: Measurement) -> None:
        t_mesh = measurement.extra.get("t_camera_mesh")
        if t_mesh is not None:
            self._reference = t_mesh
        if not measurement.extra.get("held") and measurement.position is not None:
            # A real (refined) pose: the new still-reference. Only these reset
            # the motion anchor, so slow steady motion still adds up past the
            # threshold instead of being re-anchored away every frame.
            self._held = measurement
            if self.still_motion_px > 0:
                self._flow.set_anchor()
