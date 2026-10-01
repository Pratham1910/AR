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

from app.schemas.vision import SegmentedObject
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


class MegaPoseTracker:
    can_recover = True  # refining from the last good pose can re-lock after a brief occlusion

    def __init__(self, client: ModelPoseClient, label: str, refine_iterations: int):
        self.client = client
        self.label = label
        self.refine_iterations = refine_iterations
        self._reference: np.ndarray | None = None  # last accepted T_camera_object

    @staticmethod
    def _measurement(result: PoseServiceResult, detection: SegmentedObject | None = None) -> Measurement:
        if result.t_camera_object is None:
            return Measurement(confidence=0.0, extra={"pose_service_ms": result.elapsed_ms})
        pose = cv_model_pose_to_threejs(result.t_camera_object)
        return Measurement(
            confidence=result.score,
            position=pose.position,
            quaternion=pose.quaternion,
            bbox=detection.bbox if detection else None,
            polygon=detection.polygon if detection else None,
            extra={"t_camera_object": result.t_camera_object, "pose_service_ms": result.elapsed_ms},
        )

    def initialize(self, frame: Frame, detection: SegmentedObject) -> Measurement:
        bbox = [detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2]
        result = self.client.full_search(self.label, frame.jpeg, frame.calibration.camera_matrix, bbox)
        return self._measurement(result, detection)

    def update(self, frame: Frame) -> Measurement:
        if self._reference is None:
            return Measurement(confidence=0.0)
        result = self.client.refine(
            self.label, frame.jpeg, frame.calibration.camera_matrix, self._reference, self.refine_iterations
        )
        return self._measurement(result)

    def commit(self, measurement: Measurement) -> None:
        t_co = measurement.extra.get("t_camera_object")
        if t_co is not None:
            self._reference = t_co
