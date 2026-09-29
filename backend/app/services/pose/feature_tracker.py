"""
Feature/keypoint-based 6DoF pose ("image target" tracking) — Project.md
#24's markerless upgrade, one real step closer than the bounding-box
approximation in markerless.py.

Principle (the same one Vuforia Image Targets / classic AR SDKs use for
textured planar objects, and a coarse cousin of what an industrial platform
like DELMIA Augmented Experience does against full CAD geometry): capture
one reference photo of the object's label/texture, detect distinctive
keypoints in it, and assign each a 3D coordinate on a known-size flat plane.
At runtime, detect keypoints in the live frame, match them against the
reference by descriptor similarity, and solve the resulting 2D(live)<->
3D(object) correspondences with solvePnPRansac for a REAL 6DoF pose —
position AND orientation, with outlier rejection.

Honest limits (Project.md #25/#31/#53 — never overclaim accuracy):
  - The object needs genuine visual texture (a printed label, logo, text).
    A plain glossy/matte surface with no texture will not produce enough
    keypoints to match, and this will correctly report found=False rather
    than guess.
  - The reference plane is assumed flat (a label wrapped on a cylinder is
    only approximately flat) — accuracy degrades as the viewing angle
    departs from the angle the reference photo was taken at.
  - This is still not full CAD-geometry matching (DELMIA-grade) — it
    matches against one photographed plane, not the object's actual 3D
    mesh from every angle. See docs/pose.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.services.pose.calibration import CameraCalibration

_ORB_N_FEATURES = 1500
_RATIO_TEST_THRESHOLD = 0.75
_MIN_INLIERS_FOR_POSE = 12


@dataclass
class ReferencePlane:
    """A registered reference image's features + their assigned 3D coordinates."""

    descriptors: np.ndarray  # (N, 32) uint8, ORB descriptors
    object_points: np.ndarray  # (N, 3) float64, plane-local coordinates in meters
    label_width_m: float
    label_height_m: float

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez(
            path.with_suffix(".npz"),
            descriptors=self.descriptors,
            object_points=self.object_points,
        )
        path.with_suffix(".json").write_text(
            json.dumps({"label_width_m": self.label_width_m, "label_height_m": self.label_height_m}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ReferencePlane":
        path = Path(path)
        npz = np.load(path.with_suffix(".npz"))
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(
            descriptors=npz["descriptors"],
            object_points=npz["object_points"],
            label_width_m=meta["label_width_m"],
            label_height_m=meta["label_height_m"],
        )

    @classmethod
    def exists(cls, path: str | Path) -> bool:
        path = Path(path)
        return path.with_suffix(".npz").exists() and path.with_suffix(".json").exists()


@dataclass
class FeaturePoseEstimate:
    found: bool
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    num_matches: int = 0
    num_inliers: int = 0
    inlier_points_px: list[tuple[float, float]] | None = None  # for drawing


class RegistrationQuality:
    """Feature-count feedback shown at registration time, before the user even tries live tracking."""

    @staticmethod
    def describe(feature_count: int) -> str:
        if feature_count < 30:
            return "too_few"  # very unlikely to track reliably
        if feature_count < 100:
            return "marginal"
        return "good"


class FeatureTracker:
    """Stateless wrapper around ORB detection/matching + solvePnPRansac."""

    def __init__(self, n_features: int = _ORB_N_FEATURES):
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def register_reference(self, image: np.ndarray, label_width_m: float, label_height_m: float) -> ReferencePlane:
        """
        Detects keypoints in a reference photo and assigns each a 3D
        coordinate on a `label_width_m` x `label_height_m` plane, centered at
        the plane's own local origin (X right, Y up, Z=0). The whole
        reference image is assumed to be filled by the label/texture —
        crop/frame the capture accordingly.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        if descriptors is None:
            descriptors = np.empty((0, 32), dtype=np.uint8)

        h, w = gray.shape[:2]
        object_points = np.array(
            [
                [
                    (kp.pt[0] / w - 0.5) * label_width_m,
                    -(kp.pt[1] / h - 0.5) * label_height_m,
                    0.0,
                ]
                for kp in keypoints
            ],
            dtype=np.float64,
        ).reshape(-1, 3)

        return ReferencePlane(
            descriptors=descriptors,
            object_points=object_points,
            label_width_m=label_width_m,
            label_height_m=label_height_m,
        )

    def estimate_pose(
        self, frame: np.ndarray, reference: ReferencePlane, calibration: CameraCalibration
    ) -> FeaturePoseEstimate:
        if reference.descriptors.shape[0] < 4:
            return FeaturePoseEstimate(found=False)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 4:
            return FeaturePoseEstimate(found=False)

        raw_matches = self._matcher.knnMatch(descriptors, reference.descriptors, k=2)
        good_matches = [
            m for m, n in (pair for pair in raw_matches if len(pair) == 2)
            if m.distance < _RATIO_TEST_THRESHOLD * n.distance
        ]

        if len(good_matches) < _MIN_INLIERS_FOR_POSE:
            return FeaturePoseEstimate(found=False, num_matches=len(good_matches))

        image_points = np.array([keypoints[m.queryIdx].pt for m in good_matches], dtype=np.float64)
        object_points = np.array([reference.object_points[m.trainIdx] for m in good_matches], dtype=np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            calibration.camera_matrix,
            calibration.dist_coeffs,
            reprojectionError=8.0,
            confidence=0.99,
            iterationsCount=200,
        )

        if not success or inliers is None or len(inliers) < _MIN_INLIERS_FOR_POSE:
            return FeaturePoseEstimate(found=False, num_matches=len(good_matches))

        inlier_idx = inliers.flatten()
        inlier_points_px = [tuple(image_points[i]) for i in inlier_idx]

        return FeaturePoseEstimate(
            found=True,
            rvec=rvec,
            tvec=tvec,
            num_matches=len(good_matches),
            num_inliers=len(inlier_idx),
            inlier_points_px=inlier_points_px,
        )
