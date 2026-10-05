"""
Lightweight 2D object tracker for markerless AR: Lucas-Kanade optical flow
on feature points inside the detected object's outline.

Detection answers "where is the cup?"; this answers "where did the cup I
already found move?" — initialized once from a YOLO segmentation, then
updated every frame from the previous frame alone, with no detector call.

Per update: points are tracked forward and backward (a point whose backward
track doesn't return to where it started is unreliable), a similarity
transform (translation + in-plane rotation + uniform scale) is fitted with
RANSAC, and the box/outline are moved by it. Scale is what carries depth:
the cup getting bigger in the image means closer. Confidence is the share of
points that both survived and agree with the fitted motion — it drops when
the object is occluded, leaves the frame, or moves too fast to follow.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.schemas.vision import BoundingBox, Vector2


@dataclass
class FlowUpdate:
    ok: bool
    confidence: float
    bbox: BoundingBox | None = None
    polygon: list[Vector2] | None = None
    scale: float = 1.0  # this frame's size change; > 1 = moved closer
    points: int = 0


class FlowBoxTracker:
    def __init__(
        self,
        max_points: int = 200,
        min_points: int = 8,
        forward_backward_threshold_px: float = 1.5,
        reseed_fraction: float = 0.5,
    ):
        self.max_points = max_points
        self.min_points = min_points
        self.fb_threshold = forward_backward_threshold_px
        self.reseed_fraction = reseed_fraction
        self._prev_gray: np.ndarray | None = None
        self._points: np.ndarray | None = None  # (N, 1, 2) float32
        self._polygon: np.ndarray | None = None  # (M, 2) float32
        self._seeded_count = 0
        # Where each tracked point was at the last set_anchor(), and motion
        # carried over from points dropped by a re-seed (see motion_since_anchor).
        self._anchor: np.ndarray | None = None  # (N, 2), parallel to _points
        self._carried_motion = 0.0
        # The object's 2D motion since reset_motion(), as one similarity
        # transform (3x3: scale, in-plane rotation, translation) — what moves a
        # 6DoF pose forward between pose estimates (trackers.propagate_pose).
        self._motion = np.eye(3)

    def _seed(self, gray: np.ndarray, polygon: np.ndarray) -> np.ndarray | None:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
        points = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_points, qualityLevel=0.01, minDistance=5, mask=mask
        )
        if points is None or len(points) < self.min_points:
            return None
        return points.astype(np.float32)

    def initialize(self, gray: np.ndarray, bbox: BoundingBox, polygon: list[Vector2] | None) -> bool:
        """Lock onto the object the detector found. False if it has too
        little visible texture to follow (a plain, featureless surface)."""
        if polygon and len(polygon) >= 3:
            outline = np.array([[p.x, p.y] for p in polygon], dtype=np.float32)
        else:
            outline = np.array(
                [[bbox.x1, bbox.y1], [bbox.x2, bbox.y1], [bbox.x2, bbox.y2], [bbox.x1, bbox.y2]], dtype=np.float32
            )
        points = self._seed(gray, outline)
        if points is None:
            return False
        self._prev_gray, self._points, self._polygon = gray, points, outline
        self._seeded_count = len(points)
        self.set_anchor()
        self.reset_motion()
        return True

    def reset_motion(self) -> None:
        self._motion = np.eye(3)

    @property
    def motion_matrix(self) -> np.ndarray:
        """2D similarity (3x3) taking image points at reset_motion() to where they are now."""
        return self._motion.copy()

    def set_anchor(self) -> None:
        """Remember where the object's points are now (e.g. when a pose was accepted)."""
        if self._points is not None:
            self._anchor = self._points.reshape(-1, 2).copy()
        self._carried_motion = 0.0

    @property
    def motion_since_anchor(self) -> float:
        """Median distance (px) the object's points moved since set_anchor().
        Net displacement, not a sum per frame: camera noise jitters points in
        place and doesn't add up, while real motion — including the object
        turning in place, which moves its surface points — does."""
        if self._points is None or self._anchor is None or len(self._anchor) == 0:
            return float("inf")
        moved = np.linalg.norm(self._points.reshape(-1, 2) - self._anchor, axis=1)
        return self._carried_motion + float(np.median(moved))

    def update(self, gray: np.ndarray) -> FlowUpdate:
        if self._prev_gray is None or self._points is None or self._polygon is None:
            return FlowUpdate(ok=False, confidence=0.0)

        lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        forward, status_f, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, self._points, None, **lk)
        backward, status_b, _ = cv2.calcOpticalFlowPyrLK(gray, self._prev_gray, forward, None, **lk)
        fb_error = np.linalg.norm((self._points - backward).reshape(-1, 2), axis=1)
        good = (status_f.ravel() == 1) & (status_b.ravel() == 1) & (fb_error < self.fb_threshold)

        attempted = len(self._points)
        if good.sum() < self.min_points:
            return FlowUpdate(ok=False, confidence=float(good.sum()) / attempted, points=int(good.sum()))

        old_pts, new_pts = self._points[good].reshape(-1, 2), forward[good].reshape(-1, 2)
        matrix, inliers = cv2.estimateAffinePartial2D(old_pts, new_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if matrix is None:
            return FlowUpdate(ok=False, confidence=0.0)
        inlier_mask = inliers.ravel().astype(bool)
        confidence = float(inlier_mask.sum()) / attempted

        self._motion = np.vstack([matrix, [0.0, 0.0, 1.0]]) @ self._motion
        polygon = cv2.transform(self._polygon.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        height, width = gray.shape
        x1, y1 = polygon.min(axis=0)
        x2, y2 = polygon.max(axis=0)
        if x2 < 0 or y2 < 0 or x1 > width or y1 > height:  # moved completely out of view
            return FlowUpdate(ok=False, confidence=0.0)

        self._prev_gray = gray
        self._polygon = polygon.astype(np.float32)
        self._points = new_pts[inlier_mask].reshape(-1, 1, 2).astype(np.float32)
        if self._anchor is not None:
            self._anchor = self._anchor[good][inlier_mask]
        # Points are lost over time (occlusion, rotation turning a surface
        # away). Re-seed inside the tracked outline — from the tracker's own
        # estimate, not the detector — so tracking can continue indefinitely.
        if len(self._points) < self._seeded_count * self.reseed_fraction:
            reseeded = self._seed(gray, self._polygon)
            if reseeded is not None:
                carried = self.motion_since_anchor
                self._points = reseeded
                self._seeded_count = len(reseeded)
                # New points have no history: anchor them here, keeping the motion so far.
                self.set_anchor()
                self._carried_motion = 0.0 if carried == float("inf") else carried

        return FlowUpdate(
            ok=True,
            confidence=confidence,
            bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
            polygon=[Vector2(x=float(x), y=float(y)) for x, y in polygon],
            scale=scale,
            points=len(self._points),
        )
