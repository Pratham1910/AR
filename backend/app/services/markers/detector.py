"""
Fiducial marker detection, independent of the marker family.

A MarkerDetector answers one question — "which marker ids are in this frame,
and where are their corners, in pixels?" — and nothing else. It does no pose
math (that needs camera intrinsics: app/services/pose/aruco_pose.py) and
knows nothing about products (app/api/markers.py resolves ids through the
marker_bindings table). Swapping ArUco for AprilTag is a new subclass plus one
line in factory.py; neither of those layers changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarkerDetection:
    marker_id: int
    # The marker's 4 corners in the frame's own pixel space, in the marker's
    # own order (top-left, top-right, bottom-right, bottom-left of the printed
    # pattern) — so the order follows the marker as it rotates.
    corners_px: tuple[tuple[float, float], ...]

    @property
    def center_px(self) -> tuple[float, float]:
        xs, ys = zip(*self.corners_px)
        return (sum(xs) / len(xs), sum(ys) / len(ys))


class MarkerDetector(ABC):
    # Together these namespace a marker id: id 0 of DICT_4X4_50 is not the
    # same physical marker as id 0 of another dictionary or family.
    family: str
    dictionary: str

    @property
    @abstractmethod
    def marker_count(self) -> int | None:
        """How many ids the dictionary defines (valid ids are 0..count-1), if known."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[MarkerDetection]:
        """Every marker found in a BGR or grayscale frame; empty if none."""
