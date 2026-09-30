"""
Gap/clearance measurement between two point clouds (Project.md #32): the
Measurement Engine layer (Project.md #74) — knows geometry and tolerances,
never PASS/FAIL (that stays the QA engine's job, reached separately, so a
measurement can feed into a deterministic rule without this module ever
making the disposition itself).

Uses Open3D for the actual nearest-neighbor search between point clouds —
the library Project.md's own tech stack names for Phase 6.

IMPORTANT — no plain RGB camera can produce a metric point cloud; a real gap
measurement needs an actual depth sensor (stereo, structured light, ToF,
LiDAR) to produce `points_a`/`points_b` in real-world meters in the first
place. This module's math is verified here against synthetic point clouds
with a known ground-truth gap (tests/backend/test_gap_measurement.py) —
it has NOT been run against real depth-camera hardware in this environment
(Project.md #25/#31/#53's "don't overclaim accuracy"). See docs/metrology.md.
"""

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class GapMeasurement:
    distance_m: float
    within_tolerance: bool | None  # None if no requirement/tolerance was given


def _to_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


def measure_min_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """
    The minimum distance from any point in `points_a` to the nearest point in
    `points_b` — the gap between two parts' visible surfaces, assuming both
    point sets are already expressed in the same real-world coordinate frame
    (meters).
    """
    if len(points_a) == 0 or len(points_b) == 0:
        raise ValueError("Both point clouds must be non-empty")

    cloud_a = _to_point_cloud(np.asarray(points_a, dtype=np.float64))
    cloud_b = _to_point_cloud(np.asarray(points_b, dtype=np.float64))
    distances = cloud_a.compute_point_cloud_distance(cloud_b)
    return float(np.min(np.asarray(distances)))


def is_within_tolerance(measured_m: float, required_m: float, tolerance_m: float) -> bool:
    """Pure tolerance check (Project.md #31) — reusable independent of any particular measurement source."""
    return abs(measured_m - required_m) <= tolerance_m


def measure_gap(
    points_a: np.ndarray,
    points_b: np.ndarray,
    required_m: float | None = None,
    tolerance_m: float | None = None,
) -> GapMeasurement:
    distance = measure_min_distance(points_a, points_b)
    within = (
        is_within_tolerance(distance, required_m, tolerance_m)
        if required_m is not None and tolerance_m is not None
        else None
    )
    return GapMeasurement(distance_m=distance, within_tolerance=within)
