"""
Gap/clearance measurement API (Project.md #32, Phase 6). Deliberately
generic over how the point clouds were produced (a depth camera via
app/services/metrology/depth_to_points.py, or any other source already in
real-world meters) — this endpoint is the Measurement Engine layer only, and
never returns PASS/FAIL (Project.md #74); that's the QA engine's job,
reached separately.
"""

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.schemas.pose import Vector3
from app.services.metrology.gap_measurement import measure_gap

router = APIRouter(prefix="/api/metrology", tags=["metrology"])


class GapMeasurementRequest(BaseModel):
    points_a: list[Vector3]
    points_b: list[Vector3]
    required_m: float | None = None
    tolerance_m: float | None = None


class GapMeasurementResponse(BaseModel):
    distance_m: float
    within_tolerance: bool | None


@router.post("/measure-gap", response_model=GapMeasurementResponse)
def measure_gap_endpoint(request: GapMeasurementRequest) -> GapMeasurementResponse:
    """
    Minimum distance between two point clouds already expressed in the same
    real-world coordinate frame (meters) — Project.md #32's gap/clearance
    inspection.

    IMPORTANT: verified only against synthetic point clouds with a known
    ground-truth gap (tests/backend/test_gap_measurement.py) — never against
    a real depth sensor in this environment. See docs/metrology.md.
    """
    points_a = np.array([[p.x, p.y, p.z] for p in request.points_a])
    points_b = np.array([[p.x, p.y, p.z] for p in request.points_b])

    try:
        result = measure_gap(points_a, points_b, request.required_m, request.tolerance_m)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return GapMeasurementResponse(distance_m=result.distance_m, within_tolerance=result.within_tolerance)
