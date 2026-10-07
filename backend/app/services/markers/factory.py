"""The one place that picks a marker family (MARKER_FAMILY in .env)."""

from collections.abc import Callable

from app.services.markers.aruco_detector import ArucoDetector
from app.services.markers.detector import MarkerDetector

# To add AprilTag: write AprilTagDetector(MarkerDetector) and register it here.
_FAMILIES: dict[str, Callable[[str], MarkerDetector]] = {
    "aruco": ArucoDetector,
}


def build_marker_detector(family: str, dictionary: str) -> MarkerDetector:
    if family not in _FAMILIES:
        raise ValueError(f"Unknown marker family {family!r}; supported: {list(_FAMILIES)}")
    return _FAMILIES[family](dictionary)
