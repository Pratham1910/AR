"""ArUco implementation of MarkerDetector, on OpenCV's cv2.aruco (opencv-contrib)."""

import cv2
import numpy as np

from app.services.markers.detector import MarkerDetection, MarkerDetector


def aruco_dictionary_names() -> list[str]:
    """Every predefined dictionary this OpenCV build ships, e.g. DICT_4X4_50."""
    return sorted(name for name in dir(cv2.aruco) if name.startswith("DICT_"))


class ArucoDetector(MarkerDetector):
    family = "aruco"

    def __init__(self, dictionary_name: str):
        if dictionary_name not in aruco_dictionary_names():
            raise ValueError(
                f"Unknown ArUco dictionary {dictionary_name!r}; supported: {aruco_dictionary_names()}"
            )
        self.dictionary = dictionary_name
        self._dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
        self._detector = cv2.aruco.ArucoDetector(self._dictionary, cv2.aruco.DetectorParameters())

    @property
    def marker_count(self) -> int:
        return int(self._dictionary.bytesList.shape[0])

    def detect(self, frame: np.ndarray) -> list[MarkerDetection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return []
        return [
            MarkerDetection(
                marker_id=int(marker_id),
                corners_px=tuple((float(x), float(y)) for x, y in marker_corners.reshape(4, 2)),
            )
            for marker_id, marker_corners in zip(ids.flatten(), corners)
        ]
