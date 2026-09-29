"""
Camera calibration (Project.md #25): "store calibration, do not repeatedly
assume an arbitrary camera matrix." A CameraCalibration is loaded once from a
JSON file produced by the checkerboard calibration worker
(app/workers/calibrate_camera.py) and reused for every pose estimate.

A rough default is provided purely so the pose pipeline is runnable before a
real calibration exists — every response built from it is explicitly marked
`approximate`, and Project.md #24/#25 is clear this is not the final-product
requirement.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CameraCalibration:
    camera_matrix: np.ndarray  # 3x3
    dist_coeffs: np.ndarray  # (N,)
    image_width: int
    image_height: int
    is_approximate: bool = False
    source: str = "unknown"

    @classmethod
    def from_file(cls, path: str | Path) -> "CameraCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            camera_matrix=np.array(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(data.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64),
            image_width=data["image_width"],
            image_height=data["image_height"],
            is_approximate=False,
            source=str(path),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "camera_matrix": self.camera_matrix.tolist(),
                    "dist_coeffs": self.dist_coeffs.tolist(),
                    "image_width": self.image_width,
                    "image_height": self.image_height,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def approximate(cls, image_width: int, image_height: int, assumed_hfov_deg: float = 70.0) -> "CameraCalibration":
        """
        A coarse pinhole model from an assumed horizontal field of view —
        good enough to prove the registration pipeline end-to-end, not to
        make a measurement claim (Project.md #31, #53).
        """
        hfov = math.radians(assumed_hfov_deg)
        fx = image_width / (2 * math.tan(hfov / 2))
        fy = fx  # square pixels assumed
        cx, cy = image_width / 2, image_height / 2
        matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return cls(
            camera_matrix=matrix,
            dist_coeffs=np.zeros(5, dtype=np.float64),
            image_width=image_width,
            image_height=image_height,
            is_approximate=True,
            source="approximate-default",
        )


def load_calibration(path: str | Path | None, image_width: int, image_height: int) -> CameraCalibration:
    """The one place that decides real-vs-approximate calibration (Project.md #62)."""
    if path and Path(path).exists():
        return CameraCalibration.from_file(path)
    return CameraCalibration.approximate(image_width, image_height)
