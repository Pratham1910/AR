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
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Where the in-app calibration (app/api/calibration.py) writes, and what's
# used when CAMERA_CALIBRATION_PATH isn't set.
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parents[4] / "data" / "calibration" / "camera.json"
CALIBRATION_VERSION = 2


def effective_calibration_path(configured: str | Path | None) -> Path | None:
    """The configured file, else the default one if a calibration was made, else None."""
    if configured:
        return Path(configured)
    return DEFAULT_CALIBRATION_PATH if DEFAULT_CALIBRATION_PATH.exists() else None


@dataclass
class CameraCalibration:
    camera_matrix: np.ndarray  # 3x3
    dist_coeffs: np.ndarray  # (N,)
    image_width: int
    image_height: int
    is_approximate: bool = False
    source: str = "unknown"
    camera_label: str | None = None  # the camera it was made with (browser device label)
    created_at: str | None = None  # ISO date
    rms_px: float | None = None  # reprojection error of the calibration itself
    images_used: int | None = None
    version: int = CALIBRATION_VERSION
    note: str | None = None  # why this calibration is (or isn't) the one in use

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
            camera_label=data.get("camera_label"),
            created_at=data.get("created_at"),
            rms_px=data.get("rms_px"),
            images_used=data.get("images_used"),
            version=int(data.get("version", 1)),
        )

    @property
    def has_distortion(self) -> bool:
        return not self.is_approximate and bool(np.abs(self.dist_coeffs).max() > 1e-4) if len(self.dist_coeffs) else False

    def distortion_max_px(self) -> float:
        """How far lens distortion moves a pixel at worst (image corners), in px."""
        if not self.has_distortion:
            return 0.0
        w, h = self.image_width, self.image_height
        corners = np.array([[[0, 0]], [[w - 1, 0]], [[0, h - 1]], [[w - 1, h - 1]], [[w / 2, 0]], [[0, h / 2]]], np.float64)
        undistorted = cv2.undistortPoints(corners, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
        return float(np.linalg.norm(undistorted - corners, axis=2).max())

    def vertical_fov_deg(self) -> float:
        """
        The vertical field of view this calibration implies, in degrees —
        for the frontend's Three.js overlay camera (Project.md #26). Without
        this, a hardcoded/guessed frontend FOV mismatches whatever FOV the
        backend actually used to compute position from apparent size, and
        the 3D overlay drifts visibly off the real object even when the
        underlying detection/pose math is correct.
        """
        fy = self.camera_matrix[1, 1]
        return math.degrees(2 * math.atan(self.image_height / (2 * fy)))

    def aspect_ratio(self) -> float:
        return self.image_width / self.image_height

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "camera_matrix": self.camera_matrix.tolist(),
                    "dist_coeffs": self.dist_coeffs.tolist(),
                    "image_width": self.image_width,
                    "image_height": self.image_height,
                    "camera_label": self.camera_label,
                    "created_at": self.created_at,
                    "rms_px": self.rms_px,
                    "images_used": self.images_used,
                    "version": self.version,
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


def load_calibration(
    path: str | Path | None, image_width: int, image_height: int, assumed_hfov_deg: float = 70.0,
    camera_label: str | None = None,
) -> CameraCalibration:
    """The one place that decides real-vs-approximate calibration (Project.md #62).
    `camera_label`: the camera the frames come from; a calibration recorded for
    a different camera is not used (it would be silently wrong)."""
    if path and Path(path).exists():
        calibration = CameraCalibration.from_file(path)
        if camera_label and calibration.camera_label and camera_label != calibration.camera_label:
            approx = CameraCalibration.approximate(image_width, image_height, assumed_hfov_deg)
            approx.note = f"calibration is for '{calibration.camera_label}', frames come from '{camera_label}'"
            log.warning("Calibration %s: %s — using the approximate default", path, approx.note)
            return approx
        if (calibration.image_width, calibration.image_height) == (image_width, image_height):
            return calibration
        # Same camera at another resolution: the intrinsics scale with it, as
        # long as the camera isn't cropping a different region (same aspect).
        # A different aspect means a different crop of the sensor (e.g. 4:3 vs
        # 16:9), which the file can't describe — guess rather than mislead.
        sx, sy = image_width / calibration.image_width, image_height / calibration.image_height
        if abs(sx - sy) > 0.01:
            log.warning(
                "Calibration %s is %dx%d; frames are %dx%d (different aspect) — using the approximate default",
                path, calibration.image_width, calibration.image_height, image_width, image_height,
            )
            approx = CameraCalibration.approximate(image_width, image_height, assumed_hfov_deg)
            approx.note = (f"calibration is {calibration.image_width}x{calibration.image_height}; "
                           f"frames are {image_width}x{image_height} (different aspect)")
            return approx
        matrix = calibration.camera_matrix.copy()
        matrix[0, :] *= sx
        matrix[1, :] *= sy
        matrix[0, 2] = (calibration.camera_matrix[0, 2] + 0.5) * sx - 0.5
        matrix[1, 2] = (calibration.camera_matrix[1, 2] + 0.5) * sy - 0.5
        calibration.camera_matrix = matrix
        calibration.image_width, calibration.image_height = image_width, image_height
        calibration.note = "scaled from the calibrated resolution"
        return calibration
    approx = CameraCalibration.approximate(image_width, image_height, assumed_hfov_deg)
    approx.note = "no camera calibration yet — intrinsics assumed from the field of view"
    return approx


_undistort_maps: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def undistort(image: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    """The frame with lens distortion removed, same size and camera matrix —
    so everything downstream (MegaPose, edges, features, the overlay's pinhole
    camera) works on one consistent, distortion-free image. Unchanged when
    there's no distortion to remove."""
    if not calibration.has_distortion:
        return image
    h, w = image.shape[:2]
    key = (w, h, calibration.camera_matrix.tobytes(), calibration.dist_coeffs.tobytes())
    maps = _undistort_maps.get(key)
    if maps is None:
        maps = cv2.initUndistortRectifyMap(
            calibration.camera_matrix, calibration.dist_coeffs, None, calibration.camera_matrix, (w, h), cv2.CV_16SC2
        )
        _undistort_maps.clear()
        _undistort_maps[key] = maps
    return cv2.remap(image, maps[0], maps[1], cv2.INTER_LINEAR)
