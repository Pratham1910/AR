"""
Checkerboard camera calibration (Project.md #25): produces a
CameraCalibration JSON file from a folder of chessboard images taken with the
same camera/resolution used at runtime. Without this, pose estimation falls
back to CameraCalibration.approximate() (see app/services/pose/calibration.py)
— fine for proving the pipeline, not for a measurement claim.

Usage:
    1. Print a checkerboard (default: 9x6 internal corners, e.g. OpenCV's
       standard pattern) and photograph it ~15-20 times from different
       angles/distances with the target camera, at the target resolution.
    2. Save those photos into data/calibration/images/.
    3. python -m app.workers.calibrate_camera
    4. Set CAMERA_CALIBRATION_PATH in .env to the printed output path.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from app.services.pose.calibration import CameraCalibration

DEFAULT_IMAGES_DIR = Path(__file__).resolve().parents[3] / "data" / "calibration" / "images"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[3] / "data" / "calibration" / "camera.json"


def calibrate(images_dir: Path, pattern_cols: int, pattern_rows: int, square_size_m: float) -> CameraCalibration:
    pattern_size = (pattern_cols, pattern_rows)
    object_points_template = np.zeros((pattern_cols * pattern_rows, 3), dtype=np.float32)
    object_points_template[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2) * square_size_m

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No calibration images found in {images_dir}")

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            print(f"  (no checkerboard found in {path.name}, skipping)")
            continue
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        object_points.append(object_points_template)
        image_points.append(corners)
        print(f"  + {path.name}")

    if len(object_points) < 5:
        raise ValueError(f"Only {len(object_points)} usable calibration images found — need at least 5-10.")

    assert image_size is not None
    _, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    return CameraCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs.flatten(),
        image_width=image_size[0],
        image_height=image_size[1],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--pattern-cols", type=int, default=9, help="Internal corners, horizontal")
    parser.add_argument("--pattern-rows", type=int, default=6, help="Internal corners, vertical")
    parser.add_argument("--square-size-m", type=float, default=0.025, help="Checkerboard square side, in meters")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    print(f"Calibrating from {args.images_dir} ...")
    calibration = calibrate(args.images_dir, args.pattern_cols, args.pattern_rows, args.square_size_m)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    calibration.save(args.output)
    print(f"Wrote {args.output}")
    print(f"Set CAMERA_CALIBRATION_PATH={args.output} in .env to use it.")


if __name__ == "__main__":
    main()
