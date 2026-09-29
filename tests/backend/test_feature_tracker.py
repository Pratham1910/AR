"""
Verifies the feature-tracker's full pipeline (ORB detect -> match -> RANSAC
solvePnP) with a rigorously constructed synthetic scene, not a shortcut:

  1. A richly textured reference image (a random high-contrast cell grid —
     see `_textured_reference_image`'s docstring for why this texture, not
     plain noise or solid shapes, is what actually gives ORB enough
     distinct, blur-tolerant keypoints to match reliably).
  2. A *known ground-truth* 6DoF pose for that plane, correctly composed
     against OpenCV's camera convention (Y-down, Z-forward) — see
     `_ground_truth_pose`'s docstring for the subtlety here: our reference
     plane uses a Y-up "as printed" convention (matching
     `feature_tracker.register_reference`), so "facing the camera,
     right-side up" is a 180° rotation about X, *not* the identity rotation.
  3. The plane's 4 corners are projected through that ground-truth pose with
     a real pinhole camera model (cv2.projectPoints) to get where they'd
     land in a live frame.
  4. cv2.warpPerspective uses that same homography to synthesize the "live"
     frame — mathematically exact for a flat plane under pinhole
     projection, not an approximation.
  5. The tracker is run on that synthesized frame and must recover a pose
     close to the known ground truth.

No physical camera or printed label required to verify the math end-to-end.
"""

import cv2
import numpy as np
import pytest

from app.services.pose.calibration import CameraCalibration
from app.services.pose.feature_tracker import FeatureTracker, ReferencePlane

WIDTH, HEIGHT = 640, 480
LABEL_SIZE_M = 0.10
CALIBRATION = CameraCalibration.approximate(WIDTH, HEIGHT)

# "Facing the camera, right-side up": feature_tracker.register_reference maps
# a reference photo's TOP row to object Y=+half (a Y-up, "as printed on the
# page" convention). OpenCV's camera frame is Y-DOWN, Z-forward. A proper
# rotation can't just flip the sign of Y alone (that's a reflection, not a
# rotation) — it has to flip Y *and* Z together, which is exactly a 180°
# rotation about the X axis. Skipping this and using rvec=[0,0,0] as "no
# rotation" (as an earlier version of this test did) silently renders the
# synthetic "live" frame upside-down, which ORB's descriptors are not
# invariant to — every match then fails. Composing a small perturbation on
# top of that base, rather than adding rvec components directly, keeps the
# result an exact rotation (rvec addition is only a valid approximation of
# rotation composition for very small angles).
_FACE_CAMERA_ROTATION_MATRIX, _ = cv2.Rodrigues(np.array([np.pi, 0.0, 0.0]))


def _ground_truth_pose(small_rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    small_rotation, _ = cv2.Rodrigues(small_rvec)
    total_rotation = _FACE_CAMERA_ROTATION_MATRIX @ small_rotation
    rvec, _ = cv2.Rodrigues(total_rotation)
    return rvec.flatten(), tvec


def _textured_reference_image(size: int = 640, seed: int = 42, grid: int = 20) -> np.ndarray:
    """
    A random-cell grid (like a QR code) — sharp, high-contrast corners at
    every cell boundary, randomized so no two corners share a local
    neighborhood (unlike a real checkerboard, which is periodic and
    therefore ambiguous to match). Plain pixel noise and solid shapes
    (rectangles/circles) were both tried and rejected: noise has no
    structure that survives warp+interpolation, and solid shapes' corners
    are all locally identical (a rectangle corner looks like every other
    rectangle corner), which ORB's ratio test correctly treats as ambiguous
    and discards.
    """
    rng = np.random.default_rng(seed)
    cells = rng.choice([0, 128, 255], size=(grid, grid)).astype(np.uint8)
    return cv2.resize(cells, (size, size), interpolation=cv2.INTER_NEAREST)


def _synthesize_live_frame(reference_image: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    h, w = reference_image.shape[:2]
    half = LABEL_SIZE_M / 2.0
    # Order matches ReferencePlane's own pixel->object mapping: TL, TR, BR, BL.
    object_corners = np.array(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float64
    )
    projected, _ = cv2.projectPoints(object_corners, rvec, tvec, CALIBRATION.camera_matrix, CALIBRATION.dist_coeffs)
    dst_corners = projected.reshape(4, 2).astype(np.float32)
    src_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    homography = cv2.getPerspectiveTransform(src_corners, dst_corners)
    live_frame = np.full((HEIGHT, WIDTH), 128, dtype=np.uint8)  # neutral background
    cv2.warpPerspective(reference_image, homography, (WIDTH, HEIGHT), dst=live_frame, borderMode=cv2.BORDER_TRANSPARENT)
    return cv2.cvtColor(live_frame, cv2.COLOR_GRAY2BGR)


def _rotation_angle_diff_deg(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    r_a, _ = cv2.Rodrigues(rvec_a)
    r_b, _ = cv2.Rodrigues(rvec_b)
    relative = r_a.T @ r_b
    trace = np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def test_register_reference_extracts_features_with_3d_coordinates():
    tracker = FeatureTracker()
    image = _textured_reference_image()
    reference = tracker.register_reference(image, LABEL_SIZE_M, LABEL_SIZE_M)

    assert reference.descriptors.shape[0] > 100  # a textured 640x640 grid should yield plenty of ORB keypoints
    assert reference.object_points.shape == (reference.descriptors.shape[0], 3)
    # All object points must lie within the declared label bounds.
    assert np.all(np.abs(reference.object_points[:, 0]) <= LABEL_SIZE_M / 2 + 1e-9)
    assert np.all(np.abs(reference.object_points[:, 1]) <= LABEL_SIZE_M / 2 + 1e-9)
    assert np.all(reference.object_points[:, 2] == 0.0)


def test_recovers_known_pose_from_synthesized_frame():
    tracker = FeatureTracker()
    reference_image = _textured_reference_image()
    reference = tracker.register_reference(reference_image, LABEL_SIZE_M, LABEL_SIZE_M)

    ground_truth_rvec, ground_truth_tvec = _ground_truth_pose(
        small_rvec=np.array([0.05, 0.1, 0.02]), tvec=np.array([0.0, 0.0, 0.18])
    )
    live_frame = _synthesize_live_frame(reference_image, ground_truth_rvec, ground_truth_tvec)

    estimate = tracker.estimate_pose(live_frame, reference, CALIBRATION)

    assert estimate.found is True
    assert estimate.num_inliers >= 12
    assert np.linalg.norm(estimate.tvec.flatten() - ground_truth_tvec) < 0.01  # within 1cm
    assert _rotation_angle_diff_deg(estimate.rvec, ground_truth_rvec) < 5.0  # within 5 degrees


def test_recovers_pose_with_translation_offset():
    tracker = FeatureTracker()
    reference_image = _textured_reference_image()
    reference = tracker.register_reference(reference_image, LABEL_SIZE_M, LABEL_SIZE_M)

    ground_truth_rvec, ground_truth_tvec = _ground_truth_pose(
        small_rvec=np.array([0.0, 0.0, 0.0]), tvec=np.array([0.03, -0.02, 0.18])
    )
    live_frame = _synthesize_live_frame(reference_image, ground_truth_rvec, ground_truth_tvec)

    estimate = tracker.estimate_pose(live_frame, reference, CALIBRATION)

    assert estimate.found is True
    assert np.linalg.norm(estimate.tvec.flatten() - ground_truth_tvec) < 0.01


def test_no_match_on_unrelated_frame():
    tracker = FeatureTracker()
    reference_image = _textured_reference_image(seed=1)
    reference = tracker.register_reference(reference_image, LABEL_SIZE_M, LABEL_SIZE_M)

    unrelated_frame = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)  # flat, featureless
    estimate = tracker.estimate_pose(unrelated_frame, reference, CALIBRATION)

    assert estimate.found is False


def test_reference_plane_save_and_load_round_trip(tmp_path):
    tracker = FeatureTracker()
    image = _textured_reference_image()
    reference = tracker.register_reference(image, 0.08, 0.12)

    path = tmp_path / "bottle"
    reference.save(path)
    assert ReferencePlane.exists(path)

    loaded = ReferencePlane.load(path)
    assert loaded.label_width_m == pytest.approx(0.08)
    assert loaded.label_height_m == pytest.approx(0.12)
    assert np.array_equal(loaded.descriptors, reference.descriptors)
    assert np.allclose(loaded.object_points, reference.object_points)
