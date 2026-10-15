"""Camera calibration from the app, lens distortion handling, and the camera -> renderer transform chain (Phase 8 B)."""

import base64
import json

import cv2
import numpy as np
import pytest

from app.services.pose import calibration as cal_module
from app.services.pose.calibration import CameraCalibration, load_calibration, undistort
from app.services.pose.transforms import cv_model_pose_to_threejs

W, H = 640, 480
K_TRUE = np.array([[560.0, 0, 322.0], [0, 556.0, 238.0], [0, 0, 1]])
D_TRUE = np.array([-0.21, 0.08, 0.0, 0.0, 0.0])
COLS, ROWS, SQUARE = 9, 6, 0.025  # inner corners, metres


def _board_texture(px_per_square=40):
    squares = np.indices((ROWS + 1, COLS + 1)).sum(axis=0) % 2
    tex = np.kron(squares, np.ones((px_per_square, px_per_square))) * 255
    margin = px_per_square
    return np.pad(tex, margin, constant_values=255).astype(np.uint8), px_per_square, margin


_GRID_UNDISTORTED = None


def _photo(rvec, tvec):
    """The board seen by K_TRUE through a lens with D_TRUE (exact per-pixel distortion)."""
    global _GRID_UNDISTORTED
    tex, s, m = _board_texture()
    # Texture pixel -> board plane metres: corner (0,0) of the inner grid is texture (m + s, m + s).
    tex_pts = np.array([[0, 0], [tex.shape[1], 0], [tex.shape[1], tex.shape[0]], [0, tex.shape[0]]], np.float64)
    board = (tex_pts - (m + s)) / s * SQUARE
    obj = np.c_[board, np.zeros(4)]
    img_pts = cv2.projectPoints(obj, rvec, tvec, K_TRUE, None)[0].reshape(-1, 2)
    hmat = cv2.getPerspectiveTransform(tex_pts.astype(np.float32), img_pts.astype(np.float32))
    pinhole = cv2.warpPerspective(tex, hmat, (W, H), borderValue=200)
    if _GRID_UNDISTORTED is None:
        grid = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).reshape(-1, 1, 2).astype(np.float64)
        _GRID_UNDISTORTED = cv2.undistortPoints(grid, K_TRUE, D_TRUE, P=K_TRUE).reshape(H, W, 2).astype(np.float32)
    return cv2.remap(pinhole, _GRID_UNDISTORTED[..., 0], _GRID_UNDISTORTED[..., 1], cv2.INTER_LINEAR, borderValue=200)


def _views():
    rng = np.random.default_rng(3)
    for _ in range(14):
        rvec = np.radians(rng.uniform([-30, -30, -15], [30, 30, 15]))
        tvec = np.array([rng.uniform(-0.12, 0.0), rng.uniform(-0.08, -0.02), rng.uniform(0.38, 0.55)])
        yield rvec, tvec


@pytest.fixture
def calibration_dirs(tmp_path, monkeypatch):
    import app.api.calibration as api

    images, target = tmp_path / "images", tmp_path / "camera.json"
    monkeypatch.setattr(api, "DEFAULT_IMAGES_DIR", images)
    monkeypatch.setattr(api, "DEFAULT_CALIBRATION_PATH", target)
    monkeypatch.setattr(cal_module, "DEFAULT_CALIBRATION_PATH", target)
    monkeypatch.setattr(api._settings, "camera_calibration_path", None)
    return api, target


def test_calibration_from_captured_frames_recovers_the_camera(calibration_dirs):
    api, target = calibration_dirs
    found = 0
    for rvec, tvec in _views():
        ok, jpg = cv2.imencode(".png", _photo(rvec, tvec))
        result = api.capture_image(api.CaptureRequest(image_base64=base64.b64encode(jpg.tobytes()).decode()))
        found += result.found
    assert found >= 10
    blank = cv2.imencode(".png", np.full((H, W), 128, np.uint8))[1]
    assert not api.capture_image(api.CaptureRequest(image_base64=base64.b64encode(blank.tobytes()).decode())).found

    status = api.run_calibration(api.RunRequest(square_size_m=SQUARE, camera_label="Logitech BRIO"))
    assert status.calibrated and status.camera_label == "Logitech BRIO" and status.rms_px < 0.5
    saved = CameraCalibration.from_file(target)
    assert saved.camera_matrix[0, 0] == pytest.approx(K_TRUE[0, 0], rel=0.02)
    assert saved.camera_matrix[1, 1] == pytest.approx(K_TRUE[1, 1], rel=0.02)
    assert saved.camera_matrix[0, 2] == pytest.approx(K_TRUE[0, 2], abs=4)
    assert saved.dist_coeffs[0] == pytest.approx(D_TRUE[0], abs=0.04)
    assert status.distortion_max_px > 5  # this lens bends the corners by many pixels

    # Used from then on (no CAMERA_CALIBRATION_PATH needed)...
    in_use = load_calibration(cal_module.effective_calibration_path(None), W, H)
    assert not in_use.is_approximate and in_use.camera_label == "Logitech BRIO"
    # ...but never for another camera.
    other = load_calibration(target, W, H, camera_label="Integrated Webcam")
    assert other.is_approximate and "Logitech BRIO" in other.note
    assert api.remove_calibration().calibrated is False


def test_undistortion_puts_straight_lines_back():
    """A board corner seen through the lens lands, after undistort(), where a pinhole camera puts it."""
    rvec, tvec = np.radians([10.0, -15.0, 5.0]), np.array([-0.15, -0.05, 0.42])
    photo = _photo(rvec, tvec)
    calibration = CameraCalibration(K_TRUE, D_TRUE, W, H, source="test")
    fixed = undistort(photo, calibration)
    found, corners = cv2.findChessboardCorners(fixed, (COLS, ROWS))
    assert found
    corners = cv2.cornerSubPix(fixed, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    obj = np.zeros((COLS * ROWS, 3))
    obj[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQUARE
    pinhole = cv2.projectPoints(obj, rvec, tvec, K_TRUE, None)[0].reshape(-1, 2)
    error = np.linalg.norm(np.sort(corners.reshape(-1, 2), axis=0) - np.sort(pinhole, axis=0), axis=1)
    assert np.median(error) < 0.8
    # No distortion, nothing done.
    assert undistort(photo, CameraCalibration.approximate(W, H)) is photo


def test_a_calibration_for_another_resolution_is_scaled_or_refused(tmp_path):
    path = tmp_path / "c.json"
    CameraCalibration(K_TRUE, D_TRUE, W, H, camera_label="BRIO").save(path)
    scaled = load_calibration(path, 1280, 960)
    assert scaled.camera_matrix[0, 0] == pytest.approx(2 * K_TRUE[0, 0]) and "scaled" in scaled.note
    refused = load_calibration(path, 1280, 720)  # different aspect: a different crop of the sensor
    assert refused.is_approximate and "different aspect" in refused.note
    assert json.loads(path.read_text())["version"] == 2


# ── the camera -> renderer chain ─────────────────────────────────────────


def _threejs_pixel(point_model, t_co, k, width, height, near=0.01, far=100.0):
    """What the frontend does: model placed by (position, quaternion) from
    cv_model_pose_to_threejs, camera at the origin looking down -Z, projection
    built from fx, fy, cx, cy exactly like RegistrationOverlay.applyCameraIntrinsics."""
    pose = cv_model_pose_to_threejs(t_co)
    x, y, z, w = pose.quaternion
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    p_gl = rot @ point_model + np.asarray(pose.position)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    left, right = -cx * near / fx, (width - cx) * near / fx
    top, bottom = cy * near / fy, -(height - cy) * near / fy
    proj = np.array([  # THREE.Matrix4.makePerspective(left, right, top, bottom, near, far)
        [2 * near / (right - left), 0, (right + left) / (right - left), 0],
        [0, 2 * near / (top - bottom), (top + bottom) / (top - bottom), 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],
    ])
    clip = proj @ np.append(p_gl, 1.0)
    ndc = clip[:3] / clip[3]
    return np.array([(ndc[0] + 1) / 2 * width, (1 - ndc[1]) / 2 * height])


def test_the_rendered_model_lands_exactly_where_opencv_projects_it():
    """No mirroring, axis flip or principal-point error between pose maths and rendering."""
    rng = np.random.default_rng(0)
    k = np.array([[912.0, 0, 655.0], [0, 905.0, 351.0], [0, 0, 1]])
    for _ in range(5):
        rvec = rng.uniform(-1.5, 1.5, 3)
        t_co = np.eye(4)
        t_co[:3, :3] = cv2.Rodrigues(rvec)[0]
        t_co[:3, 3] = [rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(0.3, 0.8)]
        for point in rng.uniform(-0.08, 0.08, (6, 3)):
            cam = t_co[:3, :3] @ point + t_co[:3, 3]
            opencv = (k @ cam)[:2] / cam[2]
            assert _threejs_pixel(point, t_co, k, 1280, 720) == pytest.approx(opencv, abs=1e-6)


def test_the_upper_right_of_the_model_stays_upper_right_on_screen():
    """A camera looking straight down at a box: model +X right, -Z (away) up the image."""
    t_co = np.eye(4)
    t_co[:3, :3] = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # camera above, looking down -Y
    t_co[:3, 3] = [0, 0, 0.5]
    k = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    right_far = _threejs_pixel(np.array([0.05, 0.0, -0.05]), t_co, k, 1280, 720)
    assert right_far[0] > 640 and right_far[1] < 360
