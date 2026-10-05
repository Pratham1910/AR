"""Clip replay benchmark: live-like frame picking, pose round trip, metrics."""

from pathlib import Path

import numpy as np
import pytest

from app.services.pose.transforms import cv_model_pose_to_threejs
from app.services.tracking.replay import ClipFrame, ReplayRecord, next_frame, summarize, threejs_pose_to_cv

FRAMES = [ClipFrame(i, i * 100.0, Path(f"{i}.jpg")) for i in range(10)]  # 10 fps


def test_a_fast_tracker_takes_every_frame():
    assert next_frame(FRAMES, elapsed_ms=0, after=-1) == 0
    assert next_frame(FRAMES, elapsed_ms=105, after=0) == 1


def test_a_slow_tracker_skips_to_the_newest_frame_like_the_live_loop():
    # 350 ms after the start the camera has produced frames 0-3: take 3, skip 1 and 2.
    assert next_frame(FRAMES, elapsed_ms=350, after=0) == 3


def test_ahead_of_the_camera_it_waits_for_the_next_frame():
    assert next_frame(FRAMES, elapsed_ms=150, after=1) == 2  # frame 2 is due at 200 ms


def test_end_of_clip():
    assert next_frame(FRAMES, elapsed_ms=10_000, after=9) is None


def test_shown_pose_converts_back_to_the_camera_pose():
    a = np.radians(30)
    pose = np.eye(4)
    pose[:3, :3] = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    pose[:3, 3] = (0.05, -0.02, 0.6)
    shown = cv_model_pose_to_threejs(pose)
    assert threejs_pose_to_cv(shown.position, shown.quaternion) == pytest.approx(pose, abs=1e-9)


def _record(i, state="TRACKING", x=0.0, detector=False, overlap=None):
    return ReplayRecord(i, i * 100.0, 50.0, state, detector, (x, 0.0, -0.5), (0.0, 0.0, 0.0, 1.0), overlap)


def test_summary_counts_losses_detections_and_jumps():
    records = [_record(0, detector=True, overlap=0.9)]
    records += [_record(i, x=i * 0.002) for i in range(1, 6)]  # smooth 2 mm steps
    records += [_record(6, x=0.06, overlap=0.8)]  # a 48 mm jump
    records += [_record(7, state="LOST"), _record(8, state="TRACKING", x=0.06)]
    s = summarize(records, clip_frames=12, clip_seconds=1.2)
    assert s["frames_processed"] == 9 and s["updates_per_second"] == pytest.approx(7.5)
    assert s["detector_runs"] == 1 and s["lost_events"] == 1
    assert s["jumps"] == 1
    assert s["step_mm_p50"] == pytest.approx(2.0)
    assert s["overlap_mean"] == pytest.approx(0.85) and s["overlap_samples"] == 2
