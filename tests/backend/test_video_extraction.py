"""
Verifies frame extraction against a real (tiny, synthetic) video file — MJPG
in an .avi container, which round-trips frame counts reliably in a headless
OpenCV build, unlike some mp4 container/codec combinations on small clips.
"""

import cv2
import numpy as np
import pytest

from app.services.video.extraction import VideoExtractionError, extract_frames


def _write_test_video(path, num_frames: int = 10, size: tuple[int, int] = (16, 16)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    for i in range(num_frames):
        frame = np.full((size[1], size[0], 3), (i * 20) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_extract_frames_samples_every_nth_frame(tmp_path):
    video_path = tmp_path / "test.avi"
    _write_test_video(video_path, num_frames=10)

    results = list(extract_frames(video_path, sample_every_n_frames=3))
    indices = [i for i, _ in results]

    assert indices == [0, 3, 6, 9]
    assert all(frame.shape[:2] == (16, 16) for _, frame in results)


def test_extract_frames_with_sample_rate_1_returns_every_frame(tmp_path):
    video_path = tmp_path / "test.avi"
    _write_test_video(video_path, num_frames=5)

    results = list(extract_frames(video_path, sample_every_n_frames=1))
    assert len(results) == 5
    assert [i for i, _ in results] == [0, 1, 2, 3, 4]


def test_extract_frames_raises_on_missing_file(tmp_path):
    with pytest.raises(VideoExtractionError):
        list(extract_frames(tmp_path / "does-not-exist.avi"))


def test_extract_frames_rejects_non_positive_sample_rate(tmp_path):
    video_path = tmp_path / "test.avi"
    _write_test_video(video_path, num_frames=3)
    with pytest.raises(ValueError):
        list(extract_frames(video_path, sample_every_n_frames=0))
