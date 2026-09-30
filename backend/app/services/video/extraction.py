"""
Video frame extraction (Project.md #28's pipeline: "Video -> Frame
extraction -> Detection -> ..."). Streams a video file frame-by-frame rather
than loading it all into memory, yielding every Nth frame — the temporal
resolution needed to catch a state transition is far lower than a video's
native frame rate.
"""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class VideoExtractionError(ValueError):
    pass


def extract_frames(video_path: str | Path, sample_every_n_frames: int = 15) -> Iterator[tuple[int, np.ndarray]]:
    """Yields (frame_index, frame) for every `sample_every_n_frames`-th frame in the video."""
    if sample_every_n_frames < 1:
        raise ValueError("sample_every_n_frames must be >= 1")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoExtractionError(f"Could not open video file: {video_path}")

    try:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % sample_every_n_frames == 0:
                yield frame_index, frame
            frame_index += 1
    finally:
        cap.release()
