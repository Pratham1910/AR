"""
HTTP client for pose_service/ (MegaPose model-based 6DoF, running in WSL2).

Keeps two pieces of per-process state:
  - which Model3D meshes the service already has, and at which scale (a
    corrected scale means the mesh must be re-registered, or MegaPose would
    match against the wrong-sized object and get depth wrong);
  - the last good pose per tracking session, so each new frame only runs
    MegaPose's refiner from the previous pose instead of a full search.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import numpy as np


class PoseServiceUnavailable(RuntimeError):
    pass


@dataclass
class ModelPoseResult:
    found: bool
    t_camera_object: np.ndarray | None
    score: float | None
    mode: str
    elapsed_ms: float


class ModelPoseClient:
    def __init__(self, base_url: str, timeout_s: float, transport: httpx.BaseTransport | None = None):
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)
        self._registered_scale: dict[str, float] = {}
        self._tracks: dict[str, np.ndarray] = {}  # last GOOD pose per label:session
        self._misses: dict[str, int] = {}  # consecutive low-score frames while tracking

    def _post(self, path: str, payload: dict) -> dict:
        try:
            response = self._client.post(path, json=payload)
        except httpx.HTTPError as exc:
            raise PoseServiceUnavailable(
                f"Model-based pose service not reachable at {self._client.base_url} ({exc}). "
                "Start it in WSL — see pose_service/README.md."
            ) from exc
        if response.status_code >= 400:
            raise PoseServiceUnavailable(f"Pose service error {response.status_code}: {response.text}")
        return response.json()

    def ensure_registered(self, label: str, read_glb: Callable[[], bytes], scale: float) -> None:
        """`read_glb` is only called when the mesh actually has to be sent —
        this runs on every tracking frame, so the GLB isn't re-read each time."""
        if self._registered_scale.get(label) == scale:
            return
        self._post(
            "/objects",
            {"label": label, "glb_base64": base64.b64encode(read_glb()).decode("ascii"), "scale": scale},
        )
        self._registered_scale[label] = scale
        self.reset_tracks_for(label)

    def forget(self, label: str) -> None:
        """Model deleted: drop local state and ask the service to drop its mesh.
        Best effort — if the service isn't running there's nothing to clean."""
        self._registered_scale.pop(label, None)
        self.reset_tracks_for(label)
        try:
            self._client.delete(f"/objects/{label}")
        except httpx.HTTPError:
            pass

    def reset_tracks_for(self, label: str) -> None:
        for key in [k for k in self._tracks if k.startswith(f"{label}:")]:
            del self._tracks[key]
            self._misses.pop(key, None)

    def reset(self, label: str, session_id: str) -> None:
        self._tracks.pop(f"{label}:{session_id}", None)
        self._misses.pop(f"{label}:{session_id}", None)

    def has_track(self, label: str, session_id: str) -> bool:
        return f"{label}:{session_id}" in self._tracks

    def estimate(
        self,
        label: str,
        session_id: str,
        image_jpeg: bytes,
        camera_matrix: np.ndarray,
        bbox: list[float] | None,
        min_score: float,
        track_iterations: int,
        max_misses: int = 0,
    ) -> ModelPoseResult:
        """Refines from this session's previous pose if there is one, otherwise
        does a full search inside `bbox`. While tracking, up to `max_misses`
        consecutive low-score frames (motion blur, a hand passing) keep the
        track — the next frame refines again from the last GOOD pose — instead
        of immediately paying for a ~1s full search; one more drops it."""
        key = f"{label}:{session_id}"
        prev = self._tracks.get(key)
        if prev is None and bbox is None:
            return ModelPoseResult(found=False, t_camera_object=None, score=None, mode="no_detection", elapsed_ms=0.0)

        data = self._post(
            "/estimate",
            {
                "label": label,
                "image_base64": base64.b64encode(image_jpeg).decode("ascii"),
                "K": np.asarray(camera_matrix, dtype=float).tolist(),
                "bbox": None if prev is not None else bbox,
                "prev_pose": prev.tolist() if prev is not None else None,
                "n_refiner_iterations": track_iterations if prev is not None else None,
            },
        )
        pose = np.asarray(data["pose"], dtype=float) if data.get("found") else None
        score = data.get("score")
        plausible = pose is not None and 0.02 < pose[2, 3] < 5.0  # object in front of the camera, within reach
        good = plausible and (score is None or score >= min_score)

        if good:
            self._tracks[key] = pose
            self._misses.pop(key, None)
        elif prev is not None and self._misses.get(key, 0) < max_misses:
            self._misses[key] = self._misses.get(key, 0) + 1  # keep refining from the last good pose
        else:
            self._tracks.pop(key, None)
            self._misses.pop(key, None)
        return ModelPoseResult(
            found=good,
            t_camera_object=pose if good else None,
            score=score,
            mode=data["mode"],
            elapsed_ms=float(data["elapsed_ms"]),
        )
