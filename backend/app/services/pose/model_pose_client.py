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
        self._tracks: dict[str, np.ndarray] = {}

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

    def ensure_registered(self, label: str, glb_bytes: bytes, scale: float) -> None:
        if self._registered_scale.get(label) == scale:
            return
        self._post(
            "/objects",
            {"label": label, "glb_base64": base64.b64encode(glb_bytes).decode("ascii"), "scale": scale},
        )
        self._registered_scale[label] = scale
        self.reset_tracks_for(label)

    def reset_tracks_for(self, label: str) -> None:
        for key in [k for k in self._tracks if k.startswith(f"{label}:")]:
            del self._tracks[key]

    def reset(self, label: str, session_id: str) -> None:
        self._tracks.pop(f"{label}:{session_id}", None)

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
    ) -> ModelPoseResult:
        """Refines from this session's previous pose if there is one, otherwise
        does a full search inside `bbox`. A result scoring below `min_score`
        drops the track so the next frame starts over from a fresh detection."""
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
        else:
            self._tracks.pop(key, None)
        return ModelPoseResult(
            found=good,
            t_camera_object=pose if good else None,
            score=score,
            mode=data["mode"],
            elapsed_ms=float(data["elapsed_ms"]),
        )
