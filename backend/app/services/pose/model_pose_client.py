"""
HTTP client for pose_service/ (MegaPose model-based 6DoF, running in WSL2).

Stateless apart from remembering which Model3D meshes the service already
has, and at which scale (a corrected scale means the mesh must be
re-registered, or MegaPose would match against the wrong-sized object and
get depth wrong). Tracking state lives in app/services/tracking/ar_session.py.
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
class PoseServiceResult:
    t_camera_object: np.ndarray | None  # 4x4, OpenCV camera convention, meters
    score: float  # MegaPose appearance score, 0 when nothing usable came back
    mode: str  # "coarse+refine" or "refine"
    elapsed_ms: float
    projected_bbox: tuple[float, float, float, float] | None = None  # the posed mesh's image-space box


class ModelPoseClient:
    def __init__(self, base_url: str, timeout_s: float, transport: httpx.BaseTransport | None = None):
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)
        self._registered_scale: dict[str, float] = {}

    def _post(self, path: str, payload: dict) -> dict:
        try:
            response = self._client.post(path, json=payload)
        except httpx.HTTPError as exc:
            raise PoseServiceUnavailable(
                f"Model-based pose service not reachable at {self._client.base_url} ({exc}). "
                "Start it in WSL — see pose_service/README.md."
            ) from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise PoseServiceUnavailable(f"Pose service error {response.status_code}: {detail}")
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

    def forget(self, label: str) -> None:
        """Model deleted: drop local state and ask the service to drop its mesh.
        Best effort — if the service isn't running there's nothing to clean."""
        self._registered_scale.pop(label, None)
        try:
            self._client.delete(f"/objects/{label}")
        except httpx.HTTPError:
            pass

    def _estimate(self, payload: dict) -> PoseServiceResult:
        data = self._post("/estimate", payload)
        pose = np.asarray(data["pose"], dtype=float) if data.get("found") else None
        # Object must be in front of the camera and within reach; anything
        # else is a failed solve, whatever its score.
        if pose is not None and not 0.02 < pose[2, 3] < 5.0:
            pose = None
        score = float(data["score"]) if pose is not None and data.get("score") is not None else 0.0
        box = data.get("projected_bbox") if pose is not None else None
        return PoseServiceResult(pose, score, data["mode"], float(data["elapsed_ms"]), tuple(box) if box else None)

    def full_search(self, label: str, image_jpeg: bytes, camera_matrix: np.ndarray, bbox: list[float]) -> PoseServiceResult:
        """Initial pose: coarse rotation search + refinement inside the detector's box."""
        return self._estimate(
            {
                "label": label,
                "image_base64": base64.b64encode(image_jpeg).decode("ascii"),
                "K": np.asarray(camera_matrix, dtype=float).tolist(),
                "bbox": bbox,
            }
        )

    def refine(
        self, label: str, image_jpeg: bytes, camera_matrix: np.ndarray, prev_pose: np.ndarray, iterations: int
    ) -> PoseServiceResult:
        """Tracking step: refine from the previous pose; no detector, no search."""
        return self._estimate(
            {
                "label": label,
                "image_base64": base64.b64encode(image_jpeg).decode("ascii"),
                "K": np.asarray(camera_matrix, dtype=float).tolist(),
                "prev_pose": np.asarray(prev_pose, dtype=float).tolist(),
                "n_refiner_iterations": iterations,
            }
        )
