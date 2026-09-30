"""
Model-based (CAD) 6DoF pose service — MegaPose via HappyPose, RGB-only.

Runs inside WSL2 (Linux + CUDA), separately from the Windows FastAPI backend,
because MegaPose/HappyPose is Linux-first and needs a CUDA PyTorch build that
the main backend doesn't carry. The backend talks to it over HTTP
(localhost is forwarded from Windows into WSL2).

Unlike the backend's marker/feature/markerless modes, this matches the
object's actual 3D mesh against the image, so the returned pose is the pose
of the MODEL ITSELF (its own mesh frame), not of a proxy patch — no anchor
offset calibration is needed.

Frame convention: meshes are registered in the GLB's own coordinate frame,
scaled to meters, and RECENTERED on their bounding-box center — exactly what
frontend RegistrationOverlay.tsx does to the loaded GLB (scale, then subtract
the bbox center inside a wrapper group). Returned poses are T_camera_object
in OpenCV camera convention (X right, Y down, Z forward), meters.

Run (inside WSL, from the HappyPose checkout's venv):
    HAPPYPOSE_DATA_DIR=~/tvasta-pose/data \\
    .venv/bin/python -m uvicorn server:app --app-dir /path/to/TVASTA/pose_service --port 8765
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import trimesh
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.inference.types import ObservationTensor
from happypose.toolbox.utils.load_model import NAMED_MODELS, load_named_model
from happypose.toolbox.utils.tensor_collection import PandasTensorCollection

MODEL_NAME = os.getenv("MEGAPOSE_MODEL", "megapose-1.0-RGB")
MESH_DIR = Path(os.getenv("TVASTA_POSE_MESH_DIR", Path.home() / "tvasta-pose" / "meshes"))
# Coarse search tries a grid of rotations; the full grid is the most robust
# but slowest. Subsampling (as HappyPose's own example does, [::8]) trades a
# little first-lock robustness for a much faster first lock.
COARSE_GRID_STRIDE = int(os.getenv("MEGAPOSE_COARSE_GRID_STRIDE", "4"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="TVASTA model-based pose service")

_lock = threading.Lock()  # one GPU pipeline; serialize requests
_estimator = None
_model_info = None
_labels: set[str] = set()


def _mesh_path(label: str) -> Path:
    return MESH_DIR / f"{label}.ply"


def _rebuild_estimator() -> None:
    """(Re)load MegaPose bound to every registered mesh. MegaPose binds its
    renderer to a fixed object set at load time, so registering a new object
    means rebuilding — acceptable since it only happens once per new model."""
    global _estimator, _model_info, _labels
    objects = [
        RigidObject(label=p.stem, mesh_path=p, mesh_units="m")
        for p in sorted(MESH_DIR.glob("*.ply"))
    ]
    _labels = {o.label for o in objects}
    if not objects:
        _estimator = None
        return
    _model_info = NAMED_MODELS[MODEL_NAME]
    estimator = load_named_model(MODEL_NAME, RigidObjectDataset(objects)).to(device)
    estimator._SO3_grid = estimator._SO3_grid[::COARSE_GRID_STRIDE]
    _estimator = estimator


@app.on_event("startup")
def _startup() -> None:
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        _rebuild_estimator()


class RegisterObjectRequest(BaseModel):
    label: str  # the backend's Model3D id
    glb_base64: str
    scale: float  # GLB units -> meters (Model3D.scale)


class RegisterObjectResponse(BaseModel):
    label: str
    extents_m: list[float]
    vertex_count: int


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "model": MODEL_NAME,
        "objects": sorted(_labels),
    }


@app.post("/objects", response_model=RegisterObjectResponse)
def register_object(req: RegisterObjectRequest) -> RegisterObjectResponse:
    data = base64.b64decode(req.glb_base64)
    loaded = trimesh.load(io.BytesIO(data), file_type="glb", force="scene")
    # dump(concatenate=True) bakes node transforms into world coordinates,
    # matching Box3.setFromObject() on the Three.js side.
    mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise HTTPException(status_code=400, detail="GLB contains no triangle mesh")

    mesh = mesh.copy()
    mesh.apply_scale(req.scale)
    lo, hi = mesh.bounds
    mesh.apply_translation(-(lo + hi) / 2.0)

    MESH_DIR.mkdir(parents=True, exist_ok=True)
    mesh.export(_mesh_path(req.label))
    with _lock:
        _rebuild_estimator()
    return RegisterObjectResponse(
        label=req.label,
        extents_m=[float(v) for v in mesh.extents],
        vertex_count=len(mesh.vertices),
    )


class EstimateRequest(BaseModel):
    label: str
    image_base64: str  # JPEG/PNG, RGB
    K: list[list[float]]  # 3x3 intrinsics for THIS image's resolution
    bbox: list[float] | None = None  # [x1, y1, x2, y2]; required unless prev_pose given
    prev_pose: list[list[float]] | None = None  # 4x4 T_camera_object from the previous frame
    n_refiner_iterations: int | None = None


class EstimateResponse(BaseModel):
    found: bool
    pose: list[list[float]] | None = None  # 4x4 T_camera_object, OpenCV convention, meters
    score: float | None = None  # MegaPose's pose score (higher = better match)
    mode: str  # "coarse+refine" (full search) or "refine" (tracking from prev_pose)
    elapsed_ms: float


@app.post("/estimate", response_model=EstimateResponse)
def estimate(req: EstimateRequest) -> EstimateResponse:
    if req.label not in _labels or _estimator is None:
        raise HTTPException(status_code=404, detail=f"Object {req.label!r} not registered")
    if req.bbox is None and req.prev_pose is None:
        raise HTTPException(status_code=400, detail="Provide bbox (first lock) or prev_pose (tracking)")

    rgb = np.array(Image.open(io.BytesIO(base64.b64decode(req.image_base64))).convert("RGB"), dtype=np.uint8)
    K = np.asarray(req.K, dtype=np.float32)
    infos = pd.DataFrame({"label": [req.label], "batch_im_id": [0], "instance_id": [0]})

    started = time.perf_counter()
    with _lock, torch.no_grad():
        observation = ObservationTensor.from_numpy(rgb, None, K).to(device)
        params = dict(_model_info["inference_parameters"])
        if req.n_refiner_iterations is not None:
            params["n_refiner_iterations"] = req.n_refiner_iterations

        if req.prev_pose is not None:
            mode = "refine"
            coarse = PandasTensorCollection(
                infos=infos,
                poses=torch.as_tensor(np.asarray(req.prev_pose, dtype=np.float32)).unsqueeze(0),
            ).to(device)
            params.pop("n_pose_hypotheses", None)
            output, _ = _estimator.run_inference_pipeline(observation, coarse_estimates=coarse, **params)
        else:
            mode = "coarse+refine"
            detections = PandasTensorCollection(
                infos=infos,
                bboxes=torch.as_tensor(np.asarray([req.bbox], dtype=np.float32)),
            ).to(device)
            output, _ = _estimator.run_inference_pipeline(observation, detections=detections, **params)

    output = output.cpu()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if len(output) == 0:
        return EstimateResponse(found=False, mode=mode, elapsed_ms=elapsed_ms)

    score = output.infos["pose_score"].iloc[0] if "pose_score" in output.infos else None
    return EstimateResponse(
        found=True,
        pose=output.poses[0].numpy().astype(float).tolist(),
        score=float(score) if score is not None else None,
        mode=mode,
        elapsed_ms=elapsed_ms,
    )
