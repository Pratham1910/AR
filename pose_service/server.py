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
import hashlib
import io
import json
import logging
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
# but slowest. Subsampling as HappyPose's own example does ([::8]): measured
# on the known-answer cup render, full search 1572ms -> 893ms with first-lock
# error 3.5mm/4.5deg -> 5.6mm/5.6deg, and the next tracking step brings both
# back to the same 3.7mm/3.2deg, so re-acquiring a lost object is ~43% faster.
COARSE_GRID_STRIDE = int(os.getenv("MEGAPOSE_COARSE_GRID_STRIDE", "8"))
# MegaPose pads every loaded object's data to the LARGEST mesh, so one heavy
# model slows every request, even for other objects. Measured on an RTX 4090:
# cup tracking ~185ms with small meshes loaded, ~1.7-2.7s once a 105k-face
# mesh was also loaded. It matches shape/silhouette, so detail beyond this
# buys nothing; meshes are decimated to at most this many faces.
MAX_MESH_FACES = int(os.getenv("TVASTA_POSE_MAX_FACES", "10000"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="TVASTA model-based pose service")

_lock = threading.Lock()  # one GPU pipeline; serialize requests
_estimator = None
_model_info = None
_labels: set[str] = set()
_vertices: dict[str, np.ndarray] = {}  # per label, for projecting the posed model into the image
log = logging.getLogger("uvicorn.error")  # shows up in the service's console/log


def _mesh_path(label: str) -> Path:
    return MESH_DIR / f"{label}.ply"


def _meta_path(label: str) -> Path:
    return MESH_DIR / f"{label}.json"


def _registration_key(glb: bytes, scale: float) -> dict:
    return {"glb_sha256": hashlib.sha256(glb).hexdigest(), "scale": scale, "max_faces": MAX_MESH_FACES}


def _simplified(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if len(mesh.faces) <= MAX_MESH_FACES:
        return mesh
    return mesh.simplify_quadric_decimation(face_count=MAX_MESH_FACES)


def _simplify_existing_meshes() -> None:
    """One-off for meshes registered before the face cap existed. They're
    already recentered, and decimation keeps the bounds within ~0.1%."""
    for path in MESH_DIR.glob("*.ply"):
        mesh = trimesh.load(path, force="mesh")
        if len(mesh.faces) > MAX_MESH_FACES:
            _simplified(mesh).export(path)


def _rebuild_estimator() -> None:
    """(Re)load MegaPose bound to every registered mesh. MegaPose binds its
    renderer to a fixed object set at load time, so registering a new object
    means rebuilding — acceptable since it only happens once per new model."""
    global _estimator, _model_info, _labels, _vertices
    objects = [
        RigidObject(label=p.stem, mesh_path=p, mesh_units="m")
        for p in sorted(MESH_DIR.glob("*.ply"))
    ]
    _labels = {o.label for o in objects}
    _vertices = {p.stem: np.asarray(trimesh.load(p, force="mesh").vertices) for p in MESH_DIR.glob("*.ply")}
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
    _simplify_existing_meshes()
    with _lock:
        _rebuild_estimator()


class RegisterObjectRequest(BaseModel):
    label: str  # the backend's Model3D id (plus a part suffix when tracking one part)
    glb_base64: str
    scale: float  # GLB units -> meters (Model3D.scale)
    # Only these GLB nodes (e.g. ["Cylinder"] = a bottle's body), or the whole model if omitted.
    node_names: list[str] | None = None


class RegisterObjectResponse(BaseModel):
    label: str
    extents_m: list[float]
    vertex_count: int
    # Center of the registered mesh relative to the whole assembly's center
    # (meters, assembly/glTF frame). MegaPose returns poses of the mesh as
    # registered (recentered on itself); the assembly's pose is that pose
    # composed with this offset. [0, 0, 0] for whole models.
    offset_m: list[float] = [0.0, 0.0, 0.0]


def _gpu_problem() -> str | None:
    """None if the GPU is usable, else what's wrong and how to fix it."""
    if device.type != "cuda":
        return (
            "No CUDA GPU visible inside WSL (after an NVIDIA driver update WSL reports "
            "'GPU access blocked by the operating system' until Windows is restarted or "
            "`wsl --shutdown` is run), so MegaPose would run on the CPU, far too slow to track. "
            "Fix the GPU, then restart this service."
        )
    try:
        torch.ones(1, device=device).sum().item()
    except RuntimeError as exc:
        return f"GPU error ({exc}); restart the pose service (pose_service/start.sh)"
    return None


@app.get("/health")
def health() -> dict:
    # Actually touch the GPU: after an NVIDIA driver update a running service
    # keeps reporting cuda=True but every real request fails with "CUDA
    # error: unknown error" until it's restarted.
    gpu_error = _gpu_problem()
    gpu_ok = gpu_error is None
    return {
        "ok": gpu_ok,
        "gpu_error": gpu_error,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "model": MODEL_NAME,
        "objects": sorted(_labels),
    }


def _scene_mesh(loaded, node_names: list[str] | None) -> trimesh.Trimesh | None:
    """The GLB's geometry in world coordinates (node transforms baked in, as
    Three.js's Box3.setFromObject sees it) — all of it, or only the named nodes."""
    if not isinstance(loaded, trimesh.Scene):
        return loaded if node_names is None else None
    pieces = []
    for node in loaded.graph.nodes_geometry:
        if node_names is not None and node not in node_names:
            continue
        transform, geometry = loaded.graph[node]
        piece = loaded.geometry[geometry].copy()
        piece.apply_transform(transform)
        pieces.append(piece)
    return trimesh.util.concatenate(pieces) if pieces else None


@app.post("/objects", response_model=RegisterObjectResponse)
def register_object(req: RegisterObjectRequest) -> RegisterObjectResponse:
    data = base64.b64decode(req.glb_base64)
    key = _registration_key(data, req.scale) | {"node_names": req.node_names}
    mesh_path, meta_path = _mesh_path(req.label), _meta_path(req.label)
    # The backend re-sends every model after each of ITS restarts. Reloading
    # MegaPose for an identical mesh took seconds and stalled all tracking,
    # so an unchanged (same GLB bytes, scale, parts) registration is a no-op.
    if req.label in _labels and mesh_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if {k: meta.get(k) for k in key} == key:
            existing = trimesh.load(mesh_path, force="mesh")
            return RegisterObjectResponse(
                label=req.label,
                extents_m=[float(v) for v in existing.extents],
                vertex_count=len(existing.vertices),
                offset_m=meta.get("offset_m", [0.0, 0.0, 0.0]),
            )

    loaded = trimesh.load(io.BytesIO(data), file_type="glb", force="scene")
    assembly = _scene_mesh(loaded, None)
    if not isinstance(assembly, trimesh.Trimesh) or len(assembly.vertices) == 0:
        raise HTTPException(status_code=400, detail="GLB contains no triangle mesh")
    lo, hi = assembly.bounds * req.scale
    assembly_center = (lo + hi) / 2.0  # what the frontend recenters the rendered model on

    mesh = _scene_mesh(loaded, req.node_names)
    if mesh is None or len(mesh.vertices) == 0:
        raise HTTPException(status_code=400, detail=f"No geometry for parts {req.node_names}")
    mesh = mesh.copy()
    mesh.apply_scale(req.scale)
    # Recenter on the FULL-detail bounds of what MegaPose will match, then
    # simplify. For a part, its center sits at `offset` from the assembly's
    # center; the caller converts the part's pose back to the assembly frame.
    lo, hi = mesh.bounds
    center = (lo + hi) / 2.0
    mesh.apply_translation(-center)
    mesh = _simplified(mesh)
    offset = [float(v) for v in center - assembly_center]

    MESH_DIR.mkdir(parents=True, exist_ok=True)
    mesh.export(mesh_path)
    meta_path.write_text(json.dumps(key | {"offset_m": offset}))
    log.info("registering %s (%d faces): reloading MegaPose", req.label[:8], len(mesh.faces))
    with _lock:
        _rebuild_estimator()
    return RegisterObjectResponse(
        label=req.label,
        extents_m=[float(v) for v in mesh.extents],
        vertex_count=len(mesh.vertices),
        offset_m=offset,
    )


@app.delete("/objects/{label}")
def delete_object(label: str) -> dict:
    path = _mesh_path(label)
    existed = path.exists()
    _meta_path(label).unlink(missing_ok=True)
    if existed:
        path.unlink()
        with _lock:
            _rebuild_estimator()
    return {"label": label, "deleted": existed}


class EstimateRequest(BaseModel):
    label: str
    image_base64: str  # JPEG/PNG, RGB
    K: list[list[float]]  # 3x3 intrinsics for THIS image's resolution
    bbox: list[float] | None = None  # [x1, y1, x2, y2]; required unless prev_pose given
    prev_pose: list[list[float]] | None = None  # 4x4 T_camera_object from the previous frame
    n_refiner_iterations: int | None = None
    # Full search only: refine this many of the coarse model's best rotation
    # guesses and keep the highest-scoring (HappyPose's "multi-hypothesis").
    n_pose_hypotheses: int | None = None


class EstimateResponse(BaseModel):
    found: bool
    pose: list[list[float]] | None = None  # 4x4 T_camera_object, OpenCV convention, meters
    score: float | None = None  # MegaPose's pose score (higher = better match)
    mode: str  # "coarse+refine" (full search) or "refine" (tracking from prev_pose)
    elapsed_ms: float
    # [x1, y1, x2, y2] of the mesh drawn at `pose` in this image. MegaPose's
    # appearance score proved unreliable for untextured meshes (a correct
    # upright bottle scored 0.11, a wrong sideways one 0.41), so callers
    # judge a pose by how well this box overlaps the real object instead.
    projected_bbox: list[float] | None = None


@app.post("/estimate", response_model=EstimateResponse)
def estimate(req: EstimateRequest) -> EstimateResponse:
    gpu_problem = _gpu_problem()
    if gpu_problem:
        raise HTTPException(status_code=503, detail=gpu_problem)
    if req.label not in _labels or _estimator is None:
        raise HTTPException(status_code=404, detail=f"Object {req.label!r} not registered")
    if req.bbox is None and req.prev_pose is None:
        raise HTTPException(status_code=400, detail="Provide bbox (first lock) or prev_pose (tracking)")

    rgb = np.array(Image.open(io.BytesIO(base64.b64decode(req.image_base64))).convert("RGB"), dtype=np.uint8)
    K = np.asarray(req.K, dtype=np.float32)
    infos = pd.DataFrame({"label": [req.label], "batch_im_id": [0], "instance_id": [0]})

    started = time.perf_counter()
    with _lock, torch.no_grad():
        waited_ms = (time.perf_counter() - started) * 1000.0  # queued behind another request/reload
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
            output, extra = _estimator.run_inference_pipeline(observation, coarse_estimates=coarse, **params)
        else:
            mode = "coarse+refine"
            if req.n_pose_hypotheses is not None:
                params["n_pose_hypotheses"] = req.n_pose_hypotheses
            detections = PandasTensorCollection(
                infos=infos,
                bboxes=torch.as_tensor(np.asarray([req.bbox], dtype=np.float32)),
            ).to(device)
            output, extra = _estimator.run_inference_pipeline(observation, detections=detections, **params)

    output = output.cpu()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    score = output.infos["pose_score"].iloc[0] if len(output) and "pose_score" in output.infos else None
    log.info(
        "estimate %s mode=%s score=%s total=%.0fms queued=%.0fms [%s]",
        req.label[:8], mode, "n/a" if score is None else f"{score:.2f}", elapsed_ms, waited_ms, extra.get("timing_str", ""),
    )
    if len(output) == 0:
        return EstimateResponse(found=False, mode=mode, elapsed_ms=elapsed_ms)

    pose = output.poses[0].numpy().astype(float)
    projected_bbox = None
    vertices = _vertices.get(req.label)
    if vertices is not None:
        cam = vertices @ pose[:3, :3].T + pose[:3, 3]
        in_front = cam[:, 2] > 1e-3
        if in_front.any():
            px = cam[in_front] @ K.astype(float).T
            px = px[:, :2] / px[:, 2:3]
            projected_bbox = [float(px[:, 0].min()), float(px[:, 1].min()), float(px[:, 0].max()), float(px[:, 1].max())]

    return EstimateResponse(
        found=True,
        pose=pose.tolist(),
        score=float(score) if score is not None else None,
        mode=mode,
        elapsed_ms=elapsed_ms,
        projected_bbox=projected_bbox,
    )
