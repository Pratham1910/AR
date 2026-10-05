"""
Renders a registered mesh from evenly spread viewpoints — the "templates"
the CAD detector (cad_detector.py) compares camera-image regions against.

Run as its own process (the service does this once per object and caches the
result): Panda3D's ShowBase is a process-wide singleton that must not be
driven from the web server's worker threads.

    python render_templates.py <mesh.ply> <out.npz> [--views 42] [--size 512]

Output .npz: rgb (N, S, S, 3) uint8 on black, mask (N, S, S) bool,
viewpoints (N, 3) unit vectors from the object center to the camera.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import trimesh

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # HappyPose's renderer asserts it is set

from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset  # noqa: E402
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer, make_scene_lights  # noqa: E402
from happypose.toolbox.renderer.types import Panda3dCameraData, Panda3dObjectData  # noqa: E402

MESH_COLOR = (0.7, 0.7, 0.7, 1.0)
UP = np.array([0.0, 1.0, 0.0])  # glTF is Y-up; templates are rendered upright like a hand-held camera sees


def viewpoints(count: int) -> np.ndarray:
    """Unit directions spread over the sphere: icosphere vertices (42 / 162) or a Fibonacci sphere."""
    for subdivisions in (1, 2):
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions)
        if len(sphere.vertices) == count:
            return np.asarray(sphere.vertices, dtype=float)
    i = np.arange(count) + 0.5
    phi = np.arccos(1 - 2 * i / count)
    theta = np.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], axis=1)


def look_at(eye: np.ndarray) -> np.ndarray:
    """4x4 camera-to-world (OpenCV axes: X right, Y down, Z forward) looking at the origin."""
    z = -eye / np.linalg.norm(eye)
    up = UP if abs(z @ UP) < 0.99 else np.array([0.0, 0.0, -1.0])  # straight above/below: any horizon
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([x, y, z], axis=1)
    pose[:3, 3] = eye
    return pose


def render(mesh_path: str, count: int, size: int) -> dict[str, np.ndarray]:
    mesh = trimesh.load(mesh_path, force="mesh")
    radius = float(np.linalg.norm(mesh.vertices, axis=1).max())  # meshes are registered recentered
    focal = float(size)
    # Far enough that the bounding sphere spans ~85% of the image from any side.
    distance = 2 * focal * radius / (0.85 * size)
    K = np.array([[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1]])

    # The registered PLY carries no normals, so lights only add ambient and
    # every view renders as a flat silhouette; shading needs smooth normals.
    with tempfile.TemporaryDirectory() as tmp:
        shaded = Path(tmp) / "template.ply"
        mesh.export(shaded, vertex_normal=True)
        return _render_views(shaded, K, size, distance, radius, viewpoints(count))


def _render_views(
    mesh_path: Path, K: np.ndarray, size: int, distance: float, radius: float, dirs: np.ndarray
) -> dict[str, np.ndarray]:
    dataset = RigidObjectDataset([RigidObject(label="template", mesh_path=mesh_path, mesh_units="m")])
    renderer = Panda3dSceneRenderer(dataset)
    lights = make_scene_lights(ambient_light_color=(0.35, 0.35, 0.35, 1.0), point_lights_color=(0.55, 0.55, 0.55, 1.0))
    rgbs, masks = [], []
    for direction in dirs:
        camera = Panda3dCameraData(
            K=K,
            resolution=(size, size),
            TWC=look_at(direction * distance),
            z_near=max(1e-3, distance - 2 * radius),
            z_far=distance + 2 * radius,
        )
        out = renderer.render_scene(
            # An explicit material: meshes loaded without one (e.g. GLBs with no
            # colours, after PLY export) otherwise render pure black under lights.
            [Panda3dObjectData(label="template", color=MESH_COLOR)],
            [camera],
            lights,
            render_depth=True,
            render_binary_mask=True,
        )[0]
        rgbs.append(out.rgb)
        masks.append(out.binary_mask[..., 0])
    return {"rgb": np.stack(rgbs).astype(np.uint8), "mask": np.stack(masks), "viewpoints": dirs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh")
    parser.add_argument("out")
    parser.add_argument("--views", type=int, default=42)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    np.savez_compressed(args.out, **render(args.mesh, args.views, args.size))


if __name__ == "__main__":
    main()
