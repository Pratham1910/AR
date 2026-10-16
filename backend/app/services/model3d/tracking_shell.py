"""
"Optimize for tracking": the part of a CAD model a camera can ever see.

CAD assemblies carry everything inside them — the HDD inside the enclosure,
boards, fasteners behind panels. None of it can appear in a camera image,
but it costs tracking: MegaPose renders it, the 3D edge model spends its
sample budget on its edges, symmetry and outline tests include it, and the
X-Ray overlay is cluttered with it. The tracking shell keeps only the
triangles that are the nearest surface from at least one direction outside
the model (plus tiny ones whose centre is unobstructed — screw heads, LED
holes smaller than a pixel), in the original's coordinate frame, so poses
and overlays are unchanged. The original GLB stays what is displayed and
inspected (node indices and component mappings refer to it).

    outer_shell(meshes)        -> which faces of which node are visible
    build_tracking_glb(glb)    -> (shell GLB bytes, report)
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field

import numpy as np

from app.services.component_inspection.projection import _rasterize
from app.services.model3d.glb_inspect import GlbPartMesh, glb_nodes, glb_part_meshes


def _directions(count: int) -> np.ndarray:
    """Unit view directions spread evenly over the sphere (Fibonacci)."""
    i = np.arange(count) + 0.5
    phi = np.arccos(1 - 2 * i / count)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], axis=1)


# Where on a face visibility is tested: the centre, points a third of the way
# from the centre to each corner, and to each edge midpoint.
_SAMPLE_WEIGHTS = np.array([
    [1 / 3, 1 / 3, 1 / 3],
    [5 / 9, 2 / 9, 2 / 9], [2 / 9, 5 / 9, 2 / 9], [2 / 9, 2 / 9, 5 / 9],
    [4 / 9, 4 / 9, 1 / 9], [1 / 9, 4 / 9, 4 / 9], [4 / 9, 1 / 9, 4 / 9],
])


def _visible_from(vertices: np.ndarray, faces: np.ndarray, direction: np.ndarray, resolution: int,
                  tolerance: float) -> np.ndarray:
    """Faces with a point that is the nearest surface, looking along -direction
    (orthographic depth buffer; both sides of a face count). Tested at sample
    points, not by pixel ownership, so faces smaller than a pixel count too."""
    d = direction / np.linalg.norm(direction)
    helper = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(helper, d)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    x, y = vertices @ u, vertices @ v
    depth = -(vertices @ d)  # smaller = nearer to a viewer far out along +d
    lo = min(x.min(), y.min())
    span = max(x.max(), y.max()) - lo or 1.0
    scale = (resolution - 2) / span
    gx, gy = (x - lo) * scale, (y - lo) * scale
    tri_x, tri_y = gx[faces], gy[faces]
    boxes = np.stack([
        np.floor(tri_x.min(axis=1)), np.floor(tri_y.min(axis=1)),
        np.ceil(tri_x.max(axis=1)) + 1, np.ceil(tri_y.max(axis=1)) + 1,
    ], axis=1).clip(0, resolution).astype(np.int64)
    extent = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])

    zbuf = np.full(resolution * resolution, np.inf)
    bounds = [0, 2]
    while bounds[-1] < resolution + 2:
        bounds.append(bounds[-1] * 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        for low, high in zip(bounds[:-1], bounds[1:]):
            group = np.flatnonzero((extent > low) & (extent <= high))
            chunk = max(1, 400_000 // (high * high))
            for start in range(0, len(group), chunk):
                batch = group[start : start + chunk]
                face, px, py, w0, w1 = _rasterize(tri_x[batch], tri_y[batch], boxes[batch], int(extent[batch].max()),
                                                  resolution, resolution)
                if len(face) == 0:
                    continue
                tri = faces[batch][face]
                z = w0 * depth[tri[:, 0]] + w1 * depth[tri[:, 1]] + (1 - w0 - w1) * depth[tri[:, 2]]
                np.minimum.at(zbuf, py * resolution + px, z)
    # A face is visible where one of its sample points is at the front.
    corners = vertices[faces]  # (F, 3, 3)
    points = np.einsum("sk,fkd->fsd", _SAMPLE_WEIGHTS, corners)  # (F, S, 3)
    px = np.clip(((points @ u - lo) * scale).astype(int), 0, resolution - 1)
    py = np.clip(((points @ v - lo) * scale).astype(int), 0, resolution - 1)
    front = -(points @ d) <= zbuf[py * resolution + px] + tolerance
    return front.any(axis=1)


def outer_shell(meshes: dict[int, GlbPartMesh], views: int = 48, resolution: int = 192) -> dict[int, np.ndarray]:
    """Per node: which of its faces are visible from some direction outside."""
    nodes = sorted(meshes)
    if not nodes:
        return {}
    vertex_sets, face_sets, owner, base = [], [], [], 0
    for n in nodes:
        m = meshes[n]
        vertex_sets.append(np.asarray(m.vertices, float))
        face_sets.append(np.asarray(m.faces, np.int64) + base)
        owner.append(np.full(len(m.faces), n))
        base += len(m.vertices)
    vertices, faces, owner = np.concatenate(vertex_sets), np.concatenate(face_sets), np.concatenate(owner)
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))) or 1.0
    visible = np.zeros(len(faces), bool)
    for direction in _directions(views):
        visible |= _visible_from(vertices, faces, direction, resolution, tolerance=1e-3 * diagonal)
    return {n: visible[owner == n] for n in nodes}


@dataclass
class ShellReport:
    faces_before: int
    faces_after: int
    nodes: list[dict] = field(default_factory=list)  # per node: index, name, faces before / after
    frame_center: list[float] = field(default_factory=list)  # the ORIGINAL assembly centre, GLB units
    size: list[float] = field(default_factory=list)  # bounding box of the shell, GLB units
    source_sha256: str = ""


def _write_glb(parts: list[tuple[str, np.ndarray, np.ndarray]]) -> bytes:
    """A minimal GLB: one node + mesh per part, vertices already in the assembly frame."""
    nodes_json, meshes_json, accessors, views = [], [], [], []
    binary = bytearray()

    def add_view(data: bytes, target: int) -> int:
        while len(binary) % 4:
            binary.append(0)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data), "target": target})
        binary.extend(data)
        return len(views) - 1

    for name, v, f in parts:
        used = np.unique(f)
        remap = np.full(len(v), -1, np.int64)
        remap[used] = np.arange(len(used))
        v32 = v[used].astype(np.float32)
        f32 = remap[f].astype(np.uint32)
        pos = add_view(v32.tobytes(), 34962)
        idx = add_view(f32.reshape(-1).tobytes(), 34963)
        accessors.append({"bufferView": pos, "componentType": 5126, "count": len(v32), "type": "VEC3",
                          "min": v32.min(axis=0).tolist(), "max": v32.max(axis=0).tolist()})
        accessors.append({"bufferView": idx, "componentType": 5125, "count": int(f32.size), "type": "SCALAR"})
        meshes_json.append({"primitives": [{"attributes": {"POSITION": len(accessors) - 2}, "indices": len(accessors) - 1}]})
        nodes_json.append({"name": name, "mesh": len(meshes_json) - 1})
    while len(binary) % 4:
        binary.append(0)
    doc = {
        "asset": {"version": "2.0", "generator": "TVASTA tracking shell"},
        "scene": 0, "scenes": [{"nodes": list(range(len(nodes_json)))}],
        "nodes": nodes_json, "meshes": meshes_json, "accessors": accessors,
        "bufferViews": views, "buffers": [{"byteLength": len(binary)}],
    }
    json_bytes = json.dumps(doc).encode()
    json_bytes += b" " * (-len(json_bytes) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
            + struct.pack("<I4s", len(binary), b"BIN\x00") + bytes(binary))


def build_tracking_glb(glb: bytes, views: int = 48, resolution: int = 192) -> tuple[bytes, ShellReport]:
    """The model's visible shell as a GLB (same frame as the original), and what was kept."""
    meshes = glb_part_meshes(glb)
    if not meshes:
        raise ValueError("The model has no triangle meshes")
    names = {n.node_index: n.name for n in glb_nodes(glb)}
    keep = outer_shell(meshes, views, resolution)
    every = np.concatenate([m.vertices for m in meshes.values()])
    lo, hi = every.min(axis=0), every.max(axis=0)
    parts, report_nodes, kept_vertices = [], [], []
    for n, mask in keep.items():
        m = meshes[n]
        report_nodes.append({"node_index": n, "name": names.get(n, f"node {n}"), "faces_before": int(len(m.faces)),
                             "faces_after": int(mask.sum())})
        if mask.any():
            parts.append((names.get(n, f"node {n}"), np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)[mask]))
            kept_vertices.append(np.asarray(m.vertices, float)[np.unique(np.asarray(m.faces)[mask])])
    shell_points = np.concatenate(kept_vertices)
    report = ShellReport(
        faces_before=int(sum(len(m.faces) for m in meshes.values())),
        faces_after=int(sum(int(k.sum()) for k in keep.values())),
        nodes=report_nodes,
        frame_center=[float(c) for c in (lo + hi) / 2],
        size=[float(s) for s in shell_points.max(axis=0) - shell_points.min(axis=0)],
        source_sha256=hashlib.sha256(glb).hexdigest(),
    )
    return _write_glb(parts), report
