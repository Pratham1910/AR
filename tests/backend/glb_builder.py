"""Builds small binary GLBs in memory for tests: named nodes, nested
transforms, meshes made of boxes — enough to stand in for a CAD export."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

import numpy as np


def box(center, size) -> tuple[np.ndarray, np.ndarray]:
    """An axis-aligned box as (vertices (8, 3), faces (12, 3))."""
    c, h = np.asarray(center, float), np.asarray(size, float) / 2
    vertices = np.array([[c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]]
                         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    faces = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],  # x-, x+
        [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],  # y-, y+
        [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],  # z-, z+
    ])
    return vertices, faces


def merge(*meshes: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Several meshes as one (disconnected bodies in a single primitive)."""
    vertices, faces, base = [], [], 0
    for v, f in meshes:
        vertices.append(v)
        faces.append(f + base)
        base += len(v)
    return np.concatenate(vertices), np.concatenate(faces)


@dataclass
class Node:
    name: str
    mesh: tuple[np.ndarray, np.ndarray] | None = None
    translation: tuple[float, float, float] | None = None
    scale: tuple[float, float, float] | None = None
    children: list["Node"] = field(default_factory=list)


def build_glb(roots: list[Node]) -> bytes:
    nodes_json, meshes_json, accessors, views = [], [], [], []
    binary = bytearray()

    def add_view(data: bytes, target: int) -> int:
        while len(binary) % 4:
            binary.append(0)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data), "target": target})
        binary.extend(data)
        return len(views) - 1

    def add_node(node: Node) -> int:
        index = len(nodes_json)
        entry: dict = {"name": node.name}
        nodes_json.append(entry)
        if node.translation is not None:
            entry["translation"] = list(node.translation)
        if node.scale is not None:
            entry["scale"] = list(node.scale)
        if node.mesh is not None:
            v, f = node.mesh
            v32, f32 = v.astype(np.float32), f.astype(np.uint32)
            pos = add_view(v32.tobytes(), 34962)
            idx = add_view(f32.reshape(-1).tobytes(), 34963)
            accessors.append({"bufferView": pos, "componentType": 5126, "count": len(v32), "type": "VEC3",
                              "min": v32.min(axis=0).tolist(), "max": v32.max(axis=0).tolist()})
            accessors.append({"bufferView": idx, "componentType": 5125, "count": f32.size, "type": "SCALAR"})
            meshes_json.append({"primitives": [{"attributes": {"POSITION": len(accessors) - 2}, "indices": len(accessors) - 1}]})
            entry["mesh"] = len(meshes_json) - 1
        if node.children:
            entry["children"] = [add_node(child) for child in node.children]
        return index

    root_indices = [add_node(r) for r in roots]
    while len(binary) % 4:
        binary.append(0)
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": root_indices}],
        "nodes": nodes_json,
        "meshes": meshes_json,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
    }
    json_bytes = json.dumps(doc).encode()
    json_bytes += b" " * (-len(json_bytes) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    return (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
        + struct.pack("<I4s", len(binary), b"BIN\x00") + bytes(binary)
    )


# A 0.20 x 0.12 m plate (in GLB units = metres) lying in the XZ plane, its top
# face at y = 0, with eight 1 cm screw heads standing 4 mm proud of it in two
# rows. Screw positions (x, z) by number:
SCREW_POSITIONS = [(x, z) for z in (-0.04, 0.04) for x in (-0.075, -0.025, 0.025, 0.075)]
SCREW_SIZE = (0.012, 0.004, 0.012)


def screw_mesh(x: float, z: float) -> tuple[np.ndarray, np.ndarray]:
    return box((x, SCREW_SIZE[1] / 2, z), SCREW_SIZE)


def plate_with_screws(combined: bool = False) -> bytes:
    """`combined`: all eight screws as one mesh node (as some CAD exports do)."""
    plate = Node("Base_Plate", mesh=box((0, -0.005, 0), (0.2, 0.01, 0.12)))
    if combined:
        screws = [Node("Screws", mesh=merge(*(screw_mesh(x, z) for x, z in SCREW_POSITIONS)))]
    else:
        screws = [Node(f"Screw_{i + 1:02d}", mesh=screw_mesh(x, z)) for i, (x, z) in enumerate(SCREW_POSITIONS)]
    return build_glb([Node("Assembly", children=[plate, *screws])])
