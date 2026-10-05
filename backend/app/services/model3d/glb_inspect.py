"""
Reads a GLB file's own JSON chunk to find its geometric bounds — no full
mesh/scene parser needed, just the glTF accessor `min`/`max` metadata that
every well-formed exporter writes for POSITION attributes.

Used to auto-compute a newly uploaded Model3D's `scale` (Project.md #20/#26):
given the user's measured real-world height, `scale = real_height_m /
mesh_height_units`. This is exactly what was done by hand to fix
bottle.glb's scale (Project.md's own provided placeholder model was an
unscaled ~5.3m-tall Blender export) — automated here so it doesn't require
manually inspecting the binary format again for every new model.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np


class GlbParseError(ValueError):
    pass


@dataclass
class GlbBounds:
    min: tuple[float, float, float]
    max: tuple[float, float, float]

    @property
    def width(self) -> float:  # X
        return self.max[0] - self.min[0]

    @property
    def height(self) -> float:  # Y
        return self.max[1] - self.min[1]

    @property
    def depth(self) -> float:  # Z
        return self.max[2] - self.min[2]


@dataclass
class GlbPart:
    """One mesh-carrying node of an assembly GLB (e.g. a bottle's body or cap)."""

    node_index: int  # stable id; Three.js's GLTFLoader exposes it via parser.associations
    name: str  # the node's name as authored (Three.js sanitizes it, e.g. "Cylinder.001" -> "Cylinder001")
    bounds: GlbBounds  # in the file's own units, assembly (world) frame


def compute_glb_bounds(data: bytes) -> GlbBounds:
    """
    Parses a binary .glb file's JSON chunk and aggregates the min/max of
    every POSITION accessor referenced by a mesh primitive — the overall
    bounding box of everything actually rendered, in the file's own units.

    Only handles binary .glb (magic 'glTF'), not plain-JSON .gltf — that
    covers every file this platform currently produces/consumes (Blender's
    default glTF export, and everything in data/models/).
    """
    points = np.concatenate([c for _, c in _world_corners(data)])
    return GlbBounds(
        min=tuple(float(v) for v in points.min(axis=0)),
        max=tuple(float(v) for v in points.max(axis=0)),
    )


def list_glb_parts(data: bytes) -> list[GlbPart]:
    """Every mesh-carrying node with its own bounds — the assembly's parts."""
    by_node: dict[int, list[np.ndarray]] = {}
    for node_index, corners in _world_corners(data):
        if node_index is not None:
            by_node.setdefault(node_index, []).append(corners)
    nodes = _json_chunk(data).get("nodes", [])
    parts = []
    for node_index, corner_sets in sorted(by_node.items()):
        points = np.concatenate(corner_sets)
        parts.append(
            GlbPart(
                node_index=node_index,
                name=nodes[node_index].get("name") or f"node_{node_index}",
                bounds=GlbBounds(
                    min=tuple(float(v) for v in points.min(axis=0)),
                    max=tuple(float(v) for v in points.max(axis=0)),
                ),
            )
        )
    return parts


def _json_chunk(data: bytes) -> dict:
    if len(data) < 12 or data[0:4] != b"glTF":
        raise GlbParseError("Not a binary .glb file (missing 'glTF' magic header)")

    _, version, _ = struct.unpack("<4sII", data[0:12])
    if version != 2:
        raise GlbParseError(f"Unsupported glTF version {version} (only 2 is supported)")

    offset = 12
    json_chunk: dict | None = None
    while offset + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack("<I4s", data[offset : offset + 8])
        chunk_data = data[offset + 8 : offset + 8 + chunk_len]
        if chunk_type == b"JSON":
            json_chunk = json.loads(chunk_data)
            break
        offset += 8 + chunk_len

    if json_chunk is None:
        raise GlbParseError("No JSON chunk found in .glb file")
    return json_chunk


def _world_corners(data: bytes) -> list[tuple[int | None, np.ndarray]]:
    """Each primitive's bounding-box corners in assembly coordinates, tagged
    with the node that carries it (None when there's no scene graph)."""
    json_chunk = _json_chunk(data)
    accessors = json_chunk.get("accessors", [])
    meshes = json_chunk.get("meshes", [])

    def mesh_boxes(mesh_index: int) -> list[tuple[np.ndarray, np.ndarray]]:
        boxes = []
        for prim in meshes[mesh_index].get("primitives", []):
            idx = prim.get("attributes", {}).get("POSITION")
            if idx is None:
                continue
            accessor = accessors[idx]
            if "min" in accessor and "max" in accessor:  # not every exporter writes bounds
                boxes.append((np.asarray(accessor["min"], float), np.asarray(accessor["max"], float)))
        return boxes

    if not any("POSITION" in p.get("attributes", {}) for m in meshes for p in m.get("primitives", [])):
        raise GlbParseError("No mesh with a POSITION attribute found in .glb file")

    # Walk the scene graph so node scale/rotation/translation count — an
    # exporter may keep a unit conversion or axis flip as a node transform
    # (Blender's FBX->glTF path does) instead of baking it into vertices.
    # Each primitive's accessor box is transformed corner-by-corner, the same
    # thing Three.js's Box3.setFromObject() does.
    corners: list[tuple[int | None, np.ndarray]] = []
    nodes = json_chunk.get("nodes", [])
    scenes = json_chunk.get("scenes", [])
    roots = scenes[json_chunk.get("scene", 0)].get("nodes", []) if scenes else []

    def visit(node_index: int, parent: np.ndarray) -> None:
        node = nodes[node_index]
        world = parent @ _node_matrix(node)
        if "mesh" in node:
            for lo, hi in mesh_boxes(node["mesh"]):
                box = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                corners.append((node_index, (box @ world.T)[:, :3]))
        for child in node.get("children", []):
            visit(child, world)

    for root in roots:
        visit(root, np.eye(4))

    if not corners:  # no scene graph referencing the meshes: fall back to raw accessor bounds
        for mesh_index in range(len(meshes)):
            for lo, hi in mesh_boxes(mesh_index):
                corners.append((None, np.stack([lo, hi])))

    if not corners:
        raise GlbParseError("No POSITION accessor in this .glb has min/max bounds")
    return corners


@dataclass
class GlbPartMesh:
    """A part's triangles in the file's own units, assembly (world) frame."""

    vertices: np.ndarray  # (N, 3) float
    faces: np.ndarray  # (M, 3) int, indices into vertices


_COMPONENT_DTYPES = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_TYPE_SIZES = {"SCALAR": 1, "VEC3": 3}


def glb_part_meshes(data: bytes) -> dict[int, GlbPartMesh]:
    """Every mesh-carrying node's triangles (node transforms applied), keyed by
    node index — for drawing exactly where a part appears in a camera image
    once the assembly's pose is known."""
    json_chunk = _json_chunk(data)
    binary = _bin_chunk(data)
    accessors = json_chunk.get("accessors", [])
    views = json_chunk.get("bufferViews", [])

    def read(accessor_index: int) -> np.ndarray:
        accessor = accessors[accessor_index]
        view = views[accessor["bufferView"]]
        dtype = np.dtype(_COMPONENT_DTYPES[accessor["componentType"]])
        width = _TYPE_SIZES[accessor["type"]]
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        stride = view.get("byteStride") or dtype.itemsize * width
        count = accessor["count"]
        rows = np.ndarray(
            shape=(count, width), dtype=dtype, buffer=binary, offset=start, strides=(stride, dtype.itemsize)
        )
        return np.array(rows)

    meshes = json_chunk.get("meshes", [])
    nodes = json_chunk.get("nodes", [])
    scenes = json_chunk.get("scenes", [])
    roots = scenes[json_chunk.get("scene", 0)].get("nodes", []) if scenes else []
    out: dict[int, GlbPartMesh] = {}

    def visit(node_index: int, parent: np.ndarray) -> None:
        node = nodes[node_index]
        world = parent @ _node_matrix(node)
        if "mesh" in node:
            vertex_sets, face_sets, base = [], [], 0
            for prim in meshes[node["mesh"]].get("primitives", []):
                position = prim.get("attributes", {}).get("POSITION")
                if position is None or prim.get("mode", 4) != 4:  # triangle lists only
                    continue
                vertices = read(position).astype(float)
                if "indices" in prim:
                    faces = read(prim["indices"]).reshape(-1, 3).astype(np.int64)
                else:
                    faces = np.arange(len(vertices)).reshape(-1, 3)
                homogeneous = np.c_[vertices, np.ones(len(vertices))]
                vertex_sets.append((homogeneous @ world.T)[:, :3])
                face_sets.append(faces + base)
                base += len(vertices)
            if vertex_sets:
                out[node_index] = GlbPartMesh(np.concatenate(vertex_sets), np.concatenate(face_sets))
        for child in node.get("children", []):
            visit(child, world)

    for root in roots:
        visit(root, np.eye(4))
    return out


def _bin_chunk(data: bytes) -> bytes:
    offset = 12
    while offset + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack("<I4s", data[offset : offset + 8])
        if chunk_type == b"BIN\x00":
            return data[offset + 8 : offset + 8 + chunk_len]
        offset += 8 + chunk_len
    raise GlbParseError("No binary chunk found in .glb file")


def _node_matrix(node: dict) -> np.ndarray:
    """A glTF node's local transform: `matrix` (column-major) or T * R * S."""
    if "matrix" in node:
        return np.asarray(node["matrix"], float).reshape(4, 4).T
    x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation @ np.diag(node.get("scale", [1.0, 1.0, 1.0]))
    matrix[:3, 3] = node.get("translation", [0.0, 0.0, 0.0])
    return matrix


def compute_scale_for_real_height(data: bytes, real_height_m: float) -> float:
    """The one calculation every uploaded model needs — see module docstring."""
    bounds = compute_glb_bounds(data)
    if bounds.height <= 0:
        raise GlbParseError(f"Computed a non-positive mesh height ({bounds.height}); cannot derive a scale.")
    return real_height_m / bounds.height
