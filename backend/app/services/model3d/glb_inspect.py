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


def compute_glb_bounds(data: bytes) -> GlbBounds:
    """
    Parses a binary .glb file's JSON chunk and aggregates the min/max of
    every POSITION accessor referenced by a mesh primitive — the overall
    bounding box of everything actually rendered, in the file's own units.

    Only handles binary .glb (magic 'glTF'), not plain-JSON .gltf — that
    covers every file this platform currently produces/consumes (Blender's
    default glTF export, and everything in data/models/).
    """
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

    accessors = json_chunk.get("accessors", [])
    position_accessor_indices = {
        prim["attributes"]["POSITION"]
        for mesh in json_chunk.get("meshes", [])
        for prim in mesh.get("primitives", [])
        if "POSITION" in prim.get("attributes", {})
    }

    if not position_accessor_indices:
        raise GlbParseError("No mesh with a POSITION attribute found in .glb file")

    mins: list[tuple[float, float, float]] = []
    maxs: list[tuple[float, float, float]] = []
    for idx in position_accessor_indices:
        accessor = accessors[idx]
        if "min" not in accessor or "max" not in accessor:
            continue  # not every exporter writes bounds on every accessor
        mins.append(tuple(accessor["min"]))
        maxs.append(tuple(accessor["max"]))

    if not mins:
        raise GlbParseError("No POSITION accessor in this .glb has min/max bounds")

    overall_min = tuple(min(v[axis] for v in mins) for axis in range(3))
    overall_max = tuple(max(v[axis] for v in maxs) for axis in range(3))
    return GlbBounds(min=overall_min, max=overall_max)


def compute_scale_for_real_height(data: bytes, real_height_m: float) -> float:
    """The one calculation every uploaded model needs — see module docstring."""
    bounds = compute_glb_bounds(data)
    if bounds.height <= 0:
        raise GlbParseError(f"Computed a non-positive mesh height ({bounds.height}); cannot derive a scale.")
    return real_height_m / bounds.height
