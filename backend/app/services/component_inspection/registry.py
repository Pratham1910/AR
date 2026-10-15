"""
Component registry: which parts of a GLB assembly are inspectable, under
which stable IDs, with their geometry in the tracked object's frame.

Not every scene node is a component. Components come from, in order:

1. An explicit mapping file (data/component_registry/<model_id>.json), the
   authoritative source once it exists:

       {
         "objectId": "PUMP_001",
         "components": [
           {"id": "CAP_001", "meshName": "Protective_Cap", "type": "cap"},
           {"id": "SCREW_001", "nodeIndex": 7, "type": "screw", "required": false},
           {"idPrefix": "SCREW_", "meshName": "Screws", "split": true, "type": "screw"}
         ]
       }

   `meshName` is the glTF node name as authored ("Cylinder.001") or as
   Three.js shows it ("Cylinder001"); `nodeIndex` wins when names repeat.
   `split` turns one node holding several disconnected bodies (eight screws
   exported as one mesh) into one component per body; `piece` picks one body.
2. Without a mapping file: parts the user named in the Parts panel, and
   nodes whose own name says what they are ("Screw_Top_Left").

Geometry is in the frame the tracked pose refers to: GLB node transforms
applied, recentred on the whole assembly's bounding-box centre (as the pose
service registers the mesh), scaled to metres by Model3D.scale.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from app.services.model3d.glb_inspect import GlbNode, GlbPartMesh

# Types the registry recognises in names. Order matters: "o-ring" before "ring".
COMPONENT_TYPES = (
    "screw", "bolt", "nut", "washer", "rivet", "pin", "stud", "clip", "cap", "lid", "cover",
    "plug", "connector", "terminal", "fuse", "relay", "switch", "led", "cable", "hose",
    "gasket", "o-ring", "seal", "ring", "bracket", "label",
)
_TYPE_PATTERNS = [
    (t, re.compile(rf"(?:^|[^a-z]){'[-_ ]?'.join(map(re.escape, t.split('-')))}s?(?:[^a-z]|$)", re.IGNORECASE))
    for t in COMPONENT_TYPES
]

ComponentSource = Literal["mapping", "named-part", "name-match"]
MeshStatus = Literal["ok", "missing", "ambiguous", "piece-missing", "empty"]


def infer_type(*names: str | None) -> str | None:
    """The component type a name states ("Screw_Top_Left" -> "screw"), if any."""
    for name in names:
        if not name:
            continue
        for kind, pattern in _TYPE_PATTERNS:
            if pattern.search(name.replace(".", "_")):
                return kind
    return None


def sanitize_node_name(name: str) -> str:
    """Three.js GLTFLoader's PropertyBinding.sanitizeNodeName: what the frontend shows."""
    return re.sub(r"[\[\]\.:/]", "", re.sub(r"\s", "_", name))


@dataclass
class ComponentGeometry:
    """A component's triangles in the tracked object's frame, metres."""

    vertices: np.ndarray  # (N, 3)
    faces: np.ndarray  # (M, 3) int

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        used = self.vertices[np.unique(self.faces)] if len(self.faces) else self.vertices
        return used.min(axis=0), used.max(axis=0)


@dataclass
class RegisteredComponent:
    id: str
    type: str
    required: bool
    inspection_mode: str  # "presence" for now; the verifier interface allows others
    display_name: str
    source: ComponentSource
    node_index: int | None
    node_name: str | None
    piece: int | None  # which disconnected body of the node, for combined meshes
    mesh_status: MeshStatus
    problem: str | None = None  # why the mesh couldn't be resolved, when mesh_status != "ok"
    geometry: ComponentGeometry | None = None
    color: str | None = None  # "#rrggbb" to tell it apart in the AR view (None: the UI picks one)
    node_matrix: np.ndarray | None = None  # node's transform in the object frame (metres)

    @property
    def inspectable(self) -> bool:
        return self.mesh_status == "ok" and self.geometry is not None

    @property
    def center_m(self) -> list[float] | None:
        if self.geometry is None:
            return None
        lo, hi = self.geometry.bounds
        return [float(v) for v in (lo + hi) / 2]

    @property
    def size_m(self) -> list[float] | None:
        if self.geometry is None:
            return None
        lo, hi = self.geometry.bounds
        return [float(v) for v in hi - lo]


@dataclass
class NodeInfo:
    """Every mesh node, inspectable or not — the Phase 1 debug listing."""

    node_index: int
    name: str
    sanitized_name: str
    pieces: int  # disconnected bodies in its mesh (>1: combined geometry)
    faces: int
    component_ids: list[str]


@dataclass
class SceneGeometry:
    """The whole assembly as one triangle soup, each face labelled with the
    index (into ComponentRegistry.components) of the component it belongs
    to, or -1 — for CAD occlusion tests."""

    vertices: np.ndarray
    faces: np.ndarray
    labels: np.ndarray


@dataclass
class ComponentRegistry:
    object_id: str
    components: list[RegisteredComponent]
    nodes: list[NodeInfo]
    warnings: list[str]
    source: Literal["mapping", "auto"]
    scene: SceneGeometry
    mapping_path: str | None = None

    def get(self, component_id: str) -> RegisteredComponent | None:
        return next((c for c in self.components if c.id == component_id), None)


@dataclass
class ComponentSpec:
    """One entry of a mapping file (before resolving it against the GLB)."""

    id: str | None = None
    id_prefix: str | None = None
    node_index: int | None = None
    mesh_name: str | None = None
    piece: int | None = None
    split: bool = False
    type: str | None = None
    required: bool = True
    inspection_mode: str = "presence"
    display_name: str | None = None
    color: str | None = None

    @staticmethod
    def from_json(entry: dict) -> ComponentSpec:
        def pick(*keys):
            return next((entry[k] for k in keys if k in entry), None)

        spec = ComponentSpec(
            id=pick("id"),
            id_prefix=pick("idPrefix", "id_prefix"),
            node_index=pick("nodeIndex", "node_index"),
            mesh_name=pick("meshName", "mesh_name", "nodeName", "node_name"),
            piece=pick("piece"),
            split=bool(pick("split") or False),
            type=pick("type"),
            required=bool(entry.get("required", True)),
            inspection_mode=pick("inspectionMode", "inspection_mode") or "presence",
            display_name=pick("displayName", "display_name", "name"),
            color=pick("color", "colour"),
        )
        if spec.color is not None and not re.fullmatch(r"#[0-9a-fA-F]{6}", str(spec.color)):
            raise ValueError(f"Component {spec.id or spec.id_prefix!r}: color must look like #1e90ff")
        if spec.node_index is None and not spec.mesh_name:
            raise ValueError(f"Component {spec.id or spec.id_prefix!r}: needs nodeIndex or meshName")
        if spec.split and spec.piece is not None:
            raise ValueError(f"Component {spec.id or spec.id_prefix!r}: 'split' and 'piece' are exclusive")
        if not spec.split and not spec.id:
            raise ValueError("Every component needs an 'id' (or 'split': true with an 'idPrefix')")
        return spec


def load_mapping(path: Path) -> tuple[str | None, list[ComponentSpec]] | None:
    """(objectId, specs) from a mapping file, or None when there is no file."""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("components")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: 'components' must be a list")
    return data.get("objectId"), [ComponentSpec.from_json(e) for e in entries]


def split_pieces(mesh: GlbPartMesh, weld_tolerance: float = 1e-6) -> list[np.ndarray]:
    """The mesh's disconnected bodies, each as an array of face indices.

    Vertices at the same position are welded first: exporters duplicate
    vertices along UV seams and hard edges, which would otherwise cut one
    screw into several "bodies". Ordered by centroid (x, then y, then z) so a
    piece number stays the same for the same file."""
    faces = mesh.faces
    if len(faces) == 0:
        return []
    extent = float(np.ptp(mesh.vertices, axis=0).max()) or 1.0
    keys = np.round(mesh.vertices / (extent * weld_tolerance)).astype(np.int64)
    _, welded = np.unique(keys, axis=0, return_inverse=True)
    welded = welded.reshape(-1)
    face_vertices = welded[faces]

    parent = np.arange(welded.max() + 1)

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    for a, b, c in face_vertices:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[find(rc)] = ra
    roots = np.array([find(v) for v in face_vertices[:, 0]])
    groups = [np.flatnonzero(roots == r) for r in np.unique(roots)]
    centroids = [mesh.vertices[faces[g]].reshape(-1, 3).mean(axis=0) for g in groups]
    order = sorted(range(len(groups)), key=lambda i: tuple(np.round(centroids[i], 6)))
    return [groups[i] for i in order]


def _object_frame(mesh: GlbPartMesh, faces: np.ndarray, center: np.ndarray, scale: float) -> ComponentGeometry:
    used = np.unique(mesh.faces[faces])
    remap = np.full(len(mesh.vertices), -1, np.int64)
    remap[used] = np.arange(len(used))
    return ComponentGeometry(vertices=(mesh.vertices[used] - center) * scale, faces=remap[mesh.faces[faces]])


def _type_id(kind: str, taken: set[str]) -> str:
    n = 1
    while f"{kind.upper().replace('-', '')}_{n:03d}" in taken:
        n += 1
    return f"{kind.upper().replace('-', '')}_{n:03d}"


def build_registry(
    object_id: str,
    nodes: list[GlbNode],
    meshes: dict[int, GlbPartMesh],
    center: np.ndarray,
    scale: float,
    named_parts: dict[str, str] | None = None,
    mapping: tuple[str | None, list[ComponentSpec]] | None = None,
    mapping_path: str | None = None,
) -> ComponentRegistry:
    """Resolves components against the GLB (see module docstring).

    `nodes`/`meshes`/`center` as glb_nodes / glb_part_meshes /
    vision.assembly_geometry give them; `named_parts`: authored node name ->
    the user's name for it (Parts panel)."""
    warnings: list[str] = []
    named_parts = named_parts or {}
    mesh_nodes = [n for n in nodes if n.has_mesh and n.node_index in meshes]
    by_index = {n.node_index: n for n in mesh_nodes}
    pieces_cache: dict[int, list[np.ndarray]] = {}

    def pieces(node_index: int) -> list[np.ndarray]:
        if node_index not in pieces_cache:
            pieces_cache[node_index] = split_pieces(meshes[node_index])
        return pieces_cache[node_index]

    def node_matrix(node: GlbNode) -> np.ndarray:
        m = node.world_matrix.copy()
        m[:3, 3] = (m[:3, 3] - center) * scale  # same recentring/scale as the vertices
        m[:3, :3] = m[:3, :3] * scale
        return m

    def resolve(spec: ComponentSpec) -> tuple[GlbNode | None, MeshStatus, str | None]:
        if spec.node_index is not None:
            node = by_index.get(spec.node_index)
            if node is None:
                return None, "missing", f"no mesh node with index {spec.node_index}"
            return node, "ok", None
        name = spec.mesh_name or ""
        matches = [n for n in mesh_nodes if n.name == name] or [
            n for n in mesh_nodes if sanitize_node_name(n.name) == sanitize_node_name(name)
        ]
        if not matches:
            return None, "missing", f"no mesh node named {name!r}"
        if len(matches) > 1:
            indices = ", ".join(str(n.node_index) for n in matches)
            return None, "ambiguous", f"{len(matches)} mesh nodes are named {name!r} (indices {indices}); map by nodeIndex"
        return matches[0], "ok", None

    components: list[RegisteredComponent] = []
    face_owner: dict[int, np.ndarray] = {}  # node index -> per-face component index (-1 = none)

    def add(
        cid: str, kind: str, spec_required: bool, mode: str, display: str, source: ComponentSource,
        node: GlbNode | None, piece: int | None, status: MeshStatus, problem: str | None,
        node_name: str | None = None, color: str | None = None,
    ) -> None:
        if any(c.id == cid for c in components):
            warnings.append(f"Duplicate component id {cid!r}: the later entry was ignored")
            return
        geometry = None
        if node is not None and status == "ok":
            mesh = meshes[node.node_index]
            if piece is None:
                face_ids = np.arange(len(mesh.faces))
            else:
                body = pieces(node.node_index)
                if piece >= len(body) or piece < 0:
                    status, problem = "piece-missing", f"node {node.node_index} has {len(body)} bodies; no piece {piece}"
                    face_ids = None
                else:
                    face_ids = body[piece]
            if face_ids is not None:
                if len(face_ids) == 0:
                    status, problem = "empty", "the mesh has no triangles"
                else:
                    geometry = _object_frame(mesh, face_ids, center, scale)
                    owner = face_owner.setdefault(node.node_index, np.full(len(mesh.faces), -1, np.int64))
                    owner[face_ids] = len(components)
        if problem:
            warnings.append(f"{cid}: {problem}")
        components.append(
            RegisteredComponent(
                id=cid, type=kind, required=spec_required, inspection_mode=mode, display_name=display,
                source=source, node_index=node.node_index if node else None,
                node_name=node.name if node else node_name, piece=piece, mesh_status=status,
                problem=problem, geometry=geometry, node_matrix=node_matrix(node) if node else None,
                color=color,
            )
        )

    object_name = object_id
    if mapping is not None:
        object_name = mapping[0] or object_id
        for spec in mapping[1]:
            node, status, problem = resolve(spec)
            kind = spec.type or infer_type(spec.display_name, node.name if node else spec.mesh_name) or "part"
            if spec.split:
                if node is None:
                    add(f"{spec.id_prefix or kind.upper() + '_'}*", kind, spec.required, spec.inspection_mode,
                        spec.display_name or spec.mesh_name or "", "mapping", None, None, status, problem, spec.mesh_name)
                    continue
                prefix = spec.id_prefix or f"{kind.upper()}_"
                for i in range(len(pieces(node.node_index))):
                    add(f"{prefix}{i + 1:03d}", kind, spec.required, spec.inspection_mode,
                        f"{spec.display_name or node.name} #{i + 1}", "mapping", node, i, status, problem,
                        color=spec.color)
            else:
                assert spec.id is not None
                add(spec.id, kind, spec.required, spec.inspection_mode,
                    spec.display_name or (node.name if node else spec.mesh_name or spec.id), "mapping",
                    node, spec.piece, status, problem, spec.mesh_name, color=spec.color)
    else:
        taken: set[str] = set()
        for node in mesh_nodes:
            display = named_parts.get(node.name)
            kind = infer_type(display, node.name)
            if display is None and kind is None:
                continue  # an unnamed node with no telling name: not assumed to be a component
            kind = kind or "part"
            cid = _type_id(kind, taken)
            taken.add(cid)
            add(cid, kind, True, "presence", display or node.name, "named-part" if display else "name-match",
                node, None, "ok", None)
        if not components:
            warnings.append(
                "No inspectable components: name the parts to inspect in the Parts panel, "
                "or add a component mapping file."
            )

    for c in components:
        if c.node_index is not None and len(pieces(c.node_index)) > 1 and c.piece is None:
            warnings.append(
                f"{c.id}: node {c.node_index} ({c.node_name!r}) holds {len(pieces(c.node_index))} separate bodies "
                "inspected as one component; map it with 'split': true to inspect each on its own"
            )

    node_infos = [
        NodeInfo(
            node_index=n.node_index,
            name=n.name,
            sanitized_name=sanitize_node_name(n.name),
            pieces=len(pieces(n.node_index)),
            faces=len(meshes[n.node_index].faces),
            component_ids=[c.id for c in components if c.node_index == n.node_index],
        )
        for n in mesh_nodes
    ]

    # The whole assembly with each face labelled by component, for occlusion.
    vertex_sets, face_sets, label_sets, base = [], [], [], 0
    for n in mesh_nodes:
        mesh = meshes[n.node_index]
        vertex_sets.append((mesh.vertices - center) * scale)
        face_sets.append(mesh.faces + base)
        label_sets.append(face_owner.get(n.node_index, np.full(len(mesh.faces), -1, np.int64)))
        base += len(mesh.vertices)
    scene = SceneGeometry(
        vertices=np.concatenate(vertex_sets) if vertex_sets else np.zeros((0, 3)),
        faces=np.concatenate(face_sets) if face_sets else np.zeros((0, 3), np.int64),
        labels=np.concatenate(label_sets) if label_sets else np.zeros(0, np.int64),
    )
    return ComponentRegistry(
        object_id=object_name,
        components=components,
        nodes=node_infos,
        warnings=warnings,
        source="mapping" if mapping is not None else "auto",
        scene=scene,
        mapping_path=mapping_path,
    )


def assembly_center(meshes: dict[int, GlbPartMesh]) -> np.ndarray:
    """Bounding-box centre of every mesh vertex (GLB units) — the point the
    pose service recentres the model on, so the tracked pose refers to it."""
    points = np.concatenate([m.vertices for m in meshes.values()])
    return (points.min(axis=0) + points.max(axis=0)) / 2


def registry_from_glb(
    object_id: str,
    data: bytes,
    scale: float,
    named_parts: dict[str, str] | None = None,
    mapping_path: Path | None = None,
) -> ComponentRegistry:
    from app.services.model3d.glb_inspect import glb_nodes, glb_part_meshes

    meshes = glb_part_meshes(data)
    mapping = load_mapping(mapping_path) if mapping_path is not None else None
    return build_registry(
        object_id, glb_nodes(data), meshes, assembly_center(meshes), scale, named_parts, mapping,
        str(mapping_path) if mapping is not None else None,
    )
