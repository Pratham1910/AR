"""Component registry: which GLB parts are inspectable components (app/services/component_inspection/registry.py)."""

import json

import numpy as np
import pytest

from app.services.component_inspection.registry import (
    infer_type,
    registry_from_glb,
    sanitize_node_name,
    split_pieces,
)
from app.services.model3d.glb_inspect import GlbPartMesh, glb_nodes, glb_part_meshes

from glb_builder import SCREW_POSITIONS, Node, box, build_glb, merge, plate_with_screws


def _mapping(tmp_path, components, object_id="PLATE_001"):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({"objectId": object_id, "components": components}))
    return path


# ── names ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name, kind",
    [
        ("Screw_Top_Left", "screw"),
        ("M4 screws", "screw"),
        ("Protective_Cap", "cap"),
        ("O-Ring.002", "o-ring"),
        ("seal_ring", "seal"),
        ("Cylinder.001", None),
        ("Screwdriver_Slot", None),  # a word containing "screw" is not a screw
    ],
)
def test_component_type_comes_from_whole_words_in_the_name(name, kind):
    assert infer_type(name) == kind


def test_names_are_sanitized_like_threejs_does():
    assert sanitize_node_name("Cylinder.001") == "Cylinder001"
    assert sanitize_node_name("Screw Top/Left") == "Screw_TopLeft"


# ── enumeration without a mapping file ──────────────────────────────────


def test_screws_are_found_by_name_and_the_plate_is_not_assumed_to_be_a_component():
    registry = registry_from_glb("m1", plate_with_screws(), scale=1.0)
    assert registry.source == "auto"
    assert [c.id for c in registry.components] == [f"SCREW_{i:03d}" for i in range(1, 9)]
    assert all(c.type == "screw" and c.source == "name-match" and c.inspectable for c in registry.components)
    plate = next(n for n in registry.nodes if n.name == "Base_Plate")
    assert plate.component_ids == []


def test_parts_named_in_the_parts_panel_become_components():
    glb = build_glb([Node("Cylinder", mesh=box((0, 0, 0), (1, 2, 1))), Node("Cylinder.001", mesh=box((0, 1.2, 0), (0.6, 0.4, 0.6)))])
    registry = registry_from_glb("m1", glb, scale=0.1, named_parts={"Cylinder.001": "Cap"})
    assert [(c.id, c.type, c.display_name, c.source) for c in registry.components] == [("CAP_001", "cap", "Cap", "named-part")]


def test_no_components_is_reported_not_guessed():
    glb = build_glb([Node("Cylinder", mesh=box((0, 0, 0), (1, 1, 1)))])
    registry = registry_from_glb("m1", glb, scale=1.0)
    assert registry.components == []
    assert any("No inspectable components" in w for w in registry.warnings)


# ── transforms ──────────────────────────────────────────────────────────


def test_geometry_is_in_the_tracked_frame_recentred_and_in_metres():
    """Nested node transforms apply; the frame is the whole assembly's bounding-box centre, scaled."""
    glb = build_glb([
        Node("Body", mesh=box((0, 0, 0), (2, 2, 2))),  # spans -1..1
        Node("Group", translation=(10, 0, 0), scale=(2, 2, 2), children=[
            Node("Screw_A", mesh=box((0, 0, 0), (0.5, 0.5, 0.5)), translation=(0.5, 0, 0)),
        ]),
    ])
    # Screw_A: local box +-0.25 at local x 0.5 -> group-scaled to +-0.5 at x 1 -> translated to x 11 (10.5..11.5).
    registry = registry_from_glb("m1", glb, scale=0.01)
    screw = registry.get("SCREW_001")
    # Assembly spans x -1..11.5 -> centre x 5.25; y and z -1..1 -> centre 0.
    assert screw.center_m == pytest.approx([(11 - 5.25) * 0.01, 0, 0])
    assert screw.size_m == pytest.approx([0.01, 0.01, 0.01])
    # The node's own transform in the same frame: its origin at x 11, scale 2 (x 0.01 m/unit).
    assert screw.node_matrix[:3, 3] == pytest.approx([(11 - 5.25) * 0.01, 0, 0])
    assert screw.node_matrix[0, 0] == pytest.approx(0.02)


def test_scene_geometry_labels_each_face_with_its_component():
    registry = registry_from_glb("m1", plate_with_screws(), scale=1.0)
    scene = registry.scene
    assert len(scene.faces) == 12 * 9 and len(scene.labels) == len(scene.faces)
    assert (scene.labels == -1).sum() == 12  # the plate
    assert sorted(set(scene.labels[scene.labels >= 0])) == list(range(8))


# ── mapping file ────────────────────────────────────────────────────────


def test_mapping_file_is_authoritative(tmp_path):
    path = _mapping(tmp_path, [
        {"id": "SCREW_TL", "meshName": "Screw_01", "type": "screw"},
        {"id": "PLATE", "nodeIndex": 1, "type": "cover", "required": False},
    ])
    registry = registry_from_glb("m1", plate_with_screws(), scale=1.0, mapping_path=path)
    assert registry.source == "mapping" and registry.object_id == "PLATE_001"
    assert [(c.id, c.node_name, c.required) for c in registry.components] == [
        ("SCREW_TL", "Screw_01", True),
        ("PLATE", "Base_Plate", False),
    ]


def test_mapping_accepts_the_name_threejs_shows(tmp_path):
    glb = build_glb([Node("Cylinder.001", mesh=box((0, 0, 0), (1, 1, 1)))])
    path = _mapping(tmp_path, [{"id": "CAP_001", "meshName": "Cylinder001", "type": "cap"}])
    assert registry_from_glb("m1", glb, scale=1.0, mapping_path=path).get("CAP_001").inspectable


def test_duplicate_names_must_be_mapped_by_index(tmp_path):
    glb = build_glb([Node("Screw", mesh=box((0, 0, 0), (1, 1, 1))), Node("Screw", mesh=box((3, 0, 0), (1, 1, 1)))])
    path = _mapping(tmp_path, [
        {"id": "S1", "meshName": "Screw"},
        {"id": "S2", "nodeIndex": 1},
    ])
    registry = registry_from_glb("m1", glb, scale=1.0, mapping_path=path)
    s1, s2 = registry.get("S1"), registry.get("S2")
    assert s1.mesh_status == "ambiguous" and not s1.inspectable and "nodeIndex" in s1.problem
    assert s2.inspectable and s2.center_m[0] > 0
    assert any("S1" in w for w in registry.warnings)


def test_a_missing_mesh_is_listed_with_the_reason_not_dropped(tmp_path):
    path = _mapping(tmp_path, [{"id": "SCREW_009", "meshName": "Screw_09", "type": "screw"}])
    component = registry_from_glb("m1", plate_with_screws(), scale=1.0, mapping_path=path).get("SCREW_009")
    assert component.mesh_status == "missing" and component.geometry is None
    assert "Screw_09" in component.problem


def test_combined_screws_split_into_one_component_each(tmp_path):
    path = _mapping(tmp_path, [{"idPrefix": "SCREW_", "meshName": "Screws", "split": True, "type": "screw"}])
    registry = registry_from_glb("m1", plate_with_screws(combined=True), scale=1.0, mapping_path=path)
    assert [c.id for c in registry.components] == [f"SCREW_{i:03d}" for i in range(1, 9)]
    centers = [tuple(np.round(c.center_m, 3)) for c in registry.components]
    # Every screw found once, at its own place (recentred: the assembly's centre is at y -0.003).
    assert sorted((x, z) for x, _, z in centers) == sorted((round(x, 3), round(z, 3)) for x, z in SCREW_POSITIONS)
    assert all(c.size_m == pytest.approx([0.012, 0.004, 0.012]) for c in registry.components)


def test_combined_mesh_without_split_is_flagged(tmp_path):
    path = _mapping(tmp_path, [{"id": "ALL_SCREWS", "meshName": "Screws", "type": "screw"}])
    registry = registry_from_glb("m1", plate_with_screws(combined=True), scale=1.0, mapping_path=path)
    assert any("8 separate bodies" in w for w in registry.warnings)


def test_piece_out_of_range_is_reported(tmp_path):
    path = _mapping(tmp_path, [{"id": "SCREW_X", "meshName": "Screws", "piece": 12}])
    component = registry_from_glb("m1", plate_with_screws(combined=True), scale=1.0, mapping_path=path).get("SCREW_X")
    assert component.mesh_status == "piece-missing" and "8 bodies" in component.problem


def test_invalid_mapping_entries_are_rejected(tmp_path):
    path = _mapping(tmp_path, [{"id": "X"}])
    with pytest.raises(ValueError, match="nodeIndex or meshName"):
        registry_from_glb("m1", plate_with_screws(), scale=1.0, mapping_path=path)


def test_duplicate_ids_keep_the_first(tmp_path):
    path = _mapping(tmp_path, [{"id": "S", "meshName": "Screw_01"}, {"id": "S", "meshName": "Screw_02"}])
    registry = registry_from_glb("m1", plate_with_screws(), scale=1.0, mapping_path=path)
    assert [c.node_name for c in registry.components] == ["Screw_01"]
    assert any("Duplicate component id" in w for w in registry.warnings)


# ── pieces ──────────────────────────────────────────────────────────────


def test_seam_duplicated_vertices_do_not_split_a_body():
    """Exporters duplicate vertices along seams; welding keeps one body one piece."""
    v, f = box((0, 0, 0), (1, 1, 1))
    # Give every face its own three vertices (fully unshared, like a flat-shaded export).
    unshared = GlbPartMesh(v[f.reshape(-1)], np.arange(len(f) * 3).reshape(-1, 3))
    assert len(split_pieces(unshared)) == 1
    two = GlbPartMesh(*merge((v, f), box((5, 0, 0), (1, 1, 1))))
    assert [len(p) for p in split_pieces(two)] == [12, 12]


def test_node_listing_reports_combined_geometry():
    glb = plate_with_screws(combined=True)
    nodes = {n.node_index: n for n in glb_nodes(glb)}
    assert nodes[0].name == "Assembly" and not nodes[0].has_mesh
    assert set(glb_part_meshes(glb)) == {1, 2}
    registry = registry_from_glb("m1", glb, scale=1.0)
    screws = next(n for n in registry.nodes if n.name == "Screws")
    assert screws.pieces == 8 and screws.faces == 96


def test_a_component_can_carry_a_colour(tmp_path):
    path = _mapping(tmp_path, [{"id": "S1", "meshName": "Screw_01", "color": "#1e90ff"}, {"id": "S2", "meshName": "Screw_02"}])
    registry = registry_from_glb("m1", plate_with_screws(), scale=1.0, mapping_path=path)
    assert registry.get("S1").color == "#1e90ff" and registry.get("S2").color is None
    bad = _mapping(tmp_path, [{"id": "S1", "meshName": "Screw_01", "color": "blue"}])
    with pytest.raises(ValueError, match="color must look like"):
        registry_from_glb("m1", plate_with_screws(), scale=1.0, mapping_path=bad)
