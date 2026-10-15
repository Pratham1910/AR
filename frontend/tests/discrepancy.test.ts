// Discrepancy-rendering lifecycle (src/features/viewer3d/discrepancy.ts).
// Run with `npm test` (Node's built-in test runner; no extra dependencies).
import assert from "node:assert/strict";
import { test } from "node:test";
import * as THREE from "three";
import { DiscrepancyRenderer, type MarkerTextures } from "../src/features/viewer3d/discrepancy.ts";
import type { ComponentResult, ComponentStatus, RegisteredComponent } from "../src/types/index.ts";

const created: THREE.Texture[] = [];
const disposed = new Set<THREE.Texture>();
const textures: MarkerTextures = {
  ring() {
    const t = new THREE.Texture();
    t.addEventListener("dispose", () => disposed.add(t));
    created.push(t);
    return t;
  },
  label() {
    const t = new THREE.Texture();
    t.addEventListener("dispose", () => disposed.add(t));
    created.push(t);
    return { texture: t, aspect: 4 };
  },
};

function component(id: string, node: number | null, center: number[], piece: number | null = null): RegisteredComponent {
  return {
    id, type: "screw", display_name: id, required: true, inspection_mode: "presence", source: "mapping",
    node_index: node, node_name: `node${node}`, piece, mesh_status: "ok", problem: null,
    center_m: center, size_m: [0.004, 0.002, 0.004], node_matrix: null, reference_samples: 5, calibrated: false,
  };
}

function result(id: string, status: ComponentStatus): ComponentResult {
  return {
    id, status, reason: "", required: true, score: null, confidence: null, observable: true, visibility: "visible",
    visibility_reasons: [], last_valid: null, valid_observations: 1, center_px: null, bbox_px: null, outline_px: null,
    verdicts: [], history: [], learn_samples: null,
  };
}

function scene() {
  const root = new THREE.Group();
  const part = new THREE.Group();
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial());
  part.add(mesh);
  root.add(part);
  const parts = new Map<number, THREE.Object3D>([[3, part]]);
  const renderer = new DiscrepancyRenderer(root, parts, textures);
  renderer.setComponents([component("SCREW_001", 3, [0.01, 0.02, 0.03]), component("SCREW_002", null, [0, 0, 0], 1)]);
  return { root, mesh, renderer };
}

const markers = (root: THREE.Object3D) => root.children.filter((c) => c.name.startsWith("inspection:"));

test("confirmed missing: red marker at the component, opaque part hidden, ghost shown", () => {
  const { root, mesh, renderer } = scene();
  renderer.update([result("SCREW_001", "MISSING_CONFIRMED")], { showAll: false, selected: null });
  assert.deepEqual(renderer.markerIds, ["SCREW_001"]);
  const [marker] = markers(root);
  assert.deepEqual(marker.position.toArray(), [0.01, 0.02, 0.03]);
  assert.ok(marker.children[0].scale.x >= 0.025, "tiny parts still get a visible ring");
  assert.equal(mesh.layers.test(new THREE.Layers()), false, "the reference part is not drawn");
  assert.equal(mesh.children.length, 1, "a ghost is attached to the part");
  const ghost = mesh.children[0] as THREE.Mesh;
  assert.equal(ghost.geometry, mesh.geometry, "the ghost reuses the part's geometry");
});

test("present again: marker removed, part restored, resources disposed", () => {
  const { root, mesh, renderer } = scene();
  const before = created.length;
  renderer.update([result("SCREW_001", "MISSING_CONFIRMED")], { showAll: false, selected: null });
  const made = created.slice(before);
  renderer.update([result("SCREW_001", "PRESENT")], { showAll: false, selected: null });
  assert.deepEqual(renderer.markerIds, []);
  assert.equal(markers(root).length, 0);
  assert.equal(mesh.layers.test(new THREE.Layers()), true);
  assert.equal(mesh.children.length, 0);
  assert.ok(made.length === 2 && made.every((t) => disposed.has(t)));
});

test("candidates are amber-marked without hiding anything; present ones only with show-all", () => {
  const { mesh, renderer } = scene();
  renderer.update([result("SCREW_001", "MISSING_CANDIDATE"), result("SCREW_002", "PRESENT")], { showAll: false, selected: null });
  assert.deepEqual(renderer.markerIds, ["SCREW_001"]);
  assert.equal(mesh.layers.test(new THREE.Layers()), true);
  renderer.update([result("SCREW_001", "MISSING_CANDIDATE"), result("SCREW_002", "PRESENT")], { showAll: true, selected: null });
  assert.deepEqual(renderer.markerIds.sort(), ["SCREW_001", "SCREW_002"]);
});

test("a body of a combined mesh gets a ring but no ghost", () => {
  const { root, renderer } = scene();
  renderer.update([result("SCREW_002", "MISSING_CONFIRMED")], { showAll: false, selected: null });
  assert.equal(markers(root).length, 1);
  assert.equal(markers(root)[0].children.length, 2); // ring + label only
});

test("selecting a component marks it whatever its status, and changing status rebuilds the marker", () => {
  const { root, renderer } = scene();
  renderer.update([result("SCREW_001", "PRESENT")], { showAll: false, selected: "SCREW_001" });
  const first = markers(root)[0];
  assert.ok(first);
  renderer.update([result("SCREW_001", "PRESENT")], { showAll: false, selected: "SCREW_001" });
  assert.equal(markers(root)[0], first, "unchanged results keep the same objects");
  renderer.update([result("SCREW_001", "MISSING_CONFIRMED")], { showAll: false, selected: "SCREW_001" });
  assert.notEqual(markers(root)[0], first);
});

test("clear() (inspection off / reset) leaves the overlay as it was", () => {
  const { root, mesh, renderer } = scene();
  const childrenBefore = root.children.length;
  renderer.update([result("SCREW_001", "MISSING_CONFIRMED"), result("SCREW_002", "MISSING_CONFIRMED")], { showAll: false, selected: null });
  renderer.clear();
  assert.equal(root.children.length, childrenBefore);
  assert.equal(mesh.layers.test(new THREE.Layers()), true);
  assert.equal(mesh.children.length, 0);
});

test("results for components not in the registry are ignored", () => {
  const { renderer } = scene();
  renderer.update([result("UNKNOWN_9", "MISSING_CONFIRMED")], { showAll: true, selected: null });
  assert.deepEqual(renderer.markerIds, []);
});
