import * as THREE from "three";
import type { GLTF } from "three/examples/jsm/loaders/GLTFLoader.js";

/** Which parts of an assembly to show, and which one to highlight (by glTF node index). */
export interface PartView {
  hidden: number[];
  highlight: number | null;
}

export const EMPTY_PART_VIEW: PartView = { hidden: [], highlight: null };

const OUTLINE = "tvastaPartOutline";

/**
 * glTF node index -> the Object3D GLTFLoader built for it. Indices, not
 * names: GLTFLoader sanitizes names ("Cylinder.001" becomes "Cylinder001"),
 * while the backend reports parts by node index (GET /models3d/{id}/parts).
 */
export function indexParts(gltf: GLTF): Map<number, THREE.Object3D> {
  const parts = new Map<number, THREE.Object3D>();
  gltf.scene.traverse((obj) => {
    const node = gltf.parser.associations.get(obj)?.nodes;
    if (node !== undefined && !parts.has(node)) parts.set(node, obj);
  });
  return parts;
}

/** Shows/hides parts and outlines the highlighted one, without touching materials. */
export function applyPartView(parts: Map<number, THREE.Object3D>, view: PartView): void {
  parts.forEach((obj, nodeIndex) => {
    obj.visible = !view.hidden.includes(nodeIndex);

    const old: THREE.Object3D[] = [];
    obj.traverse((child) => {
      if (child.userData[OUTLINE]) old.push(child);
    });
    for (const outline of old) {
      outline.removeFromParent();
      (outline as THREE.LineSegments).geometry.dispose();
      ((outline as THREE.LineSegments).material as THREE.Material).dispose();
    }

    if (view.highlight !== nodeIndex) return;
    const meshes: THREE.Mesh[] = [];
    obj.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) meshes.push(child as THREE.Mesh);
    });
    for (const mesh of meshes) {
      const outline = new THREE.LineSegments(
        new THREE.EdgesGeometry(mesh.geometry, 25),
        // Drawn on top of everything so the part stays visible even when it
        // sits inside another (a bottle's neck under its cap).
        new THREE.LineBasicMaterial({ color: 0xffd600, depthTest: false, transparent: true })
      );
      outline.renderOrder = 999;
      outline.userData[OUTLINE] = true;
      mesh.add(outline);
    }
  });
}

/** The part (node index) an object belongs to: the nearest ancestor that is a part node. */
export function partOf(obj: THREE.Object3D, parts: Map<number, THREE.Object3D>): number | null {
  const byObject = new Map<THREE.Object3D, number>();
  parts.forEach((o, i) => byObject.set(o, i));
  for (let cur: THREE.Object3D | null = obj; cur; cur = cur.parent) {
    const index = byObject.get(cur);
    if (index !== undefined) return index;
  }
  return null;
}
