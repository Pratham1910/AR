import * as THREE from "three";

export type RenderStyle = "solid" | "wireframe" | "xray";

const ORIGINAL_MATERIAL = "tvastaOriginalMaterial";
const XRAY_EDGES = "tvastaXrayEdges";
const OVERLAY_COLOR = 0x00e5ff;

/**
 * Switches how an AR-overlaid model is drawn, for judging registration
 * accuracy against the real object behind it:
 *  - solid: the model's own materials (restored exactly when switching back);
 *  - wireframe: every triangle edge, real object visible between the lines;
 *  - xray: faint see-through fill plus crisp outline edges, so the model's
 *    silhouette can be compared directly with the real object's.
 */
export function applyRenderStyle(root: THREE.Object3D, style: RenderStyle): void {
  const meshes: THREE.Mesh[] = [];
  const oldEdges: THREE.LineSegments[] = [];
  root.traverse((obj) => {
    if ((obj as THREE.Mesh).isMesh) meshes.push(obj as THREE.Mesh);
    if (obj.userData[XRAY_EDGES]) oldEdges.push(obj as THREE.LineSegments);
  });

  for (const edges of oldEdges) {
    edges.removeFromParent();
    edges.geometry.dispose();
    (edges.material as THREE.Material).dispose();
  }

  for (const mesh of meshes) {
    const original = (mesh.userData[ORIGINAL_MATERIAL] ??= mesh.material) as THREE.Material | THREE.Material[];
    if (mesh.material !== original) {
      (Array.isArray(mesh.material) ? mesh.material : [mesh.material]).forEach((m) => m.dispose());
    }

    if (style === "solid") {
      mesh.material = original;
    } else if (style === "wireframe") {
      mesh.material = new THREE.MeshBasicMaterial({ color: OVERLAY_COLOR, wireframe: true, transparent: true, opacity: 0.9 });
    } else {
      mesh.material = new THREE.MeshBasicMaterial({
        color: OVERLAY_COLOR,
        transparent: true,
        opacity: 0.25,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      // Only creases sharper than 30 degrees, so a smooth curved body shows
      // its outline and rims, not every triangle.
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(mesh.geometry, 30),
        new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 })
      );
      edges.userData[XRAY_EDGES] = true;
      mesh.add(edges);
    }
  }
}
