import * as THREE from "three";

export type RenderStyle = "solid" | "wireframe" | "xray";

const ORIGINAL_MATERIAL = "tvastaOriginalMaterial";
const XRAY_EDGES = "tvastaXrayEdges";
const OVERLAY_COLOR = 0x00e5ff;
const FEATURE_EDGES = "tvastaFeatureEdges";

/** Creases sharper than this are drawn as lines: the same rule the backend's
 *  3D edge alignment fits to the camera image (ar_edge_crease_deg). */
export const CREASE_DEG = 30;

/**
 * Draws every mesh's feature edges (sharp creases and open borders) as lines
 * over its own surface, so flat-shaded panels, steps and lids read as 3D
 * shape instead of one grey silhouette. Idempotent; `color` per call.
 */
export function addFeatureEdges(root: THREE.Object3D, color = 0x1f2937, opacity = 0.85): void {
  const meshes: THREE.Mesh[] = [];
  root.traverse((obj) => {
    if ((obj as THREE.Mesh).isMesh && !obj.userData[FEATURE_EDGES]) meshes.push(obj as THREE.Mesh);
  });
  for (const mesh of meshes) {
    for (const child of [...mesh.children]) {
      if (child.userData[FEATURE_EDGES]) {
        child.removeFromParent();
        (child as THREE.LineSegments).geometry.dispose();
        ((child as THREE.LineSegments).material as THREE.Material).dispose();
      }
    }
    // Push the surface back a hair so its own edges aren't half-hidden by depth ties.
    for (const m of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
      m.polygonOffset = true;
      m.polygonOffsetFactor = 1;
      m.polygonOffsetUnits = 1;
    }
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(mesh.geometry, CREASE_DEG),
      new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity })
    );
    edges.userData[FEATURE_EDGES] = true;
    edges.raycast = () => {}; // clicks pick the part, not its outline
    mesh.add(edges);
  }
}

/**
 * Switches how an AR-overlaid model is drawn, for judging registration
 * accuracy against the real object behind it:
 *  - solid: the model's own materials (restored exactly when switching back),
 *    with its crease edges outlined;
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
      // Its creases outlined, so steps and panel borders can be checked
      // against the real object's edges.
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(mesh.geometry, CREASE_DEG),
        new THREE.LineBasicMaterial({ color: OVERLAY_COLOR, transparent: true, opacity: 0.95 })
      );
      edges.userData[XRAY_EDGES] = true;
      mesh.add(edges);
    } else if (style === "wireframe") {
      mesh.material = new THREE.MeshBasicMaterial({
        color: OVERLAY_COLOR,
        wireframe: true,
        transparent: true,
        opacity: 0.9,
        side: THREE.DoubleSide, // no back-face culling, like the solid style (ensureVisibleMaterials)
      });
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
        new THREE.EdgesGeometry(mesh.geometry, CREASE_DEG),
        new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 })
      );
      edges.userData[XRAY_EDGES] = true;
      mesh.add(edges);
    }
  }
}
