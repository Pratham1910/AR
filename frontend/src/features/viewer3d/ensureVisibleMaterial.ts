import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

/**
 * Makes every mesh of a loaded GLB actually visible:
 *
 * - Both sides drawn (no back-face culling). Exports, scans and generated
 *   models often have inward-facing normals, open edges or single-sided
 *   surfaces; with the default front-side-only materials those show as
 *   see-through holes or parts that vanish from some angles. Three.js lights
 *   back faces with flipped normals, so they shade correctly. (The pose
 *   service's renderer already draws both sides, so the overlay now matches
 *   what pose estimation sees.)
 *
 * - When a GLB ships with no materials (like the provided bottle.glb — a
 *   single untextured mesh), Three.js's GLTFLoader assigns a fallback
 *   MeshStandardMaterial with metalness=1, roughness=1. A fully metallic
 *   material has no diffuse component — it only shows color via environment
 *   reflections — and none of our scenes set an environment/reflection map,
 *   so that fallback renders almost pure black under simple point/hemisphere
 *   lights. That fallback is swapped for a visible, mildly glossy one, without
 *   touching meshes that have real authored materials/textures.
 */
export function ensureVisibleMaterials(root: THREE.Object3D, color = 0x4a90c4): void {
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh) return;

    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    materials.forEach((mat) => {
      if (mat.side !== THREE.DoubleSide) {
        mat.side = THREE.DoubleSide;
        mat.needsUpdate = true; // the shader is compiled per side mode
      }
      const standard = mat as THREE.MeshStandardMaterial;
      const looksLikeUntexturedFallback =
        standard.isMeshStandardMaterial && standard.metalness >= 0.9 && standard.roughness >= 0.9 && !standard.map;
      if (looksLikeUntexturedFallback) {
        standard.metalness = 0.15;
        standard.roughness = 0.55;
        standard.color.set(color);
      }
    });
  });
}

/**
 * Soft studio surroundings for the scene to reflect. glTF materials default
 * to fully metallic when a file doesn't say otherwise (the joystick GLB's
 * textured body has no metalness value), and a metallic surface shows only
 * reflections of its environment — with none it renders almost black. With
 * this, materials look as authored instead of being overridden.
 * Returns a cleanup function.
 */
export function addStudioEnvironment(scene: THREE.Scene, renderer: THREE.WebGLRenderer): () => void {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const room = new RoomEnvironment();
  const target = pmrem.fromScene(room, 0.04);
  scene.environment = target.texture;
  room.dispose();
  return () => {
    scene.environment = null;
    target.dispose();
    pmrem.dispose();
  };
}
