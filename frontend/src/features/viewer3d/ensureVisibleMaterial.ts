import * as THREE from "three";

/**
 * When a GLB ships with no materials (like the provided bottle.glb — a
 * single untextured mesh), Three.js's GLTFLoader assigns a fallback
 * MeshStandardMaterial with metalness=1, roughness=1. A fully metallic
 * material has no diffuse component — it only shows color via environment
 * reflections — and none of our scenes set an environment/reflection map,
 * so that fallback renders almost pure black under simple point/hemisphere
 * lights. This looks exactly like "the model failed to load", even though
 * it's rendering correctly, just invisibly.
 *
 * Swaps any such fallback material for a visible, mildly glossy one so the
 * model is actually recognizable, without touching meshes that already
 * have real authored materials/textures.
 */
export function ensureVisibleMaterials(root: THREE.Object3D, color = 0x4a90c4): void {
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh) return;

    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    materials.forEach((mat) => {
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
