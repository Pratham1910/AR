import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { ensureVisibleMaterials } from "./ensureVisibleMaterial";

interface Props {
  modelUrl: string;
  height?: number;
}

/**
 * Phase 4 (Project.md #22): a standalone Three.js viewer — load GLB, orbit,
 * zoom, pan. No camera, no pose, no registration; that's Phase 5
 * (RegistrationOverlay.tsx). Kept as plain Three.js (not react-three-fiber)
 * so the same scene-setup pattern is reused for the AR overlay, which needs
 * direct control over the camera/object matrices from OpenCV pose output.
 */
export function ThreeViewer({ modelUrl, height = 480 }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nodeNames, setNodeNames] = useState<string[]>([]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    const camera = new THREE.PerspectiveCamera(50, container.clientWidth / height, 0.01, 100);
    camera.position.set(0.2, 0.2, 0.4);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, height);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.5));
    const directional = new THREE.DirectionalLight(0xffffff, 1.0);
    directional.position.set(1, 1, 1);
    scene.add(directional);
    scene.add(new THREE.AxesHelper(0.1));

    let disposed = false;
    let frameId = 0;

    const loader = new GLTFLoader();
    loader.load(
      modelUrl,
      (gltf) => {
        if (disposed) return;
        ensureVisibleMaterials(gltf.scene);
        scene.add(gltf.scene);

        // Frame the camera on the loaded model's bounding box, and surface
        // its node names — useful for confirming target.componentId ->
        // cadNodeId mapping (Project.md #20) against what the GLB actually
        // contains.
        const box = new THREE.Box3().setFromObject(gltf.scene);
        const size = box.getSize(new THREE.Vector3()).length();
        const center = box.getCenter(new THREE.Vector3());
        controls.target.copy(center);
        camera.position.copy(center).add(new THREE.Vector3(size * 0.6, size * 0.6, size * 0.6));
        camera.near = size / 100;
        camera.far = size * 100;
        camera.updateProjectionMatrix();

        const names: string[] = [];
        gltf.scene.traverse((obj) => {
          if (obj.name) names.push(obj.name);
        });
        setNodeNames(names);
      },
      undefined,
      (err) => setError(err instanceof Error ? err.message : String(err))
    );

    const animate = () => {
      frameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      camera.aspect = container.clientWidth / height;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, height);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      window.removeEventListener("resize", handleResize);
      controls.dispose();
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, [modelUrl, height]);

  return (
    <div>
      <div ref={containerRef} style={{ width: "100%", height }} />
      {error && <p className="error">Failed to load model: {error}</p>}
      {nodeNames.length > 0 && (
        <p className="node-names">Scene nodes: {nodeNames.join(", ")}</p>
      )}
    </div>
  );
}
