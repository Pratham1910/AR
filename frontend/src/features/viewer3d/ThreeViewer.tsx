import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { addStudioEnvironment, ensureVisibleMaterials } from "./ensureVisibleMaterial";
import { applyPartView, EMPTY_PART_VIEW, indexParts, partOf, type PartView } from "./parts";
import type { ProcedureHost } from "./procedure";

interface Props {
  modelUrl: string;
  height?: number;
  partView?: PartView;
  onPartClick?: (nodeIndex: number | null) => void; // click a part in 3D to select it
  procedureHost?: ProcedureHost; // plays an attached procedure's animations on this model
}

/**
 * Phase 4 (Project.md #22): a standalone Three.js viewer — load GLB, orbit,
 * zoom, pan. No camera, no pose, no registration; that's Phase 5
 * (RegistrationOverlay.tsx). Kept as plain Three.js (not react-three-fiber)
 * so the same scene-setup pattern is reused for the AR overlay, which needs
 * direct control over the camera/object matrices from OpenCV pose output.
 */
export function ThreeViewer({ modelUrl, height = 480, partView = EMPTY_PART_VIEW, onPartClick, procedureHost }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nodeNames, setNodeNames] = useState<string[]>([]);
  const partsRef = useRef<Map<number, THREE.Object3D>>(new Map());
  const partViewRef = useRef(partView);
  const onPartClickRef = useRef(onPartClick);
  onPartClickRef.current = onPartClick;

  useEffect(() => {
    partViewRef.current = partView;
    applyPartView(partsRef.current, partView);
  }, [partView]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1117);

    const camera = new THREE.PerspectiveCamera(50, container.clientWidth / height, 0.01, 100);
    camera.position.set(0.2, 0.2, 0.4);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    const removeEnvironment = addStudioEnvironment(scene, renderer);
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
        partsRef.current = indexParts(gltf);
        applyPartView(partsRef.current, partViewRef.current);
        procedureHost?.attach(partsRef.current);

        // Frame the camera on the loaded model's bounding box, and surface
        // its node names — useful for confirming target.componentId ->
        // cadNodeId mapping (Project.md #20) against what the GLB actually
        // contains.
        const box = new THREE.Box3().setFromObject(gltf.scene);
        const size = box.getSize(new THREE.Vector3()).length();
        const center = box.getCenter(new THREE.Vector3());
        controls.target.copy(center);
        // Far enough that the whole bounding sphere fits the narrower of the
        // vertical / horizontal field of view, with a little margin.
        const radius = size / 2;
        const vFov = THREE.MathUtils.degToRad(camera.fov) / 2;
        const hFov = Math.atan(Math.tan(vFov) * camera.aspect);
        const distance = (radius / Math.sin(Math.min(vFov, hFov))) * 1.1;
        camera.position.copy(center).add(new THREE.Vector3(1, 0.6, 1).normalize().multiplyScalar(distance));
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
      procedureHost?.tick?.(performance.now());
      renderer.render(scene, camera);
    };
    animate();

    // Click (not drag) a part to select it; clicking empty space clears.
    const raycaster = new THREE.Raycaster();
    let downAt: { x: number; y: number } | null = null;
    const onPointerDown = (e: PointerEvent) => (downAt = { x: e.clientX, y: e.clientY });
    const onPointerUp = (e: PointerEvent) => {
      if (!downAt || Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y) > 4) return; // that was an orbit drag
      const rect = renderer.domElement.getBoundingClientRect();
      const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
      raycaster.setFromCamera(ndc, camera);
      const hit = raycaster
        .intersectObjects(scene.children, true)
        .find((h) => (h.object as THREE.Mesh).isMesh && h.object.visible);
      onPartClickRef.current?.(hit ? partOf(hit.object, partsRef.current) : null);
    };
    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointerup", onPointerUp);

    const handleResize = () => {
      camera.aspect = container.clientWidth / height;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, height);
    };
    // The container's width follows the page layout (sidebar, window), not just the window.
    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);

    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      removeEnvironment();
      procedureHost?.detach(partsRef.current);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      controls.dispose();
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, [modelUrl, height, procedureHost]);

  return (
    <div>
      <div ref={containerRef} className="viewer-canvas" style={{ width: "100%", height }} />
      {error && <p className="error">Failed to load model: {error}</p>}
      {nodeNames.length > 0 && (
        <p className="node-names">Nodes: {nodeNames.join(", ")}</p>
      )}
    </div>
  );
}
