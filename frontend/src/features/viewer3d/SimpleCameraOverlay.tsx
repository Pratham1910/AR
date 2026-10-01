import { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { CameraSelect } from "../../components/CameraSelect";
import { CameraStatusBadge } from "../../components/CameraStatusBadge";
import { useCamera } from "../../hooks/useCamera";
import { ensureVisibleMaterials } from "./ensureVisibleMaterial";
import { applyPartView, EMPTY_PART_VIEW, indexParts, type PartView } from "./parts";

interface Props {
  modelUrl: string;
  partView?: PartView;
}

/**
 * The simplest possible view: the 3D model rendered on top of the live
 * camera feed, with NO detection/tracking/registration at all — no marker,
 * no object recognition, no feature matching, no backend calls. The model
 * just floats at a fixed spot in front of the virtual camera, auto-framed
 * the same way ThreeViewer.tsx frames it for standalone viewing, and you
 * drag/scroll (OrbitControls) to look at it. It does not know where the
 * physical object is and does not try to align to it — see
 * RegistrationOverlay.tsx for the modes that do that.
 */
export function SimpleCameraOverlay({ modelUrl, partView = EMPTY_PART_VIEW }: Props) {
  const partsRef = useRef<Map<number, THREE.Object3D>>(new Map());
  const partViewRef = useRef(partView);
  useEffect(() => {
    partViewRef.current = partView;
    applyPartView(partsRef.current, partView);
  }, [partView]);
  const {
    videoRef,
    ready,
    status: cameraStatus,
    error: cameraError,
    deviceLabel,
    resolution,
    devices,
    selectedDeviceId,
    selectDevice,
  } = useCamera();
  const overlayRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = overlayRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 1000);
    camera.position.set(0.3, 0.2, 0.5);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setClearColor(0x000000, 0); // transparent, so the live video shows through
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2.0));
    const directional = new THREE.DirectionalLight(0xffffff, 1.2);
    directional.position.set(1, 1, 1);
    scene.add(directional);

    const loader = new GLTFLoader();
    loader.load(modelUrl, (gltf) => {
      ensureVisibleMaterials(gltf.scene);
      scene.add(gltf.scene);
      partsRef.current = indexParts(gltf);
      applyPartView(partsRef.current, partViewRef.current);

      // Auto-frame on the model's own bounding box (same approach as
      // ThreeViewer.tsx) — since there's no real-world registration here,
      // "correct size" just means "nicely visible", not "true to scale".
      const box = new THREE.Box3().setFromObject(gltf.scene);
      const size = box.getSize(new THREE.Vector3()).length();
      const center = box.getCenter(new THREE.Vector3());
      controls.target.copy(center);
      camera.position.copy(center).add(new THREE.Vector3(size * 0.6, size * 0.6, size * 0.6));
      camera.near = size / 100;
      camera.far = size * 100;
      camera.updateProjectionMatrix();
    });

    let frameId = 0;
    const animate = () => {
      frameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Observes the container, not the window: it also resizes when the webcam
    // video finishes loading, which a window "resize" event never reports.
    const resizeObserver = new ResizeObserver(() => {
      if (container.clientWidth === 0 || container.clientHeight === 0) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    });
    resizeObserver.observe(container);

    return () => {
      cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, [modelUrl]);

  return (
    <div>
      <div style={{ position: "relative", width: "100%", maxWidth: 640, height: 480 }}>
        {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }}
        />
        <div ref={overlayRef} style={{ position: "absolute", inset: 0 }} />
      </div>

      <CameraStatusBadge status={cameraStatus} deviceLabel={deviceLabel} resolution={resolution} error={cameraError} />
      <CameraSelect devices={devices} selectedDeviceId={selectedDeviceId} onSelect={selectDevice} />

      <p className="hint">
        Drag to rotate, scroll to zoom. The model is not registered to anything in the camera feed — it's
        just floating on top of it. Use the other tabs for actual physical↔3D alignment.
      </p>

      {!ready && !cameraError && <p>Requesting camera access…</p>}
    </div>
  );
}
