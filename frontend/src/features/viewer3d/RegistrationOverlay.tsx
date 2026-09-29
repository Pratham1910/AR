import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { CameraSelect } from "../../components/CameraSelect";
import { CameraStatusBadge } from "../../components/CameraStatusBadge";
import { useCamera } from "../../hooks/useCamera";
import { VisionApi } from "../../services/api";
import type { PoseResponse } from "../../types";

interface Props {
  modelUrl: string;
  targetMarkerId?: number;
}

/**
 * Phase 5 (Project.md #23-#26): physical <-> 3D registration.
 *
 * Camera -> detect ArUco marker -> solvePnP (backend) -> 6DoF pose ->
 * apply transform to the 3D model -> 3D model overlays the physical object.
 *
 * The Three.js camera is kept at the identity transform; only the model's
 * position/quaternion move, using the pose the backend already converted
 * into Three.js space (app/services/pose/transforms.py) — this component
 * does no coordinate-system math of its own (Project.md #26).
 *
 * Known limitation: the overlay camera's FOV is a fixed guess, not derived
 * from the backend's actual camera_matrix, so the overlay's *perspective*
 * won't exactly match the live video's — only the marker-relative pose is
 * accurate (and only as accurate as the calibration in use; see
 * PoseResponse.calibration_is_approximate).
 *
 * A 2D outline (the marker's detected corners, drawn on `outlineCanvasRef`)
 * is rendered on every attempt regardless of whether a pose was found — this
 * is the "what did the detector actually see" debug signal (Project.md #57),
 * distinct from the 3D model overlay which only appears once a pose exists.
 */
export function RegistrationOverlay({ modelUrl, targetMarkerId }: Props) {
  const {
    videoRef,
    canvasRef,
    ready,
    status: cameraStatus,
    error: cameraError,
    deviceLabel,
    resolution,
    devices,
    selectedDeviceId,
    selectDevice,
    captureFrameBase64,
  } = useCamera();
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const outlineCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const modelObjectRef = useRef<THREE.Object3D | null>(null);
  const [pose, setPose] = useState<PoseResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [liveTracking, setLiveTracking] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  // --- Three.js overlay scene setup (once) ---
  useEffect(() => {
    const container = overlayRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 100);
    // Camera stays at the origin, looking down -Z — see module docstring.

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setClearColor(0x000000, 0); // transparent, so the video shows through
    container.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2.0));

    const loader = new GLTFLoader();
    loader.load(modelUrl, (gltf) => {
      gltf.scene.visible = false; // hidden until a pose is found
      modelObjectRef.current = gltf.scene;
      scene.add(gltf.scene);
    });

    let frameId = 0;
    const animate = () => {
      frameId = requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(frameId);
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, [modelUrl]);

  const drawOutline = (result: PoseResponse) => {
    const canvas = outlineCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || video.videoWidth === 0) return;

    // Match the canvas's pixel buffer to the video's native resolution — the
    // corner coordinates are in that same native space (see
    // useCamera.captureFrameBase64), and CSS then scales both identically.
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!result.found || !result.corners_px || result.corners_px.length !== 4) return;

    ctx.strokeStyle = "#00e676";
    ctx.lineWidth = 3;
    ctx.beginPath();
    result.corners_px.forEach((pt, i) => {
      if (i === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    });
    ctx.closePath();
    ctx.stroke();

    ctx.fillStyle = "#00e676";
    result.corners_px.forEach((pt) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
      ctx.fill();
    });

    if (result.marker_id !== null) {
      ctx.font = "24px sans-serif";
      ctx.fillText(`id ${result.marker_id}`, result.corners_px[0].x, Math.max(20, result.corners_px[0].y - 10));
    }
  };

  const detectAndAlign = async () => {
    setBusy(true);
    setApiError(null);
    try {
      const frame = captureFrameBase64();
      if (!frame) throw new Error("Could not capture a frame from the camera.");
      const result = await VisionApi.estimatePose(frame, targetMarkerId);
      setPose(result);
      drawOutline(result);

      const model = modelObjectRef.current;
      if (model) {
        if (result.found && result.position && result.quaternion) {
          model.position.set(result.position.x, result.position.y, result.position.z);
          model.quaternion.set(result.quaternion.x, result.quaternion.y, result.quaternion.z, result.quaternion.w);
          model.visible = true;
        } else {
          model.visible = false;
        }
      }
    } catch (err) {
      setApiError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (!liveTracking) return;
    const interval = setInterval(() => {
      if (!busy) void detectAndAlign();
    }, 800);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveTracking, busy]);

  return (
    <div>
      <div style={{ position: "relative", width: "100%", maxWidth: 640 }}>
        {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
        <video ref={videoRef} autoPlay playsInline muted style={{ width: "100%", display: "block" }} />
        <div
          ref={overlayRef}
          style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
        />
        <canvas
          ref={outlineCanvasRef}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
        />
        <canvas ref={canvasRef} style={{ display: "none" }} />
      </div>

      <CameraStatusBadge status={cameraStatus} deviceLabel={deviceLabel} resolution={resolution} error={cameraError} />
      <CameraSelect devices={devices} selectedDeviceId={selectedDeviceId} onSelect={selectDevice} />

      <div className="registration-controls">
        <button onClick={detectAndAlign} disabled={busy || !ready}>
          {busy ? "Detecting…" : "Detect & Align"}
        </button>
        <label>
          <input type="checkbox" checked={liveTracking} onChange={(e) => setLiveTracking(e.target.checked)} />
          Live tracking (every ~0.8s)
        </label>
      </div>

      {pose && (
        <div className={`pose-status ${pose.found ? "pose-found" : "pose-not-found"}`}>
          {pose.found ? (
            <>
              <p>Marker {pose.marker_id} found — reprojection error {pose.reprojection_error_px?.toFixed(2)}px</p>
              <p>
                position: ({pose.position!.x.toFixed(3)}, {pose.position!.y.toFixed(3)}, {pose.position!.z.toFixed(3)}) m
              </p>
            </>
          ) : (
            <p>No marker detected in frame.</p>
          )}
          {pose.calibration_is_approximate && (
            <p className="warning">
              Using an approximate default camera calibration ({pose.calibration_source}) — run
              `python -m app.workers.calibrate_camera` for accurate pose.
            </p>
          )}
        </div>
      )}

      {apiError && <p className="error">{apiError}</p>}
    </div>
  );
}
