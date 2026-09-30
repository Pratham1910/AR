import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { CameraSelect } from "../../components/CameraSelect";
import { CameraStatusBadge } from "../../components/CameraStatusBadge";
import { useCamera } from "../../hooks/useCamera";
import { ensureVisibleMaterials } from "./ensureVisibleMaterial";
import { VisionApi } from "../../services/api";
import type {
  FeaturePoseResponse,
  ObjectRegistrationResponse,
  PoseResponse,
  RegisterReferenceImageResponse,
  Vector2,
} from "../../types";

interface Props {
  assetId: string;
  modelUrl: string;
  /**
   * Multiplier converting the GLB's own mesh units into real-world meters
   * (Model3DInfo.scale). AR overlays place the model directly at a
   * real-world position — without this, a GLB not authored at 1 unit = 1
   * meter renders wildly wrong-sized (often invisible: the camera ends up
   * inside an oversized mesh and backface culling hides everything).
   */
  modelScale?: number;
  targetMarkerId?: number;
  /**
   * The currently selected asset's own detection class (Model3DInfo.
   * component_class_label), e.g. "cup" for a coffee mug asset. When this
   * changes (the user picked a different asset), "Object class" below is
   * reset to match it — without this, switching assets silently left the
   * PREVIOUS asset's object class behind, so the overlay could show one
   * object's 3D model while detecting a completely different class in the
   * live feed (a real bug this fixes).
   */
  defaultTargetClassLabel?: string | null;
}

type RegistrationMode = "marker" | "markerless" | "feature";

/**
 * Phase 5 (Project.md #23-#26): physical <-> 3D registration.
 *
 * Two modes, both ending in the same place — the model's Three.js
 * position/quaternion are set directly from whatever the backend computed,
 * with the Three.js camera kept at the identity transform (Project.md #26:
 * this component does no coordinate-system math of its own):
 *
 *   - "marker": print an ArUco marker (app/workers/generate_marker.py),
 *     place it near the object -> accurate 6DoF pose via solvePnP
 *     (app/services/pose/aruco_pose.py). This is the spec's stated initial
 *     registration method (#24).
 *   - "markerless": no marker needed — detects `targetClassLabel` (e.g.
 *     "bottle") via segmentation and estimates an APPROXIMATE position from
 *     its apparent size vs. `realWorldHeightM` (app/services/pose/markerless.py).
 *     Position only, no orientation — see the warning shown in this mode.
 *   - "feature": no marker needed either, but real 6DoF (position AND
 *     orientation) via ORB keypoint matching + solvePnPRansac against a
 *     reference photo you register first (app/services/pose/feature_tracker.py).
 *     Needs the object to have real visual texture (a label, logo, text) —
 *     the closest of the three to what an industrial platform like DELMIA
 *     Augmented Experience actually does, though still plane-based rather
 *     than matched against the full 3D CAD geometry.
 *
 * A 2D outline is drawn on every attempt regardless of whether a pose was
 * found (Project.md #57's "show me what the detector actually saw"),
 * distinct from the 3D model overlay which only appears once a pose exists.
 */
export function RegistrationOverlay({
  assetId,
  modelUrl,
  modelScale = 1,
  targetMarkerId,
  defaultTargetClassLabel,
}: Props) {
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
  const cameraObjectRef = useRef<THREE.PerspectiveCamera | null>(null);
  // Latest pose the backend reported. The render loop eases the model's
  // ACTUAL displayed transform toward this every frame (below), rather than
  // snapping to it the instant a new detection arrives — detections only
  // arrive every few hundred ms (a full camera->backend->response round
  // trip), so without this the model visibly "pops" between positions
  // instead of gliding. This does not make tracking more ACCURATE — see the
  // module docstring's honest limits on markerless mode — it only makes the
  // existing per-detection estimates feel continuous instead of jerky.
  const targetPositionRef = useRef(new THREE.Vector3());
  const targetQuaternionRef = useRef(new THREE.Quaternion());
  const hasTargetRef = useRef(false);

  const [mode, setMode] = useState<RegistrationMode>("markerless");
  const [targetClassLabel, setTargetClassLabel] = useState(defaultTargetClassLabel || "bottle");
  const [realWorldHeightM, setRealWorldHeightM] = useState(0.2);

  // Re-sync "Object class" whenever the selected asset's own detection class
  // changes (i.e. the user switched assets) — see the Props doc above for
  // exactly what silent-mismatch bug this prevents.
  useEffect(() => {
    if (defaultTargetClassLabel) setTargetClassLabel(defaultTargetClassLabel);
  }, [defaultTargetClassLabel]);

  const [labelWidthM, setLabelWidthM] = useState(0.08);
  const [labelHeightM, setLabelHeightM] = useState(0.1);
  const [registration, setRegistration] = useState<RegisterReferenceImageResponse | null>(null);
  const [registering, setRegistering] = useState(false);

  const [markerPose, setMarkerPose] = useState<PoseResponse | null>(null);
  const [objectPose, setObjectPose] = useState<ObjectRegistrationResponse | null>(null);
  const [featurePose, setFeaturePose] = useState<FeaturePoseResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [liveTracking, setLiveTracking] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [modelStatus, setModelStatus] = useState<"loading" | "loaded" | "error">("loading");
  const [modelError, setModelError] = useState<string | null>(null);

  // --- Three.js overlay scene setup (once per model) ---
  useEffect(() => {
    const container = overlayRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    const scene = new THREE.Scene();
    // FOV/aspect start as a placeholder and are corrected to the backend's
    // exact calibration on the first pose response (see applyCameraModel
    // below) — a guessed constant here visibly misaligns the overlay.
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 100);
    cameraObjectRef.current = camera;
    // Camera stays at the origin, looking down -Z — see module docstring.

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setClearColor(0x000000, 0); // transparent, so the video shows through
    container.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2.0));

    setModelStatus("loading");
    setModelError(null);
    const loader = new GLTFLoader();
    loader.load(
      modelUrl,
      (gltf) => {
        gltf.scene.scale.setScalar(modelScale); // GLB units -> real-world meters
        ensureVisibleMaterials(gltf.scene);

        // The GLB's own local origin isn't necessarily its geometric
        // center (bottle.glb's mesh spans Y roughly -1.6..3.6, i.e. its
        // origin sits well below the visual middle) — but the position we
        // place the model at *is* meant to represent the object's center
        // (a bounding-box-center back-projection, or a reference plane's
        // own center). Recenter the loaded scene inside a wrapper group so
        // "place the group at position P" actually means "the model's
        // visible center ends up at P", not "the model's arbitrary
        // authoring-time origin ends up at P".
        gltf.scene.updateMatrixWorld(true);
        const box = new THREE.Box3().setFromObject(gltf.scene);
        const center = box.getCenter(new THREE.Vector3());
        gltf.scene.position.sub(center);

        const group = new THREE.Group();
        group.visible = false; // hidden until a pose is found
        group.add(gltf.scene);
        modelObjectRef.current = group;
        scene.add(group);
        setModelStatus("loaded");
      },
      undefined,
      (err) => {
        setModelStatus("error");
        setModelError(err instanceof Error ? err.message : String(err));
      }
    );

    let frameId = 0;
    const SMOOTHING = 0.25; // fraction of the remaining distance closed per frame — higher = snappier, lower = smoother/laggier
    const animate = () => {
      frameId = requestAnimationFrame(animate);

      const model = modelObjectRef.current;
      if (model && hasTargetRef.current) {
        model.position.lerp(targetPositionRef.current, SMOOTHING);
        model.quaternion.slerp(targetQuaternionRef.current, SMOOTHING);
      }

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(frameId);
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, [modelUrl, modelScale]);

  const clearOutline = () => {
    const ctx = outlineCanvasRef.current?.getContext("2d");
    const canvas = outlineCanvasRef.current;
    if (ctx && canvas) ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const drawPolygon = (points: Vector2[], color: string, label?: string) => {
    const canvas = outlineCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || video.videoWidth === 0 || points.length === 0) return;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    points.forEach((pt, i) => (i === 0 ? ctx.moveTo(pt.x, pt.y) : ctx.lineTo(pt.x, pt.y)));
    ctx.closePath();
    ctx.stroke();

    ctx.fillStyle = color;
    points.forEach((pt) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
      ctx.fill();
    });

    if (label) {
      ctx.font = "24px sans-serif";
      ctx.fillText(label, points[0].x, Math.max(20, points[0].y - 10));
    }
  };

  const drawPoints = (points: Vector2[], color: string, label?: string) => {
    const canvas = outlineCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || video.videoWidth === 0) return;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = color;
    points.forEach((pt) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2);
      ctx.fill();
    });

    if (label && points.length > 0) {
      ctx.font = "20px sans-serif";
      ctx.fillText(label, 10, 28);
    }
  };

  const applyCameraModel = (verticalFovDeg: number, aspect: number) => {
    const camera = cameraObjectRef.current;
    if (!camera) return;
    // Must match the backend's calibration exactly (not the container's own
    // measured aspect) — that calibration is what the position/pose math
    // itself was computed against.
    if (camera.fov !== verticalFovDeg || camera.aspect !== aspect) {
      camera.fov = verticalFovDeg;
      camera.aspect = aspect;
      camera.updateProjectionMatrix();
    }
  };

  const applyModelTransform = (
    position: { x: number; y: number; z: number } | null,
    quaternion: { x: number; y: number; z: number; w: number } | null
  ) => {
    const model = modelObjectRef.current;
    if (!model) return;
    if (position && quaternion) {
      const wasVisible = model.visible;
      targetPositionRef.current.set(position.x, position.y, position.z);
      targetQuaternionRef.current.set(quaternion.x, quaternion.y, quaternion.z, quaternion.w);
      if (!wasVisible || !hasTargetRef.current) {
        // First sighting (or reappearing after being lost) — snap instead of
        // gliding in from wherever it last was (or the origin).
        model.position.copy(targetPositionRef.current);
        model.quaternion.copy(targetQuaternionRef.current);
      }
      hasTargetRef.current = true;
      model.visible = true;
    } else {
      model.visible = false;
      hasTargetRef.current = false;
    }
  };

  const registerReference = async () => {
    setRegistering(true);
    setApiError(null);
    try {
      const frame = captureFrameBase64();
      if (!frame) throw new Error("Could not capture a frame from the camera.");
      const result = await VisionApi.registerReferenceImage(assetId, frame, labelWidthM, labelHeightM);
      setRegistration(result);
    } catch (err) {
      setApiError(err instanceof Error ? err.message : String(err));
    } finally {
      setRegistering(false);
    }
  };

  const detectAndAlign = async () => {
    setBusy(true);
    setApiError(null);
    try {
      const frame = captureFrameBase64();
      if (!frame) throw new Error("Could not capture a frame from the camera.");

      if (mode === "marker") {
        const result = await VisionApi.estimatePose(frame, targetMarkerId);
        setMarkerPose(result);
        applyCameraModel(result.camera_vertical_fov_deg, result.camera_aspect);
        if (result.found && result.corners_px) {
          drawPolygon(result.corners_px, "#00e676", result.marker_id !== null ? `id ${result.marker_id}` : undefined);
        } else {
          clearOutline();
        }
        applyModelTransform(result.position, result.quaternion);
      } else if (mode === "markerless") {
        const result = await VisionApi.registerObject(frame, targetClassLabel, realWorldHeightM);
        setObjectPose(result);
        applyCameraModel(result.camera_vertical_fov_deg, result.camera_aspect);
        if (result.found && result.polygon) {
          drawPolygon(
            result.polygon,
            "#29b6f6",
            `${result.class_label} ${result.confidence ? (result.confidence * 100).toFixed(0) + "%" : ""}`
          );
        } else {
          clearOutline();
        }
        applyModelTransform(result.position, result.quaternion);
      } else {
        const result = await VisionApi.estimateFeaturePose(assetId, frame);
        setFeaturePose(result);
        applyCameraModel(result.camera_vertical_fov_deg, result.camera_aspect);
        if (result.found && result.inlier_points_px) {
          drawPoints(result.inlier_points_px, "#ffca28", `${result.num_inliers}/${result.num_matches} matched`);
        } else {
          clearOutline();
        }
        applyModelTransform(result.position, result.quaternion);
      }
    } catch (err) {
      setApiError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (!liveTracking) return;
    let cancelled = false;

    // Self-rescheduling rather than a fixed setInterval: each detection is a
    // full camera -> backend -> response round trip, so a fixed interval
    // either wastes time waiting when the backend was already fast, or
    // stacks up overlapping requests when it's slow. This runs exactly as
    // fast as the backend can actually keep up, with only a small minimum
    // gap (50ms) so an instant response doesn't spin needlessly.
    const loop = async () => {
      while (!cancelled) {
        await detectAndAlign();
        if (cancelled) break;
        await new Promise((resolve) => setTimeout(resolve, 50));
      }
    };
    void loop();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveTracking, mode, targetClassLabel, realWorldHeightM]);

  return (
    <div>
      <div className="mode-toggle">
        <button className={mode === "markerless" ? "active" : ""} onClick={() => setMode("markerless")}>
          Markerless (place object, no marker)
        </button>
        <button className={mode === "marker" ? "active" : ""} onClick={() => setMode("marker")}>
          ArUco Marker (accurate 6DoF)
        </button>
        <button className={mode === "feature" ? "active" : ""} onClick={() => setMode("feature")}>
          Feature Tracking (real 6DoF, needs texture)
        </button>
      </div>

      {mode === "markerless" && (
        <div className="registration-controls">
          <label>
            Object class
            <input value={targetClassLabel} onChange={(e) => setTargetClassLabel(e.target.value)} />
          </label>
          <label>
            Real height (m)
            <input
              type="number"
              step="0.01"
              min="0.01"
              value={realWorldHeightM}
              onChange={(e) => setRealWorldHeightM(Number(e.target.value))}
            />
          </label>
        </div>
      )}

      {mode === "feature" && (
        <div className="registration-controls">
          <label>
            Label width (m)
            <input
              type="number"
              step="0.01"
              min="0.01"
              value={labelWidthM}
              onChange={(e) => setLabelWidthM(Number(e.target.value))}
            />
          </label>
          <label>
            Label height (m)
            <input
              type="number"
              step="0.01"
              min="0.01"
              value={labelHeightM}
              onChange={(e) => setLabelHeightM(Number(e.target.value))}
            />
          </label>
          <button onClick={registerReference} disabled={registering || !ready}>
            {registering ? "Registering…" : "Register Reference Image"}
          </button>
        </div>
      )}

      {mode === "feature" && registration && (
        <p className={`registration-quality quality-${registration.quality}`}>
          {registration.feature_count} features extracted —{" "}
          {registration.quality === "good" && "good, should track reliably."}
          {registration.quality === "marginal" && "marginal — tracking may be unreliable; use a more textured label."}
          {registration.quality === "too_few" &&
            "too few — this surface likely lacks enough texture to track. Use a labeled/printed surface, filling the frame with it."}
        </p>
      )}

      <div style={{ position: "relative", width: "100%", maxWidth: 640 }}>
        {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
        <video ref={videoRef} autoPlay playsInline muted style={{ width: "100%", display: "block" }} />
        <div ref={overlayRef} style={{ position: "absolute", inset: 0, pointerEvents: "none" }} />
        <canvas
          ref={outlineCanvasRef}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
        />
        <canvas ref={canvasRef} style={{ display: "none" }} />
      </div>

      <CameraStatusBadge status={cameraStatus} deviceLabel={deviceLabel} resolution={resolution} error={cameraError} />
      <CameraSelect devices={devices} selectedDeviceId={selectedDeviceId} onSelect={selectDevice} />

      {modelStatus === "loading" && <p className="camera-status camera-status-pending">🟡 Loading 3D model…</p>}
      {modelStatus === "loaded" && <p className="camera-status camera-status-ok">🟢 3D model loaded (hidden until a pose is found)</p>}
      {modelStatus === "error" && (
        <p className="camera-status camera-status-error">🔴 3D model failed to load — {modelError}</p>
      )}

      <div className="registration-controls">
        <button onClick={detectAndAlign} disabled={busy || !ready}>
          {busy ? "Detecting…" : "Detect & Align"}
        </button>
        <label>
          <input type="checkbox" checked={liveTracking} onChange={(e) => setLiveTracking(e.target.checked)} />
          Live tracking (as fast as detection responds, smoothed)
        </label>
      </div>

      {mode === "markerless" && (
        <p className="warning">
          Approximate: position only, estimated from the object's apparent size — no orientation is
          estimated (the model is always placed upright, as authored). Accuracy depends on the
          "Real height" value above being correct and on camera calibration; run
          `python -m app.workers.calibrate_camera` for a real one instead of the default approximation.
        </p>
      )}

      {mode === "marker" && markerPose && (
        <div className={`pose-status ${markerPose.found ? "pose-found" : "pose-not-found"}`}>
          {markerPose.found ? (
            <>
              <p>
                Marker {markerPose.marker_id} found — reprojection error{" "}
                {markerPose.reprojection_error_px?.toFixed(2)}px
              </p>
              <p>
                position: ({markerPose.position!.x.toFixed(3)}, {markerPose.position!.y.toFixed(3)},{" "}
                {markerPose.position!.z.toFixed(3)}) m
              </p>
            </>
          ) : (
            <p>No marker detected in frame.</p>
          )}
          {markerPose.calibration_is_approximate && (
            <p className="warning">
              Using an approximate default camera calibration ({markerPose.calibration_source}).
            </p>
          )}
        </div>
      )}

      {mode === "markerless" && objectPose && (
        <div className={`pose-status ${objectPose.found ? "pose-found" : "pose-not-found"}`}>
          {objectPose.found ? (
            <>
              <p>
                {objectPose.class_label} found — confidence{" "}
                {objectPose.confidence !== null ? (objectPose.confidence * 100).toFixed(0) : "?"}%
              </p>
              <p>
                approx. position: ({objectPose.position!.x.toFixed(3)}, {objectPose.position!.y.toFixed(3)},{" "}
                {objectPose.position!.z.toFixed(3)}) m
              </p>
            </>
          ) : (
            <p>No "{targetClassLabel}" detected in frame.</p>
          )}
          {objectPose.calibration_is_approximate && (
            <p className="warning">
              Using an approximate default camera calibration ({objectPose.calibration_source}).
            </p>
          )}
        </div>
      )}

      {mode === "feature" && !registration && (
        <p className="warning">Register a reference image of the object's labeled/textured surface first.</p>
      )}

      {mode === "feature" && featurePose && (
        <div className={`pose-status ${featurePose.found ? "pose-found" : "pose-not-found"}`}>
          {featurePose.found ? (
            <>
              <p>
                Matched — {featurePose.num_inliers}/{featurePose.num_matches} inlier features
              </p>
              <p>
                position: ({featurePose.position!.x.toFixed(3)}, {featurePose.position!.y.toFixed(3)},{" "}
                {featurePose.position!.z.toFixed(3)}) m
              </p>
            </>
          ) : (
            <p>No match ({featurePose.num_matches} candidate matches, not enough to solve a pose).</p>
          )}
          {featurePose.calibration_is_approximate && (
            <p className="warning">
              Using an approximate default camera calibration ({featurePose.calibration_source}).
            </p>
          )}
        </div>
      )}

      {apiError && <p className="error">{apiError}</p>}
    </div>
  );
}
