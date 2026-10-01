import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { CameraSelect } from "../../components/CameraSelect";
import { CameraStatusBadge } from "../../components/CameraStatusBadge";
import { useCamera } from "../../hooks/useCamera";
import { ensureVisibleMaterials } from "./ensureVisibleMaterial";
import { applyRenderStyle, type RenderStyle } from "./renderStyle";
import { applyPartView, EMPTY_PART_VIEW, indexParts, type PartView } from "./parts";
import { Models3DApi, VisionApi, apiErrorMessage } from "../../services/api";
import { ClassSelect } from "../../components/ClassSelect";
import type {
  AnchorOffset,
  ARFrameResponse,
  CameraIntrinsics,
  Model3DPart,
  FeaturePoseResponse,
  PoseAxes,
  PoseResponse,
  RegisterReferenceImageResponse,
  Vector2,
} from "../../types";

interface Props {
  assetId: string;
  modelId: string;
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
  /**
   * The model's saved anchor offset (Model3DInfo.anchor_*) — its own local
   * transform relative to the tracked reference plane. Tracking gives you
   * the pose of a flat patch on the object (a marker or a photographed
   * label), not the 3D model's own origin/orientation, so this offset is
   * what actually pins the model onto the real object instead of onto that
   * patch's raw pose. Defaults to zero/identity (no correction) until
   * calibrated via the on-screen alignment controls below.
   */
  initialAnchor?: AnchorOffset;
  /** Assembly parts to hide / highlight (e.g. show only the cap on the real bottle). */
  partView?: PartView;
  /** The model's assembly parts, offered as "Track by" choices in model-based mode. */
  parts?: Model3DPart[];
}

type RegistrationMode = "marker" | "markerless" | "feature" | "model";

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
  modelId,
  modelUrl,
  modelScale = 1,
  targetMarkerId,
  defaultTargetClassLabel,
  initialAnchor,
  partView = EMPTY_PART_VIEW,
  parts = [],
}: Props) {
  // Model-based only: match just this part (e.g. the body, which looks the
  // same with the cap on or off); the whole assembly is still rendered.
  const [trackPart, setTrackPart] = useState<number | null>(null);
  const partsRef = useRef<Map<number, THREE.Object3D>>(new Map());
  const partViewRef = useRef(partView);
  useEffect(() => {
    partViewRef.current = partView;
    applyPartView(partsRef.current, partView);
  }, [partView]);
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

  // Anchor offset (see Props.initialAnchor doc): edited as position (m) +
  // Euler degrees for intuitive nudge buttons, converted to a quaternion
  // only when composing onto the tracked pose or saving. Mirrored into a ref
  // so the self-rescheduling live-tracking loop (below) always reads the
  // latest value instead of whatever was current when that effect last ran.
  const [anchorPos, setAnchorPos] = useState({
    x: initialAnchor?.anchor_offset_x ?? 0,
    y: initialAnchor?.anchor_offset_y ?? 0,
    z: initialAnchor?.anchor_offset_z ?? 0,
  });
  const [anchorEulerDeg, setAnchorEulerDeg] = useState(() => {
    const q = initialAnchor
      ? new THREE.Quaternion(
          initialAnchor.anchor_rotation_x,
          initialAnchor.anchor_rotation_y,
          initialAnchor.anchor_rotation_z,
          initialAnchor.anchor_rotation_w
        )
      : new THREE.Quaternion();
    const e = new THREE.Euler().setFromQuaternion(q, "XYZ");
    return { rx: THREE.MathUtils.radToDeg(e.x), ry: THREE.MathUtils.radToDeg(e.y), rz: THREE.MathUtils.radToDeg(e.z) };
  });
  const [anchorSaving, setAnchorSaving] = useState(false);
  const [anchorSaved, setAnchorSaved] = useState(false);
  const anchorRef = useRef({ pos: anchorPos, eulerDeg: anchorEulerDeg });
  useEffect(() => {
    anchorRef.current = { pos: anchorPos, eulerDeg: anchorEulerDeg };
  }, [anchorPos, anchorEulerDeg]);

  const nudgePos = (axis: "x" | "y" | "z", deltaM: number) => {
    setAnchorPos((p) => ({ ...p, [axis]: p[axis] + deltaM }));
    setAnchorSaved(false);
  };
  const nudgeRot = (axis: "rx" | "ry" | "rz", deltaDeg: number) => {
    setAnchorEulerDeg((r) => ({ ...r, [axis]: r[axis] + deltaDeg }));
    setAnchorSaved(false);
  };
  const resetAnchor = () => {
    setAnchorPos({ x: 0, y: 0, z: 0 });
    setAnchorEulerDeg({ rx: 0, ry: 0, rz: 0 });
    setAnchorSaved(false);
  };
  const saveAnchor = async () => {
    setAnchorSaving(true);
    try {
      const q = new THREE.Quaternion().setFromEuler(
        new THREE.Euler(
          THREE.MathUtils.degToRad(anchorEulerDeg.rx),
          THREE.MathUtils.degToRad(anchorEulerDeg.ry),
          THREE.MathUtils.degToRad(anchorEulerDeg.rz),
          "XYZ"
        )
      );
      await Models3DApi.updateAnchor(modelId, {
        anchor_offset_x: anchorPos.x,
        anchor_offset_y: anchorPos.y,
        anchor_offset_z: anchorPos.z,
        anchor_rotation_x: q.x,
        anchor_rotation_y: q.y,
        anchor_rotation_z: q.z,
        anchor_rotation_w: q.w,
      });
      setAnchorSaved(true);
    } catch (err) {
      setApiError(apiErrorMessage(err));
    } finally {
      setAnchorSaving(false);
    }
  };

  const [mode, setMode] = useState<RegistrationMode>("markerless");
  const [targetClassLabel, setTargetClassLabel] = useState(defaultTargetClassLabel || "");
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
  const [featurePose, setFeaturePose] = useState<FeaturePoseResponse | null>(null);
  // Markerless + model-based modes go through the backend's detect-once /
  // track state machine (one session per mounted overlay, so two tabs or
  // cameras never share tracking state).
  const sessionIdRef = useRef(`s-${Math.random().toString(36).slice(2)}`);
  const [arResult, setArResult] = useState<ARFrameResponse | null>(null);
  const [arEvents, setArEvents] = useState<string[]>([]);
  // Detector runs / tracker runs per second, from the session's counters.
  const [rates, setRates] = useState({ detect: 0, track: 0, render: 0, renderMs: 0 });
  const renderMsRef = useRef(0);
  const [frameMs, setFrameMs] = useState(0); // camera frame -> backend -> pose, round trip
  // Cache key of the intrinsics currently in the render camera's projection.
  const intrinsicsKeyRef = useRef("");
  const counterSamplesRef = useRef<{ t: number; detections: number; tracks: number }[]>([]);
  const renderFramesRef = useRef(0);
  const [busy, setBusy] = useState(false);
  // "Save frame": keeps the raw camera frame (no overlay) under a label for offline analysis.
  const [frameLabel, setFrameLabel] = useState("cap-on");
  const [frameSaved, setFrameSaved] = useState<string | null>(null);
  const saveFrame = async () => {
    const frame = captureFrameBase64();
    if (!frame) return;
    try {
      const saved = await VisionApi.saveFrame(frame, frameLabel);
      setFrameSaved(`Saved ${saved.saved} (${saved.width}×${saved.height})`);
    } catch (err) {
      setApiError(apiErrorMessage(err));
    }
  };
  const [liveTracking, setLiveTracking] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [renderStyle, setRenderStyle] = useState<RenderStyle>("solid");
  // Read by the model loader too, so a model (re)loaded while in X-Ray or
  // Wireframe comes up in that style instead of reverting to solid.
  const renderStyleRef = useRef<RenderStyle>("solid");
  useEffect(() => {
    renderStyleRef.current = renderStyle;
    if (modelObjectRef.current) applyRenderStyle(modelObjectRef.current, renderStyle);
  }, [renderStyle]);
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
    intrinsicsKeyRef.current = ""; // new camera: the next result must set its projection
    // Camera stays at the origin, looking down -Z — see module docstring.

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setClearColor(0x000000, 0); // transparent, so the video shows through
    container.appendChild(renderer.domElement);

    // The container only reaches its real height once the webcam video has
    // loaded (after this effect runs), so the canvas must follow it — sized
    // once at mount it stays at a placeholder height and the whole render is
    // squashed into the top of the frame. Only the canvas size follows the
    // container; the camera's aspect stays the backend calibration's.
    const resizeObserver = new ResizeObserver(() => {
      if (container.clientWidth > 0 && container.clientHeight > 0) {
        renderer.setSize(container.clientWidth, container.clientHeight);
      }
    });
    resizeObserver.observe(container);

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
        applyRenderStyle(group, renderStyleRef.current);
        partsRef.current = indexParts(gltf);
        applyPartView(partsRef.current, partViewRef.current);
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
    // Fraction of the remaining distance closed per rendered frame. This only
    // interpolates between pose updates (which arrive slower than 60 fps);
    // the actual smoothing is the backend's Kalman filter, so this stays
    // quick to avoid stacking a second layer of lag on top of it.
    const INTERPOLATION = 0.4;
    const animate = () => {
      frameId = requestAnimationFrame(animate);

      const model = modelObjectRef.current;
      if (model && hasTargetRef.current) {
        model.position.lerp(targetPositionRef.current, INTERPOLATION);
        model.quaternion.slerp(targetQuaternionRef.current, INTERPOLATION);
      }

      const started = performance.now();
      renderer.render(scene, camera);
      renderMsRef.current += performance.now() - started;
      renderFramesRef.current += 1;
    };
    animate();

    return () => {
      cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
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

  /**
   * Debug pose gizmo (Project.md's debug-mode requirement): draws the
   * estimated pose's own X/Y/Z axes directly on the live feed, X=red,
   * Y=green, Z=blue — call AFTER drawPolygon/drawPoints (this does not clear
   * the canvas, so it layers on top of whatever outline was just drawn).
   */
  const drawAxes = (axes: PoseAxes) => {
    const ctx = outlineCanvasRef.current?.getContext("2d");
    if (!ctx) return;

    const drawLine = (to: Vector2, color: string) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(axes.origin.x, axes.origin.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
    };
    drawLine(axes.x_axis, "#ff1744");
    drawLine(axes.y_axis, "#00e676");
    drawLine(axes.z_axis, "#2979ff");

    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(axes.origin.x, axes.origin.y, 4, 0, Math.PI * 2);
    ctx.fill();
  };

  const applyCameraModel = (verticalFovDeg: number, aspect: number) => {
    const camera = cameraObjectRef.current;
    if (!camera) return;
    // Must match the backend's calibration exactly (not the container's own
    // measured aspect) — that calibration is what the position/pose math
    // itself was computed against.
    if (camera.fov !== verticalFovDeg || camera.aspect !== aspect || intrinsicsKeyRef.current) {
      camera.fov = verticalFovDeg;
      camera.aspect = aspect;
      camera.updateProjectionMatrix();
      intrinsicsKeyRef.current = ""; // the intrinsics projection was replaced
    }
  };

  /**
   * Builds the render camera's projection straight from the pinhole
   * intrinsics the backend used (fx, fy, cx, cy for this frame size), so the
   * rendered model lines up pixel-for-pixel with the pose math — including
   * an off-center principal point, which a vertical-FOV + aspect camera
   * can't represent. Camera at the origin looking down -Z (docs/coordinates.md).
   */
  const applyCameraIntrinsics = (k: CameraIntrinsics) => {
    const camera = cameraObjectRef.current;
    if (!camera) return;
    const key = `${k.fx},${k.fy},${k.cx},${k.cy},${k.width},${k.height}`;
    if (key === intrinsicsKeyRef.current) return;
    intrinsicsKeyRef.current = key;
    const near = camera.near;
    // Image y grows downward, so the frustum's top edge is at +cy and its
    // bottom at -(height - cy), scaled to the near plane.
    camera.projectionMatrix.makePerspective(
      (-k.cx * near) / k.fx,
      ((k.width - k.cx) * near) / k.fx,
      (k.cy * near) / k.fy,
      (-(k.height - k.cy) * near) / k.fy,
      near,
      camera.far
    );
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  };

  const applyModelTransform = (
    position: { x: number; y: number; z: number } | null,
    quaternion: { x: number; y: number; z: number; w: number } | null,
    // false for model-based (CAD) mode: its pose is already the model's own.
    useAnchor = true
  ) => {
    const model = modelObjectRef.current;
    if (!model) return;
    if (position && quaternion) {
      const wasVisible = model.visible;

      // Compose the calibrated anchor offset onto the tracked reference
      // plane's pose: tracking only knows where a flat patch (a marker, or a
      // photographed label) is, not where the 3D model's own origin should
      // sit relative to that patch. anchorPos is expressed in the plane's
      // OWN local frame, so it must be rotated by the plane's orientation
      // before being added to its world position (not just added directly).
      const { pos, eulerDeg } = useAnchor ? anchorRef.current : { pos: { x: 0, y: 0, z: 0 }, eulerDeg: { rx: 0, ry: 0, rz: 0 } };
      const planeQuat = new THREE.Quaternion(quaternion.x, quaternion.y, quaternion.z, quaternion.w);
      const anchorQuat = new THREE.Quaternion().setFromEuler(
        new THREE.Euler(
          THREE.MathUtils.degToRad(eulerDeg.rx),
          THREE.MathUtils.degToRad(eulerDeg.ry),
          THREE.MathUtils.degToRad(eulerDeg.rz),
          "XYZ"
        )
      );
      const offsetWorld = new THREE.Vector3(pos.x, pos.y, pos.z).applyQuaternion(planeQuat);
      const finalPosition = new THREE.Vector3(position.x, position.y, position.z).add(offsetWorld);
      const finalQuaternion = planeQuat.clone().multiply(anchorQuat);

      targetPositionRef.current.copy(finalPosition);
      targetQuaternionRef.current.copy(finalQuaternion);
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
      setApiError(apiErrorMessage(err));
    } finally {
      setRegistering(false);
    }
  };

  // Rendering FPS is sampled from the animation loop, independent of how
  // often tracking results arrive (the model is re-rendered, smoothed, every frame).
  useEffect(() => {
    const timer = setInterval(() => {
      const frames = renderFramesRef.current;
      const renderMs = frames > 0 ? renderMsRef.current / frames : 0;
      renderFramesRef.current = 0;
      renderMsRef.current = 0;
      setRates((r) => ({ ...r, render: frames, renderMs }));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // The backend starts a fresh session when mode/class/height changes; drop
  // the previous session's numbers here too so they're never shown as current.
  useEffect(() => {
    setArResult(null);
    setArEvents([]);
    counterSamplesRef.current = [];
  }, [mode, targetClassLabel, realWorldHeightM, trackPart]);

  // End the backend session when this view goes away (switching model/asset remounts it).
  useEffect(() => {
    const sessionId = sessionIdRef.current;
    return () => {
      void VisionApi.endArSession(sessionId).catch(() => undefined);
    };
  }, []);

  const resetTracking = async () => {
    await VisionApi.endArSession(sessionIdRef.current).catch(() => undefined);
    counterSamplesRef.current = [];
    setArResult(null);
    setArEvents((events) => [...events, "[UI] Tracking reset — searching again"].slice(-12));
    applyModelTransform(null, null);
    clearOutline();
  };

  const applyArResult = (result: ARFrameResponse, roundTripMs: number) => {
    setArResult(result);
    setFrameMs(roundTripMs);
    applyCameraIntrinsics(result.intrinsics);
    if (result.events.length) setArEvents((events) => [...events, ...result.events].slice(-12));

    // Detection vs tracking rate, over the last ~2 seconds of responses.
    const now = performance.now();
    const samples = counterSamplesRef.current;
    samples.push({ t: now, detections: result.counters.detection_runs, tracks: result.counters.tracking_frames });
    while (samples.length > 2 && now - samples[0].t > 2000) samples.shift();
    const first = samples[0];
    const seconds = (now - first.t) / 1000;
    if (seconds > 0.25) {
      setRates((r) => ({
        ...r,
        detect: Math.max(0, result.counters.detection_runs - first.detections) / seconds,
        track: Math.max(0, result.counters.tracking_frames - first.tracks) / seconds,
      }));
    }

    clearOutline();
    const obj = result.object;
    if (result.detection) {
      // Detector hit on this frame: the pose is initialized on the next one.
      const d = result.detection;
      const [x1, y1, x2, y2] = d.bbox;
      drawPolygon(
        d.polygon && d.polygon.length >= 3
          ? d.polygon
          : [
              { x: x1, y: y1 },
              { x: x2, y: y1 },
              { x: x2, y: y2 },
              { x: x1, y: y2 },
            ],
        "#42a5f5",
        `detected ${d.class_label} ${(d.confidence * 100).toFixed(0)}% — initializing pose`
      );
    } else if (obj && result.visible) {
      const held = result.state === "LOST" || result.state === "RECOVERING" || result.state === "INITIALIZING";
      const color = held ? "#ff7043" : result.monitoring ? "#ffca28" : "#00e676";
      const label = `${held ? "held" : "tracking"} ${obj.class_label} #${obj.object_id} ${(obj.confidence * 100).toFixed(0)}%`;
      if (obj.polygon && obj.polygon.length >= 3) {
        drawPolygon(obj.polygon, color, label);
      } else if (obj.bbox) {
        const [x1, y1, x2, y2] = obj.bbox;
        drawPolygon(
          [
            { x: x1, y: y1 },
            { x: x2, y: y1 },
            { x: x2, y: y2 },
            { x: x1, y: y2 },
          ],
          color,
          label
        );
      }
    }
    if (result.visible && result.axes) drawAxes(result.axes);
    // The exact region each part check measured, colored by its verdict
    // (layered on top of the outline, like the axes).
    const ctx = outlineCanvasRef.current?.getContext("2d");
    if (ctx) {
      for (const check of result.part_checks ?? []) {
        const [x1, y1, x2, y2] = check.region_px;
        const color = check.state === "present" ? "#00e676" : check.state === "absent" ? "#ff1744" : "#ffca28";
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillStyle = color;
        ctx.font = "bold 18px sans-serif";
        ctx.fillText(`${check.part_name}: ${check.state}`, x2 + 6, (y1 + y2) / 2 + 6);
      }
    }

    // Hidden only once the session gives up (SEARCHING); while LOST the
    // backend keeps returning the last valid pose, so the model stays put
    // instead of flickering on a missed frame.
    if (result.visible) {
      applyModelTransform(result.position, result.quaternion, mode !== "model");
    } else {
      applyModelTransform(null, null);
    }
  };

  const detectAndAlign = async () => {
    if ((mode === "markerless" || mode === "model") && !targetClassLabel) {
      setApiError("Pick an object class first — it's what the camera looks for.");
      return;
    }
    setBusy(true);
    setApiError(null);
    const frameStarted = performance.now();
    try {
      const frame = captureFrameBase64();
      if (!frame) throw new Error("Could not capture a frame from the camera.");

      if (mode === "marker") {
        const result = await VisionApi.estimatePose(frame, targetMarkerId);
        setMarkerPose(result);
        applyCameraModel(result.camera_vertical_fov_deg, result.camera_aspect);
        if (result.found && result.corners_px) {
          drawPolygon(result.corners_px, "#00e676", result.marker_id !== null ? `id ${result.marker_id}` : undefined);
          if (result.axes) drawAxes(result.axes);
        } else {
          clearOutline();
        }
        applyModelTransform(result.position, result.quaternion);
      } else if (mode === "markerless" || mode === "model") {
        const result = await VisionApi.arFrame(
          sessionIdRef.current,
          mode,
          frame,
          targetClassLabel,
          // model_id in markerless mode only enables the model's calibrated part checks.
          mode === "model" ? { modelId, trackPart } : { realWorldHeightM, modelId }
        );
        applyArResult(result, performance.now() - frameStarted);
      } else {
        const result = await VisionApi.estimateFeaturePose(assetId, frame);
        setFeaturePose(result);
        applyCameraModel(result.camera_vertical_fov_deg, result.camera_aspect);
        if (result.found && result.inlier_points_px) {
          drawPoints(result.inlier_points_px, "#ffca28", `${result.num_inliers}/${result.num_matches} matched`);
          if (result.axes) drawAxes(result.axes);
        } else {
          clearOutline();
        }
        applyModelTransform(result.position, result.quaternion);
      }
    } catch (err) {
      setApiError(apiErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (!liveTracking) return;
    let cancelled = false;

    // Self-rescheduling rather than a fixed setInterval: each frame is a
    // camera -> backend -> response round trip, so a fixed interval either
    // wastes time when the backend is fast or stacks up requests when it's
    // slow. While TRACKING the backend only runs the (cheap) tracker, so this
    // sends frames as fast as it answers; the detector's own rate cap while
    // SEARCHING/LOST lives in the backend state machine, not here.
    const loop = async () => {
      while (!cancelled) {
        await detectAndAlign();
        if (cancelled) break;
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
    };
    void loop();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveTracking, mode, targetClassLabel, realWorldHeightM, trackPart]);

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
        <button className={mode === "model" ? "active" : ""} onClick={() => setMode("model")}>
          Model-based CAD (MegaPose, GPU)
        </button>
      </div>

      {mode === "model" && (
        <div className="registration-controls">
          <label>
            Object class
            <ClassSelect value={targetClassLabel} onChange={setTargetClassLabel} />
          </label>
          {parts.length > 1 && (
            <label title="Match only this part's shape; the whole assembly is still drawn. Pick a part that stays on the object (e.g. the body), so tracking holds when other parts are removed.">
              Track by
              <select
                value={trackPart ?? ""}
                onChange={(e) => setTrackPart(e.target.value === "" ? null : Number(e.target.value))}
              >
                <option value="">Whole object</option>
                {parts.map((p) => (
                  <option key={p.node_index} value={p.node_index}>
                    {p.display_name}
                  </option>
                ))}
              </select>
            </label>
          )}
          <button onClick={() => void resetTracking()}>Reset tracking</button>
        </div>
      )}

      {mode === "markerless" && (
        <div className="registration-controls">
          <label>
            Object class
            <ClassSelect value={targetClassLabel} onChange={setTargetClassLabel} />
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
          <button onClick={() => void resetTracking()}>Reset tracking</button>
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
        {(mode === "markerless" || mode === "model") && (arResult?.calibrated_parts ?? []).length > 0 && (
          <div className="part-badges">
            {arResult!.calibrated_parts.map((name) => {
              const check = (arResult!.part_checks ?? []).find((c) => c.part_name === name);
              const cls = check ? `part-badge part-badge-${check.state}` : "part-badge part-badge-waiting";
              const text = !check
                ? `⏳ ${name}: waiting for tracking`
                : check.state === "present"
                  ? `✅ ${name}: present`
                  : check.state === "absent"
                    ? `❌ ${name}: REMOVED`
                    : `❓ ${name}: unsure`;
              return (
                <div key={name} className={cls}>
                  {text}
                  {check && <span> {(check.confidence * 100).toFixed(0)}%</span>}
                </div>
              );
            })}
          </div>
        )}
      </div>

      <CameraStatusBadge status={cameraStatus} deviceLabel={deviceLabel} resolution={resolution} error={cameraError} />
      <CameraSelect devices={devices} selectedDeviceId={selectedDeviceId} onSelect={selectDevice} />

      {modelStatus === "loading" && <p className="camera-status camera-status-pending">🟡 Loading 3D model…</p>}
      {modelStatus === "loaded" && <p className="camera-status camera-status-ok">🟢 3D model loaded once — shown while the object is tracked</p>}
      {modelStatus === "error" && (
        <p className="camera-status camera-status-error">🔴 3D model failed to load — {modelError}</p>
      )}

      <div className="registration-controls">
        <button onClick={detectAndAlign} disabled={busy || !ready || liveTracking}>
          {mode === "markerless" || mode === "model"
            ? arResult?.state === "TRACKING"
              ? "Track one frame"
              : "Detect & track one frame"
            : busy
              ? "Detecting…"
              : "Detect & Align"}
        </button>
        <label>
          <input type="checkbox" checked={liveTracking} onChange={(e) => setLiveTracking(e.target.checked)} />
          {mode === "markerless" || mode === "model"
            ? "Live tracking (detect once, then track every frame)"
            : "Live tracking (as fast as detection responds, smoothed)"}
        </label>
        <span className="save-frame" title="Saves the raw camera frame (full resolution, without any overlay) to data/debug_frames/ for analysis">
          <input value={frameLabel} onChange={(e) => setFrameLabel(e.target.value)} placeholder="label" />
          <button onClick={() => void saveFrame()} disabled={!ready}>
            Save frame
          </button>
          {frameSaved && <span className="hint">{frameSaved}</span>}
        </span>
        <span className="view-style-toggle" title="How the 3D model is drawn — X-Ray and Wireframe let you see the real object through it to judge alignment">
          View:
          {(
            [
              ["solid", "Solid"],
              ["wireframe", "Wireframe"],
              ["xray", "X-Ray"],
            ] as const
          ).map(([value, label]) => (
            <button key={value} className={renderStyle === value ? "active" : ""} onClick={() => setRenderStyle(value)}>
              {label}
            </button>
          ))}
        </span>
      </div>

      {mode === "markerless" && (
        <p className="warning">
          Approximate: position only, estimated from the object's apparent size — no orientation is
          estimated (the model is always placed upright, as authored). Accuracy depends on the
          "Real height" value above being correct and on camera calibration; run
          `python -m app.workers.calibrate_camera` for a real one instead of the default approximation.
        </p>
      )}

      {(mode === "marker" || mode === "feature") && (
        <div className="anchor-calibration">
          <p className="hint">
            The pose above is the tracked <em>reference patch's</em> pose (a marker's face, or the registered
            photo's plane) — not necessarily where the 3D model's own origin should sit. If the model looks
            offset, oversized-looking, or tilted relative to the real object even though tracking itself
            looks stable, nudge it here until it lines up, then save — this is calibrated once per model,
            not per session.
          </p>
          <div className="anchor-calibration-grid">
            <div>
              <strong>Position (m)</strong>
              {(["x", "y", "z"] as const).map((axis) => (
                <div key={axis} className="anchor-row">
                  <span>{axis.toUpperCase()}: {anchorPos[axis].toFixed(3)}</span>
                  <button onClick={() => nudgePos(axis, -0.01)}>−</button>
                  <button onClick={() => nudgePos(axis, 0.01)}>+</button>
                </div>
              ))}
            </div>
            <div>
              <strong>Rotation (°)</strong>
              {(["rx", "ry", "rz"] as const).map((axis) => (
                <div key={axis} className="anchor-row">
                  <span>{axis.toUpperCase()}: {anchorEulerDeg[axis].toFixed(0)}</span>
                  <button onClick={() => nudgeRot(axis, -5)}>−</button>
                  <button onClick={() => nudgeRot(axis, 5)}>+</button>
                </div>
              ))}
            </div>
          </div>
          <div className="anchor-calibration-actions">
            <button onClick={saveAnchor} disabled={anchorSaving}>
              {anchorSaving ? "Saving…" : "Save alignment"}
            </button>
            <button onClick={resetAnchor}>Reset</button>
            {anchorSaved && <span className="camera-status-ok">Saved</span>}
          </div>
        </div>
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
              {markerPose.rotation_deg && (
                <p>
                  rotation: Rx={markerPose.rotation_deg.rx.toFixed(1)}° Ry={markerPose.rotation_deg.ry.toFixed(1)}°
                  Rz={markerPose.rotation_deg.rz.toFixed(1)}°
                </p>
              )}
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

      {mode === "feature" && !registration && (
        <p className="warning">Register a reference image of the object's labeled/textured surface first.</p>
      )}

      {mode === "model" && (
        <p className="hint">
          Matches this 3D model's own shape against the camera image (MegaPose on the GPU pose service), so the
          model lands on the real object itself — no marker, reference photo, or alignment offset. The object
          is detected once (YOLO + a full pose search, ~1s); after that only the tracker runs, refining from the
          last pose every frame, and the detector runs again only if tracking is lost. Needs the pose service
          running in WSL (see <code>pose_service/README.md</code>), and the model's real-world size must be
          correct, since distance is inferred from it.
        </p>
      )}

      {(mode === "markerless" || mode === "model") && (arResult || arEvents.length > 0) && (
        <div
          className={`pose-status ${
            arResult?.state === "TRACKING"
              ? "pose-found"
              : arResult?.state === "LOST" || arResult?.state === "RECOVERING"
                ? "pose-lost"
                : "pose-not-found"
          }`}
        >
          {arResult && (
            <>
              <p className="tracking-state">
                {arResult.state === "SEARCHING" && `🔵 Searching for ${targetClassLabel || "object"}...`}
                {arResult.state === "INITIALIZING" && "🟡 Initializing pose..."}
                {arResult.state === "TRACKING" &&
                  `🟢 Tracking: ${arResult.object?.class_label}` +
                    (arResult.monitoring ? " — ⚠ low confidence, monitoring" : "")}
                {arResult.state === "LOST" && "🟠 Tracking lost — holding last pose"}
                {arResult.state === "RECOVERING" && "🔵 Reacquiring object... (model held at last pose)"}
              </p>
              {arResult.object && (
                <p>
                  Confidence: <strong>{(arResult.object.confidence * 100).toFixed(0)}%</strong>{" "}
                  <span className="hint">
                    {mode === "model" ? "model ↔ real object overlap" : "share of tracked points still agreeing"} (good ≥{" "}
                    {(arResult.good_confidence * 100).toFixed(0)}%, lost &lt; {(arResult.lost_confidence * 100).toFixed(0)}%)
                  </span>
                </p>
              )}
              {(arResult.part_checks ?? []).map((check) => (
                <p key={check.node_index} className={`part-check part-${check.state}`}>
                  {check.state === "present" ? "✅" : check.state === "absent" ? "❌" : "❓"} {check.part_name}:{" "}
                  <strong>{check.state === "absent" ? "removed" : check.state}</strong>{" "}
                  <span className="hint">
                    ({(check.confidence * 100).toFixed(0)}% sure, region brightness {check.brightness.toFixed(0)})
                  </span>
                </p>
              ))}
              <div className="ar-stats">
                <span>Object ID</span>
                <strong>{arResult.object?.object_id ?? "—"}</strong>
                <span>Detection count</span>
                <strong>{arResult.counters.detection_count}</strong>
                <span>Detector runs</span>
                <strong>{arResult.counters.detection_runs}</strong>
                <span>Tracking frames</span>
                <strong>{arResult.counters.tracking_frames}</strong>
                <span>Frames since detection</span>
                <strong>{arResult.counters.frames_since_detection ?? "—"}</strong>
                <span>Detection / Tracking / Rendering</span>
                <strong>
                  {rates.detect.toFixed(1)}/s · {rates.track.toFixed(1)}/s · {rates.render} fps
                </strong>
              </div>
              <p className="hint">
                This frame — detection {arResult.timings_ms.detection.toFixed(0)} ms · initialization{" "}
                {arResult.timings_ms.initialization.toFixed(0)} ms · tracking {arResult.timings_ms.tracking.toFixed(0)} ms ·
                refinement {arResult.timings_ms.refinement.toFixed(0)} ms · server total{" "}
                {arResult.timings_ms.total.toFixed(0)} ms · rendering {rates.renderMs.toFixed(1)} ms/frame · total frame{" "}
                {frameMs.toFixed(0)} ms
              </p>
              {arResult.position && arResult.rotation_deg && (
                <p className="pose-readout">
                  X={arResult.position.x.toFixed(3)} Y={arResult.position.y.toFixed(3)} Z={arResult.position.z.toFixed(3)} m
                  · Rx={arResult.rotation_deg.rx.toFixed(1)}° Ry={arResult.rotation_deg.ry.toFixed(1)}° Rz=
                  {arResult.rotation_deg.rz.toFixed(1)}°
                  <span className="hint">
                    {" "}
                    (filtered, renderer frame: +X right, +Y up, camera looks down −Z
                    {arResult.approximate ? "; markerless: position only, no rotation" : ""})
                  </span>
                </p>
              )}
              {arResult.calibration_is_approximate && (
                <p className="hint">Using an approximate default camera calibration ({arResult.calibration_source}).</p>
              )}
            </>
          )}
          {arEvents.length > 0 && (
            <pre className="tracking-log">{arEvents.join("\n")}</pre>
          )}
        </div>
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
              {featurePose.rotation_deg && (
                <p>
                  rotation: Rx={featurePose.rotation_deg.rx.toFixed(1)}° Ry={featurePose.rotation_deg.ry.toFixed(1)}°
                  Rz={featurePose.rotation_deg.rz.toFixed(1)}°
                </p>
              )}
              {featurePose.num_matches > 0 && featurePose.num_inliers / featurePose.num_matches < 0.25 && (
                <p className="warning">
                  Low inlier ratio ({Math.round((100 * featurePose.num_inliers) / featurePose.num_matches)}%) — the
                  rotation solve is likely unstable. This usually means the reference photo covers a curved surface
                  (e.g. a mug's label wrapping around it) rather than a flat one, or was taken from an angle that
                  doesn't match the current view. Re-register using a reference photo of the flattest, most
                  front-on part of the surface.
                </p>
              )}
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
