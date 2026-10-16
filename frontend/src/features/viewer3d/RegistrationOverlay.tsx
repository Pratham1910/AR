import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { CameraSelect } from "../../components/CameraSelect";
import { CameraStatusBadge } from "../../components/CameraStatusBadge";
import { useCamera } from "../../hooks/useCamera";
import { addStudioEnvironment, ensureVisibleMaterials } from "./ensureVisibleMaterial";
import { applyRenderStyle, type RenderStyle } from "./renderStyle";
import { applyPartView, EMPTY_PART_VIEW, indexParts, type PartView } from "./parts";
import type { ProcedureHost, StepGuide } from "./procedure";
import { FitCheck } from "./FitCheck";
import { QAInspectionPanel } from "./QAInspectionPanel";
import { ComponentInspectionPanel } from "./ComponentInspectionPanel";
import { canvasMarkerTextures, componentColor, DiscrepancyRenderer, STATUS_COLOR } from "./discrepancy";
import { BoxSelector } from "./BoxSelector";
import { CalibrationPanel } from "./CalibrationPanel";
import { dolly, matrixToRows, projectAxes, rotateAbout, rowsToMatrix, slide, tilt, toThreePose, turn, type ModelAxis } from "./guidePose";
import {
  InspectionReportsApi,
  Models3DApi,
  StationApi,
  StationSettingsApi,
  VisionApi,
  apiErrorMessage,
  apiWebSocketUrl,
  GuideViewApi,
} from "../../services/api";
import { readWorkstation } from "../../workstation";
import { ClassSelect } from "../../components/ClassSelect";
import type {
  AnchorOffset,
  ARFrameResponse,
  CameraIntrinsics,
  ComponentRegistry,
  InspectionMode,
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
  /** Plays an attached procedure on the overlaid model and receives the camera's part checks. */
  procedureHost?: ProcedureHost;
  /** Shown in the sidebar's Procedure tab (the page's procedure and parts panels). */
  sidebarExtra?: ReactNode;
  /** The page toolbar's slot: the tracking controls render there, in the same row as asset and view. */
  toolbarSlot?: HTMLElement | null;
  /** Opens a saved inspection report (Reports screen). */
  onOpenReport?: (reportId: string) => void;
}

type RegistrationMode = "marker" | "markerless" | "feature" | "model";

const AR_MODES = [
  [
    "model",
    "Model-based (CAD)",
    "Finds and follows the object by its 3D model alone (GPU pose service in WSL): detects once, then tracks every frame. Works for any object with a GLB; the model's real size must be set correctly.",
  ],
  [
    "markerless",
    "Markerless",
    "Follows the object with optical flow and places the model from its apparent size — position only; the model stays upright.",
  ],
  [
    "marker",
    "ArUco marker",
    "Accurate 6DoF from a printed marker next to the object (python -m app.workers.generate_marker; set ARUCO_MARKER_LENGTH_M to its printed size).",
  ],
  ["feature", "Image target", "6DoF from a registered photo of a flat, textured surface on the object (e.g. a label)."],
] as const;

const STATE_LABEL: Record<ARFrameResponse["state"], string> = {
  SEARCHING: "Searching…",
  INITIALIZING: "Initializing pose…",
  TRACKING: "Tracking",
  TRACKING_OCCLUDED: "Occluded — holding last pose",
  LOST: "Tracking lost — holding last pose",
  RECOVERING: "Re-acquiring…",
};

const DIAGNOSTICS_KEY = "tvasta.ar.diagnosticsOpen";

const newSessionId = () => `s-${Math.random().toString(36).slice(2)}`;

/** Chaikin corner cutting: a rounder closed outline through the same shape (display only). */
function chaikin(points: Vector2[], iterations: number): Vector2[] {
  let out = points;
  for (let k = 0; k < iterations && out.length >= 3; k++) {
    const next: Vector2[] = [];
    for (let i = 0; i < out.length; i++) {
      const a = out[i];
      const b = out[(i + 1) % out.length];
      next.push({ x: 0.75 * a.x + 0.25 * b.x, y: 0.75 * a.y + 0.25 * b.y });
      next.push({ x: 0.25 * a.x + 0.75 * b.x, y: 0.25 * a.y + 0.75 * b.y });
    }
    out = next;
  }
  return out;
}

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
  procedureHost,
  sidebarExtra,
  onOpenReport,
  toolbarSlot,
}: Props) {
  // Model-based only: match just this part (e.g. the body, which looks the
  // same with the cap on or off); the whole assembly is still rendered.
  const [trackPart, setTrackPart] = useState<number | null>(null);
  // Which side of the model faces where (settles top/bottom for symmetric
  // objects like a flat box). Saved per model on the server: it's a property
  // of the station, the same in every browser.
  const [resting, setRestingState] = useState<"any" | "camera" | "upright" | "flat">("any");
  useEffect(() => {
    StationSettingsApi.get(modelId)
      .then((s) => setRestingState(s.resting))
      .catch(() => undefined);
  }, [modelId]);
  const setResting = (value: "any" | "camera" | "upright" | "flat") => {
    setRestingState(value);
    StationSettingsApi.save(modelId, { resting: value }).catch((err) => setApiError(apiErrorMessage(err)));
  };
  // The procedure's current step, shown over the video.
  const [guide, setGuide] = useState<StepGuide | null>(procedureHost?.guide ?? null);
  useEffect(() => {
    if (!procedureHost) return;
    return procedureHost.subscribe((event) => {
      if (event.type === "guide") setGuide(event.guide);
    });
  }, [procedureHost]);
  const [diagnosticsOpen, setDiagnosticsOpenState] = useState(() => {
    try {
      return localStorage.getItem(DIAGNOSTICS_KEY) === "1";
    } catch {
      return false;
    }
  });
  const setDiagnosticsOpen = (open: boolean) => {
    setDiagnosticsOpenState(open);
    try {
      localStorage.setItem(DIAGNOSTICS_KEY, open ? "1" : "0");
    } catch {
      // a per-viewer convenience only
    }
  };
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
    captureFrameJpeg,
    retry: retryCamera,
    takeOver: takeOverCamera,
    retryNote: cameraRetryNote,
    otherTabHasCamera,
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
  // Streamed poses carry their frame's capture time: the last one plus its
  // velocity lets every rendered frame draw the pose for "now" (see animate).
  const motionRef = useRef<{
    t: number; // capture time (performance.now() clock)
    p: THREE.Vector3;
    q: THREE.Quaternion;
    v: THREE.Vector3; // m/s, renderer frame
    w: THREE.Vector3; // rad/s, axis * speed
  } | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

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

  const [mode, setMode] = useState<RegistrationMode>("model");
  const [targetClassLabel, setTargetClassLabel] = useState(defaultTargetClassLabel || "");
  // Model-based mode: find the object by its 3D model (no class needed — works
  // for anything with a GLB, e.g. a junction box), or by a detector class.
  // How the object is found (both markerless and model-based): by its 3D model,
  // by a detector class, or from a box drawn around it on the video — the last
  // works for any object, with no class or model.
  const [detectBy, setDetectBy] = useState<"model" | "class" | "box" | "guide">("model");
  // Guide view on screen: the model drawn as an outline at the guide pose (X-Ray style).
  const guideShownRef = useRef(false);
  const [guideBusy, setGuideBusy] = useState(false);
  // Guide outline placed by the operator: T_camera_object (OpenCV camera
  // frame, metres) as a THREE.Matrix4. Moved here in the browser for smooth
  // dragging, and sent to the backend (stream config) so the edge check and
  // the green/red feedback follow it. null = the saved / default outline.
  const guidePlacedRef = useRef<THREE.Matrix4 | null>(null);
  const [guidePlaced, setGuidePlaced] = useState(false);
  const guideDragRef = useRef<{ x: number; y: number; shift: boolean; alt: boolean } | null>(null);
  // The model axis the rotate buttons and Alt+drag turn about.
  const [guideAxis, setGuideAxis] = useState<ModelAxis>("y");
  // How closely the object must match the outline to lock (0.2-0.9), the
  // operator's choice per model, remembered in this browser; null = default.
  const [guideLockAt, setGuideLockAtState] = useState<number | null>(null);
  useEffect(() => {
    try {
      const saved = modelId ? localStorage.getItem(`tvasta.guideLockAt.${modelId}`) : null;
      setGuideLockAtState(saved ? Number(saved) : null);
    } catch {
      setGuideLockAtState(null);
    }
  }, [modelId]);
  const setGuideLockAt = (value: number | null) => {
    setGuideLockAtState(value);
    try {
      if (modelId && value != null) localStorage.setItem(`tvasta.guideLockAt.${modelId}`, String(value));
      else if (modelId) localStorage.removeItem(`tvasta.guideLockAt.${modelId}`);
    } catch {
      // private window / blocked storage: just not remembered
    }
  };
  const guideAxisRef = useRef<ModelAxis>("y");
  guideAxisRef.current = guideAxis;
  const guideSentAtRef = useRef(0);
  const [initBox, setInitBox] = useState<[number, number, number, number] | null>(null);
  const needsClass = (mode === "markerless" || mode === "model") && detectBy === "class";
  const needsBox = (mode === "markerless" || mode === "model") && detectBy === "box";
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
  const sessionIdRef = useRef(newSessionId());
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
  // Clip recording for the replay benchmark (app/workers/replay_clip.py).
  const [clipName, setClipName] = useState("");
  const [recording, setRecording] = useState<{ frames: number; seconds: number; pending: number } | null>(null);
  const [clipSaved, setClipSaved] = useState<string | null>(null);
  const recordTimer = useRef<number | null>(null);
  const uploadChain = useRef<Promise<unknown>>(Promise.resolve());
  const uploadFailures = useRef<{ count: number; message: string }>({ count: 0, message: "" });
  const RECORD_FPS = 15;
  const RECORD_MAX_S = 60;
  const stopRecording = () => {
    if (recordTimer.current !== null) window.clearInterval(recordTimer.current);
    recordTimer.current = null;
    const name = clipName.trim();
    void uploadChain.current.then(() => {
      const failed = uploadFailures.current;
      setRecording((r) => {
        if (r) {
          setClipSaved(
            failed.count > 0
              ? `Clip "${name}" NOT saved: ${failed.count} of ${r.frames} frames failed to upload (${failed.message})`
              : `Saved clip "${name}" — ${r.frames} frames, ${r.seconds.toFixed(1)} s`
          );
        }
        return null;
      });
    });
  };
  const startRecording = () => {
    const name = clipName.trim();
    if (!name) return;
    setClipSaved(null);
    uploadFailures.current = { count: 0, message: "" };
    const started = performance.now();
    let index = 0;
    setRecording({ frames: 0, seconds: 0, pending: 0 });
    recordTimer.current = window.setInterval(() => {
      const tMs = performance.now() - started;
      if (tMs > RECORD_MAX_S * 1000) {
        stopRecording();
        return;
      }
      const frame = captureFrameBase64();
      if (!frame) return;
      const i = index++;
      setRecording((r) => (r ? { frames: i + 1, seconds: tMs / 1000, pending: r.pending + 1 } : r));
      // Strictly in order: frame 0 resets the clip on the server.
      uploadChain.current = uploadChain.current
        .then(() => VisionApi.addClipFrame(name, i, tMs, frame))
        .catch((err) => {
          uploadFailures.current = { count: uploadFailures.current.count + 1, message: apiErrorMessage(err) };
        })
        .finally(() => setRecording((r) => (r ? { ...r, pending: r.pending - 1 } : r)));
    }, 1000 / RECORD_FPS);
  };
  useEffect(() => () => {
    if (recordTimer.current !== null) window.clearInterval(recordTimer.current);
  }, []);

  // No default: a pre-filled "cap-on" got cap-off frames saved under the wrong label.
  const [frameLabel, setFrameLabel] = useState("");
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
  const [sideTab, setSideTab] = useState<"inspect" | "procedure" | "tracking" | "tools">("inspect");
  // Fixed-station mode: this model's pose is locked (camera + fixtured part don't move).
  const [stationLocked, setStationLocked] = useState(false);
  // Where the operator last clicked to pick the object (stage %, shown briefly).
  const [pickMark, setPickMark] = useState<{ left: number; top: number } | null>(null);
  const [stationBusy, setStationBusy] = useState(false);
  // Inspect only the current procedure step's components.
  const [stepFocus, setStepFocus] = useState(true);
  // "Save report": records the live inspection (Reports screen).
  const [reportStatus, setReportStatus] = useState<{ id?: string; error?: string; saving?: boolean } | null>(null);
  const saveReport = async () => {
    setReportStatus({ saving: true });
    try {
      const { station, operator } = readWorkstation();
      const report = await InspectionReportsApi.save(sessionIdRef.current, station || null, operator || null);
      setReportStatus({ id: report.id });
    } catch (err) {
      setReportStatus({ error: apiErrorMessage(err) });
    }
  };
  // Debug: draw the hand mask the tracker excludes (read by the stream handler).
  const [showHands, setShowHands] = useState(false);
  const showHandsRef = useRef(showHands);
  showHandsRef.current = showHands;
  // Phase 8 diagnostics: pose features predicted (dashed) vs found (solid), model corners.
  const [showPoseFeatures, setShowPoseFeatures] = useState(true);
  const showPoseFeaturesRef = useRef(showPoseFeatures);
  showPoseFeaturesRef.current = showPoseFeatures;
  const [calibrationOpen, setCalibrationOpen] = useState(false);

  // Component inspection (model-based mode): runs on the tracking stream's
  // frames; results come back with each frame. Kept apart from the overlay
  // itself — its markers are drawn by DiscrepancyRenderer, cleared when off.
  const [inspectionMode, setInspectionMode] = useState<InspectionMode>("off");
  const [inspectionRun, setInspectionRun] = useState(() => newSessionId());
  const [componentRegistry, setComponentRegistry] = useState<ComponentRegistry | null>(null);
  const [registryError, setRegistryError] = useState<string | null>(null);
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
  const [showAllMarkers, setShowAllMarkers] = useState(false);
  const [inspectionDebug, setInspectionDebug] = useState(false);
  // Each component's outline with its name on the video, coloured by status.
  const [outlineNames, setOutlineNames] = useState(true);
  const discrepancyRef = useRef<DiscrepancyRenderer | null>(null);
  // Read by the stream's message handler, which outlives renders.
  const inspectionViewRef = useRef({ selected: selectedComponent, showAll: showAllMarkers, debug: inspectionDebug, names: outlineNames });
  inspectionViewRef.current = { selected: selectedComponent, showAll: showAllMarkers, debug: inspectionDebug, names: outlineNames };
  const componentRegistryRef = useRef<ComponentRegistry | null>(null);
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
    const removeEnvironment = addStudioEnvironment(scene, renderer);
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
        procedureHost?.attach(partsRef.current);
        discrepancyRef.current?.dispose();
        discrepancyRef.current = new DiscrepancyRenderer(group, partsRef.current, canvasMarkerTextures);
        if (componentRegistryRef.current) discrepancyRef.current.setComponents(componentRegistryRef.current.components);
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
    // Streamed poses: predict at most this far past the newest pose's capture
    // time, and follow the prediction this quickly (hides correction steps).
    const MAX_PREDICT_MS = 100;
    const PREDICTED_FOLLOW = 0.5;
    const predictedPosition = new THREE.Vector3();
    const predictedQuaternion = new THREE.Quaternion();
    const turn = new THREE.Quaternion();
    const axis = new THREE.Vector3();
    const animate = () => {
      frameId = requestAnimationFrame(animate);

      const model = modelObjectRef.current;
      const motion = motionRef.current;
      const age = motion ? performance.now() - motion.t : Infinity;
      if (model && hasTargetRef.current && motion && age < 2 * MAX_PREDICT_MS) {
        // The newest pose is from a frame captured `age` ms ago: move it on by
        // its velocity, so the model is where the object is now, at the
        // screen's refresh rate rather than the camera's.
        const horizon = Math.min(age, MAX_PREDICT_MS) / 1000;
        predictedPosition.copy(motion.p).addScaledVector(motion.v, horizon);
        predictedQuaternion.copy(motion.q);
        const angle = motion.w.length() * horizon;
        if (angle > 1e-6) predictedQuaternion.premultiply(turn.setFromAxisAngle(axis.copy(motion.w).normalize(), angle));
        model.position.lerp(predictedPosition, PREDICTED_FOLLOW);
        model.quaternion.slerp(predictedQuaternion, PREDICTED_FOLLOW);
      } else if (model && hasTargetRef.current) {
        model.position.lerp(targetPositionRef.current, INTERPOLATION);
        model.quaternion.slerp(targetQuaternionRef.current, INTERPOLATION);
      }

      procedureHost?.tick?.(performance.now());
      const started = performance.now();
      renderer.render(scene, camera);
      renderMsRef.current += performance.now() - started;
      renderFramesRef.current += 1;
    };
    animate();

    return () => {
      cancelAnimationFrame(frameId);
      removeEnvironment();
      procedureHost?.detach(partsRef.current);
      discrepancyRef.current?.dispose();
      discrepancyRef.current = null;
      resizeObserver.disconnect();
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, [modelUrl, modelScale, procedureHost]);

  const clearOutline = () => {
    const ctx = outlineCanvasRef.current?.getContext("2d");
    const canvas = outlineCanvasRef.current;
    if (ctx && canvas) ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const drawPolygon = (points: Vector2[], color: string, label?: string, smooth = false) => {
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
    // Display only: the tracked outline drawn rounded, without its vertex
    // dots; the geometry the backend works with is unchanged.
    const drawn = smooth ? chaikin(points, 2) : points;
    drawn.forEach((pt, i) => (i === 0 ? ctx.moveTo(pt.x, pt.y) : ctx.lineTo(pt.x, pt.y)));
    ctx.closePath();
    ctx.stroke();

    if (!smooth) {
      ctx.fillStyle = color;
      points.forEach((pt) => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
        ctx.fill();
      });
    }
    ctx.fillStyle = color;

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
    useAnchor = true,
    // Streamed frames: when the frame behind this pose was captured (enables prediction).
    captureMs?: number
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
      if (captureMs !== undefined) {
        // Velocity from the previous streamed pose (lightly smoothed), for prediction.
        const prev = motionRef.current;
        if (wasVisible && prev && captureMs > prev.t && captureMs - prev.t < 250) {
          const dt = (captureMs - prev.t) / 1000;
          const v = finalPosition.clone().sub(prev.p).divideScalar(dt);
          const dq = finalQuaternion.clone().multiply(prev.q.clone().invert());
          if (dq.w < 0) dq.set(-dq.x, -dq.y, -dq.z, -dq.w);
          const angle = 2 * Math.acos(Math.min(1, dq.w));
          const s = Math.sqrt(Math.max(0, 1 - dq.w * dq.w));
          const w = s < 1e-6 ? new THREE.Vector3() : new THREE.Vector3(dq.x / s, dq.y / s, dq.z / s).multiplyScalar(angle / dt);
          motionRef.current = {
            t: captureMs,
            p: finalPosition.clone(),
            q: finalQuaternion.clone(),
            v: prev.v.clone().lerp(v, 0.5),
            w: prev.w.clone().lerp(w, 0.5),
          };
        } else {
          motionRef.current = {
            t: captureMs,
            p: finalPosition.clone(),
            q: finalQuaternion.clone(),
            v: new THREE.Vector3(),
            w: new THREE.Vector3(),
          };
        }
      }
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
      motionRef.current = null; // never predict from a pose that's no longer shown
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
  }, [mode, targetClassLabel, realWorldHeightM, trackPart, detectBy, initBox]);

  // End the backend session when this view goes away (switching model/asset remounts it).
  useEffect(
    () => () => {
      void VisionApi.endArSession(sessionIdRef.current).catch(() => undefined);
    },
    []
  );

  const resetTracking = async () => {
    // Switch to a fresh session rather than only deleting the current one:
    // during live tracking a frame is almost always in flight, and the backend
    // stores that frame's session back when it finishes, undoing the delete.
    const previous = sessionIdRef.current;
    sessionIdRef.current = newSessionId();
    guidePlacedRef.current = null;
    setGuidePlaced(false);
    void VisionApi.endArSession(previous).catch(() => undefined);
    motionRef.current = null;
    if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(streamConfig());
    counterSamplesRef.current = [];
    setArResult(null);
    setArEvents((events) => [...events, "[UI] Tracking reset — searching again"].slice(-12));
    applyModelTransform(null, null);
    clearOutline();
  };

  // Components the inspection has decided also answer the procedure's "part
  // present / removed" facts, for parts without their own calibration.
  const withInspectionChecks = (result: ARFrameResponse) => {
    const checks = [...(result.part_checks ?? [])];
    const registry = componentRegistryRef.current;
    for (const c of result.inspection?.components ?? []) {
      const info = registry?.components.find((r) => r.id === c.id);
      const name = info?.display_name ?? c.id;
      if (checks.some((k) => k.part_name === name)) continue;
      const state = c.status === "PRESENT" ? "present" : c.status === "MISSING_CONFIRMED" ? "absent" : "uncertain";
      checks.push({
        node_index: info?.node_index ?? -1,
        part_name: name,
        state,
        confidence: c.confidence ?? 0,
        brightness: 0,
        region_px: ((c.bbox_px ?? [0, 0, 0, 0]) as [number, number, number, number]),
        measure: "inspection",
        region_polygon: null,
      });
    }
    return checks;
  };

  /** Draw the model at a pose given in the OpenCV camera frame (the guide outline the operator moves). */
  const showGuidePose = (tCameraObject: THREE.Matrix4) => {
    const { position, quaternion } = toThreePose(tCameraObject);
    applyModelTransform(position, quaternion, false);
  };

  const applyArResult = (result: ARFrameResponse, roundTripMs: number) => {
    setArResult(result);
    procedureHost?.publishChecks(withInspectionChecks(result));
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
      const held =
        result.state === "LOST" || result.state === "RECOVERING" || result.state === "INITIALIZING" || result.state === "TRACKING_OCCLUDED";
      const color = held ? "#ff7043" : result.monitoring ? "#ffca28" : "#00e676";
      const occluded = result.quality?.occlusion ? ` · ${(result.quality.occlusion * 100).toFixed(0)}% covered` : "";
      const label = `${result.state === "TRACKING_OCCLUDED" ? "occluded" : held ? "held" : "tracking"} ${obj.class_label} #${obj.object_id} ${(obj.confidence * 100).toFixed(0)}%${occluded}`;
      if (obj.polygon && obj.polygon.length >= 3) {
        drawPolygon(obj.polygon, color, label, true);
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
    // Debug: the hand mask kept out of tracking and inspection.
    const handCtx = outlineCanvasRef.current?.getContext("2d");
    if (handCtx && showHandsRef.current && result.hands?.outlines.length) {
      handCtx.fillStyle = "rgba(255, 64, 129, 0.28)";
      handCtx.strokeStyle = "#ff4081";
      handCtx.lineWidth = 2;
      for (const outline of result.hands.outlines) {
        handCtx.beginPath();
        outline.forEach((p, i) => (i === 0 ? handCtx.moveTo(p.x, p.y) : handCtx.lineTo(p.x, p.y)));
        handCtx.closePath();
        handCtx.fill();
        handCtx.stroke();
      }
    }
    // Guide view: each outline edge point green where it already sits on a
    // real edge of the object, red where it doesn't — shows which side to move.
    const guideCtx = outlineCanvasRef.current?.getContext("2d");
    if (guideCtx && !result.visible && result.guide?.edge_points?.length) {
      for (const [x, y, on] of result.guide.edge_points) {
        guideCtx.fillStyle = on ? "rgba(0, 230, 118, 0.95)" : "rgba(255, 82, 82, 0.9)";
        guideCtx.beginPath();
        guideCtx.arc(x, y, on ? 2.6 : 2.2, 0, Math.PI * 2);
        guideCtx.fill();
      }
    }

    // Guide view: the outline's own axes (X red, Y green, Z blue; the picked
    // one thicker), so the rotate buttons have something to refer to.
    if (guideCtx && !result.visible && result.guide && result.intrinsics) {
      const m = guidePlacedRef.current ?? (result.guide.t_camera_object ? rowsToMatrix(result.guide.t_camera_object) : null);
      const axes = m ? projectAxes(m, result.intrinsics, 0.07) : null;
      if (axes) {
        const colours: Record<ModelAxis, string> = { x: "#ff5252", y: "#00e676", z: "#448aff" };
        for (const axis of ["x", "y", "z"] as ModelAxis[]) {
          guideCtx.strokeStyle = colours[axis];
          guideCtx.lineWidth = guideAxisRef.current === axis ? 5 : 2.5;
          guideCtx.beginPath();
          guideCtx.moveTo(...axes.o);
          guideCtx.lineTo(...axes[axis]);
          guideCtx.stroke();
          guideCtx.fillStyle = colours[axis];
          guideCtx.font = "bold 15px 'IBM Plex Sans', sans-serif";
          guideCtx.fillText(axis.toUpperCase(), axes[axis][0] + 4, axes[axis][1] - 4);
        }
      }
    }

    // Pose features: where the pose puts each one (dashed ring, +) and the
    // circle actually found there (solid dot), so a wrong orientation shows
    // as a ring on empty surface and a dot somewhere else.
    const featureCtx = outlineCanvasRef.current?.getContext("2d");
    if (featureCtx && showPoseFeaturesRef.current && result.pose_features?.length) {
      const FEATURE_COLOR: Record<string, string> = {
        detected: "#00e676",
        not_detected: "#ff5252",
        occluded: "#9e9e9e",
        hidden: "#9e9e9e",
        facing_away: "#9e9e9e",
        outside: "#9e9e9e",
      };
      for (const f of result.pose_features) {
        const color = FEATURE_COLOR[f.status] ?? "#ffca28";
        featureCtx.strokeStyle = color;
        featureCtx.lineWidth = 2;
        if (f.predicted_contour_px.length >= 3 && !["facing_away", "outside"].includes(f.status)) {
          featureCtx.setLineDash([6, 4]);
          featureCtx.beginPath();
          f.predicted_contour_px.forEach((p, i) => (i === 0 ? featureCtx.moveTo(p.x, p.y) : featureCtx.lineTo(p.x, p.y)));
          featureCtx.closePath();
          featureCtx.stroke();
          featureCtx.setLineDash([]);
        }
        const c = f.predicted_center_px;
        if (c) {
          featureCtx.beginPath();
          featureCtx.moveTo(c.x - 7, c.y);
          featureCtx.lineTo(c.x + 7, c.y);
          featureCtx.moveTo(c.x, c.y - 7);
          featureCtx.lineTo(c.x, c.y + 7);
          featureCtx.stroke();
          featureCtx.font = "bold 13px 'IBM Plex Sans', sans-serif";
          featureCtx.fillStyle = color;
          const residual = f.residual_px != null ? ` · ${f.residual_px.toFixed(1)} px` : "";
          featureCtx.fillText(`${f.feature_id}: ${f.status.replace("_", " ")}${residual}`, c.x + 10, c.y - 10);
        }
        if (f.observed_center_px) {
          featureCtx.fillStyle = "#00e676";
          featureCtx.beginPath();
          featureCtx.arc(f.observed_center_px.x, f.observed_center_px.y, 4, 0, Math.PI * 2);
          featureCtx.fill();
        }
      }
    }

    // The exact region each part check measured, colored by its verdict
    // (layered on top of the outline, like the axes).
    const ctx = outlineCanvasRef.current?.getContext("2d");
    if (ctx) {
      for (const check of result.part_checks ?? []) {
        const [x1, y1, x2, y2] = check.region_px;
        const color = check.state === "present" ? "#00e676" : check.state === "absent" ? "#ff1744" : "#ffca28";
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        if (check.region_polygon && check.region_polygon.length >= 3) {
          // The part's exact outline at the tracked pose — what was measured.
          ctx.beginPath();
          check.region_polygon.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
          ctx.closePath();
          ctx.stroke();
        } else {
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        }
        ctx.fillStyle = color;
        ctx.font = "bold 18px sans-serif";
        ctx.fillText(`${check.part_name}: ${check.state}`, x2 + 6, (y1 + y2) / 2 + 6);
      }
    }

    // Component inspection: markers on the model, and (debug view) what each
    // component's check looked at in this frame.
    const inspection = result.inspection;
    if (inspection) {
      discrepancyRef.current?.update(inspection.components, inspectionViewRef.current);
      if (ctx && inspectionViewRef.current.names && !inspectionViewRef.current.debug) {
        // Outline + name per component, in its status colour.
        const registered = componentRegistryRef.current?.components ?? [];
        const names = new Map(registered.map((r) => [r.id, r.display_name]));
        const own = new Map(registered.map((r, i) => [r.id, componentColor(r, i)]));
        const prefix: Partial<Record<string, string>> = {
          MISSING_CONFIRMED: "MISSING: ",
          MISSING_CANDIDATE: "CHECKING: ",
          OCCLUDED: "OCCLUDED: ",
          UNCERTAIN: "? ",
        };
        for (const c of inspection.components) {
          if (!c.outline_px || c.outline_px.length < 3) continue;
          const missing = c.status === "MISSING_CONFIRMED";
          // Each part in its own colour; a missing / checking part switches to red / amber.
          const color =
            missing || c.status === "MISSING_CANDIDATE" ? STATUS_COLOR[c.status] : own.get(c.id) ?? STATUS_COLOR[c.status];
          ctx.lineWidth = missing || c.id === inspectionViewRef.current.selected ? 4 : 2.5;
          ctx.strokeStyle = color;
          ctx.beginPath();
          c.outline_px.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
          ctx.closePath();
          if (missing) {
            ctx.fillStyle = "rgba(255, 92, 92, 0.22)";
            ctx.fill();
          }
          ctx.stroke();
          const label = `${c.status === "PRESENT" ? "✓ " : prefix[c.status] ?? ""}${names.get(c.id) ?? c.id}`;
          const [x1, y1] = c.bbox_px ?? [c.outline_px[0][0], c.outline_px[0][1]];
          ctx.font = "bold 18px 'IBM Plex Sans', sans-serif";
          const w = ctx.measureText(label).width + 12;
          const ly = Math.max(24, y1 - 8);
          ctx.fillStyle = "rgba(5, 8, 11, 0.8)";
          ctx.fillRect(x1, ly - 19, w, 26);
          ctx.fillStyle = color;
          ctx.fillText(label, x1 + 6, ly);
        }
      }
      if (ctx && inspectionViewRef.current.debug) {
        for (const c of inspection.components) {
          if (!c.bbox_px) continue;
          const color = STATUS_COLOR[c.status];
          ctx.strokeStyle = color;
          ctx.lineWidth = c.id === inspectionViewRef.current.selected ? 3 : 1.5;
          if (c.outline_px && c.outline_px.length >= 3) {
            ctx.beginPath();
            c.outline_px.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
            ctx.closePath();
            ctx.stroke();
          }
          const [x1, y1, x2, y2] = c.bbox_px;
          ctx.setLineDash([4, 3]);
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          ctx.setLineDash([]);
          if (c.center_px) {
            const [cx, cy] = c.center_px;
            ctx.beginPath();
            ctx.moveTo(cx - 5, cy);
            ctx.lineTo(cx + 5, cy);
            ctx.moveTo(cx, cy - 5);
            ctx.lineTo(cx, cy + 5);
            ctx.stroke();
          }
          ctx.fillStyle = color;
          ctx.font = "bold 13px sans-serif";
          ctx.fillText(`${c.id} ${c.visibility ?? ""}`, x2 + 4, y1 + 12);
        }
      }
    } else {
      discrepancyRef.current?.clear();
    }

    // Hidden only once the session gives up (SEARCHING); while LOST the
    // backend keeps returning the last valid pose, so the model stays put
    // instead of flickering on a missed frame.
    // Short-term prediction only while tracking is healthy: a held pose
    // (occluded, lost, re-acquiring) stays put instead of drifting on.
    const healthy = result.state === "TRACKING" && !result.monitoring;
    if (!healthy) motionRef.current = null;
    const showGuide = !result.visible && !!result.guide;
    if (showGuide !== guideShownRef.current && modelObjectRef.current) {
      applyRenderStyle(modelObjectRef.current, showGuide ? "xray" : renderStyleRef.current);
    }
    guideShownRef.current = showGuide;
    if (result.visible) {
      applyModelTransform(result.position, result.quaternion, mode !== "model", healthy ? result.capture_ms ?? undefined : undefined);
    } else if (result.guide) {
      // Guide view: the outline to line the real object up with (where the operator put it, if they moved it).
      if (guidePlacedRef.current) showGuidePose(guidePlacedRef.current);
      else applyModelTransform(result.guide.position, result.guide.quaternion, false);
    } else {
      applyModelTransform(null, null);
    }
    // Faded while the pose is only held, so it never reads as a live fit.
    if (overlayRef.current) overlayRef.current.style.opacity = result.state === "TRACKING" || showGuide ? "1" : "0.45";
  };

  const detectAndAlign = async () => {
    if (needsClass && !targetClassLabel) {
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
        const sessionId = sessionIdRef.current;
        const result = await VisionApi.arFrame(
          sessionId,
          mode,
          frame,
          needsClass ? targetClassLabel : null,
          // model_id in markerless mode only enables the model's calibrated part checks.
          mode === "model"
            ? { modelId, trackPart, detectBy, initBox: needsBox ? initBox : null, inspection: inspectionMode, inspectionRun }
            : { realWorldHeightM, modelId, detectBy, initBox: needsBox ? initBox : null }
        );
        if (sessionId !== sessionIdRef.current) return; // tracking was reset while this frame was in flight
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

  // What the stream tracks — sent when it opens and again whenever it changes or tracking is reset.
  const streamConfig = () =>
    JSON.stringify({
      session_id: sessionIdRef.current,
      mode,
      detect_by: detectBy,
      init_box: needsBox ? initBox : null,
      class_label: needsClass ? targetClassLabel : null,
      model_id: modelId,
      track_part: mode === "model" ? trackPart : null,
      resting: mode === "model" ? resting : "any",
      real_world_height_m: mode === "markerless" ? realWorldHeightM : null,
      inspection: mode === "model" ? inspectionMode : "off",
      inspection_run: inspectionRun,
      inspection_focus: focusIds.length ? focusIds : null,
      station_lock: mode === "model" && stationLocked,
      camera_label: deviceLabel || null,
      guide_pose: detectBy === "guide" && guidePlacedRef.current ? matrixToRows(guidePlacedRef.current) : null,
      guide_min_score: detectBy === "guide" ? guideLockAt : null,
    });

  // Live tracking (markerless / model-based) as a stream: every new camera
  // frame goes up as JPEG with its capture time, poses come back as they're
  // computed; the backend always works on the newest frame. At most two
  // frames in flight, so the browser never encodes frames the tracker would drop.
  useEffect(() => {
    if (!liveTracking || !(mode === "markerless" || mode === "model")) return;
    if (needsClass && !targetClassLabel) {
      setApiError("Pick an object class first — it's what the camera looks for.");
      return;
    }
    if (needsBox && !initBox) {
      setApiError("Drag a box around the object on the video first.");
      return;
    }
    setApiError(null);
    const ws = new WebSocket(apiWebSocketUrl("/api/vision/ar-session/stream"));
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;
    const video = videoRef.current;
    let stopped = false;
    let outstanding = 0;
    let lastReply = performance.now();
    let handle = 0;

    const sendFrame = async () => {
      if (stopped || ws.readyState !== WebSocket.OPEN) return;
      if (outstanding >= 2 && performance.now() - lastReply < 1000) return;
      const captured = performance.now();
      const blob = await captureFrameJpeg();
      if (!blob || stopped || ws.readyState !== WebSocket.OPEN) return;
      const jpeg = new Uint8Array(await blob.arrayBuffer());
      const message = new Uint8Array(8 + jpeg.length);
      new DataView(message.buffer).setFloat64(0, captured, true);
      message.set(jpeg, 8);
      ws.send(message);
      outstanding += 1;
    };
    // One send per new camera image where supported (fires once per video frame), else per animation frame.
    const schedule = () => {
      if (stopped) return;
      if (video && "requestVideoFrameCallback" in video) {
        handle = video.requestVideoFrameCallback(() => {
          void sendFrame();
          schedule();
        });
      } else {
        handle = requestAnimationFrame(() => {
          void sendFrame();
          schedule();
        });
      }
    };
    ws.onopen = () => {
      ws.send(streamConfig());
      schedule();
    };
    ws.onmessage = (event) => {
      outstanding = 0; // anything older was processed or dropped in favour of a newer frame
      lastReply = performance.now();
      const message = JSON.parse(event.data as string);
      if (message.error) {
        setApiError(message.error);
        return;
      }
      const result = message as ARFrameResponse;
      applyArResult(result, result.capture_ms != null ? performance.now() - result.capture_ms : 0);
    };
    ws.onerror = () =>
      setApiError("Tracking stream failed — is the backend running (restarted after the latest update)?");
    return () => {
      stopped = true;
      wsRef.current = null;
      if (video && "cancelVideoFrameCallback" in video) video.cancelVideoFrameCallback(handle);
      cancelAnimationFrame(handle);
      ws.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveTracking, mode, targetClassLabel, realWorldHeightM, trackPart, detectBy, modelId, initBox]);

  useEffect(() => {
    // Marker / image-target modes: one HTTP request per frame.
    if (!liveTracking || mode === "markerless" || mode === "model") return;
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
  }, [liveTracking, mode, targetClassLabel, realWorldHeightM, trackPart, detectBy]);

  // The model's component registry (what can be inspected, and where on the model).
  useEffect(() => {
    if (mode !== "model") return;
    let cancelled = false;
    setRegistryError(null);
    Models3DApi.components(modelId)
      .then((registry) => {
        if (cancelled) return;
        componentRegistryRef.current = registry;
        setComponentRegistry(registry);
        discrepancyRef.current?.setComponents(registry.components);
      })
      .catch((err) => !cancelled && setRegistryError(apiErrorMessage(err)));
    return () => {
      cancelled = true;
    };
  }, [modelId, mode, inspectionRun]);

  // The current step's components (matched by name to the registry), when step focus is on.
  const focusIds = (() => {
    if (!stepFocus || !guide || guide.state === "complete" || !componentRegistry) return [] as string[];
    const norm = (v: string | null | undefined) => (v ?? "").toLowerCase().replace(/[^a-z0-9]/g, "");
    const wanted = new Set(guide.parts.map(norm));
    return componentRegistry.components
      .filter((c) => wanted.has(norm(c.display_name)) || wanted.has(norm(c.node_name)) || wanted.has(norm(c.id)))
      .map((c) => c.id);
  })();

  // Click the object on the video to lock onto it (model mode): the backend
  // tries the image segments under the click and keeps the one the 3D model
  // fits best. For plain objects that searching by appearance can't find.
  const canPick = mode === "model" && liveTracking && !needsBox && !stationLocked && detectBy !== "guide";

  // ── Moving the guide outline with the mouse ────────────────────────────
  // Drag: across the table · wheel: nearer/farther · Shift+drag: turn
  // (left/right, about the model's up axis) and tilt (up/down). Only rough
  // placement is needed: the edge check snaps it onto the real object.
  const guideActive = mode === "model" && detectBy === "guide" && !!arResult?.guide && !arResult.visible;
  const sendGuidePose = (force = false) => {
    const ws = wsRef.current;
    const now = performance.now();
    if (!force && now - guideSentAtRef.current < 80) return;
    guideSentAtRef.current = now;
    if (ws?.readyState === WebSocket.OPEN) ws.send(streamConfig());
  };
  const placedGuide = (): THREE.Matrix4 | null => {
    if (guidePlacedRef.current) return guidePlacedRef.current;
    const rows = arResult?.guide?.t_camera_object;
    if (!rows) return null;
    guidePlacedRef.current = rowsToMatrix(rows);
    setGuidePlaced(true);
    return guidePlacedRef.current;
  };
  const moveGuide = (change: (m: THREE.Matrix4) => void, force = false) => {
    const m = placedGuide();
    if (!m) return;
    change(m);
    showGuidePose(m);
    sendGuidePose(force);
  };
  const rotateGuide = (degrees: number) => moveGuide((m) => rotateAbout(m, guideAxisRef.current, degrees), true);
  const onGuidePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!guideActive || e.button !== 0) return;
    if ((e.target as HTMLElement).closest("button, a, input, select, label, .hud")) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    guideDragRef.current = { x: e.clientX, y: e.clientY, shift: e.shiftKey, alt: e.altKey };
  };
  const onGuidePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const drag = guideDragRef.current;
    const video = videoRef.current;
    const k = arResult?.intrinsics;
    if (!drag || !video || !video.videoWidth || !k) return;
    const toFrame = video.videoWidth / video.getBoundingClientRect().width;
    const du = (e.clientX - drag.x) * toFrame;
    const dv = (e.clientY - drag.y) * toFrame;
    guideDragRef.current = { x: e.clientX, y: e.clientY, shift: drag.shift || e.shiftKey, alt: drag.alt || e.altKey };
    moveGuide((m) => {
      if (drag.alt || e.altKey) {
        rotateAbout(m, guideAxisRef.current, (du - dv) * 0.4); // free rotation about the picked axis
        return;
      }
      if (!(drag.shift || e.shiftKey)) {
        slide(m, du, dv, k.fx, k.fy);
        return;
      }
      turn(m, du * 0.4); // left / right: turn on the table
      tilt(m, -dv * 0.3); // up / down: tilt
    });
  };
  const onGuidePointerUp = () => {
    if (!guideDragRef.current) return;
    guideDragRef.current = null;
    sendGuidePose(true);
  };
  const onGuideWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    if (!guideActive) return;
    moveGuide((m) => dolly(m, Math.exp(e.deltaY * 0.0008)), true); // nearer / farther along the viewing ray
  };
  const resetPlacedGuide = () => {
    guidePlacedRef.current = null;
    setGuidePlaced(false);
    sendGuidePose(true);
  };
  const pickObjectAt = (e: React.MouseEvent<HTMLDivElement>) => {
    const video = videoRef.current;
    const ws = wsRef.current;
    if (!canPick || !video || !video.videoWidth || ws?.readyState !== WebSocket.OPEN) return;
    if ((e.target as HTMLElement).closest("button, a, input, select, label, .hud")) return; // a control, not the video
    const rect = video.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return;
    const point = [Math.round(fx * video.videoWidth), Math.round(fy * video.videoHeight)];
    ws.send(JSON.stringify({ ...JSON.parse(streamConfig()), select_point: point }));
    // Later configs carry no point: the backend selects once.
    ws.send(streamConfig());
    setPickMark({ left: fx * 100, top: fy * 100 });
    window.setTimeout(() => setPickMark(null), 4000);
    setArEvents((events) => [...events, `[UI] Selecting the object at (${point[0]}, ${point[1]})…`].slice(-12));
  };

  // Inspection and station choices reach the backend with the stream's configuration.
  useEffect(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(streamConfig());
    if (inspectionMode === "off") discrepancyRef.current?.clear();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inspectionMode, inspectionRun, stationLocked, focusIds.join(","), resting, guideLockAt]);

  // Is this model's station pose locked? (Saved on the backend, survives restarts.)
  useEffect(() => {
    if (mode !== "model") return;
    StationApi.status(modelId)
      .then((s) => setStationLocked(s.locked))
      .catch(() => setStationLocked(false));
  }, [modelId, mode]);

  const saveGuideView = async () => {
    if (!modelId) return;
    setGuideBusy(true);
    try {
      if (arResult?.state !== "TRACKING" && guidePlacedRef.current) {
        await GuideViewApi.savePose(modelId, matrixToRows(guidePlacedRef.current));
        setArEvents((events) => [...events, "[UI] Guide view saved where you placed the outline"].slice(-12));
        return;
      }
      await GuideViewApi.saveCurrent(modelId, sessionIdRef.current);
      setArEvents((events) => [...events, "[UI] Guide view saved from the current pose"].slice(-12));
    } catch (err) {
      setApiError(apiErrorMessage(err));
    } finally {
      setGuideBusy(false);
    }
  };

  const resetGuideView = async () => {
    if (!modelId) return;
    setGuideBusy(true);
    try {
      await GuideViewApi.reset(modelId);
      setArEvents((events) => [...events, "[UI] Guide view reset to the default"].slice(-12));
    } catch (err) {
      setApiError(apiErrorMessage(err));
    } finally {
      setGuideBusy(false);
    }
  };

  const lockStation = async () => {
    const frame = captureFrameBase64();
    if (!frame) return;
    setStationBusy(true);
    try {
      await StationApi.lock(sessionIdRef.current, modelId, frame);
      setStationLocked(true);
      setArEvents((events) => [...events, "[STATION] Pose locked — tracking off, inspection uses the fixed pose"].slice(-12));
    } catch (err) {
      setApiError(apiErrorMessage(err));
    } finally {
      setStationBusy(false);
    }
  };

  const unlockStation = async () => {
    setStationBusy(true);
    try {
      await StationApi.unlock(modelId);
      setStationLocked(false);
      setArEvents((events) => [...events, "[STATION] Unlocked — tracking again"].slice(-12));
    } catch (err) {
      setApiError(apiErrorMessage(err));
    } finally {
      setStationBusy(false);
    }
  };

  // A new selection / "mark all" applies to the markers straight away.
  useEffect(() => {
    const inspection = arResult?.inspection;
    if (inspection) discrepancyRef.current?.update(inspection.components, inspectionViewRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedComponent, showAllMarkers]);

  const resetInspection = () => {
    discrepancyRef.current?.clear();
    setSelectedComponent(null);
    setInspectionRun(newSessionId()); // the backend starts the run over
  };

  const forgetReference = async () => {
    try {
      await Models3DApi.forgetComponentReference(modelId);
      resetInspection();
    } catch (err) {
      setApiError(apiErrorMessage(err));
    }
  };

  const isAr = mode === "markerless" || mode === "model";
  // The stage keeps the camera's own aspect ratio, so the overlay lines up pixel for pixel.
  const stageAspect = resolution ? (resolution.width / resolution.height).toFixed(4) : "1.7778";
  const sideTabs = (
    [
      ["inspect", "Inspection", mode === "model"],
      ["procedure", "Procedure", Boolean(sidebarExtra)],
      ["tracking", "Tracking", true],
      ["tools", "Tools", true],
    ] as const
  ).filter(([, , shown]) => shown);
  const activeSideTab = sideTabs.some(([value]) => value === sideTab) ? sideTab : sideTabs[0][0];

  return (
    <div className="ar-workspace">
      <div className="ar-primary">
      {(() => {
        const topline = (
      <div className="ar-topline">
        <label className="field inline" title={AR_MODES.find(([value]) => value === mode)?.[2]}>
          Tracking
          <select value={mode} onChange={(e) => setMode(e.target.value as RegistrationMode)}>
            {AR_MODES.map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>
      {mode !== "marker" && (
        <div className="settings-row inline">
          {(mode === "model" || mode === "markerless") && (
            <label title="3D model: matches renders of this model against the camera image — any object with a GLB. Object class: the stock detector's categories (bottle, cup, …). Draw a box: drag a rectangle around the object on the video — any object, no class or model; if tracking is lost it's found again by the 3D model.">
              Find object by
              <span className="segmented">
                <button className={detectBy === "model" ? "active" : ""} onClick={() => setDetectBy("model")}>
                  3D model
                </button>
                <button className={detectBy === "class" ? "active" : ""} onClick={() => setDetectBy("class")}>
                  Object class
                </button>
                <button className={detectBy === "box" ? "active" : ""} onClick={() => setDetectBy("box")}>
                  Draw a box
                </button>
                {mode === "model" && (
                  <button
                    className={detectBy === "guide" ? "active" : ""}
                    onClick={() => setDetectBy("guide")}
                    title="Nothing is detected: the model's outline is shown, you line the real object up with it, and tracking starts once its 3D edges match — the most reliable start for plain objects"
                  >
                    Guide view
                  </button>
                )}
              </span>
            </label>
          )}
          {needsBox && (
            <span className="hint" style={{ alignSelf: "center" }}>
              {initBox ? "Box set — drag again on the video to redraw it." : "Drag a box around the object on the video."}
            </span>
          )}
          {needsClass && (
            <label>
              Object class
              <ClassSelect value={targetClassLabel} onChange={setTargetClassLabel} />
            </label>
          )}
          {mode === "model" && (
            <label title="Which side of the object faces where. Objects whose top and bottom look alike (a flat box) can otherwise lock upside down. Camera above: top faces the camera. Camera at the side: top is up in the image.">
              Resting side
              <select value={resting} onChange={(e) => setResting(e.target.value as "any" | "camera" | "upright" | "flat")}>
                <option value="any">Any</option>
                <option value="camera">Top faces camera</option>
                <option value="flat">Flat, camera straight above</option>
                <option value="upright">Upright (side view)</option>
              </select>
            </label>
          )}
          {mode === "model" && parts.length > 1 && (
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
          {mode === "markerless" && (
            <label title="Distance is estimated from the object's apparent size, so this must be right.">
              Real height (m)
              <input
                type="number"
                step="0.01"
                min="0.01"
                value={realWorldHeightM}
                onChange={(e) => setRealWorldHeightM(Number(e.target.value))}
              />
            </label>
          )}
          {mode === "feature" && (
            <>
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
                {registering ? "Registering…" : "Register reference image"}
              </button>
            </>
          )}
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
      {mode === "feature" && !registration && (
        <p className="warning">Register a reference image of the object's labeled/textured surface first.</p>
      )}

      </div>
        );
        return toolbarSlot ? createPortal(topline, toolbarSlot) : topline;
      })()}

      {calibrationOpen && (
        <CalibrationPanel captureFrameJpeg={captureFrameJpeg} cameraLabel={deviceLabel} onClose={() => setCalibrationOpen(false)} />
      )}
      <div className="stage-box">
      <div
        className={`ar-stage${canPick ? " pickable" : ""}${guideActive ? " guide-draggable" : ""}`}
        style={{ "--stage-ar": stageAspect } as React.CSSProperties}
        onClick={pickObjectAt}
        onPointerDown={onGuidePointerDown}
        onPointerMove={onGuidePointerMove}
        onPointerUp={onGuidePointerUp}
        onPointerCancel={onGuidePointerUp}
        onWheel={onGuideWheel}
      >
        {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
        <video ref={videoRef} autoPlay playsInline muted />
        <div ref={overlayRef} style={{ position: "absolute", inset: 0, pointerEvents: "none" }} />
        <canvas
          ref={outlineCanvasRef}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
        />
        <canvas ref={canvasRef} style={{ display: "none" }} />
        {needsBox && (
          <BoxSelector
            videoRef={videoRef}
            box={arResult?.state === "TRACKING" ? null : initBox}
            onBox={(box) => {
              setInitBox(box);
              sessionIdRef.current = newSessionId(); // a new box is a new object to lock onto
            }}
          />
        )}

        {pickMark && <div className="pick-mark" style={{ left: `${pickMark.left}%`, top: `${pickMark.top}%` }} />}

        <div className="hud hud-top-left">
          {isAr && arResult?.visible && arResult.quality?.orientation_status === "ambiguous" && (
            <div className="hud-pill state-RECOVERING" title={arResult.quality.orientation_reason ?? ""}>
              <span className="dot" />
              Orientation unconfirmed
              <span className="sub">{arResult.quality.orientation_reason}</span>
            </div>
          )}
          {isAr && arResult?.visible && arResult.quality?.orientation_status === "confirmed" && (
            <div className="hud-pill state-TRACKING" title="The pose features show which way round the object is">
              <span className="dot" />
              Orientation confirmed
            </div>
          )}
          {arResult?.guide && !arResult.visible && (
            <div className="hud-pill state-INITIALIZING">
              <span className="dot" />
              Line the object up with the outline — or drag the outline onto it
              <span
                className="guide-meter"
                role="meter"
                aria-label="Match with the outline"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round((arResult.guide.score ?? 0) * 100)}
                title={`Match ${Math.round((arResult.guide.score ?? 0) * 100)}% — locks at ${Math.round(arResult.guide.min_score * 100)}%`}
              >
                <span
                  className={`guide-meter-fill${(arResult.guide.score ?? 0) >= arResult.guide.min_score ? " ok" : ""}`}
                  style={{ width: `${Math.min(100, Math.round((arResult.guide.score ?? 0) * 100))}%` }}
                />
                <span className="guide-meter-mark" style={{ left: `${Math.round(arResult.guide.min_score * 100)}%` }} />
              </span>
              <span className="sub">
                match {arResult.guide.score != null ? `${Math.round(arResult.guide.score * 100)}%` : "—"} · locks at{" "}
                {Math.round(arResult.guide.min_score * 100)}%
                {arResult.guide.edge_points?.length
                  ? ` · edges on target ${Math.round(
                      (100 * arResult.guide.edge_points.filter((p) => p[2] > 0).length) / arResult.guide.edge_points.length
                    )}% (green)`
                  : ""}
                {arResult.guide.saved ? " · saved view" : ""}
              </span>
            </div>
          )}
          {canPick && (!arResult || arResult.state === "SEARCHING" || arResult.state === "LOST") && (
            <div className="hud-pill">
              <span className="dot" />
              Click the object to lock onto it
            </div>
          )}
          {needsBox && !initBox && (
            <div className="hud-pill">
              <span className="dot" />
              Drag a box around the object
            </div>
          )}
          {arResult?.station?.locked && (
            <div className={`hud-pill ${arResult.station.moved ? "state-LOST" : "state-TRACKING"}`}>
              <span className="dot" />
              {arResult.station.moved ? "Part moved — re-align" : "Station locked"}
              <span className="sub">
                shift {arResult.station.shift_px != null ? `${arResult.station.shift_px.toFixed(1)} px` : "—"}
              </span>
            </div>
          )}
          {isAr && arResult && (
            <div className={`hud-pill state-${arResult.state}`}>
              <span className="dot" />
              {STATE_LABEL[arResult.state]}
              {arResult.state === "TRACKING" && arResult.object && (
                <span className="sub">
                  {arResult.object.class_label} · {(arResult.object.confidence * 100).toFixed(0)}%
                  {arResult.monitoring ? " · low confidence" : ""}
                  {arResult.quality?.occlusion ? ` · ${(arResult.quality.occlusion * 100).toFixed(0)}% under a hand` : ""}
                </span>
              )}
              {arResult.state === "TRACKING_OCCLUDED" && arResult.quality && (
                <span className="sub">
                  {((arResult.quality.occlusion ?? 0) * 100).toFixed(0)}% covered
                  {arResult.quality.held_ms != null ? ` · ${(arResult.quality.held_ms / 1000).toFixed(1)} s` : ""}
                </span>
              )}
            </div>
          )}
          {isAr && !arResult && liveTracking && (
            <div className="hud-pill state-SEARCHING">
              <span className="dot" />
              Starting…
            </div>
          )}
          {modelStatus !== "loaded" && (
            <div className="hud-pill">{modelStatus === "loading" ? "Loading 3D model…" : "3D model failed to load"}</div>
          )}
        </div>

        {isAr && (arResult?.calibrated_parts ?? []).length > 0 && (
          <div className="hud hud-top-right part-badges">
            {arResult!.calibrated_parts.map((name) => {
              const check = (arResult!.part_checks ?? []).find((c) => c.part_name === name);
              const cls = check ? `part-badge part-badge-${check.state}` : "part-badge part-badge-waiting";
              const text = !check
                ? `${name}: waiting for tracking`
                : check.state === "present"
                  ? `✓ ${name} present`
                  : check.state === "absent"
                    ? `✕ ${name} REMOVED`
                    : `? ${name} unsure`;
              return (
                <div key={name} className={cls}>
                  {text}
                  {check && <span> {(check.confidence * 100).toFixed(0)}%</span>}
                </div>
              );
            })}
          </div>
        )}

        {guide && (
          <div className={`step-banner ${guide.state === "done" || guide.state === "complete" ? "done" : ""}`}>
            <div className="step-no">{guide.state === "complete" ? "✓" : guide.number}</div>
            <div className="step-text">
              <strong>
                {guide.state === "complete" ? "Procedure complete" : `Step ${guide.number}/${guide.total} · ${guide.title}`}
              </strong>
              {guide.instruction && guide.state !== "complete" && <span>{guide.instruction}</span>}
            </div>
            <div className="step-status">{guide.statusText}</div>
          </div>
        )}
      </div>
      </div>

      <div className="control-bar">
        <button
          className={`big ${liveTracking ? "stop" : "primary"}`}
          onClick={() => setLiveTracking((v) => !v)}
          disabled={!ready && !liveTracking}
          title={isAr ? "Detect once, then track every frame" : "As fast as detection responds, smoothed"}
        >
          {liveTracking ? "■ Stop tracking" : "▶ Start tracking"}
        </button>
        <button onClick={detectAndAlign} disabled={busy || !ready || liveTracking}>
          {isAr ? (arResult?.state === "TRACKING" ? "Track once" : "Detect once") : busy ? "Detecting…" : "Detect & align"}
        </button>
        {isAr && <button onClick={() => void resetTracking()}>Reset</button>}
        {mode === "model" &&
          (stationLocked ? (
            <button
              className="active"
              onClick={() => void unlockStation()}
              disabled={stationBusy}
              title="Fixed station: the pose is locked. Unlock to track again (e.g. after moving the camera or fixture)."
            >
              🔒 Station locked
            </button>
          ) : (
            <button
              onClick={() => void lockStation()}
              disabled={stationBusy || arResult?.state !== "TRACKING" || arResult.monitoring}
              title="Fixed camera + part in a fixture: freeze this pose. Tracking stops; inspection uses the locked pose; a moved part is reported."
            >
              Lock station
            </button>
          ))}
        {mode === "model" && modelId && (
          <button
            onClick={() => void saveGuideView()}
            disabled={guideBusy || !((arResult?.state === "TRACKING" && !arResult.monitoring) || (guidePlaced && guideActive))}
            title="Use the pose tracked now - or where you placed the outline - as this model's guide view: its outline is shown there next time (Find object by: Guide view)"
          >
            Save guide view
          </button>
        )}
        {guideActive && (
          <>
          <label
            className="guide-lock-at"
            title="How closely the real object must match the outline before it locks. Lower locks sooner (but may lock on look-alikes); higher waits for a closer match. Remembered for this model in this browser."
          >
            Lock at
            <input
              type="range"
              min={20}
              max={90}
              step={1}
              value={Math.round((guideLockAt ?? arResult?.guide?.min_score ?? 0.58) * 100)}
              onChange={(e) => setGuideLockAt(Number(e.target.value) / 100)}
              aria-label="Match needed to lock"
            />
            <span className="mono">{Math.round((guideLockAt ?? arResult?.guide?.min_score ?? 0.58) * 100)}%</span>
            {guideLockAt != null && (
              <button onClick={() => setGuideLockAt(null)} title="Back to the default threshold">
                Default
              </button>
            )}
          </label>
          <span
            className="segmented guide-rotate"
            aria-label="Rotate the guide outline"
            title="Rotate the outline about one of the model's own axes (drawn on it: X red, Y green, Z blue). Or drag: drag moves it, the wheel brings it nearer / farther, Shift+drag turns and tilts, Alt+drag rotates about the picked axis"
          >
            {(["x", "y", "z"] as ModelAxis[]).map((axis) => (
              <button
                key={axis}
                className={`axis-${axis}${guideAxis === axis ? " active" : ""}`}
                onClick={() => setGuideAxis(axis)}
                title={`Rotate about the model's ${axis.toUpperCase()} axis`}
              >
                {axis.toUpperCase()}
              </button>
            ))}
            <button onClick={() => rotateGuide(-90)} title="-90° about the picked axis">−90°</button>
            <button onClick={() => rotateGuide(-15)} title="-15° about the picked axis">−15°</button>
            <button onClick={() => rotateGuide(15)} title="+15° about the picked axis">+15°</button>
            <button onClick={() => rotateGuide(90)} title="+90° about the picked axis">+90°</button>
            <button onClick={() => rotateGuide(180)} title="Turn round (180°) about the picked axis">180°</button>
            {guidePlaced && (
              <button onClick={resetPlacedGuide} title="Back to the saved / default outline">
                Undo moves
              </button>
            )}
          </span>
          </>
        )}
        {mode === "model" && modelId && detectBy === "guide" && arResult?.guide?.saved && !guidePlaced && (
          <button onClick={() => void resetGuideView()} disabled={guideBusy} title="Back to the default guide view">
            Reset guide
          </button>
        )}
        <span
          className="view-style-toggle"
          title="How the 3D model is drawn — X-Ray and Wireframe let you see the real object through it to judge alignment"
        >
          <span className="segmented" aria-label="Model display">
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
        </span>
        <CameraStatusBadge status={cameraStatus} deviceLabel={deviceLabel} resolution={resolution} error={cameraError} retryNote={cameraRetryNote} otherTabHasCamera={otherTabHasCamera} onRetry={retryCamera} onTakeOver={takeOverCamera} />
        <CameraSelect devices={devices} selectedDeviceId={selectedDeviceId} onSelect={selectDevice} />
        {isAr && (
          <label
            className="hint"
            title={
              arResult?.hands
                ? arResult.hands.available
                  ? `Hand mask from ${arResult.hands.sources.join(" + ") || "—"}: ${arResult.hands.hands} hand(s), ${arResult.hands.persons} person region(s)`
                  : `No hand detector: ${arResult.hands.error ?? "disabled"}`
                : "Draw the hand mask that tracking and inspection leave out"
            }
          >
            <input type="checkbox" checked={showHands} onChange={(e) => setShowHands(e.target.checked)} /> Show hands
            {arResult?.hands && !arResult.hands.available && " (no detector)"}
          </label>
        )}
        {mode === "model" && (
          <label className="hint" title="Draw each pose feature where the pose puts it (dashed) and the circle actually found (dot)">
            <input type="checkbox" checked={showPoseFeatures} onChange={(e) => setShowPoseFeatures(e.target.checked)} /> Pose features
          </label>
        )}
        <button
          onClick={() => setCalibrationOpen((open) => !open)}
          className={calibrationOpen ? "active" : ""}
          title={
            arResult
              ? arResult.calibration_is_approximate
                ? `Camera not calibrated: ${arResult.calibration_note ?? "intrinsics assumed"}`
                : `Calibrated (${arResult.calibration_source}${arResult.calibration_rms_px != null ? `, ${arResult.calibration_rms_px.toFixed(2)} px RMS` : ""})`
              : "Camera calibration"
          }
        >
          {arResult && !arResult.calibration_is_approximate ? "Camera calibrated" : "Calibrate camera"}
        </button>
        {modelStatus === "error" && <span className="camera-status camera-status-error">3D model: {modelError}</span>}
      </div>
      {apiError && <p className="error">{apiError}</p>}
      </div>

      <aside className="ar-sidebar" aria-label="Inspection, procedure and tracking">
        <div className="side-tabs" role="tablist">
          {sideTabs.map(([value, label]) => (
            <button
              key={value}
              role="tab"
              aria-selected={activeSideTab === value}
              className={activeSideTab === value ? "active" : ""}
              onClick={() => setSideTab(value)}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="side-body">
          {activeSideTab === "inspect" && (
            <>
        {mode === "model" && (
          <ComponentInspectionPanel
            registry={componentRegistry}
            registryError={registryError}
            inspection={inspectionMode === "off" ? null : arResult?.inspection ?? null}
            mode={inspectionMode}
            onMode={setInspectionMode}
            onReset={resetInspection}
            tracking={liveTracking}
            selected={selectedComponent}
            onSelect={setSelectedComponent}
            showAll={showAllMarkers}
            onShowAll={setShowAllMarkers}
            debug={inspectionDebug}
            onDebug={setInspectionDebug}
            outlineNames={outlineNames}
            onOutlineNames={setOutlineNames}
            onForgetReference={() => void forgetReference()}
            stepFocus={guide && guide.state !== "complete" ? { on: stepFocus, ids: focusIds, onChange: setStepFocus } : null}
            onSaveReport={() => void saveReport()}
            reportStatus={reportStatus}
            onOpenReport={onOpenReport}
          />
        )}

            </>
          )}
          {activeSideTab === "procedure" && sidebarExtra}
          {activeSideTab === "tracking" && (
            <>
          <div className="card">
            <div className="section-label" style={{ marginBottom: 10 }}>
              Tracking health
            </div>
            <div className="health-grid">
              <div>
                <span>Status</span>
                <strong>{arResult?.tracking_status?.replace("_", " ") ?? (liveTracking ? "STARTING" : "IDLE")}</strong>
              </div>
              <div>
                <span>Confidence</span>
                <strong>{arResult?.object ? `${(arResult.object.confidence * 100).toFixed(0)}%` : "—"}</strong>
              </div>
              <div>
                <span>Model fit</span>
                <strong>{arResult?.quality?.overlap != null ? arResult.quality.overlap.toFixed(2) : "—"}</strong>
              </div>
              <div>
                <span>Flow support</span>
                <strong>{arResult?.quality?.flow_support != null ? arResult.quality.flow_support.toFixed(2) : "—"}</strong>
              </div>
              <div>
                <span>Hand cover</span>
                <strong>
                  {arResult?.quality?.occlusion != null ? `${(arResult.quality.occlusion * 100).toFixed(0)}%` : "—"}
                </strong>
              </div>
              <div>
                <span>Frame round trip</span>
                <strong>{frameMs ? `${frameMs.toFixed(0)} ms` : "—"}</strong>
              </div>
            </div>
          </div>

          {arEvents.length > 0 && (
            <div className="card event-log">
              <div className="section-label" style={{ marginBottom: 8 }}>
                Event log
              </div>
              <pre className="tracking-log">{[...arEvents].reverse().join("\n")}</pre>
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
              <p className="pose-readout">
                position: ({markerPose.position!.x.toFixed(3)}, {markerPose.position!.y.toFixed(3)},{" "}
                {markerPose.position!.z.toFixed(3)}) m
                {markerPose.rotation_deg &&
                  ` · Rx=${markerPose.rotation_deg.rx.toFixed(1)}° Ry=${markerPose.rotation_deg.ry.toFixed(1)}° Rz=${markerPose.rotation_deg.rz.toFixed(1)}°`}
              </p>
            </>
          ) : (
            <p>No marker detected in frame.</p>
          )}
          {markerPose.calibration_is_approximate && (
            <p className="hint">Using an approximate default camera calibration ({markerPose.calibration_source}).</p>
          )}
        </div>
      )}

      {isAr && (arResult || arEvents.length > 0) && (
        <details
          className="card diagnostics"
          open={diagnosticsOpen}
          onToggle={(e) => setDiagnosticsOpen((e.currentTarget as HTMLDetailsElement).open)}
        >
          <summary>
            Tracking details
            <span className="hint">
              {rates.detect.toFixed(1)}/s detect · {rates.track.toFixed(1)}/s track · {rates.render} fps render ·{" "}
              {frameMs.toFixed(0)} ms/frame
            </span>
          </summary>
          <div className="diagnostics-body">
            {arResult && (
              <>
                {arResult.object && (
                  <p>
                    Confidence <strong>{(arResult.object.confidence * 100).toFixed(0)}%</strong>{" "}
                    <span className="hint">
                      {mode === "model" ? "model ↔ real object overlap" : "share of tracked points still agreeing"} (good ≥{" "}
                      {(arResult.good_confidence * 100).toFixed(0)}%, lost &lt; {(arResult.lost_confidence * 100).toFixed(0)}%)
                    </span>
                  </p>
                )}
                {(arResult.part_checks ?? []).map((check) => (
                  <p key={check.node_index} className={`part-check part-${check.state}`}>
                    {check.part_name}: <strong>{check.state === "absent" ? "removed" : check.state}</strong>{" "}
                    <span className="hint">
                      ({(check.confidence * 100).toFixed(0)}% sure, {check.measure ?? "mean"} brightness{" "}
                      {check.brightness.toFixed(1)}
                      {check.region_polygon ? ", measured on its 3D outline" : ""})
                    </span>
                  </p>
                ))}
                <div className="metric-grid">
                  {(
                    [
                      ["Object ID", arResult.object?.object_id ?? "—"],
                      ["Detections", arResult.counters.detection_count],
                      ["Detector runs", arResult.counters.detection_runs],
                      ["Tracking frames", arResult.counters.tracking_frames],
                      ["Frames since detection", arResult.counters.frames_since_detection ?? "—"],
                      ["Server time", `${arResult.timings_ms.total.toFixed(0)} ms`],
                      ["Render time", `${rates.renderMs.toFixed(1)} ms`],
                    ] as const
                  ).map(([label, value]) => (
                    <div key={label} className="metric">
                      <span>{label}</span>
                      <strong>{value}</strong>
                    </div>
                  ))}
                </div>
                <p className="hint">
                  This frame — detection {arResult.timings_ms.detection.toFixed(0)} ms · initialization{" "}
                  {arResult.timings_ms.initialization.toFixed(0)} ms · tracking {arResult.timings_ms.tracking.toFixed(0)} ms ·
                  refinement {arResult.timings_ms.refinement.toFixed(0)} ms
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
            {arEvents.length > 0 && <pre className="tracking-log">{arEvents.join("\n")}</pre>}
          </div>
        </details>
      )}

      {mode === "feature" && featurePose && (
        <div className={`pose-status ${featurePose.found ? "pose-found" : "pose-not-found"}`}>
          {featurePose.found ? (
            <>
              <p>
                Matched — {featurePose.num_inliers}/{featurePose.num_matches} inlier features
              </p>
              <p className="pose-readout">
                position: ({featurePose.position!.x.toFixed(3)}, {featurePose.position!.y.toFixed(3)},{" "}
                {featurePose.position!.z.toFixed(3)}) m
                {featurePose.rotation_deg &&
                  ` · Rx=${featurePose.rotation_deg.rx.toFixed(1)}° Ry=${featurePose.rotation_deg.ry.toFixed(1)}° Rz=${featurePose.rotation_deg.rz.toFixed(1)}°`}
              </p>
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
            <p className="hint">Using an approximate default camera calibration ({featurePose.calibration_source}).</p>
          )}
        </div>
      )}

            </>
          )}
          {activeSideTab === "tools" && (
            <>
      {mode === "model" && <FitCheck modelId={modelId} captureFrame={captureFrameBase64} disabled={!ready} />}


      {mode === "model" && <QAInspectionPanel modelId={modelId} captureFrame={captureFrameBase64} disabled={!ready} />}

      {(mode === "marker" || mode === "feature") && (
        <div className="anchor-calibration">
          <div className="panel-title">Model alignment</div>
          <p className="hint">
            Tracking gives the pose of the reference patch (the marker's face or the registered photo's plane), not
            the model's own origin. If the model sits offset or tilted while tracking is stable, nudge it here and
            save — once per model.
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
            <button className="primary" onClick={saveAnchor} disabled={anchorSaving}>
              {anchorSaving ? "Saving…" : "Save alignment"}
            </button>
            <button onClick={resetAnchor}>Reset</button>
            {anchorSaved && <span className="camera-status camera-status-ok">Saved</span>}
          </div>
        </div>
      )}

              <div className="card tools-card">
                <div className="section-label">Capture</div>
        <span
          className="save-frame"
          title="Records raw camera frames (15/s, up to 60 s) to data/clips/<name>/ for the replay benchmark: python -m app.workers.replay_clip <name> --model <id>"
        >
          <input
            value={clipName}
            onChange={(e) => setClipName(e.target.value)}
            placeholder="clip name"
            disabled={recording !== null}
          />
          {recording ? (
            <button className="stop" onClick={stopRecording}>
              ■ {recording.seconds.toFixed(0)} s · {recording.frames} frames
            </button>
          ) : (
            <button onClick={startRecording} disabled={!ready || !clipName.trim()}>
              ● Record clip
            </button>
          )}
          {clipSaved && <span className="hint">{clipSaved}</span>}
        </span>
        <span className="save-frame" title="Saves the raw camera frame (full resolution, without any overlay) to data/debug_frames/ for analysis">
          <input
            value={frameLabel}
            onChange={(e) => setFrameLabel(e.target.value)}
            placeholder="label, e.g. cap-off"
          />
          <button onClick={() => void saveFrame()} disabled={!ready || !frameLabel.trim()}>
            Save frame
          </button>
          {frameSaved && <span className="hint">{frameSaved}</span>}
        </span>
              </div>
            </>
          )}
        </div>
      </aside>
    </div>
  );
}
