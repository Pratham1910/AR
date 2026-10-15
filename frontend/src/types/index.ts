// Mirrors backend/app/schemas + app/models/enums.py — keep in sync manually
// until an OpenAPI-generated client is introduced (Project.md #63).

export type QAResult = "PASS" | "FAIL" | "UNCERTAIN" | "NOT_EVALUATED" | "MANUAL_REVIEW";

export interface Asset {
  id: string;
  name: string;
  description: string | null;
}

export interface ProcedureRevisionSummary {
  id: string;
  revision_label: string;
  status: "draft" | "published";
  step_count: number;
}

export interface Procedure {
  id: string;
  procedure_id_str: string;
  title: string;
  asset_id: string;
  revisions: ProcedureRevisionSummary[];
}

export interface InspectionStepStatus {
  step_id: string;
  step_id_str: string;
  title: string;
  result: QAResult;
  confidence: number | null;
}

export interface InspectionRun {
  id: string;
  status: "in_progress" | "completed" | "aborted";
  steps: InspectionStepStatus[];
}

export interface ValidationDetail {
  objectDetected: boolean;
  trackingStable: boolean | null;
  poseValid: boolean | null;
  stateMatched: boolean;
}

export interface StepValidationResponse {
  stepId: string;
  expectedState: string;
  observedState: string | null;
  confidence: number;
  result: QAResult;
  validation: ValidationDetail;
  evidence: string[];
  reason: string | null;
}

export interface ObserveResponse {
  observed_state: string | null;
  confidence: number;
  detections: unknown[];
  frame_ref: string | null;
}

// Phase 4/5 — 3D viewer + physical<->3D registration (see backend
// app/api/models3d.py, app/api/vision.py's /pose endpoint).

export interface Model3DInfo {
  id: string;
  asset_id: string;
  component_id: string | null;
  name: string;
  format: string;
  storage_key: string;
  // Multiplier to convert the GLB's own mesh units into real-world meters —
  // AR overlays MUST apply this before placing the model at a real-world
  // position, or a GLB not authored at 1 unit = 1 meter renders wildly
  // wrong-sized (backend/app/models/model3d.py).
  scale: number;
  url: string; // relative to the API base URL, e.g. /static/models/bottle.glb
  // The linked Component's class_label, if any (e.g. "cup", "bottle") — the
  // vision layer's name for this object. Use this to auto-fill markerless
  // registration's "Object class" instead of leaving a free-typed field
  // that can silently disagree with whichever asset is actually selected.
  component_class_label: string | null;
  // How tall the model renders in the AR overlay, in meters (mesh height x
  // scale) — null if the GLB couldn't be read.
  real_height_m: number | null;
  // The model's own local transform relative to the tracked reference plane
  // (marker or feature-tracking target) — see backend app/models/model3d.py.
  // Defaults to zero offset/identity rotation (model planted directly at the
  // tracked pose), which is only correct by coincidence; calibrate via the
  // AR overlay's alignment controls.
  anchor_offset_x: number;
  anchor_offset_y: number;
  anchor_offset_z: number;
  anchor_rotation_x: number;
  anchor_rotation_y: number;
  anchor_rotation_z: number;
  anchor_rotation_w: number;
}

// One part of an assembly GLB (backend GET /api/models3d/{id}/parts).
export interface Model3DPart {
  node_index: number; // stable id; matched in Three.js via GLTFLoader's parser.associations
  node_name: string; // as authored in the GLB
  display_name: string; // the user's name for it ("Cap"), or node_name until named
  component_id: string | null;
  size_m: [number, number, number]; // width, height, depth in meters
  center_m: [number, number, number];
}

export interface AnchorOffset {
  anchor_offset_x: number;
  anchor_offset_y: number;
  anchor_offset_z: number;
  anchor_rotation_x: number;
  anchor_rotation_y: number;
  anchor_rotation_z: number;
  anchor_rotation_w: number;
}

export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

export interface QuaternionXYZW {
  x: number;
  y: number;
  z: number;
  w: number;
}

export interface Vector2 {
  x: number;
  y: number;
}

// Debug display only (Project.md's debug-mode requirement) — never used for
// the actual placement math, which stays in quaternion form.
export interface RotationDeg {
  rx: number;
  ry: number;
  rz: number;
}

// 2D-projected XYZ pose gizmo, in the frame's own pixel space — draw as
// origin->x_axis (red), origin->y_axis (green), origin->z_axis (blue).
export interface PoseAxes {
  origin: Vector2;
  x_axis: Vector2;
  y_axis: Vector2;
  z_axis: Vector2;
}

export interface PoseResponse {
  found: boolean;
  marker_id: number | null;
  position: Vector3 | null;
  quaternion: QuaternionXYZW | null;
  reprojection_error_px: number | null;
  // Marker corners in the captured frame's own pixel space
  // (top-left/top-right/bottom-right/bottom-left) — draw as a quadrilateral
  // to show exactly what was detected.
  corners_px: Vector2[] | null;
  calibration_is_approximate: boolean;
  calibration_source: string;
  // The overlay's Three.js camera MUST use this exact FOV/aspect, not a
  // guessed constant, or the 3D model visibly drifts off the real object
  // even when position/orientation are computed correctly.
  camera_vertical_fov_deg: number;
  camera_aspect: number;
  rotation_deg: RotationDeg | null;
  axes: PoseAxes | null;
}

// Live AR session (backend /api/vision/ar-session/frame,
// app/services/tracking/ar_session.py): detect once, then track. The detector
// only runs while SEARCHING/LOST; while TRACKING only the tracker runs —
// optical flow ("markerless") or the MegaPose refiner ("model").
export type ARMode = "markerless" | "model";
export type TrackingState = "SEARCHING" | "INITIALIZING" | "TRACKING" | "TRACKING_OCCLUDED" | "LOST" | "RECOVERING";

// The tracking state in the operator's terms (backend vision._tracking_status).
export type TrackingStatus = "SEARCHING" | "INITIALIZING" | "TRACKING" | "TRACKING_OCCLUDED" | "REACQUIRING" | "TRACKING_LOST";

// The exact pinhole camera the backend computed the pose with; the overlay
// builds its projection from these (see docs/coordinates.md).
export interface CameraIntrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
}

export interface TrackedObject {
  object_id: number;
  class_label: string;
  confidence: number;
  first_seen_frame: number;
  last_seen_frame: number;
  bbox: [number, number, number, number] | null;
  polygon: Vector2[] | null;
  velocity_px_s: Vector2 | null;
}

// Is a calibrated part (e.g. the cap) still on the tracked object?
export interface ProcedureSummary {
  procedure_id: string;
  title: string;
  steps: number;
  actions: number;
  target_parts: string[]; // parts the animations move / highlight
  vision_parts: string[]; // parts whose presence the camera checks per step
  unmatched_parts: string[]; // named in the procedure but not found among the model's parts
}

export interface FitCheckResult {
  iou: number; // how much of the real outline the posed model covers
  good_fit: boolean;
  findings: string[]; // what to change in the GLB, biggest first
  real_height_cm: number;
  model_height_cm: number;
  parts: { part: string; from_top_cm: [number, number]; real_cm: number; model_cm: number; diff_cm: number; diff_rel: number }[];
  bands: { from_top_cm: number; real_cm: number; model_cm: number; part: string | null }[];
  overlay_jpeg_base64: string; // real outline red, model green
}

export interface PartCheck {
  node_index: number;
  part_name: string;
  state: "present" | "absent" | "uncertain";
  confidence: number;
  brightness: number; // the measured value of `measure`
  region_px: [number, number, number, number];
  measure?: string;
  region_polygon?: Vector2[] | null; // "pose" checks: the part's exact outline at the tracked pose
}

export interface QAPartResult {
  name: string;
  node_index: number;
  status: "present" | "missing" | "partial" | "occluded" | "unknown";
  coverage: number;    // 0–1 fraction of projected area covered by real object
  visibility: number;  // 0–1 fraction of part faces facing the camera
  projected_area: number;
  outline: [number, number][]; // [[x,y],…] projected outline
}

export interface QAInspectResult {
  parts: QAPartResult[];
  verdict: "PASS" | "FAIL" | "UNCERTAIN";
  missing_count: number;
  partial_count: number;
  present_count: number;
  overlay_jpeg_base64: string;
}

export interface PresenceCalibrationResult {
  node_index: number;
  part_name: string;
  present_mean: number;
  absent_mean: number;
  present_samples: number;
  absent_samples: number;
  separation: number;
  verdict: string;
  frames_without_object: string[]; // not used: "<frame> (object not found / pose rejected / …)"
  method: "pose" | "box"; // pose: the part's exact pixels at the solved 3D pose
  measure: string; // brightness statistic that separated on/off best ("mean", "p90", "std", …)
  frames: { frame: string; state: "present" | "absent"; value: number }[];
}

export interface ARFrameResponse {
  state: TrackingState;
  capture_ms?: number | null; // streamed frames: when this frame was captured (the sender's performance.now())
  // Draw the model: tracking, or holding the last valid pose while lost / re-acquiring.
  visible: boolean;
  monitoring: boolean; // TRACKING but confidence below good_confidence ("tracking with warning")
  object: TrackedObject | null;
  // What the detector found on this frame (shown while INITIALIZING).
  detection: { class_label: string; confidence: number; bbox: number[]; polygon: Vector2[] | null } | null;
  // Filtered pose, renderer (Three.js) space. "model": the model's own pose
  // (no anchor offset). "markerless": approximate, position only.
  position: Vector3 | null;
  quaternion: QuaternionXYZW | null;
  rotation_deg: RotationDeg | null; // XYZ Euler of `quaternion`
  axes: PoseAxes | null; // raw measurement gizmo (model-based)
  approximate: boolean;
  detector_ran: boolean;
  tracker_ran: boolean;
  timings_ms: { detection: number; initialization: number; tracking: number; refinement: number; total: number };
  counters: {
    frame_index: number;
    detection_runs: number;
    detection_count: number; // 1 at first lock, +1 per re-acquisition — must not climb while tracking
    tracking_frames: number;
    frames_since_detection: number | null;
  };
  events: string[];
  part_checks: PartCheck[];
  calibrated_parts: string[]; // parts with a presence calibration, sent in every state
  good_confidence: number;
  lost_confidence: number;
  intrinsics: CameraIntrinsics;
  calibration_is_approximate: boolean;
  calibration_source: string;
  camera_vertical_fov_deg: number;
  camera_aspect: number;
  inspection?: ComponentInspection | null; // when ARFrameRequest.inspection is on
  tracking_status?: TrackingStatus;
  // Hands found in this frame (the mask kept out of tracking and inspection).
  hands?: {
    available: boolean;
    sources: string[];
    hands: number;
    persons: number;
    age_ms: number | null;
    outlines: Vector2[][];
    error: string | null;
  } | null;
  // Guide view (find by "guide"): where to draw the model's outline for the
  // operator to line the object up with, and how well it fits right now.
  guide?: {
    position: { x: number; y: number; z: number };
    quaternion: { x: number; y: number; z: number; w: number };
    saved: boolean;
    score: number | null;
    min_score: number;
  } | null;
  // Fixed-station mode: the frozen pose in use, and whether the part moved off it.
  station?: { locked: boolean; shift_px: number | null; moved: boolean; created_at: string | null; max_shift_px: number | null } | null;
  // What the model-based tracking confidence is made of.
  quality?: {
    overlap: number | null;
    flow_support: number | null;
    occlusion: number | null;
    held_ms: number | null;
    refine_skipped_for_hand: boolean;
    edge_fit?: number | null;
    edge_adjusted?: boolean;
    // Phase 8: which way round the tracked pose is, and how sure.
    orientation_status?: "unique" | "confirmed" | "ambiguous" | null;
    orientation_reason?: string | null;
    twin_flips_suppressed?: number;
    corrections_rejected?: number;
    orientation_switches?: number;
  } | null;
  // Pose features where the pose puts them vs what the image shows (diagnostics).
  pose_features?: PoseFeatureView[];
  calibration_note?: string | null;
  calibration_rms_px?: number | null;
  distortion_removed_px?: number;
}

export interface PoseFeatureView {
  feature_id: string;
  status: "detected" | "not_detected" | "occluded" | "hidden" | "facing_away" | "outside";
  evidence: number;
  predicted_center_px: Vector2 | null;
  predicted_contour_px: Vector2[];
  observed_center_px: Vector2 | null;
  residual_px: number | null;
}

// Feature/keypoint ("image target") tracking — see backend
// app/services/pose/feature_tracker.py. Real 6DoF (position + orientation),
// but requires a registered reference photo of a textured surface.
export interface RegisterReferenceImageResponse {
  feature_count: number;
  quality: "too_few" | "marginal" | "good";
}

export interface FeaturePoseResponse {
  found: boolean;
  position: Vector3 | null;
  quaternion: QuaternionXYZW | null;
  num_matches: number;
  num_inliers: number;
  inlier_points_px: Vector2[] | null;
  calibration_is_approximate: boolean;
  calibration_source: string;
  camera_vertical_fov_deg: number;
  camera_aspect: number;
  rotation_deg: RotationDeg | null;
  axes: PoseAxes | null;
}

// Component-level missing-part inspection (backend app/services/component_inspection).
export type ComponentStatus =
  | "NOT_INSPECTED"
  | "PRESENT"
  | "MISSING_CANDIDATE"
  | "MISSING_CONFIRMED"
  | "UNCERTAIN"
  | "OCCLUDED";

export type InspectionMode = "off" | "inspect" | "learn" | "paused";

export interface RegisteredComponent {
  id: string;
  type: string;
  display_name: string;
  required: boolean;
  inspection_mode: string;
  source: "mapping" | "named-part" | "name-match";
  node_index: number | null;
  node_name: string | null;
  piece: number | null; // which body of a combined mesh node
  mesh_status: "ok" | "missing" | "ambiguous" | "piece-missing" | "empty";
  problem: string | null;
  center_m: number[] | null; // in the tracked object's frame = the overlay group's local frame
  size_m: number[] | null;
  node_matrix: number[][] | null; // row-major 4x4, same frame
  color: string | null; // chosen in the mapping; null = automatic
  reference_samples: number;
  calibrated: boolean;
}

export interface ComponentRegistry {
  model_id: string;
  object_id: string;
  source: "mapping" | "auto";
  mapping_path: string | null;
  components: RegisteredComponent[];
  nodes: { node_index: number; name: string; sanitized_name: string; pieces: number; faces: number; component_ids: string[] }[];
  warnings: string[];
  detector: string | null;
}

export interface ComponentResult {
  id: string;
  status: ComponentStatus;
  reason: string;
  required: boolean;
  score: number | null; // smoothed probability present
  confidence: number | null;
  observable: boolean;
  visibility: string | null;
  visibility_reasons: string[];
  last_valid: string | null;
  valid_observations: number;
  center_px: number[] | null;
  bbox_px: number[] | null;
  outline_px: number[][] | null;
  verdicts: { verifier: string; decision: "present" | "absent" | "uncertain"; confidence: number; reason: string }[];
  history: string[];
  learn_samples: number | null;
}

export interface ComponentInspection {
  run_id: string;
  mode: "inspect" | "learn" | "paused";
  passes: number;
  inspected_this_frame: boolean;
  summary: {
    required: number;
    present: number;
    missing: number;
    candidates: number;
    uncertain: number;
    not_inspected: number;
    verdict: "PASS" | "FAIL" | "INCOMPLETE";
  };
  components: ComponentResult[];
  message: string | null;
  notes: string[];
}

// Saved inspection reports (backend app/api/inspection_reports.py).
export interface InspectionReportComponent {
  id: string;
  type: string | null;
  display_name: string;
  required: boolean;
  status: ComponentStatus;
  reason: string;
  score: number | null;
  valid_observations: number;
  last_valid: string | null;
  visibility: string | null;
  verdicts: { verifier: string; decision: string; confidence: number; reason: string }[];
}

export interface InspectionReport {
  id: string;
  created_at: string;
  model_id: string;
  model_name: string;
  object_id: string;
  station: string | null;
  operator: string | null;
  run_id: string;
  passes: number;
  verdict: "PASS" | "FAIL" | "INCOMPLETE";
  summary: ComponentInspection["summary"];
  components?: InspectionReportComponent[]; // detail only, not in the list
  notes: string[];
  note: string;
  disposition: "approved" | "rework" | null;
  disposition_at?: string;
  disposition_by?: string | null;
  evidence: boolean;
}

export interface SystemStatus {
  camera: { assumed_hfov_deg: number; vertical_fov_deg_720p: number; calibration_source: string; approximate: boolean };
  pose_service: { url: string; ok: boolean; device?: string; cuda?: boolean; model?: string; objects?: number; error?: string | null };
  hands: { method: string; available: boolean; mediapipe: boolean; person: boolean; error: string | null } | null;
  component_detector: string | null;
  inspection: { confirm_missing_after: number; confirm_present_after: number; interval_s: number };
}
