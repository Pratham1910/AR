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
export type TrackingState = "SEARCHING" | "INITIALIZING" | "TRACKING" | "LOST" | "RECOVERING";

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
export interface PartCheck {
  node_index: number;
  part_name: string;
  state: "present" | "absent" | "uncertain";
  confidence: number;
  brightness: number;
  region_px: [number, number, number, number];
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
  frames_without_object: string[];
}

export interface ARFrameResponse {
  state: TrackingState;
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
