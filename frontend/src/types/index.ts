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
  url: string; // relative to the API base URL, e.g. /static/models/bottle.glb
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
}

// Markerless (approximate) registration — see backend
// app/services/pose/markerless.py for exactly what is and isn't estimated
// (position only from apparent size, no orientation).
export interface ObjectRegistrationResponse {
  found: boolean;
  class_label: string | null;
  confidence: number | null;
  bbox: [number, number, number, number] | null;
  polygon: Vector2[] | null;
  position: Vector3 | null;
  quaternion: QuaternionXYZW | null;
  approximate: boolean;
  calibration_is_approximate: boolean;
  calibration_source: string;
}
