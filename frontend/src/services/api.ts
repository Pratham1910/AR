import axios from "axios";
import type {
  AnchorOffset,
  ARFrameResponse,
  ARMode,
  Asset,
  FeaturePoseResponse,
  InspectionRun,
  Model3DInfo,
  Model3DPart,
  FitCheckResult,
  ObserveResponse,
  PresenceCalibrationResult,
  PoseResponse,
  Procedure,
  ProcedureSummary,
  RegisterReferenceImageResponse,
  StepValidationResponse,
} from "../types";
import type { ProcedurePackage } from "../features/viewer3d/procedure";

// 127.0.0.1, not "localhost": browsers try IPv6 [::1] first, and on this
// machine Docker can hold [::1]:8000 for another container (the FoundationPose
// image publishes 8000-8010), which accepts and then drops the connection
// (ERR_EMPTY_RESPONSE) while the real backend listens on 127.0.0.1.
const baseURL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export const api = axios.create({ baseURL });

// Absolute URL for a model/static path returned relative to the API (e.g.
// "/static/models/bottle.glb") — the <model-viewer>/Three.js loader needs a
// fully-qualified URL, not one relative to the frontend's own origin.
export const resolveApiUrl = (path: string) => `${baseURL}${path}`;
export const apiWebSocketUrl = (path: string) => `${baseURL.replace(/^http/, "ws")}${path}`;

export const AssetsApi = {
  list: () => api.get<Asset[]>("/api/assets").then((r) => r.data),
  create: (name: string, description?: string) =>
    api.post<Asset>("/api/assets", { name, description }).then((r) => r.data),
};

export const ProceduresApi = {
  list: () => api.get<Procedure[]>("/api/procedures").then((r) => r.data),
};

export const InspectionApi = {
  start: (assetId: string, procedureRevisionId: string, operator?: string) =>
    api
      .post<{ inspection_run_id: string; first_step_id: string; total_steps: number }>(
        "/api/inspection/start",
        { asset_id: assetId, procedure_revision_id: procedureRevisionId, operator }
      )
      .then((r) => r.data),

  get: (inspectionId: string) => api.get<InspectionRun>(`/api/inspection/${inspectionId}`).then((r) => r.data),

  observe: (inspectionId: string, stepId: string, imageBase64: string) =>
    api
      .post<ObserveResponse>(`/api/inspection/${inspectionId}/step/${stepId}/observe`, {
        image_base64: imageBase64,
      })
      .then((r) => r.data),

  validate: (inspectionId: string, stepId: string) =>
    api
      .post<StepValidationResponse>(`/api/inspection/${inspectionId}/step/${stepId}/validate`)
      .then((r) => r.data),

  complete: (inspectionId: string) =>
    api.post<{ id: string; status: string }>(`/api/inspection/${inspectionId}/complete`).then((r) => r.data),
};

export const Models3DApi = {
  listForAsset: (assetId: string) =>
    api.get<Model3DInfo[]>("/api/models3d", { params: { asset_id: assetId } }).then((r) => r.data),

  upload: (
    assetId: string,
    name: string,
    file: File,
    realWorldHeightM?: number,
    detectionClassLabel?: string
  ) => {
    const form = new FormData();
    form.append("asset_id", assetId);
    form.append("name", name);
    form.append("file", file);
    if (realWorldHeightM !== undefined) form.append("real_world_height_m", String(realWorldHeightM));
    if (detectionClassLabel) form.append("detection_class_label", detectionClassLabel);
    // No explicit Content-Type here — axios sets the multipart boundary itself from the FormData.
    return api.post<Model3DInfo>("/api/models3d/upload", form).then((r) => r.data);
  },

  updateAnchor: (modelId: string, anchor: AnchorOffset) =>
    api.patch<Model3DInfo>(`/api/models3d/${modelId}/anchor`, anchor).then((r) => r.data),

  updateSettings: (modelId: string, settings: { real_world_height_m?: number; detection_class_label?: string }) =>
    api.patch<Model3DInfo>(`/api/models3d/${modelId}`, settings).then((r) => r.data),

  remove: (modelId: string) => api.delete(`/api/models3d/${modelId}`).then(() => undefined),

  parts: (modelId: string) => api.get<Model3DPart[]>(`/api/models3d/${modelId}/parts`).then((r) => r.data),

  calibratePresence: (modelId: string, nodeIndex: number, presentLabel: string, absentLabel: string) =>
    api
      .post<PresenceCalibrationResult>(`/api/models3d/${modelId}/parts/${nodeIndex}/presence-calibration`, {
        present_label: presentLabel,
        absent_label: absentLabel,
      })
      .then((r) => r.data),

  renamePart: (modelId: string, nodeIndex: number, displayName: string) =>
    api
      .put<Model3DPart[]>(`/api/models3d/${modelId}/parts/${nodeIndex}`, { display_name: displayName })
      .then((r) => r.data),

  // Does the GLB have the real object's shape? Compares outlines in one camera frame.
  fitCheck: (modelId: string, imageBase64: string) =>
    api
      .post<FitCheckResult>(`/api/models3d/${modelId}/fit-check`, { image_base64: imageBase64 })
      .then((r) => r.data),

  // Procedure authored in Vishwa (.procedure.json), one per model.
  procedure: (modelId: string) =>
    api.get<ProcedurePackage>(`/api/models3d/${modelId}/procedure`).then((r) => r.data),

  procedureSummary: (modelId: string) =>
    api.get<ProcedureSummary>(`/api/models3d/${modelId}/procedure/summary`).then((r) => r.data),

  uploadProcedure: (modelId: string, json: unknown) =>
    api.put<ProcedureSummary>(`/api/models3d/${modelId}/procedure`, json).then((r) => r.data),

  removeProcedure: (modelId: string) => api.delete(`/api/models3d/${modelId}/procedure`).then(() => undefined),
};

// The server's `detail` message when there is one (e.g. "'bot' is not a class
// the detector knows…"), instead of axios's generic "status code 400".
export const apiErrorMessage = (err: unknown): string => {
  const detail = axios.isAxiosError(err) ? err.response?.data?.detail : undefined;
  if (typeof detail === "string") return detail;
  return err instanceof Error ? err.message : String(err);
};

export const VisionApi = {
  // Class labels the object detector can find — anything else is never detected.
  classes: () => api.get<string[]>("/api/vision/classes").then((r) => r.data),

  estimatePose: (imageBase64: string, targetMarkerId?: number) =>
    api
      .post<PoseResponse>("/api/vision/pose", { image_base64: imageBase64, target_marker_id: targetMarkerId })
      .then((r) => r.data),

  registerReferenceImage: (assetId: string, imageBase64: string, labelWidthM: number, labelHeightM: number) =>
    api
      .post<RegisterReferenceImageResponse>("/api/vision/reference-image", {
        asset_id: assetId,
        image_base64: imageBase64,
        label_width_m: labelWidthM,
        label_height_m: labelHeightM,
      })
      .then((r) => r.data),

  estimateFeaturePose: (assetId: string, imageBase64: string) =>
    api
      .post<FeaturePoseResponse>("/api/vision/feature-pose", { asset_id: assetId, image_base64: imageBase64 })
      .then((r) => r.data),

  // One frame through the session's SEARCHING/TRACKING/LOST state machine.
  arFrame: (
    sessionId: string,
    mode: ARMode,
    imageBase64: string,
    classLabel: string | null, // null: find the object by its 3D model (model-based mode)
    options: { modelId?: string; realWorldHeightM?: number; trackPart?: number | null; detectBy?: "model" | "class" }
  ) =>
    api
      .post<ARFrameResponse>("/api/vision/ar-session/frame", {
        session_id: sessionId,
        mode,
        image_base64: imageBase64,
        detect_by: options.detectBy ?? "class",
        class_label: classLabel,
        model_id: options.modelId,
        real_world_height_m: options.realWorldHeightM,
        track_part: options.trackPart ?? null,
      })
      .then((r) => r.data),

  // One raw frame of a recorded clip (replayed by app/workers/replay_clip.py). Index 0 starts the clip afresh.
  addClipFrame: (clip: string, index: number, tMs: number, imageBase64: string) =>
    api
      .post<{ clip: string; frames: number }>(`/api/vision/clips/${encodeURIComponent(clip)}/frames`, {
        index,
        t_ms: tMs,
        image_base64: imageBase64,
      })
      .then((r) => r.data),

  // Keeps the raw camera frame (full resolution, no overlay) for offline analysis.
  saveFrame: (imageBase64: string, label: string) =>
    api
      .post<{ saved: string; width: number; height: number }>("/api/vision/debug-frame", {
        image_base64: imageBase64,
        label,
      })
      .then((r) => r.data),

  // Drops the session's tracking state; the next frame starts SEARCHING.
  endArSession: (sessionId: string) => api.delete(`/api/vision/ar-session/${sessionId}`).then(() => undefined),
};
