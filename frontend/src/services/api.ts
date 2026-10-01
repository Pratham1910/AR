import axios from "axios";
import type {
  AnchorOffset,
  ARFrameResponse,
  ARMode,
  Asset,
  FeaturePoseResponse,
  InspectionRun,
  Model3DInfo,
  ObserveResponse,
  PoseResponse,
  Procedure,
  RegisterReferenceImageResponse,
  StepValidationResponse,
} from "../types";

const baseURL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export const api = axios.create({ baseURL });

// Absolute URL for a model/static path returned relative to the API (e.g.
// "/static/models/bottle.glb") — the <model-viewer>/Three.js loader needs a
// fully-qualified URL, not one relative to the frontend's own origin.
export const resolveApiUrl = (path: string) => `${baseURL}${path}`;

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
    classLabel: string,
    options: { modelId?: string; realWorldHeightM?: number }
  ) =>
    api
      .post<ARFrameResponse>("/api/vision/ar-session/frame", {
        session_id: sessionId,
        mode,
        image_base64: imageBase64,
        class_label: classLabel,
        model_id: options.modelId,
        real_world_height_m: options.realWorldHeightM,
      })
      .then((r) => r.data),

  // Drops the session's tracking state; the next frame starts SEARCHING.
  endArSession: (sessionId: string) => api.delete(`/api/vision/ar-session/${sessionId}`).then(() => undefined),
};
