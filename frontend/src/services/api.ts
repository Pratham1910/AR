import axios from "axios";
import type {
  Asset,
  InspectionRun,
  ObserveResponse,
  Procedure,
  StepValidationResponse,
} from "../types";

const baseURL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export const api = axios.create({ baseURL });

export const AssetsApi = {
  list: () => api.get<Asset[]>("/api/assets").then((r) => r.data),
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
