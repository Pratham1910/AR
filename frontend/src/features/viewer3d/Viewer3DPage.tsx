import { useEffect, useMemo, useState } from "react";
import { Models3DApi, resolveApiUrl } from "../../services/api";
import type { Asset, Model3DInfo, Model3DPart } from "../../types";
import { PartsPanel } from "./PartsPanel";
import { ProcedurePanel } from "./ProcedurePanel";
import { ProcedureHost } from "./procedure";
import { EMPTY_PART_VIEW, type PartView } from "./parts";
import { ThreeViewer } from "./ThreeViewer";
import { SimpleCameraOverlay } from "./SimpleCameraOverlay";
import { RegistrationOverlay } from "./RegistrationOverlay";
import { UploadModelForm } from "./UploadModelForm";
import { ModelSettingsPanel } from "./ModelSettingsPanel";

interface Props {
  assets: Asset[];
  onAssetsChanged: () => void;
  /** Preselect this asset (e.g. the product a scanned marker identified). */
  initialAssetId?: string;
  /** With initialAssetId: open AR Registration in marker mode, locked onto this marker id. */
  targetMarkerId?: number;
}

/**
 * Phase 4 + 5 test page: pick any asset that has a registered Model3D (the
 * seed script registers BOTTLE-001 -> bottle.glb, Project.md's provided
 * placeholder model) and either view it standalone or attempt physical<->3D
 * registration against a live camera feed. UploadModelForm lets you add
 * more assets/models without hand-writing API calls.
 */
export function Viewer3DPage({ assets, onAssetsChanged, initialAssetId, targetMarkerId }: Props) {
  const [selectedAssetId, setSelectedAssetId] = useState(initialAssetId ?? "");
  const [models, setModels] = useState<Model3DInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [mode, setMode] = useState<"view" | "overlay" | "register">(
    targetMarkerId !== undefined ? "register" : "view"
  );
  const [error, setError] = useState<string | null>(null);
  const [showUpload, setShowUpload] = useState(false);

  const refreshModels = (assetId: string, keepModelId?: string) => {
    if (!assetId) {
      setModels([]);
      return;
    }
    Models3DApi.listForAsset(assetId)
      .then((list) => {
        setModels(list);
        // Backend returns newest-first; default to it, but let the user
        // pick a different one explicitly when an asset has several models —
        // silently picking models[0] with no visible indicator was the root
        // cause of at least one asset/model mismatch bug.
        const keep = keepModelId && list.some((m) => m.id === keepModelId);
        setSelectedModelId(keep ? keepModelId : (list[0]?.id ?? ""));
      })
      .catch((err: Error) => setError(err.message));
  };

  useEffect(() => {
    refreshModels(selectedAssetId);
  }, [selectedAssetId]);

  const model = models.find((m) => m.id === selectedModelId) ?? models[0];

  // The selected model's assembly parts (body, cap, …) and which are shown /
  // highlighted — shared by all three views, reset when the model changes.
  const [parts, setParts] = useState<Model3DPart[]>([]);
  const [partView, setPartView] = useState<PartView>(EMPTY_PART_VIEW);
  const modelId = model?.id;
  // Links the procedure panel to whichever view is drawing the model.
  const procedureHost = useMemo(() => new ProcedureHost(), []);
  useEffect(() => {
    setParts([]);
    setPartView(EMPTY_PART_VIEW);
    if (!modelId) return;
    Models3DApi.parts(modelId)
      .then(setParts)
      .catch(() => setParts([])); // a single-part model simply has no parts panel
  }, [modelId]);

  const VIEWS = [
    ["view", "3D model"],
    ["overlay", "Camera overlay"],
    ["register", "AR tracking"],
  ] as const;

  return (
    <div>
      <div className="workspace-toolbar">
        <label className="field">
          Asset
          <select value={selectedAssetId} onChange={(e) => setSelectedAssetId(e.target.value)}>
            <option value="">Select an asset…</option>
            {assets.map((a) => (
              <option key={a.id} value={a.id}>
                {a.name}
              </option>
            ))}
          </select>
        </label>
        {models.length > 1 && (
          <label className="field">
            Model
            <select value={selectedModelId} onChange={(e) => setSelectedModelId(e.target.value)}>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                  {m.real_height_m !== null ? ` · ${(m.real_height_m * 100).toFixed(1)} cm` : ""}
                </option>
              ))}
            </select>
          </label>
        )}
        <div className="spacer" />
        <button className={showUpload ? "" : "primary"} onClick={() => setShowUpload((v) => !v)}>
          {showUpload ? "Cancel upload" : "+ Upload 3D model"}
        </button>
      </div>

      {showUpload && (
        <UploadModelForm
          assets={assets}
          onUploaded={(assetId) => {
            onAssetsChanged(); // in case a new asset was created
            setSelectedAssetId(assetId);
            refreshModels(assetId);
            setShowUpload(false);
          }}
        />
      )}

      {error && <p className="error">{error}</p>}

      {!selectedAssetId && (
        <div className="card">
          <p className="hint">Pick an asset above to open its 3D model, or upload a new one.</p>
        </div>
      )}
      {selectedAssetId && models.length === 0 && !error && (
        <div className="card">
          <p className="hint">No 3D model registered for this asset yet — use "+ Upload 3D model".</p>
        </div>
      )}

      {model && (
        <>
          <ModelSettingsPanel
            model={model}
            onSaved={() => refreshModels(selectedAssetId, model.id)}
            onDeleted={() => refreshModels(selectedAssetId)}
          />

          <div className="view-bar">
            <div className="segmented" role="tablist">
              {VIEWS.map(([value, label]) => (
                <button
                  key={value}
                  role="tab"
                  data-view={value}
                  className={mode === value ? "active" : ""}
                  onClick={() => setMode(value)}
                >
                  {label}
                </button>
              ))}
            </div>
            <span className="hint">
              {mode === "view" && "Orbit, zoom and click parts to select them."}
              {mode === "overlay" && "The model floats over the camera feed — no tracking."}
              {mode === "register" && "Lock the model onto the real object and follow it live."}
            </span>
          </div>

          <div className="viewer-with-parts">
            <div className="viewer-main">
              {mode === "view" && (
                <ThreeViewer
                  modelUrl={resolveApiUrl(model.url)}
                  height={560}
                  partView={partView}
                  onPartClick={(nodeIndex) => setPartView((v) => ({ ...v, highlight: nodeIndex }))}
                  procedureHost={procedureHost}
                />
              )}
              {mode === "overlay" && <SimpleCameraOverlay modelUrl={resolveApiUrl(model.url)} partView={partView} />}
              {mode === "register" && (
                <RegistrationOverlay
                  key={model.id}
                  assetId={selectedAssetId}
                  modelId={model.id}
                  modelUrl={resolveApiUrl(model.url)}
                  initialAnchor={{
                    anchor_offset_x: model.anchor_offset_x,
                    anchor_offset_y: model.anchor_offset_y,
                    anchor_offset_z: model.anchor_offset_z,
                    anchor_rotation_x: model.anchor_rotation_x,
                    anchor_rotation_y: model.anchor_rotation_y,
                    anchor_rotation_z: model.anchor_rotation_z,
                    anchor_rotation_w: model.anchor_rotation_w,
                  }}
                  modelScale={model.scale}
                  targetMarkerId={targetMarkerId}
                initialMode={targetMarkerId !== undefined ? "marker" : undefined}
                defaultTargetClassLabel={model.component_class_label}
                  partView={partView}
                  parts={parts}
                  procedureHost={procedureHost}
                />
              )}
            </div>
            <aside className="side-panels">
              {mode !== "overlay" && (
                <ProcedurePanel
                  key={model.id}
                  modelId={model.id}
                  parts={parts}
                  host={procedureHost}
                  cameraChecks={mode === "register"}
                />
              )}
              {parts.length > 1 && (
                <PartsPanel
                  modelId={model.id}
                  parts={parts}
                  view={partView}
                  onViewChange={setPartView}
                  onPartsChange={setParts}
                />
              )}
            </aside>
          </div>
        </>
      )}
    </div>
  );
}
