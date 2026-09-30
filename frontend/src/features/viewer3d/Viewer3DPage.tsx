import { useEffect, useState } from "react";
import { Models3DApi, resolveApiUrl } from "../../services/api";
import type { Asset, Model3DInfo } from "../../types";
import { ThreeViewer } from "./ThreeViewer";
import { SimpleCameraOverlay } from "./SimpleCameraOverlay";
import { RegistrationOverlay } from "./RegistrationOverlay";
import { UploadModelForm } from "./UploadModelForm";

interface Props {
  assets: Asset[];
  onAssetsChanged: () => void;
}

/**
 * Phase 4 + 5 test page: pick any asset that has a registered Model3D (the
 * seed script registers BOTTLE-001 -> bottle.glb, Project.md's provided
 * placeholder model) and either view it standalone or attempt physical<->3D
 * registration against a live camera feed. UploadModelForm lets you add
 * more assets/models without hand-writing API calls.
 */
export function Viewer3DPage({ assets, onAssetsChanged }: Props) {
  const [selectedAssetId, setSelectedAssetId] = useState("");
  const [models, setModels] = useState<Model3DInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [mode, setMode] = useState<"view" | "overlay" | "register">("view");
  const [error, setError] = useState<string | null>(null);
  const [showUpload, setShowUpload] = useState(false);

  const refreshModels = (assetId: string) => {
    if (!assetId) {
      setModels([]);
      return;
    }
    Models3DApi.listForAsset(assetId)
      .then((list) => {
        setModels(list);
        // Backend now returns newest-first; default to it, but let the user
        // pick a different one explicitly when an asset has several models —
        // silently picking models[0] with no visible indicator was the root
        // cause of at least one asset/model mismatch bug.
        setSelectedModelId(list[0]?.id ?? "");
      })
      .catch((err: Error) => setError(err.message));
  };

  useEffect(() => {
    refreshModels(selectedAssetId);
  }, [selectedAssetId]);

  const model = models.find((m) => m.id === selectedModelId) ?? models[0];

  return (
    <div>
      <div className="setup">
        <label>
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
          <label>
            Model
            <select value={selectedModelId} onChange={(e) => setSelectedModelId(e.target.value)}>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} ({m.storage_key}, scale={m.scale})
                </option>
              ))}
            </select>
          </label>
        )}
        <button onClick={() => setShowUpload((v) => !v)}>{showUpload ? "Hide upload" : "+ Upload 3D model"}</button>
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

      {selectedAssetId && models.length === 0 && !error && (
        <p>No 3D model registered for this asset yet — use "+ Upload 3D model" above.</p>
      )}

      {model && (
        <>
          <div className="mode-toggle">
            <button className={mode === "view" ? "active" : ""} onClick={() => setMode("view")}>
              3D Viewer (Phase 4)
            </button>
            <button className={mode === "overlay" ? "active" : ""} onClick={() => setMode("overlay")}>
              Camera Overlay (no detection)
            </button>
            <button className={mode === "register" ? "active" : ""} onClick={() => setMode("register")}>
              AR Registration (Phase 5)
            </button>
          </div>

          {mode === "view" && <ThreeViewer modelUrl={resolveApiUrl(model.url)} />}
          {mode === "overlay" && <SimpleCameraOverlay modelUrl={resolveApiUrl(model.url)} />}
          {mode === "register" && (
            <>
              <p className="hint">
                Print a marker with <code>python -m app.workers.generate_marker</code>, place it next to the
                physical object, and measure its printed side length against <code>ARUCO_MARKER_LENGTH_M</code>{" "}
                in <code>.env</code>.
              </p>
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
                defaultTargetClassLabel={model.component_class_label}
              />
            </>
          )}
        </>
      )}
    </div>
  );
}
