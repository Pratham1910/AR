import { useEffect, useState } from "react";
import { Models3DApi, resolveApiUrl } from "../../services/api";
import type { Asset, Model3DInfo } from "../../types";
import { ThreeViewer } from "./ThreeViewer";
import { RegistrationOverlay } from "./RegistrationOverlay";

interface Props {
  assets: Asset[];
}

/**
 * Phase 4 + 5 test page: pick any asset that has a registered Model3D (the
 * seed script registers BOTTLE-001 -> bottle.glb, Project.md's provided
 * placeholder model) and either view it standalone or attempt physical<->3D
 * registration against a live camera feed of the printed ArUco marker.
 */
export function Viewer3DPage({ assets }: Props) {
  const [selectedAssetId, setSelectedAssetId] = useState("");
  const [models, setModels] = useState<Model3DInfo[]>([]);
  const [mode, setMode] = useState<"view" | "register">("view");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedAssetId) {
      setModels([]);
      return;
    }
    Models3DApi.listForAsset(selectedAssetId)
      .then(setModels)
      .catch((err: Error) => setError(err.message));
  }, [selectedAssetId]);

  const model = models[0];

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
      </div>

      {error && <p className="error">{error}</p>}

      {selectedAssetId && models.length === 0 && !error && (
        <p>No 3D model registered for this asset yet (POST /api/models3d to add one).</p>
      )}

      {model && (
        <>
          <div className="mode-toggle">
            <button className={mode === "view" ? "active" : ""} onClick={() => setMode("view")}>
              3D Viewer (Phase 4)
            </button>
            <button className={mode === "register" ? "active" : ""} onClick={() => setMode("register")}>
              AR Registration (Phase 5)
            </button>
          </div>

          {mode === "view" && <ThreeViewer modelUrl={resolveApiUrl(model.url)} />}
          {mode === "register" && (
            <>
              <p className="hint">
                Print a marker with <code>python -m app.workers.generate_marker</code>, place it next to the
                physical object, and measure its printed side length against <code>ARUCO_MARKER_LENGTH_M</code>{" "}
                in <code>.env</code>.
              </p>
              <RegistrationOverlay
                assetId={selectedAssetId}
                modelUrl={resolveApiUrl(model.url)}
                modelScale={model.scale}
              />
            </>
          )}
        </>
      )}
    </div>
  );
}
