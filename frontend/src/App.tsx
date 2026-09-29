import { useEffect, useState } from "react";
import { AssetsApi, ProceduresApi } from "./services/api";
import { InspectionRunner } from "./features/inspection/InspectionRunner";
import { Viewer3DPage } from "./features/viewer3d/Viewer3DPage";
import type { Asset, Procedure } from "./types";
import "./App.css";

type Tab = "inspection" | "3d";

export default function App() {
  const [tab, setTab] = useState<Tab>("inspection");
  const [assets, setAssets] = useState<Asset[]>([]);
  const [procedures, setProcedures] = useState<Procedure[]>([]);
  const [selectedAssetId, setSelectedAssetId] = useState<string>("");
  const [selectedRevisionId, setSelectedRevisionId] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshAssets = () => {
    AssetsApi.list()
      .then(setAssets)
      .catch((err: Error) => setError(err.message));
  };

  useEffect(() => {
    refreshAssets();
    ProceduresApi.list()
      .then(setProcedures)
      .catch((err: Error) => setError(err.message));
  }, []);

  const proceduresForAsset = procedures.filter((p) => p.asset_id === selectedAssetId);

  return (
    <div className="app">
      <header>
        <h1>TVASTA — Procedure & QA Platform</h1>
        <p className="subtitle">
          Phase 1: camera-based procedure execution and deterministic QA · Phase 4/5: 3D viewer and
          marker-based physical↔3D registration
        </p>
      </header>

      <nav className="tabs">
        <button className={tab === "inspection" ? "active" : ""} onClick={() => setTab("inspection")}>
          Procedure Inspection
        </button>
        <button className={tab === "3d" ? "active" : ""} onClick={() => setTab("3d")}>
          3D / AR Registration
        </button>
      </nav>

      {error && <p className="error">{error}</p>}

      {tab === "inspection" && (
        <>
          {!running && (
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

              <label>
                Procedure Revision
                <select value={selectedRevisionId} onChange={(e) => setSelectedRevisionId(e.target.value)}>
                  <option value="">Select a procedure…</option>
                  {proceduresForAsset.flatMap((p) =>
                    p.revisions
                      .filter((r) => r.status === "published")
                      .map((r) => (
                        <option key={r.id} value={r.id}>
                          {p.title} — rev {r.revision_label} ({r.step_count} steps)
                        </option>
                      ))
                  )}
                </select>
              </label>

              <button disabled={!selectedAssetId || !selectedRevisionId} onClick={() => setRunning(true)}>
                Start Inspection
              </button>
            </div>
          )}

          {running && selectedAssetId && selectedRevisionId && (
            <InspectionRunner assetId={selectedAssetId} procedureRevisionId={selectedRevisionId} />
          )}
        </>
      )}

      {tab === "3d" && <Viewer3DPage assets={assets} onAssetsChanged={refreshAssets} />}
    </div>
  );
}
