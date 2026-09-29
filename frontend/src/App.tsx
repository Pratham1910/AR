import { useEffect, useState } from "react";
import { AssetsApi, ProceduresApi } from "./services/api";
import { InspectionRunner } from "./features/inspection/InspectionRunner";
import type { Asset, Procedure } from "./types";
import "./App.css";

export default function App() {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [procedures, setProcedures] = useState<Procedure[]>([]);
  const [selectedAssetId, setSelectedAssetId] = useState<string>("");
  const [selectedRevisionId, setSelectedRevisionId] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([AssetsApi.list(), ProceduresApi.list()])
      .then(([a, p]) => {
        setAssets(a);
        setProcedures(p);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  const proceduresForAsset = procedures.filter((p) => p.asset_id === selectedAssetId);

  return (
    <div className="app">
      <header>
        <h1>TVASTA — Procedure & QA Platform</h1>
        <p className="subtitle">Phase 1: Camera-based procedure execution and deterministic QA</p>
      </header>

      {error && <p className="error">{error}</p>}

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
    </div>
  );
}
