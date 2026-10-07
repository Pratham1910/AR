import { useEffect, useState } from "react";
import { AssetsApi, ProceduresApi } from "./services/api";
import { InspectionRunner } from "./features/inspection/InspectionRunner";
import { Viewer3DPage } from "./features/viewer3d/Viewer3DPage";
import { MarkerScanPage } from "./features/markers/MarkerScanPage";
import type { Asset, Procedure } from "./types";
import "./App.css";

type Tab = "inspection" | "3d" | "markers";

export default function App() {
  const [tab, setTab] = useState<Tab>("inspection");
  const [assets, setAssets] = useState<Asset[]>([]);
  const [procedures, setProcedures] = useState<Procedure[]>([]);
  const [selectedAssetId, setSelectedAssetId] = useState<string>("");
  const [selectedRevisionId, setSelectedRevisionId] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Set when a scanned marker hands its product over to the 3D / AR tab.
  const [scanned3D, setScanned3D] = useState<{ assetId: string; markerId?: number } | null>(null);

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
    <>
      <header className="app-header">
        <div className="brand">
          <div className="brand-mark">T</div>
          <div>
            <h1>TVASTA</h1>
            <span>Procedure &amp; QA platform</span>
          </div>
        </div>
        <nav className="tabs">
          <button className={tab === "inspection" ? "active" : ""} onClick={() => setTab("inspection")}>
            Procedure inspection
          </button>
          <button
          className={tab === "3d" ? "active" : ""}
          onClick={() => {
            setScanned3D(null);
            setTab("3d");
          }}
        >
            3D &amp; AR workspace
          </button>
          <button className={tab === "markers" ? "active" : ""} onClick={() => setTab("markers")}>
          Marker Scan
        </button>
      </nav>
      </header>

    <div className="app">
      {error && <p className="error">{error}</p>}

      {tab === "inspection" && (
        <>
          {!running && (
            <div className="setup card">
              <div>
                <h2>Start an inspection</h2>
                <p className="hint">Pick the asset and the published procedure revision to run against the camera.</p>
              </div>
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

              <button className="primary big" disabled={!selectedAssetId || !selectedRevisionId} onClick={() => setRunning(true)}>
                Start inspection
              </button>
            </div>
          )}

          {running && selectedAssetId && selectedRevisionId && (
            <InspectionRunner assetId={selectedAssetId} procedureRevisionId={selectedRevisionId} />
          )}
        </>
      )}

      {tab === "3d" && (
        <Viewer3DPage
          assets={assets}
          onAssetsChanged={refreshAssets}
          initialAssetId={scanned3D?.assetId}
          targetMarkerId={scanned3D?.markerId}
        />
      )}

      {tab === "markers" && (
        <MarkerScanPage
          assets={assets}
          onOpenIn3D={(assetId, markerId) => {
            setScanned3D({ assetId, markerId });
            setTab("3d");
          }}
          onStartProcedure={(assetId, revisionId) => {
            setSelectedAssetId(assetId);
            setSelectedRevisionId(revisionId);
            setRunning(false);
            setTab("inspection");
          }}
        />
      )}
    </div>
    </>
  );
}
