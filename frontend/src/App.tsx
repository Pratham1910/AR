import { useEffect, useState } from "react";
import { AssetsApi, ProceduresApi } from "./services/api";
import { InspectionRunner } from "./features/inspection/InspectionRunner";
import { Viewer3DPage } from "./features/viewer3d/Viewer3DPage";
import { ComponentsPage } from "./features/components/ComponentsPage";
import { ReportsPage } from "./features/reports/ReportsPage";
import { SettingsPage } from "./features/settings/SettingsPage";
import { useWorkstation } from "./workstation";
import type { Asset, Procedure } from "./types";
import "./App.css";

type Tab = "live" | "components" | "reports" | "procedure" | "settings";

const NAV_KEY = "tvasta.navCollapsed";

const ICONS: Record<Tab, JSX.Element> = {
  live: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2" />
      <circle cx="12" cy="12" r="3.5" />
    </svg>
  ),
  components: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <path d="M12 2 3 7v10l9 5 9-5V7z" />
      <path d="M3 7l9 5 9-5M12 12v10" />
    </svg>
  ),
  reports: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
      <path d="M14 3v6h6M8 13h8M8 17h5" />
    </svg>
  ),
  procedure: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <path d="M9 6h11M9 12h11M9 18h11" />
      <path d="M4 6h.01M4 12h.01M4 18h.01" strokeWidth="3" strokeLinecap="round" />
    </svg>
  ),
  settings: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <circle cx="12" cy="12" r="3" />
      <path d="M12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1" />
    </svg>
  ),
};

const NAV: [Tab, string][] = [
  ["live", "Live AR inspection"],
  ["components", "Models & components"],
  ["reports", "Reports"],
  ["procedure", "Procedure inspection"],
  ["settings", "Settings"],
];

export default function App() {
  const [tab, setTab] = useState<Tab>("live");
  const [assets, setAssets] = useState<Asset[]>([]);
  const [procedures, setProcedures] = useState<Procedure[]>([]);
  const [selectedAssetId, setSelectedAssetId] = useState<string>("");
  const [selectedRevisionId, setSelectedRevisionId] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openReportId, setOpenReportId] = useState<string | null>(null);
  const workstation = useWorkstation();
  const [navCollapsed, setNavCollapsedState] = useState(() => {
    try {
      return localStorage.getItem(NAV_KEY) === "1";
    } catch {
      return false;
    }
  });
  const setNavCollapsed = (collapsed: boolean) => {
    setNavCollapsedState(collapsed);
    try {
      localStorage.setItem(NAV_KEY, collapsed ? "1" : "0");
    } catch {
      // a per-device convenience only
    }
  };

  const refreshAssets = () => {
    AssetsApi.list()
      .then((list) => {
        setAssets(list);
        setError(null);
      })
      .catch((err: Error) => setError(err.message));
  };

  useEffect(() => {
    refreshAssets();
    ProceduresApi.list()
      .then(setProcedures)
      .catch((err: Error) => setError(err.message));
  }, []);

  const openReport = (reportId: string) => {
    setOpenReportId(reportId);
    setTab("reports");
  };

  const proceduresForAsset = procedures.filter((p) => p.asset_id === selectedAssetId);

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <svg className="brand-mark" width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
            <rect x="2" y="2" width="24" height="24" rx="4" fill="none" stroke="var(--accent)" strokeWidth="2" />
            <path d="M8 9h12M14 9v11" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="square" />
          </svg>
          <div>
            <h1>TVASTA</h1>
            <span>AR inspection</span>
          </div>
        </div>
        <div className="topbar-meta">
          <span>
            Station{" "}
            <button className="link-button" onClick={() => setTab("settings")}>
              {workstation.station || "set station"}
            </button>
          </span>
          <span>
            Operator{" "}
            <button className="link-button" onClick={() => setTab("settings")}>
              {workstation.operator || "set operator"}
            </button>
          </span>
        </div>
        <div className="topbar-chips">
          <span className={`chip-status ${error ? "bad" : "ok"}`} title={error ?? "The backend answered"}>
            <span className="dot" />
            {error ? "Backend unreachable" : "Backend online"}
          </span>
        </div>
      </header>

      <div className={`shell${navCollapsed ? " nav-collapsed" : ""}`}>
        <nav className="side-nav" aria-label="Main">
          {NAV.map(([value, label]) => (
            <button
              key={value}
              className={tab === value ? "active" : ""}
              onClick={() => setTab(value)}
              aria-current={tab === value ? "page" : undefined}
              title={label}
            >
              {ICONS[value]}
              <span className="nav-label">{label}</span>
            </button>
          ))}
          <div className="nav-spacer" />
          <button
            className="nav-collapse"
            onClick={() => setNavCollapsed(!navCollapsed)}
            aria-label={navCollapsed ? "Expand navigation" : "Collapse navigation"}
            title={navCollapsed ? "Expand navigation" : "Collapse navigation"}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
              <path d={navCollapsed ? "M9 6l6 6-6 6" : "M15 6l-6 6 6 6"} />
            </svg>
            <span className="nav-label">Collapse</span>
          </button>
        </nav>

        <div className="app">
          {error && tab !== "live" && <p className="error">{error}</p>}

          {tab === "procedure" && (
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

          {tab === "live" && <Viewer3DPage assets={assets} onAssetsChanged={refreshAssets} onOpenReport={openReport} />}
          {tab === "components" && <ComponentsPage assets={assets} />}
          {tab === "reports" && <ReportsPage openReportId={openReportId} onOpenReport={setOpenReportId} />}
          {tab === "settings" && <SettingsPage />}
        </div>
      </div>
    </>
  );
}
