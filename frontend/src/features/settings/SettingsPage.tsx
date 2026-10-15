import { useEffect, useState } from "react";
import { SystemApi, apiErrorMessage } from "../../services/api";
import { saveWorkstation, useWorkstation } from "../../workstation";
import type { SystemStatus } from "../../types";

/*
 * Settings: who and where this workstation is (stamped on reports), and what
 * the system is running with — lens model, pose service, hand and component
 * detectors (GET /api/system/status).
 */

export function SettingsPage() {
  const workstation = useWorkstation();
  const [station, setStation] = useState(workstation.station);
  const [operator, setOperator] = useState(workstation.operator);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  const refresh = () => {
    setError(null);
    SystemApi.status()
      .then(setStatus)
      .catch((err) => setError(apiErrorMessage(err)));
  };
  useEffect(refresh, []);

  const ok = (value: boolean) => <span className={`tag ${value ? "ok" : "bad"}`}>{value ? "online" : "unavailable"}</span>;

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <div className="section-label">Settings</div>
          <h2 className="page-title">Workstation</h2>
        </div>
      </div>

      <div className="page-grid">
        <section className="card" aria-label="Station and operator">
          <h3>Station and operator</h3>
          <p className="hint">Shown in the top bar and stamped on every saved report. Kept on this device.</p>
          <div className="form-grid">
            <label className="field">
              Station ID
              <input value={station} onChange={(e) => { setStation(e.target.value); setSaved(false); }} placeholder="e.g. LINE2-CELL-A" />
            </label>
            <label className="field">
              Operator
              <input value={operator} onChange={(e) => { setOperator(e.target.value); setSaved(false); }} placeholder="Name or badge number" />
            </label>
          </div>
          <div className="button-row">
            <button
              className="primary"
              onClick={() => {
                saveWorkstation({ station: station.trim(), operator: operator.trim() });
                setSaved(true);
              }}
            >
              Save
            </button>
            {saved && <span className="tag ok">Saved</span>}
          </div>
        </section>

        <div className="page-side">
          <section className="card" aria-label="System status">
            <div className="card-head">
              <h3>System</h3>
              <div className="spacer" />
              <button className="ghost" onClick={refresh}>
                Refresh
              </button>
            </div>
            {error && <p className="error">{error}</p>}
            {status && (
              <dl className="kv">
                <dt>Pose service (MegaPose)</dt>
                <dd>
                  {ok(status.pose_service.ok)}{" "}
                  <span className="hint mono">
                    {status.pose_service.ok ? `${status.pose_service.device} · ${status.pose_service.objects} models` : status.pose_service.error}
                  </span>
                </dd>
                <dt>Hand detector</dt>
                <dd>
                  {status.hands ? (
                    <>
                      {ok(status.hands.available)}{" "}
                      <span className="hint mono">
                        {[status.hands.mediapipe && "MediaPipe", status.hands.person && "person masks"].filter(Boolean).join(" + ") ||
                          status.hands.error}
                      </span>
                    </>
                  ) : (
                    <span className="tag">off</span>
                  )}
                </dd>
                <dt>Component detector</dt>
                <dd className="hint mono">{status.component_detector ?? "not configured"}</dd>
                <dt>Lens model</dt>
                <dd className="mono">
                  {status.camera.assumed_hfov_deg}° horizontal · {status.camera.vertical_fov_deg_720p}° vertical{" "}
                  <span className={`tag ${status.camera.approximate ? "warn" : "ok"}`}>
                    {status.camera.approximate ? "approximate" : "calibrated"}
                  </span>
                </dd>
                <dt>Missing confirmed after</dt>
                <dd className="mono">{status.inspection.confirm_missing_after} valid observations</dd>
              </dl>
            )}
          </section>
          <section className="card" aria-label="Camera tips">
            <h3>Camera</h3>
            <ul className="hint tips">
              <li>Logitech BRIO at 90° (Logi Tune) matches the 82° lens model; for 78° set CAMERA_ASSUMED_HFOV_DEG=70, for 65° set 58.</li>
              <li>Turn RightLight off in Logi Tune and use "Lock exposure" before learning or inspecting.</li>
              <li>Plug the BRIO into a rear USB 3.0 port.</li>
            </ul>
          </section>
        </div>
      </div>
    </div>
  );
}
