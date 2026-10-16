import { useEffect, useState } from "react";
import { apiErrorMessage, TrackingModelApi, type TrackingModelInfo } from "../../services/api";

/**
 * "Optimize for tracking" (backend app/services/model3d/tracking_shell.py):
 * keep only the triangles a camera can ever see — the drive inside an
 * enclosure, boards behind panels are dropped. Tracking (MegaPose, the 3D
 * edge fit, the orientation check) uses that shell; the original stays what
 * is displayed and inspected.
 */
export function TrackingModelCard({ modelId }: { modelId: string }) {
  const [info, setInfo] = useState<TrackingModelInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setInfo(null);
    TrackingModelApi.get(modelId)
      .then(setInfo)
      .catch((err) => setError(apiErrorMessage(err)));
  }, [modelId]);

  const run = async (action: () => Promise<TrackingModelInfo>) => {
    setBusy(true);
    setError(null);
    try {
      setInfo(await action());
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const cm = (v: number) => (v * 100).toFixed(1);

  return (
    <section className="card" aria-label="Tracking model">
      <h3>Tracking model</h3>
      <p className="hint">
        Optimize for tracking keeps only what a camera can see (outer surfaces, ports, holes); hidden internals such as a
        drive inside its enclosure are dropped. Tracking uses it from the next session; the model shown and inspected is
        unchanged.
      </p>
      {info?.built && (
        <div className="hint">
          {info.stale ? (
            <p className="warning">Built from an older version of this model — not used. Optimize again.</p>
          ) : (
            <p>
              In use: {info.faces_before?.toLocaleString()} → <strong>{info.faces_after?.toLocaleString()}</strong> triangles
              {info.size_m && ` · ${cm(info.size_m[0])} × ${cm(info.size_m[1])} × ${cm(info.size_m[2])} cm (check against the real part)`}
            </p>
          )}
          <ul className="node-names">
            {info.nodes.map((n) => (
              <li key={n.node_index}>
                #{n.node_index} {n.name}: {n.faces_before} → {n.faces_after}
                {n.faces_after === 0 ? " (hidden — dropped)" : ""}
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="calibration-row">
        <button onClick={() => void run(() => TrackingModelApi.build(modelId))} disabled={busy}>
          {busy ? "Optimizing… (~20 s)" : info?.built ? "Optimize again" : "Optimize for tracking"}
        </button>
        {info?.built && (
          <button onClick={() => void run(() => TrackingModelApi.remove(modelId))} disabled={busy}>
            Use the full model
          </button>
        )}
      </div>
      {error && <p className="error">{error}</p>}
    </section>
  );
}
