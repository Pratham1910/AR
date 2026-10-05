import { useState } from "react";
import { Models3DApi, apiErrorMessage } from "../../services/api";
import type { FitCheckResult } from "../../types";

interface Props {
  modelId: string;
  captureFrame: () => string | null;
  disabled?: boolean;
}

/**
 * "Does the 3D model have the real object's shape?" — the overlay can only be
 * as tight as that. Compares the real outline with the model drawn at its
 * solved pose, part by part, and says what to change in the GLB.
 */
export function FitCheck({ modelId, captureFrame, disabled }: Props) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<FitCheckResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    const frame = captureFrame();
    if (!frame) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await Models3DApi.fitCheck(modelId, frame));
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card fit-check">
      <div className="panel-title">
        Model fit
        <button onClick={() => void run()} disabled={busy || disabled}>
          {busy ? "Checking…" : result ? "Check again" : "Check model fit"}
        </button>
      </div>
      {!result && !error && (
        <p className="hint">
          Compares the 3D model's shape with the real object in the current camera view. If the outlines don't line up,
          this says which part of the GLB to change, in cm.
        </p>
      )}
      {error && <p className="error">{error}</p>}
      {result && (
        <div className="fit-check-body">
          <img src={`data:image/jpeg;base64,${result.overlay_jpeg_base64}`} alt="Real outline (red) and model outline (green)" />
          <div>
            <p>
              <span className={`chip ${result.good_fit ? "ok" : result.iou >= 0.9 ? "warn" : "bad"}`}>
                {(result.iou * 100).toFixed(0)}% outline overlap
              </span>{" "}
              <span className="hint">red: real object · green: model</span>
            </p>
            <ul className="fit-findings">
              {result.findings.map((f) => (
                <li key={f}>{f}</li>
              ))}
            </ul>
            <table className="calibration-frames">
              <thead>
                <tr>
                  <th>Part</th>
                  <th>Real</th>
                  <th>Model</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {result.parts.map((p) => (
                  <tr key={p.part}>
                    <td>{p.part}</td>
                    <td>{p.real_cm.toFixed(1)} cm</td>
                    <td>{p.model_cm.toFixed(1)} cm</td>
                    <td className={Math.abs(p.diff_rel) > 0.05 ? "warning" : "hint"}>{(p.diff_rel * 100).toFixed(0)}%</td>
                  </tr>
                ))}
                <tr>
                  <td>Height</td>
                  <td>{result.real_height_cm.toFixed(1)} cm</td>
                  <td>{result.model_height_cm.toFixed(1)} cm</td>
                  <td className="hint">
                    {(((result.model_height_cm - result.real_height_cm) / result.real_height_cm) * 100).toFixed(0)}%
                  </td>
                </tr>
              </tbody>
            </table>
            <p className="hint">
              Widths are at the object's distance, measured on this frame — check from the front, with the whole object in
              view. Anything the detector wrongly includes (a strap, a hand) reads as "model narrower" there.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
