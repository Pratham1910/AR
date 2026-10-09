import { useState } from "react";
import { Models3DApi, apiErrorMessage } from "../../services/api";
import type { QAInspectResult, QAPartResult } from "../../types";

interface Props {
  modelId: string;
  captureFrame: () => string | null;
  disabled?: boolean;
}

const STATUS_COLOR: Record<QAPartResult["status"], string> = {
  present:  "#22c55e",
  missing:  "#ef4444",
  partial:  "#eab308",
  occluded: "#f97316",
  unknown:  "#9ca3af",
};

const STATUS_LABEL: Record<QAPartResult["status"], string> = {
  present:  "PRESENT",
  missing:  "MISSING",
  partial:  "PARTIAL",
  occluded: "OCCLUDED",
  unknown:  "UNKNOWN",
};

const VERDICT_COLOR: Record<QAInspectResult["verdict"], string> = {
  PASS:      "#22c55e",
  FAIL:      "#ef4444",
  UNCERTAIN: "#f97316",
};

function PartRow({ part }: { part: QAPartResult }) {
  const color = STATUS_COLOR[part.status];
  return (
    <tr>
      <td>{part.name}</td>
      <td>
        <span
          className="chip"
          style={{ background: color + "22", color, border: `1px solid ${color}`, fontWeight: 600, fontSize: "0.78em" }}
        >
          {STATUS_LABEL[part.status]}
        </span>
      </td>
      <td className="hint">{part.status !== "occluded" && part.status !== "unknown" ? `${(part.coverage * 100).toFixed(0)}%` : "—"}</td>
      <td className="hint">{(part.visibility * 100).toFixed(0)}%</td>
    </tr>
  );
}

/**
 * Phase 5 QA: compares named assembly parts against the real object in the
 * camera view. Missing parts are highlighted in the overlay image; present
 * parts shown in green. Requires parts to be named in the Parts panel first.
 */
export function QAInspectionPanel({ modelId, captureFrame, disabled }: Props) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<QAInspectResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    const frame = captureFrame();
    if (!frame) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await Models3DApi.qaInspect(modelId, frame));
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const verdictColor = result ? VERDICT_COLOR[result.verdict] : undefined;

  return (
    <div className="card fit-check">
      <div className="panel-title">
        QA inspection
        <button onClick={() => void run()} disabled={busy || disabled}>
          {busy ? "Inspecting…" : result ? "Inspect again" : "Run QA inspection"}
        </button>
      </div>

      {!result && !error && (
        <p className="hint">
          Checks that every named assembly part is present on the real object. Name the parts in the Parts panel first — unnamed
          mesh nodes are skipped. Point the camera at the full object and click Inspect.
        </p>
      )}

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="fit-check-body">
          <img
            src={`data:image/jpeg;base64,${result.overlay_jpeg_base64}`}
            alt="QA inspection overlay — parts coloured by status"
            style={{ borderRadius: 6 }}
          />
          <div>
            <p>
              <span
                className="chip"
                style={{
                  background: verdictColor + "22",
                  color: verdictColor,
                  border: `1px solid ${verdictColor}`,
                  fontWeight: 700,
                  fontSize: "0.9em",
                }}
              >
                {result.verdict}
              </span>
              <span className="hint" style={{ marginLeft: 8 }}>
                {result.present_count} present · {result.missing_count} missing · {result.partial_count} partial
              </span>
            </p>

            <table className="calibration-frames">
              <thead>
                <tr>
                  <th>Part</th>
                  <th>Status</th>
                  <th title="Fraction of projected area covered by the real object">Coverage</th>
                  <th title="Fraction of part faces visible from this viewpoint">Visible</th>
                </tr>
              </thead>
              <tbody>
                {result.parts.map((p) => (
                  <PartRow key={p.node_index} part={p} />
                ))}
              </tbody>
            </table>

            <p className="hint" style={{ marginTop: 8 }}>
              Green = present · Red = missing · Yellow = partial · Orange = occluded (rotate to verify) · Gray = too small to
              judge.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
