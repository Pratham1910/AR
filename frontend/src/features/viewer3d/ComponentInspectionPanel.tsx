import type { ComponentInspection, ComponentRegistry, ComponentResult, InspectionMode } from "../../types";
import { componentColor } from "./discrepancy";

/*
 * Component-level missing-part inspection on the live AR session
 * (backend app/services/component_inspection): controls, the object's
 * verdict, and every component's status with the reason it got it. Results
 * come with each tracked frame; this panel only shows them and sends the
 * operator's choices back through the overlay's stream configuration.
 * Laid out for the AR workspace's sidebar.
 */

interface Props {
  registry: ComponentRegistry | null;
  registryError: string | null;
  inspection: ComponentInspection | null;
  mode: InspectionMode;
  onMode: (mode: InspectionMode) => void;
  onReset: () => void;
  tracking: boolean; // live tracking running (inspection needs it)
  selected: string | null;
  onSelect: (id: string | null) => void;
  showAll: boolean;
  onShowAll: (showAll: boolean) => void;
  debug: boolean;
  onDebug: (debug: boolean) => void;
  outlineNames?: boolean;
  onOutlineNames?: (on: boolean) => void;
  onForgetReference: () => void;
  onSaveReport?: () => void;
  reportStatus?: { id?: string; error?: string; saving?: boolean } | null;
  onOpenReport?: (reportId: string) => void;
  /** A procedure is running: inspect only the current step's components? */
  stepFocus?: { on: boolean; ids: string[]; onChange: (on: boolean) => void } | null;
}

const STATUS_TEXT: Record<ComponentResult["status"], string> = {
  NOT_INSPECTED: "NOT INSPECTED",
  PRESENT: "PRESENT",
  MISSING_CANDIDATE: "CHECKING",
  MISSING_CONFIRMED: "MISSING",
  UNCERTAIN: "UNCERTAIN",
  OCCLUDED: "OCCLUDED",
};

const ago = (iso: string | null) => {
  if (!iso) return null;
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  return s < 1 ? "now" : s < 60 ? `${s.toFixed(0)} s ago` : new Date(iso).toLocaleTimeString();
};

export function ComponentInspectionPanel({
  registry,
  registryError,
  inspection,
  mode,
  onMode,
  onReset,
  tracking,
  selected,
  onSelect,
  showAll,
  onShowAll,
  debug,
  onDebug,
  outlineNames,
  onOutlineNames,
  onForgetReference,
  onSaveReport,
  reportStatus,
  onOpenReport,
  stepFocus,
}: Props) {
  const components = registry?.components ?? [];
  const byId = new Map((inspection?.components ?? []).map((c) => [c.id, c]));
  const required = components.filter((c) => c.required);
  const learned = required.filter((c) => c.reference_samples > 0 || c.calibrated).length;
  const summary = inspection?.summary;
  const running = mode === "inspect" || mode === "learn";

  return (
    <div className="card component-inspection" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
        <span className="section-label">Component inspection</span>
        {inspection && (
          <span className="hint mono">
            run {inspection.run_id} · {inspection.passes} passes
          </span>
        )}
      </div>

      <div className="verdict">
        <div className="verdict-row">
          <span
            className={`verdict-badge ${summary?.verdict ?? ""}`}
            title="PASS only when every required component is confirmed present"
          >
            {mode === "off" ? "IDLE" : mode === "learn" ? "LEARN" : summary?.verdict ?? "—"}
          </span>
          <span className="hint" style={{ lineHeight: 1.5 }}>
            {mode === "off"
              ? "Inspection is off."
              : mode === "learn"
                ? "Learning the good state — keep every component installed and move slowly around the object."
                : summary
                  ? `${summary.required} required · ${summary.not_inspected} not inspected yet`
                  : "Waiting for the first inspected frame…"}
          </span>
        </div>
        {summary && mode !== "learn" && (
          <div className="count-grid">
            <div className="ok">
              <strong>{summary.present}</strong>Present
            </div>
            <div className="bad">
              <strong>{summary.missing}</strong>Missing
            </div>
            <div className="warn">
              <strong>{summary.candidates}</strong>Checking
            </div>
            <div>
              <strong>{summary.uncertain}</strong>Uncertain
            </div>
          </div>
        )}
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
        {running ? (
          mode === "learn" ? (
            <button className="primary" onClick={() => onMode("inspect")} title="Saves what was seen and starts inspecting">
              Finish learning
            </button>
          ) : (
            <button onClick={() => onMode("paused")}>Pause</button>
          )
        ) : (
          <button className="primary" onClick={() => onMode("inspect")} disabled={!components.length}>
            {mode === "paused" ? "Resume" : "Start inspection"}
          </button>
        )}
        <button onClick={onReset} disabled={mode === "off"}>
          Reset
        </button>
        {onSaveReport && (inspection?.passes ?? 0) > 0 && mode !== "learn" && (
          <button onClick={onSaveReport} disabled={reportStatus?.saving} title="Record this inspection as it stands (Reports)">
            {reportStatus?.saving ? "Saving…" : "Save report"}
          </button>
        )}
        {mode !== "learn" && (
          <button
            onClick={() => onMode("learn")}
            disabled={!components.length}
            title="With every component installed, move the camera slowly around the tracked object: each component's look is learned from several viewpoints. Inspection then flags components that stop looking like this."
          >
            Learn good state
          </button>
        )}
      </div>

      {reportStatus?.id && (
        <p className="registration-quality">
          Report saved.{" "}
          {onOpenReport && (
            <button className="ghost" onClick={() => onOpenReport(reportStatus.id!)}>
              Open report
            </button>
          )}
        </p>
      )}
      {reportStatus?.error && <p className="error">{reportStatus.error}</p>}
      {stepFocus && (
        <label style={{ display: "inline-flex", alignItems: "center", gap: 8, minHeight: 36 }}>
          <input type="checkbox" checked={stepFocus.on} onChange={(e) => stepFocus.onChange(e.target.checked)} />
          Current step only
          <span className="hint">
            {stepFocus.on ? (stepFocus.ids.length ? `(${stepFocus.ids.join(", ")})` : "(no component matches this step's parts)") : ""}
          </span>
        </label>
      )}
      {registryError && <p className="error">{registryError}</p>}
      {registry && !components.length && (
        <p className="warning">
          No inspectable components. Name the parts to inspect in the Parts panel (e.g. "Cap", "Screw 1"), or add a mapping file{" "}
          <code>data/component_registry/{registry.model_id}.json</code>.
        </p>
      )}
      {mode !== "off" && !tracking && <p className="warning">Start live tracking (model-based): inspection runs on its frames.</p>}
      {components.length > 0 && learned < required.length && mode !== "learn" && (
        <p className="hint">
          {learned}/{required.length} required components have a presence check. The rest stay Uncertain until you "Learn good
          state" (or calibrate them in the Parts panel).
        </p>
      )}
      {inspection?.message && <p className="registration-quality">{inspection.message}</p>}
      {inspection?.notes.map((n) => (
        <p key={n} className="hint">
          ⚠ {n}
        </p>
      ))}

      {components.length > 0 && (
        <div className="component-list" role="list">
          {components.map((c, index) => {
            const r = byId.get(c.id);
            const status = r?.status ?? "NOT_INSPECTED";
            const when = ago(r?.last_valid ?? null);
            const swatch = componentColor(c, index);
            const reason = c.mesh_status !== "ok" ? `Cannot inspect: ${c.problem}` : r?.reason;
            return (
              <button
                key={c.id}
                role="listitem"
                className={`component-row ${status}${selected === c.id ? " selected" : ""}`}
                onClick={() => onSelect(selected === c.id ? null : c.id)}
                aria-pressed={selected === c.id}
                title="Highlight it in the AR view"
              >
                <span className="head">
                  <span>
                    <span className="swatch" style={{ background: swatch }} aria-hidden="true" />
                    {c.id}
                  </span>
                  <span className="state">{STATUS_TEXT[status]}</span>
                </span>
                <span className="meta">
                  {c.display_name} · {c.type}
                  {!c.required && " · optional"}
                  {r?.score != null && ` · score ${(r.score * 100).toFixed(0)}%`}
                  {when && ` · ${when}`}
                  {r && !r.observable && status !== "NOT_INSPECTED" && " · not in view"}
                  {mode === "learn" && r?.learn_samples != null && ` · ${r.learn_samples} views`}
                </span>
                {reason && status !== "PRESENT" && <span className="why">{reason}</span>}
              </button>
            );
          })}
        </div>
      )}

      {components.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "4px 16px" }}>
          <label style={{ display: "inline-flex", alignItems: "center", gap: 8, minHeight: 36 }}>
            <input type="checkbox" checked={showAll} onChange={(e) => onShowAll(e.target.checked)} /> Mark all
          </label>
          {onOutlineNames && (
            <label style={{ display: "inline-flex", alignItems: "center", gap: 8, minHeight: 36 }}>
              <input type="checkbox" checked={outlineNames} onChange={(e) => onOutlineNames(e.target.checked)} /> Outline with names
            </label>
          )}
          <label style={{ display: "inline-flex", alignItems: "center", gap: 8, minHeight: 36 }}>
            <input type="checkbox" checked={debug} onChange={(e) => onDebug(e.target.checked)} /> Debug view
          </label>
          {learned > 0 && (
            <button className="ghost" onClick={onForgetReference} title="Forget the learned good state of every component">
              Forget learned state
            </button>
          )}
        </div>
      )}

      {debug && registry && (
        <details className="diagnostics" open>
          <summary>
            Registry · {registry.object_id}
            <span className="hint">
              {registry.source === "mapping" ? "mapping file" : "named parts"} ·{" "}
              {registry.detector ? `detector ${registry.detector}` : "no component detector"}
            </span>
          </summary>
          {registry.warnings.map((w) => (
            <p key={w} className="warning">
              {w}
            </p>
          ))}
          <div style={{ overflowX: "auto" }}>
            <table className="calibration-frames">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>GLB node</th>
                  <th>Centre (mm)</th>
                  <th>Size (mm)</th>
                  <th>Reference</th>
                  <th>Evidence (latest)</th>
                </tr>
              </thead>
              <tbody>
                {components.map((c) => {
                  const r = byId.get(c.id);
                  return (
                    <tr key={c.id}>
                      <td className="mono">{c.id}</td>
                      <td>
                        #{c.node_index ?? "—"} {c.node_name}
                        {c.piece !== null ? ` · body ${c.piece}` : ""} <span className="hint">({c.source})</span>
                      </td>
                      <td className="mono">{c.center_m ? c.center_m.map((v) => (v * 1000).toFixed(1)).join(", ") : "—"}</td>
                      <td className="mono">{c.size_m ? c.size_m.map((v) => (v * 1000).toFixed(1)).join(" × ") : "—"}</td>
                      <td>
                        {c.reference_samples ? `${c.reference_samples} views` : "—"}
                        {c.calibrated ? " · calibrated" : ""}
                      </td>
                      <td className="hint">
                        {r?.visibility ?? "—"}
                        {r?.visibility_reasons.length ? `: ${r.visibility_reasons.join("; ")}` : ""}
                        {r?.verdicts.map((v) => (
                          <div key={v.verifier}>
                            {v.verifier}: {v.decision} ({(v.confidence * 100).toFixed(0)}%) — {v.reason}
                          </div>
                        ))}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="hint">
            Nodes: {registry.nodes.map((n) => `#${n.node_index} ${n.name}${n.pieces > 1 ? ` (${n.pieces} bodies)` : ""}`).join(" · ")}
          </p>
        </details>
      )}
    </div>
  );
}
