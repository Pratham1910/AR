import { useEffect, useMemo, useState } from "react";
import { Models3DApi, apiErrorMessage } from "../../services/api";
import type { Asset, ComponentRegistry, Model3DInfo, RegisteredComponent } from "../../types";
import { componentColor } from "../viewer3d/discrepancy";

/*
 * Models & components: which parts of a model's GLB are inspected, under
 * which IDs (backend app/services/component_inspection/registry.py). The
 * editor writes the explicit mapping file through PUT /components; "Use
 * automatic" removes it again (named parts + parts whose names say what they
 * are).
 */

const TYPES = [
  "part", "screw", "bolt", "nut", "washer", "rivet", "pin", "stud", "clip", "cap", "lid", "cover", "plug",
  "connector", "terminal", "fuse", "relay", "switch", "led", "cable", "hose", "gasket", "o-ring", "seal", "ring",
  "bracket", "label",
];

interface Row {
  nodeIndex: number;
  nodeName: string;
  pieces: number;
  faces: number;
  inspect: boolean;
  id: string;
  type: string;
  required: boolean;
  split: boolean;
  prefix: string;
  color: string;
}

const prefixOf = (ids: string[]) => {
  const m = ids[0]?.match(/^(.*?)(\d+)$/);
  return m ? m[1] : `${ids[0] ?? "PART"}_`;
};

function rowsFrom(registry: ComponentRegistry): Row[] {
  return registry.nodes.map((n, nodeIndex) => {
    const own = registry.components.filter((c) => c.node_index === n.node_index);
    const first = own[0];
    const split = own.length > 1 || (first?.piece ?? null) !== null;
    return {
      nodeIndex: n.node_index,
      nodeName: n.name,
      pieces: n.pieces,
      faces: n.faces,
      inspect: own.length > 0,
      id: first?.id ?? "",
      type: first?.type ?? "part",
      required: first?.required ?? true,
      split,
      prefix: split ? prefixOf(own.map((c) => c.id)) : `${(first?.type ?? "part").toUpperCase().replace("-", "")}_`,
      color: first ? componentColor(first, registry.components.indexOf(first)) : componentColor(undefined, nodeIndex),
    };
  });
}

function presence(c: RegisteredComponent) {
  if (c.mesh_status !== "ok") return { cls: "bad", text: `Cannot inspect: ${c.problem}` };
  if (c.calibrated) return { cls: "ok", text: "Calibrated" };
  if (c.reference_samples > 0) return { cls: "ok", text: `Learned · ${c.reference_samples} views` };
  return { cls: "warn", text: "No presence check yet" };
}

export function ComponentsPage({ assets }: { assets: Asset[] }) {
  const [assetId, setAssetId] = useState(() => {
    try {
      return localStorage.getItem("tvasta.lastAssetId") ?? "";
    } catch {
      return "";
    }
  });
  const [models, setModels] = useState<Model3DInfo[]>([]);
  const [modelId, setModelId] = useState("");
  const [registry, setRegistry] = useState<ComponentRegistry | null>(null);
  const [rows, setRows] = useState<Row[]>([]);
  const [objectId, setObjectId] = useState("");
  const [editing, setEditing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);

  useEffect(() => {
    setModels([]);
    setModelId("");
    setRegistry(null);
    if (!assetId) return;
    Models3DApi.listForAsset(assetId)
      .then((list) => {
        setModels(list);
        setModelId(list[0]?.id ?? "");
      })
      .catch((err) => setError(apiErrorMessage(err)));
  }, [assetId]);

  const load = (id: string) => {
    setError(null);
    Models3DApi.components(id)
      .then((r) => {
        setRegistry(r);
        setRows(rowsFrom(r));
        setObjectId(r.object_id);
      })
      .catch((err) => setError(apiErrorMessage(err)));
  };
  useEffect(() => {
    setRegistry(null);
    setEditing(false);
    if (modelId) load(modelId);
  }, [modelId]);

  const model = models.find((m) => m.id === modelId);
  const update = (index: number, patch: Partial<Row>) => setRows((rs) => rs.map((r, i) => (i === index ? { ...r, ...patch } : r)));

  const problems = useMemo(() => {
    const ids = new Map<string, number>();
    const out: string[] = [];
    for (const r of rows.filter((r) => r.inspect)) {
      if (r.split) {
        if (!r.prefix.trim()) out.push(`${r.nodeName}: give the split bodies an ID prefix`);
        continue;
      }
      const id = r.id.trim();
      if (!id) out.push(`${r.nodeName}: needs a component ID`);
      else if (!/^[A-Za-z0-9_.-]+$/.test(id)) out.push(`${id}: use letters, digits, _ . - only`);
      ids.set(id, (ids.get(id) ?? 0) + 1);
    }
    for (const [id, n] of ids) if (id && n > 1) out.push(`${id} is used ${n} times`);
    return out;
  }, [rows]);

  const save = async () => {
    if (!modelId) return;
    setBusy(true);
    setError(null);
    setSaved(null);
    try {
      const components = rows
        .filter((r) => r.inspect)
        .map((r) =>
          r.split
            ? { idPrefix: r.prefix.trim(), nodeIndex: r.nodeIndex, split: true, type: r.type, required: r.required, color: r.color }
            : { id: r.id.trim(), nodeIndex: r.nodeIndex, type: r.type, required: r.required, color: r.color }
        );
      const r = await Models3DApi.saveComponentMapping(modelId, { objectId: objectId.trim() || undefined, components });
      setRegistry(r);
      setRows(rowsFrom(r));
      setEditing(false);
      setSaved(`Saved ${r.components.length} components.`);
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const useAutomatic = async () => {
    if (!modelId) return;
    setBusy(true);
    try {
      await Models3DApi.resetComponentMapping(modelId);
      load(modelId);
      setEditing(false);
      setSaved("Back to automatic: named parts and parts whose names say what they are.");
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const forget = async () => {
    if (!modelId) return;
    await Models3DApi.forgetComponentReference(modelId);
    load(modelId);
    setSaved("Learned good state forgotten — learn it again on the Live AR screen.");
  };

  const required = registry?.components.filter((c) => c.required) ?? [];
  const unchecked = required.filter((c) => c.mesh_status === "ok" && !c.calibrated && c.reference_samples === 0);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <div className="section-label">Models &amp; components</div>
          <h2 className="page-title">{model?.name ?? "Pick a model"}</h2>
          {model && (
            <div className="hint mono">
              {model.storage_key}
              {registry ? ` · ${registry.nodes.length} mesh nodes` : ""}
              {model.real_height_m != null ? ` · ${(model.real_height_m * 100).toFixed(1)} cm tall` : ""}
            </div>
          )}
        </div>
        <div className="spacer" />
        <label className="field">
          Asset
          <select value={assetId} onChange={(e) => setAssetId(e.target.value)}>
            <option value="">Select an asset…</option>
            {assets.map((a) => (
              <option key={a.id} value={a.id}>
                {a.name}
              </option>
            ))}
          </select>
        </label>
        {models.length > 1 && (
          <label className="field">
            Model
            <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      {error && <p className="error">{error}</p>}
      {saved && <p className="registration-quality">{saved}</p>}
      {assetId && models.length === 0 && <div className="card empty-state hint">No 3D model for this asset yet.</div>}

      {registry && (
        <div className="page-grid">
          <section className="card" aria-label="Component registry">
            <div className="card-head">
              <h3>Component registry</h3>
              <span className="tag mono">{registry.source === "mapping" ? "mapping file" : "automatic"}</span>
              <div className="spacer" />
              {!editing && <button onClick={() => setEditing(true)}>Edit mapping</button>}
              {!editing && registry.source === "mapping" && (
                <button className="ghost" onClick={() => void useAutomatic()} disabled={busy}>
                  Use automatic
                </button>
              )}
            </div>

            {!editing && (
              <>
                <div className="table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Name</th>
                        <th>Type</th>
                        <th>GLB node</th>
                        <th>Size (mm)</th>
                        <th>Required</th>
                        <th>Presence check</th>
                      </tr>
                    </thead>
                    <tbody>
                      {registry.components.map((c, index) => {
                        const p = presence(c);
                        return (
                          <tr key={c.id}>
                            <td className="mono strong">
                              <span className="swatch" style={{ background: componentColor(c, index) }} aria-hidden="true" />
                              {c.id}
                            </td>
                            <td>{c.display_name}</td>
                            <td>{c.type}</td>
                            <td className="mono">
                              #{c.node_index ?? "—"} {c.node_name}
                              {c.piece !== null ? ` · body ${c.piece + 1}` : ""}
                            </td>
                            <td className="mono">{c.size_m ? c.size_m.map((v) => (v * 1000).toFixed(0)).join(" × ") : "—"}</td>
                            <td>{c.required ? "Yes" : "Optional"}</td>
                            <td>
                              <span className={`tag ${p.cls}`}>{p.text}</span>
                            </td>
                          </tr>
                        );
                      })}
                      {registry.components.length === 0 && (
                        <tr>
                          <td colSpan={7} className="hint">
                            No components yet — use "Edit mapping" to choose which parts are inspected.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
                {unchecked.length > 0 && (
                  <p className="warning">
                    {unchecked.length} of {required.length} required components have no presence check and will stay Uncertain. Track the
                    object with every part installed, then use "Learn good state" on the Live AR screen.
                  </p>
                )}
                {registry.warnings.map((w) => (
                  <p key={w} className="warning">
                    {w}
                  </p>
                ))}
              </>
            )}

            {editing && (
              <>
                <label className="field inline-field">
                  Object ID
                  <input value={objectId} onChange={(e) => setObjectId(e.target.value)} placeholder="e.g. PUMP_001" />
                </label>
                <div className="table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Inspect</th>
                        <th>GLB node</th>
                        <th>Component ID</th>
                        <th>Type</th>
                        <th>Colour</th>
                        <th>Required</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((r, i) => (
                        <tr key={r.nodeIndex} className={r.inspect ? "" : "muted-row"}>
                          <td>
                            <input
                              type="checkbox"
                              checked={r.inspect}
                              onChange={(e) => update(i, { inspect: e.target.checked })}
                              aria-label={`Inspect ${r.nodeName}`}
                            />
                          </td>
                          <td className="mono">
                            #{r.nodeIndex} {r.nodeName}
                            <div className="hint">
                              {r.faces} faces{r.pieces > 1 ? ` · ${r.pieces} separate bodies` : ""}
                            </div>
                          </td>
                          <td>
                            {r.pieces > 1 && (
                              <label className="check-inline">
                                <input type="checkbox" checked={r.split} disabled={!r.inspect} onChange={(e) => update(i, { split: e.target.checked })} />
                                One per body
                              </label>
                            )}
                            {r.split ? (
                              <input
                                value={r.prefix}
                                disabled={!r.inspect}
                                onChange={(e) => update(i, { prefix: e.target.value })}
                                aria-label={`ID prefix for ${r.nodeName}`}
                                placeholder="SCREW_"
                              />
                            ) : (
                              <input
                                value={r.id}
                                disabled={!r.inspect}
                                onChange={(e) => update(i, { id: e.target.value })}
                                aria-label={`Component ID for ${r.nodeName}`}
                                placeholder="e.g. CAP_001"
                              />
                            )}
                            {r.split && <div className="hint">→ {r.prefix}001 … {r.prefix}{String(r.pieces).padStart(3, "0")}</div>}
                          </td>
                          <td>
                            <select value={r.type} disabled={!r.inspect} onChange={(e) => update(i, { type: e.target.value })} aria-label={`Type of ${r.nodeName}`}>
                              {TYPES.map((t) => (
                                <option key={t} value={t}>
                                  {t}
                                </option>
                              ))}
                            </select>
                          </td>
                          <td>
                            <input
                              type="color"
                              value={r.color}
                              disabled={!r.inspect}
                              onChange={(e) => update(i, { color: e.target.value })}
                              aria-label={`Colour of ${r.nodeName}`}
                              className="color-input"
                            />
                          </td>
                          <td>
                            <input
                              type="checkbox"
                              checked={r.required}
                              disabled={!r.inspect}
                              onChange={(e) => update(i, { required: e.target.checked })}
                              aria-label={`${r.nodeName} required`}
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {problems.map((p) => (
                  <p key={p} className="error">
                    {p}
                  </p>
                ))}
                <div className="button-row">
                  <button className="primary" onClick={() => void save()} disabled={busy || problems.length > 0}>
                    {busy ? "Saving…" : "Save mapping"}
                  </button>
                  <button
                    onClick={() => {
                      setEditing(false);
                      setRows(rowsFrom(registry));
                    }}
                  >
                    Cancel
                  </button>
                </div>
              </>
            )}
          </section>

          <div className="page-side">
            <section className="card" aria-label="Assembly structure">
              <h3>Assembly structure</h3>
              <ul className="tree mono">
                {registry.nodes.map((n) => (
                  <li key={n.node_index}>
                    #{n.node_index} {n.name}
                    <span className="hint">
                      {" "}
                      · {n.faces} faces{n.pieces > 1 ? ` · ${n.pieces} bodies` : ""}
                      {n.component_ids.length ? ` → ${n.component_ids.length > 3 ? `${n.component_ids.length} components` : n.component_ids.join(", ")}` : ""}
                    </span>
                  </li>
                ))}
              </ul>
            </section>
            <section className="card" aria-label="Presence references">
              <h3>Presence references</h3>
              <p className="hint">
                Components are judged against a calibration (Parts panel) or the learned good state (Live AR → Learn good state).
                {registry.detector ? ` Component detector: ${registry.detector}.` : " No component detector is configured."}
              </p>
              {registry.components.some((c) => c.reference_samples > 0) && (
                <button className="danger" onClick={() => void forget()}>
                  Forget learned good state
                </button>
              )}
            </section>
          </div>
        </div>
      )}
    </div>
  );
}

