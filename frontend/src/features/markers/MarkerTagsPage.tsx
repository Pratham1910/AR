import { useEffect, useMemo, useState } from "react";
import { MarkerTagsApi, apiErrorMessage } from "../../services/api";
import type { Asset, TagConfig, TagCorner, TagSet, TagSetIds } from "../../types";
import { CORNERS_GRID, CORNER_LABEL, printImage } from "./corners";

interface Props {
  assets: Asset[];
}

type Draft = Record<TagCorner, string>; // text inputs; "" = not filled in yet
const EMPTY: Draft = { top_left: "", top_right: "", bottom_right: "", bottom_left: "" };
const draftOf = (set: TagSet): Draft => ({
  top_left: String(set.top_left),
  top_right: String(set.top_right),
  bottom_right: String(set.bottom_right),
  bottom_left: String(set.bottom_left),
});

/**
 * Navbar → ArUco tags: give a product its four corner tags, print them, and
 * manage existing mappings (stored in the marker_tags table). The live side —
 * product scan and four-tag tracking — is in 3D & AR workspace → AR tracking
 * → ArUco marker.
 */
export function MarkerTagsPage({ assets }: Props) {
  const [config, setConfig] = useState<TagConfig | null>(null);
  const [sets, setSets] = useState<TagSet[]>([]);
  const [assetId, setAssetId] = useState("");
  const [draft, setDraft] = useState<Draft>(EMPTY);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [previewId, setPreviewId] = useState<string | null>(null); // product whose sheet is shown
  const [singleId, setSingleId] = useState("0");
  const [version, setVersion] = useState(0); // bumps image URLs after a save

  const refresh = () => {
    Promise.all([MarkerTagsApi.config(), MarkerTagsApi.sets()])
      .then(([c, s]) => {
        setConfig(c);
        setSets(s);
      })
      .catch((err) => setError(apiErrorMessage(err)));
  };
  useEffect(refresh, []);

  const existing = sets.find((s) => s.asset_id === assetId) ?? null;
  // Picking a product loads its current tags (or clears the form for a new one).
  useEffect(() => {
    setDraft(existing ? draftOf(existing) : EMPTY);
    setNotice(null);
    setError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [assetId, sets]);

  const size = config?.dictionary_size ?? 50;
  const parsed = useMemo(() => {
    const ids: Partial<TagSetIds> = {};
    for (const corner of CORNERS_GRID) {
      const text = draft[corner].trim();
      if (/^\d+$/.test(text)) ids[corner] = Number(text);
    }
    return ids;
  }, [draft]);

  // Problems found before saving (the backend checks the same and more).
  const problems = useMemo(() => {
    const list: string[] = [];
    const owners = new Map<number, string>();
    sets.forEach((s) =>
      CORNERS_GRID.forEach((c) => {
        if (s.asset_id !== assetId) owners.set(s[c], s.asset_name);
      })
    );
    const seen = new Map<number, TagCorner>();
    for (const corner of CORNERS_GRID) {
      const id = parsed[corner];
      if (draft[corner].trim() === "") continue;
      if (id === undefined) list.push(`${CORNER_LABEL[corner]}: not a whole number.`);
      else if (id >= size) list.push(`${CORNER_LABEL[corner]}: id ${id} is outside ${config?.dictionary} (0–${size - 1}).`);
      else if (seen.has(id)) list.push(`Id ${id} is used for both ${CORNER_LABEL[seen.get(id)!]} and ${CORNER_LABEL[corner]}.`);
      else if (owners.has(id)) list.push(`Id ${id} (${CORNER_LABEL[corner]}) already belongs to ${owners.get(id)}.`);
      if (id !== undefined) seen.set(id, corner);
    }
    return list;
  }, [parsed, draft, sets, assetId, size, config]);

  const complete = CORNERS_GRID.every((c) => parsed[c] !== undefined);

  const suggest = () => {
    setError(null);
    MarkerTagsApi.suggest(assetId || undefined)
      .then(({ ids }) =>
        setDraft({ top_left: String(ids[0]), top_right: String(ids[1]), bottom_right: String(ids[2]), bottom_left: String(ids[3]) })
      )
      .catch((err) => setError(apiErrorMessage(err)));
  };

  const save = () => {
    if (!assetId || !complete) return;
    setBusy(true);
    setError(null);
    MarkerTagsApi.saveSet(assetId, parsed as TagSetIds)
      .then((saved) => {
        setNotice(`Saved: ${saved.asset_name} now has tags ${saved.top_left}, ${saved.top_right}, ${saved.bottom_right}, ${saved.bottom_left}.`);
        setPreviewId(saved.asset_id);
        setVersion((v) => v + 1);
        refresh();
      })
      .catch((err) => setError(apiErrorMessage(err)))
      .finally(() => setBusy(false));
  };

  const remove = (set: TagSet) => {
    if (!window.confirm(`Remove the four tags of ${set.asset_name}? Printed copies will no longer identify it.`)) return;
    MarkerTagsApi.removeSet(set.asset_id)
      .then(() => {
        if (previewId === set.asset_id) setPreviewId(null);
        setNotice(`Removed the tags of ${set.asset_name}.`);
        refresh();
      })
      .catch((err) => setError(apiErrorMessage(err)));
  };

  const preview = sets.find((s) => s.asset_id === previewId) ?? null;
  const singleValid = /^\d+$/.test(singleId) && Number(singleId) < size;

  return (
    <div className="tag-page">
      <div className="card">
        <div className="tag-header">
          <div>
            <h2>ArUco tags</h2>
            <p className="hint">
              A product carries four printed tags, one on each corner, around the object. Show <strong>one</strong> tag
              in AR tracking → ArUco marker to look the product up; show <strong>all four</strong> to track the object
              between them.
            </p>
          </div>
          {config && (
            <div className="tag-dictionary">
              <span className="chip">{config.dictionary}</span>
              <span className="hint">
                {config.free_ids} of {config.dictionary_size} ids free
              </span>
            </div>
          )}
        </div>

        <div className="tag-form">
          <label className="field">
            Product
            <select value={assetId} onChange={(e) => setAssetId(e.target.value)}>
              <option value="">Select a product…</option>
              {assets.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.name}
                  {sets.some((s) => s.asset_id === a.id) ? "  (tagged)" : ""}
                </option>
              ))}
            </select>
          </label>

          <div className="tag-grid">
            {CORNERS_GRID.map((corner) => {
              const id = parsed[corner];
              const showable = id !== undefined && id < size;
              return (
                <div key={corner} className="tag-cell">
                  <label className="field">
                    {CORNER_LABEL[corner]}
                    <input
                      inputMode="numeric"
                      value={draft[corner]}
                      placeholder="id"
                      disabled={!assetId}
                      onChange={(e) => setDraft((d) => ({ ...d, [corner]: e.target.value }))}
                    />
                  </label>
                  <div className="tag-thumb">
                    {showable ? <img src={MarkerTagsApi.tagPngUrl(id, { sizePx: 160 })} alt={`tag ${id}`} /> : <span>—</span>}
                  </div>
                </div>
              );
            })}
          </div>

          {problems.length > 0 && (
            <ul className="warning tag-problems">
              {problems.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>
          )}

          <div className="tag-actions">
            <button onClick={suggest} disabled={!assetId}>
              Suggest free ids
            </button>
            <button className="primary" onClick={save} disabled={!assetId || !complete || problems.length > 0 || busy}>
              {busy ? "Saving…" : existing ? "Save changes" : "Save mapping"}
            </button>
            {existing && <button onClick={() => setDraft(draftOf(existing))}>Undo edits</button>}
          </div>
          {notice && <p className="camera-status camera-status-ok">{notice}</p>}
          {error && <p className="error">{error}</p>}
        </div>
      </div>

      {preview && (
        <div className="card tag-preview">
          <div className="panel-title">
            {preview.asset_name} — printable sheet
            <button className="link" onClick={() => setPreviewId(null)}>
              Close
            </button>
          </div>
          <p className="hint">
            Print at 100% scale. Cut the tags out with their white margin, keep them upright as printed, and fix each to
            the corner named under it, around the object and all facing the camera.
          </p>
          <img
            className="tag-sheet"
            src={MarkerTagsApi.sheetPngUrl(preview.asset_id, { sizePx: 300, bust: String(version) })}
            alt={`${preview.asset_name} tag sheet`}
          />
          <div className="tag-actions">
            <a className="button" href={MarkerTagsApi.sheetPngUrl(preview.asset_id, { download: true, bust: String(version) })}>
              Download sheet (PNG)
            </a>
            <button onClick={() => printImage(MarkerTagsApi.sheetPngUrl(preview.asset_id, { bust: String(version) }), preview.asset_name)}>
              Print sheet
            </button>
            {CORNERS_GRID.map((c) => (
              <a key={c} className="button" href={MarkerTagsApi.tagPngUrl(preview[c], { download: true })}>
                {CORNER_LABEL[c]} ({preview[c]})
              </a>
            ))}
          </div>
        </div>
      )}

      <div className="card">
        <div className="panel-title">Tagged products</div>
        {sets.length === 0 ? (
          <p className="hint">No product has tags yet. Pick a product above, suggest four ids and save.</p>
        ) : (
          <table className="tag-table">
            <thead>
              <tr>
                <th>Product</th>
                {CORNERS_GRID.map((c) => (
                  <th key={c}>{CORNER_LABEL[c]}</th>
                ))}
                <th>Object class</th>
                <th>Opens</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {sets.map((s) => (
                <tr key={s.asset_id} className={s.asset_id === assetId ? "selected" : ""}>
                  <td>{s.asset_name}</td>
                  {CORNERS_GRID.map((c) => (
                    <td key={c} className="mono">
                      {s[c]}
                    </td>
                  ))}
                  <td>
                    {s.detection_class ?? (
                      <span className="hint" title="Set a detection class in the model's settings to require that class between the tags">
                        any object
                      </span>
                    )}
                  </td>
                  <td>
                    {[s.model_name && "3D model", s.has_model_procedure && "procedure", s.procedure_count > 0 && `${s.procedure_count} inspection`]
                      .filter(Boolean)
                      .join(" · ") || <span className="hint">nothing yet</span>}
                  </td>
                  <td className="row-actions">
                    <button onClick={() => setPreviewId(s.asset_id)}>Sheet</button>
                    <button onClick={() => setAssetId(s.asset_id)}>Edit</button>
                    <button className="danger" onClick={() => remove(s)}>
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="card">
        <div className="panel-title">Single tag</div>
        <p className="hint">Any id of {config?.dictionary ?? "the dictionary"}, e.g. a spare or a replacement for a damaged print.</p>
        <div className="tag-single">
          <label className="field">
            Tag id
            <input inputMode="numeric" value={singleId} onChange={(e) => setSingleId(e.target.value)} />
          </label>
          {singleValid ? (
            <>
              <img src={MarkerTagsApi.tagPngUrl(Number(singleId), { sizePx: 200 })} alt={`tag ${singleId}`} />
              <div className="tag-actions">
                <a className="button" href={MarkerTagsApi.tagPngUrl(Number(singleId), { download: true })}>
                  Download PNG
                </a>
                <button onClick={() => printImage(MarkerTagsApi.tagPngUrl(Number(singleId)), `tag ${singleId}`)}>Print</button>
              </div>
            </>
          ) : (
            <p className="warning">Enter an id from 0 to {size - 1}.</p>
          )}
        </div>
      </div>
    </div>
  );
}
