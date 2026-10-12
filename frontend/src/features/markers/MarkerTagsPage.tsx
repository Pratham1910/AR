import { useEffect, useMemo, useState } from "react";
import { useDetectableClasses } from "../../hooks/useDetectableClasses";
import { MarkerTagsApi, apiErrorMessage } from "../../services/api";
import type { Asset, TagConfig, TagCorner, TagSet } from "../../types";
import { CORNERS, CORNER_LABEL } from "./corners";

interface Props {
  assets: Asset[];
}

type Draft = Record<TagCorner, string>; // text inputs; "" = not filled in yet

const EMPTY_DRAFT: Draft = { top_left: "", top_right: "", bottom_left: "", bottom_right: "" };

const draftOf = (set: TagSet): Draft => ({
  top_left: String(set.top_left),
  top_right: String(set.top_right),
  bottom_left: String(set.bottom_left),
  bottom_right: String(set.bottom_right),
});

/**
 * ArUco tag management: give a product four tags (one per corner), download
 * them to print, and see every product's mapping. The scan itself lives in
 * 3D & AR workspace → AR tracking → ArUco marker; this page only maintains
 * what the scan looks up.
 *
 * The form flags clashes as you type, but the server decides
 * (PUT /api/marker-tags/sets/{asset}): a tag id can mark only one corner of
 * one product.
 */
export function MarkerTagsPage({ assets }: Props) {
  const [config, setConfig] = useState<TagConfig | null>(null);
  const [sets, setSets] = useState<TagSet[]>([]);
  const [assetId, setAssetId] = useState("");
  const [draft, setDraft] = useState<Draft>(EMPTY_DRAFT);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const { classes } = useDetectableClasses();

  const refresh = () =>
    Promise.all([MarkerTagsApi.config(), MarkerTagsApi.sets()])
      .then(([c, s]) => {
        setConfig(c);
        setSets(s);
      })
      .catch((err) => setError(apiErrorMessage(err)));

  useEffect(() => {
    void refresh();
  }, []);

  const currentSet = sets.find((s) => s.asset_id === assetId);

  // Picking a product loads its tags into the form (or clears it for a new one).
  const selectAsset = (id: string) => {
    setAssetId(id);
    setSaved(null);
    setError(null);
    const existing = sets.find((s) => s.asset_id === id);
    setDraft(existing ? draftOf(existing) : EMPTY_DRAFT);
  };

  // Which product already uses a tag id (other than the one being edited).
  const ownerOf = useMemo(() => {
    const owners = new Map<number, string>();
    for (const set of sets) {
      if (set.asset_id === assetId) continue;
      for (const corner of CORNERS) owners.set(set[corner], set.asset_name);
    }
    return owners;
  }, [sets, assetId]);

  const problemFor = (corner: TagCorner): string | null => {
    const text = draft[corner].trim();
    if (text === "") return null;
    const id = Number(text);
    if (!Number.isInteger(id) || id < 0) return "Must be a whole number";
    if (config && id >= config.dictionary_size) return `${config.dictionary} has ids 0–${config.dictionary_size - 1}`;
    if (CORNERS.some((other) => other !== corner && draft[other].trim() === text)) return "Used on another corner";
    const owner = ownerOf.get(id);
    return owner ? `Already assigned to ${owner}` : null;
  };

  const complete = CORNERS.every((c) => draft[c].trim() !== "");
  const valid = complete && CORNERS.every((c) => problemFor(c) === null);
  const unchanged = currentSet !== undefined && CORNERS.every((c) => String(currentSet[c]) === draft[c].trim());

  const suggest = async () => {
    setError(null);
    try {
      const { ids } = await MarkerTagsApi.suggest(assetId || undefined);
      setDraft({ top_left: String(ids[0]), top_right: String(ids[1]), bottom_left: String(ids[2]), bottom_right: String(ids[3]) });
      setSaved(null);
    } catch (err) {
      setError(apiErrorMessage(err));
    }
  };

  const save = async () => {
    if (!assetId || !valid) return;
    setBusy(true);
    setError(null);
    try {
      const result = await MarkerTagsApi.saveSet(assetId, {
        top_left: Number(draft.top_left),
        top_right: Number(draft.top_right),
        bottom_left: Number(draft.bottom_left),
        bottom_right: Number(draft.bottom_right),
      });
      setSaved(`Saved — ${result.asset_name} is identified by tags ${CORNERS.map((c) => result[c]).join(", ")}.`);
      await refresh();
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const remove = async (set: TagSet) => {
    if (!window.confirm(`Remove the tag mapping for ${set.asset_name}? Its tags will no longer identify it.`)) return;
    setError(null);
    try {
      await MarkerTagsApi.removeSet(set.asset_id);
      if (set.asset_id === assetId) setDraft(EMPTY_DRAFT);
      setSaved(null);
      await refresh();
    } catch (err) {
      setError(apiErrorMessage(err));
    }
  };

  const classNote = (set: TagSet): { text: string; cls: string } => {
    if (!set.detection_class) return { text: "none set — cannot be verified", cls: "bad" };
    if (classes.length > 0 && !classes.includes(set.detection_class))
      return { text: `${set.detection_class} — not known to the detector`, cls: "bad" };
    return { text: set.detection_class, cls: "ok" };
  };

  return (
    <div className="tag-page">
      <div className="card">
        <div className="panel-title">ArUco tags for a product</div>
        <p className="hint">
          A product is identified by four printed tags, one on each corner. Pick the product, assign four tag ids, then
          download and print them. Scan in <strong>3D &amp; AR workspace → AR tracking → ArUco marker</strong>.
          {config && (
            <>
              {" "}
              Dictionary <code>{config.dictionary}</code>: {config.free_ids} of {config.dictionary_size} ids free (
              {Math.floor(config.free_ids / 4)} more products).
            </>
          )}
        </p>

        <div className="settings-row">
          <label>
            Product
            <select value={assetId} onChange={(e) => selectAsset(e.target.value)}>
              <option value="">Select a product…</option>
              {assets.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.name}
                  {sets.some((s) => s.asset_id === a.id) ? " (has tags)" : ""}
                </option>
              ))}
            </select>
          </label>
          <button onClick={() => void suggest()} disabled={!assetId || busy} title="Fill in the four lowest unused tag ids">
            Suggest free ids
          </button>
        </div>

        {assetId && (
          <>
            <div className="tag-grid">
              {CORNERS.map((corner) => {
                const problem = problemFor(corner);
                const id = draft[corner].trim();
                const downloadable = id !== "" && problem === null;
                return (
                  <div key={corner} className={`tag-cell${problem ? " tag-cell-bad" : ""}`}>
                    <label>
                      {CORNER_LABEL[corner]}
                      <input
                        type="number"
                        min={0}
                        max={config ? config.dictionary_size - 1 : undefined}
                        value={draft[corner]}
                        placeholder="tag id"
                        onChange={(e) => {
                          setDraft((d) => ({ ...d, [corner]: e.target.value }));
                          setSaved(null);
                        }}
                      />
                    </label>
                    {problem ? (
                      <span className="error">{problem}</span>
                    ) : downloadable ? (
                      <a href={MarkerTagsApi.tagPngUrl(Number(id))} download>
                        Download tag {id} (PNG)
                      </a>
                    ) : (
                      <span className="hint">—</span>
                    )}
                  </div>
                );
              })}
            </div>

            <div className="settings-row">
              <button className="primary" onClick={() => void save()} disabled={!valid || busy || unchanged}>
                {busy ? "Saving…" : currentSet ? "Update mapping" : "Save mapping"}
              </button>
              {currentSet && unchanged && (
                <a href={MarkerTagsApi.sheetPngUrl(assetId)} download>
                  Download all four as one sheet (PNG)
                </a>
              )}
              {!complete && <span className="hint">All four corners need a tag id.</span>}
            </div>
            <p className="hint">
              Print at 100% scale with the white margin intact, and fix each tag to the corner named under it, upright as
              printed. All four must be visible to the camera at once.
            </p>
          </>
        )}

        {saved && <p className="tag-saved">{saved}</p>}
        {error && <p className="error">{error}</p>}
      </div>

      <div className="card">
        <div className="panel-title">Tagged products</div>
        {sets.length === 0 ? (
          <p className="hint">No product has tags yet.</p>
        ) : (
          <table className="calibration-frames tag-table">
            <thead>
              <tr>
                <th>Product</th>
                {CORNERS.map((c) => (
                  <th key={c}>{CORNER_LABEL[c]}</th>
                ))}
                <th>YOLO class</th>
                <th>3D model</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {sets.map((set) => {
                const note = classNote(set);
                return (
                  <tr key={set.asset_id}>
                    <td>{set.asset_name}</td>
                    {CORNERS.map((c) => (
                      <td key={c}>
                        <a href={MarkerTagsApi.tagPngUrl(set[c])} download title={`Download tag ${set[c]}`}>
                          {set[c]}
                        </a>
                      </td>
                    ))}
                    <td>
                      <span className={`chip ${note.cls}`}>{note.text}</span>
                    </td>
                    <td>{set.model_name ?? <span className="hint">none</span>}</td>
                    <td className="tag-actions">
                      <button onClick={() => selectAsset(set.asset_id)}>Edit</button>
                      <a href={MarkerTagsApi.sheetPngUrl(set.asset_id)} download>
                        Sheet
                      </a>
                      <button className="danger" onClick={() => void remove(set)}>
                        Remove
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
        <p className="hint">
          The YOLO class is the product's detection class, set on its 3D model (3D &amp; AR workspace → Model settings). A
          scan confirms a product only when its four tags <em>and</em> an object of that class are seen together.
        </p>
      </div>
    </div>
  );
}
