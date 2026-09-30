import { useState } from "react";
import axios from "axios";
import { AssetsApi, Models3DApi } from "../../services/api";
import type { Asset } from "../../types";

interface Props {
  assets: Asset[];
  onUploaded: (assetId: string) => void;
}

/**
 * Lets a user add a new 3D model without hand-writing API calls: pick an
 * existing asset or name a new one, choose a .glb file, and optionally give
 * its real-world height so the backend can auto-compute `scale`
 * (app/services/model3d/glb_inspect.py) — the exact correction bottle.glb
 * needed by hand (Project.md #20/#26). Skipping the height just uploads at
 * scale=1.0, which is very likely wrong for anything not already authored
 * at 1 unit = 1 meter.
 *
 * "Detection class" is separate from all of that: it's what the markerless
 * registration mode should call this object (a COCO class like "cup" or
 * "bottle"). Giving it here links a Component to the model so switching
 * assets in the 3D/AR page auto-fills the right object class instead of
 * leaving whatever was typed for a previous asset.
 */
export function UploadModelForm({ assets, onUploaded }: Props) {
  const [assetChoice, setAssetChoice] = useState<string>("__new__");
  const [newAssetName, setNewAssetName] = useState("");
  const [modelName, setModelName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [realHeightM, setRealHeightM] = useState<string>("");
  const [detectionClassLabel, setDetectionClassLabel] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccessMessage(null);

    if (!file) {
      setError("Choose a .glb or .fbx file first.");
      return;
    }
    if (assetChoice === "__new__" && !newAssetName.trim()) {
      setError("Name the new asset, or pick an existing one.");
      return;
    }

    setBusy(true);
    try {
      const assetId = assetChoice === "__new__" ? (await AssetsApi.create(newAssetName.trim())).id : assetChoice;

      const heightM = realHeightM.trim() ? Number(realHeightM) : undefined;
      const model = await Models3DApi.upload(
        assetId,
        modelName.trim() || file.name,
        file,
        heightM,
        detectionClassLabel.trim() || undefined
      );

      const isFbx = file.name.toLowerCase().endsWith(".fbx");
      setSuccessMessage(
        `Uploaded "${model.name}"${isFbx ? " (converted from FBX)" : ""} — scale ${model.scale.toFixed(5)}` +
          (heightM !== undefined
            ? "."
            : isFbx
              ? " (FBX units applied; give a real height if the overlay size looks wrong)."
              : " (default 1.0 — no real height given, likely wrong).") +
          (model.component_class_label ? ` Detection class: "${model.component_class_label}".` : "")
      );
      setFile(null);
      setModelName("");
      setRealHeightM("");
      setDetectionClassLabel("");
      onUploaded(assetId);
    } catch (err) {
      const detail = axios.isAxiosError(err) ? err.response?.data?.detail : undefined;
      setError(typeof detail === "string" ? detail : err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="upload-model-form">
      <h3>Upload a 3D Model</h3>

      <label>
        Asset
        <select value={assetChoice} onChange={(e) => setAssetChoice(e.target.value)}>
          <option value="__new__">+ New asset…</option>
          {assets.map((a) => (
            <option key={a.id} value={a.id}>
              {a.name}
            </option>
          ))}
        </select>
      </label>

      {assetChoice === "__new__" && (
        <label>
          New asset name
          <input value={newAssetName} onChange={(e) => setNewAssetName(e.target.value)} placeholder="e.g. PUMP-002" />
        </label>
      )}

      <label>
        Model name
        <input value={modelName} onChange={(e) => setModelName(e.target.value)} placeholder="(defaults to filename)" />
      </label>

      <label>
        3D model file (.glb or .fbx)
        <input type="file" accept=".glb,.fbx" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      </label>

      <label>
        Real height (m) — optional but recommended
        <input
          type="number"
          step="0.01"
          min="0.001"
          value={realHeightM}
          onChange={(e) => setRealHeightM(e.target.value)}
          placeholder="e.g. 0.15"
        />
      </label>

      <label>
        Detection class (for Markerless mode) — optional but recommended
        <input
          value={detectionClassLabel}
          onChange={(e) => setDetectionClassLabel(e.target.value)}
          placeholder="e.g. cup, bottle"
        />
      </label>

      <button type="submit" disabled={busy}>
        {busy ? (file?.name.toLowerCase().endsWith(".fbx") ? "Converting FBX…" : "Uploading…") : "Upload"}
      </button>

      {error && <p className="error">{error}</p>}
      {successMessage && <p className="camera-status camera-status-ok">🟢 {successMessage}</p>}
    </form>
  );
}
