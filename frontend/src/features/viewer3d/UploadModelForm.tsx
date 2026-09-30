import { useState } from "react";
import { ClassSelect } from "../../components/ClassSelect";
import { AssetsApi, Models3DApi, apiErrorMessage } from "../../services/api";
import type { Asset } from "../../types";

interface Props {
  assets: Asset[];
  onUploaded: (assetId: string) => void;
}

/**
 * Lets a user add a new 3D model without hand-writing API calls: pick an
 * existing asset or name a new one, choose a .glb/.fbx file, and give the
 * physical object's real height so the backend can compute `scale`
 * (app/services/model3d/glb_inspect.py). Required for .glb — skipping it
 * left models at scale=1.0, rendering e.g. a bottle 20m tall that was
 * "detected" but never visibly overlapped. Optional for .fbx, which carries
 * its own units.
 *
 * "Detection class" is how the camera finds the object (markerless and
 * model-based modes), so it's picked from the classes the detector actually
 * knows — free-typed labels like "bot" were silently never detected.
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
    // Both checked before creating a new asset, so a rejected upload doesn't leave an empty asset behind.
    const isFbxFile = file.name.toLowerCase().endsWith(".fbx");
    if (!isFbxFile && !(Number(realHeightM) > 0)) {
      setError("Enter the object's real height in meters — a .glb's own units are rarely meters, so without it the overlay comes out the wrong size.");
      return;
    }
    if (!detectionClassLabel) {
      setError("Pick a detection class — it's how the camera finds this object.");
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
      setError(apiErrorMessage(err));
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
        Real height of the physical object (m){file?.name.toLowerCase().endsWith(".fbx") ? " — optional for FBX" : " — required"}
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
        Detection class — what the camera looks for (pick the closest match)
        <ClassSelect value={detectionClassLabel} onChange={setDetectionClassLabel} />
      </label>

      <button type="submit" disabled={busy}>
        {busy ? (file?.name.toLowerCase().endsWith(".fbx") ? "Converting FBX…" : "Uploading…") : "Upload"}
      </button>

      {error && <p className="error">{error}</p>}
      {successMessage && <p className="camera-status camera-status-ok">🟢 {successMessage}</p>}
    </form>
  );
}
