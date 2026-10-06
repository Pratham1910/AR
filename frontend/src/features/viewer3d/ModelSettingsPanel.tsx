import { useEffect, useState } from "react";
import { ClassSelect } from "../../components/ClassSelect";
import { useDetectableClasses } from "../../hooks/useDetectableClasses";
import { Models3DApi, apiErrorMessage } from "../../services/api";
import type { Model3DInfo } from "../../types";

interface Props {
  model: Model3DInfo;
  onSaved: () => void;
  onDeleted: () => void;
}

/**
 * Shows and fixes the two settings that make a model silently fail in AR:
 * its real-world size (wrong -> overlay renders at e.g. 20m and never visibly
 * overlaps) and its detection class (not a class the detector knows -> the
 * object is never found). Also deletes the model.
 */
export function ModelSettingsPanel({ model, onSaved, onDeleted }: Props) {
  const { classes } = useDetectableClasses();
  const currentHeightCm = model.real_height_m !== null ? model.real_height_m * 100 : null;
  const [heightCm, setHeightCm] = useState("");
  const [classLabel, setClassLabel] = useState(model.component_class_label ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setHeightCm(currentHeightCm !== null ? currentHeightCm.toFixed(1) : "");
    setClassLabel(model.component_class_label ?? "");
    setError(null);
  }, [model.id, model.scale, model.component_class_label]); // eslint-disable-line react-hooks/exhaustive-deps

  const problems: string[] = [];
  // The class only matters for markerless mode and "Find object by: Object
  // class"; model-based tracking finds the object by its 3D model by default.
  if (model.component_class_label && classes.length && !classes.includes(model.component_class_label)) {
    problems.push(
      `"${model.component_class_label}" isn't a class the detector knows — fine for model-based tracking ` +
        "(found by its 3D model), but markerless mode and \"Find object by: Object class\" won't find it."
    );
  }
  if (currentHeightCm !== null && (currentHeightCm > 300 || currentHeightCm < 1)) {
    problems.push(
      `Renders ${currentHeightCm >= 100 ? (currentHeightCm / 100).toFixed(1) + " m" : currentHeightCm.toFixed(2) + " cm"} tall — ` +
        "almost certainly the wrong real height, so the overlay can't line up."
    );
  } else if (currentHeightCm !== null && currentHeightCm < 3) {
    // Model-based tracking infers distance from this size: a hand-sized object
    // entered as ~1 cm gets posed a few cm from the camera, where perspective is
    // so strong no candidate orientation fits (seen: a controller at 1.1 cm locked
    // facing the wrong way). Tiny parts are possible, so this is a warning.
    problems.push(
      `Renders only ${currentHeightCm.toFixed(1)} cm tall. If the real object is bigger, set its real height — ` +
        "distance and orientation are worked out from it, and a wrong size makes the model face the wrong way."
    );
  }

  const heightChanged = heightCm !== "" && currentHeightCm !== null && Math.abs(Number(heightCm) - currentHeightCm) > 0.05;
  const classChanged = classLabel !== "" && classLabel !== (model.component_class_label ?? "");

  const save = async () => {
    if (heightCm !== "" && !(Number(heightCm) > 0)) {
      setError("Height must be a positive number of centimeters.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await Models3DApi.updateSettings(model.id, {
        ...(heightChanged ? { real_world_height_m: Number(heightCm) / 100 } : {}),
        ...(classChanged ? { detection_class_label: classLabel } : {}),
      });
      onSaved();
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    if (!window.confirm(`Delete model "${model.name}" (${model.storage_key})? Its file is moved to data/models/.deleted/.`)) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await Models3DApi.remove(model.id);
      onDeleted();
    } catch (err) {
      setError(apiErrorMessage(err));
      setBusy(false);
    }
  };

  return (
    <div className="model-settings">
      <div className="model-name">
        <strong>{model.name}</strong>
        <span className="hint">
          {model.storage_key}
          {model.real_height_m !== null && ` · ${(model.real_height_m * 100).toFixed(1)} cm tall`}
        </span>
      </div>
      {problems.map((p) => (
        <p key={p} className="warning">
          {p}
        </p>
      ))}
      <div className="registration-controls">
        <label>
          Real height (cm)
          <input type="number" step="0.1" min="0.1" value={heightCm} onChange={(e) => setHeightCm(e.target.value)} />
        </label>
        <label>
          Detection class
          <ClassSelect value={classLabel} onChange={setClassLabel} />
        </label>
        <button onClick={save} disabled={busy || (!heightChanged && !classChanged)}>
          Save
        </button>
        <button className="danger" onClick={remove} disabled={busy}>
          Delete model
        </button>
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  );
}
