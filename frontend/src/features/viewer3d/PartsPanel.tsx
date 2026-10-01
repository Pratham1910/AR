import { useEffect, useState } from "react";
import { Models3DApi, apiErrorMessage } from "../../services/api";
import type { Model3DPart } from "../../types";
import type { PartView } from "./parts";

interface Props {
  modelId: string;
  parts: Model3DPart[];
  view: PartView;
  onViewChange: (view: PartView) => void;
  onPartsChange: (parts: Model3DPart[]) => void;
}

/**
 * The assembly's parts (body, cap, …) next to the 3D view: show/hide each,
 * select one to highlight it, and give it a real name. The model is loaded
 * once; this only changes what's visible / outlined.
 */
export function PartsPanel({ modelId, parts, view, onViewChange, onPartsChange }: Props) {
  const [names, setNames] = useState<Record<number, string>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setNames(Object.fromEntries(parts.map((p) => [p.node_index, p.display_name])));
  }, [parts]);

  const toggleVisible = (nodeIndex: number) => {
    const hidden = view.hidden.includes(nodeIndex)
      ? view.hidden.filter((i) => i !== nodeIndex)
      : [...view.hidden, nodeIndex];
    onViewChange({ ...view, hidden });
  };

  const showOnly = (nodeIndex: number) =>
    onViewChange({ hidden: parts.map((p) => p.node_index).filter((i) => i !== nodeIndex), highlight: nodeIndex });

  const saveName = async (part: Model3DPart) => {
    const name = (names[part.node_index] ?? "").trim();
    if (!name || name === part.display_name) return;
    setError(null);
    try {
      onPartsChange(await Models3DApi.renamePart(modelId, part.node_index, name));
    } catch (err) {
      setError(apiErrorMessage(err));
    }
  };

  return (
    <div className="parts-panel">
      <strong>Parts</strong>
      <table>
        <thead>
          <tr>
            <th title="Show / hide">👁</th>
            <th>Name</th>
            <th>Size (cm)</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {parts.map((part) => {
            const selected = view.highlight === part.node_index;
            return (
              <tr
                key={part.node_index}
                className={selected ? "selected" : ""}
                onClick={() => onViewChange({ ...view, highlight: selected ? null : part.node_index })}
              >
                <td onClick={(e) => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    checked={!view.hidden.includes(part.node_index)}
                    onChange={() => toggleVisible(part.node_index)}
                  />
                </td>
                <td onClick={(e) => e.stopPropagation()}>
                  <input
                    value={names[part.node_index] ?? ""}
                    onChange={(e) => setNames({ ...names, [part.node_index]: e.target.value })}
                    onBlur={() => void saveName(part)}
                    onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
                    title={`GLB node: ${part.node_name}`}
                  />
                </td>
                <td>{part.size_m.map((v) => (v * 100).toFixed(1)).join(" × ")}</td>
                <td onClick={(e) => e.stopPropagation()}>
                  <button onClick={() => showOnly(part.node_index)}>Only</button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="parts-actions">
        <button onClick={() => onViewChange({ hidden: [], highlight: null })}>Show all</button>
        <span className="hint">Click a row (or the part in the 3D view) to highlight it.</span>
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  );
}
