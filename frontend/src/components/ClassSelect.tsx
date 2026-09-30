import { useDetectableClasses } from "../hooks/useDetectableClasses";

interface Props {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

/**
 * Detection-class picker limited to classes the detector actually knows —
 * a free-typed class (e.g. "bot") was silently never detected. A current
 * value outside the list is still shown, flagged, so it can be corrected.
 */
export function ClassSelect({ value, onChange, placeholder = "Select a class…" }: Props) {
  const { classes, error } = useDetectableClasses();
  const unknown = value !== "" && classes.length > 0 && !classes.includes(value);

  if (error) {
    return <input value={value} onChange={(e) => onChange(e.target.value)} title={`Class list unavailable: ${error}`} />;
  }
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">{classes.length ? placeholder : "Loading classes…"}</option>
      {unknown && <option value={value}>{value} (not detectable!)</option>}
      {classes.map((c) => (
        <option key={c} value={c}>
          {c}
        </option>
      ))}
    </select>
  );
}
