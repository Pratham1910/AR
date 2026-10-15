import { useEffect, useState } from "react";

/*
 * Which station this browser is and who is operating it: shown in the top
 * bar and stamped on saved inspection reports. A per-device setting kept in
 * localStorage (Settings screen); the app works the same without it.
 */

export interface Workstation {
  station: string;
  operator: string;
}

const KEY = "tvasta.workstation";
const EVENT = "tvasta:workstation";

export function readWorkstation(): Workstation {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<Workstation>;
      return { station: parsed.station ?? "", operator: parsed.operator ?? "" };
    }
  } catch {
    // private window / blocked storage: no saved identity
  }
  return { station: "", operator: "" };
}

export function saveWorkstation(value: Workstation): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(value));
  } catch {
    // not remembered on this device; still applied for this session below
  }
  window.dispatchEvent(new CustomEvent(EVENT, { detail: value }));
}

export function useWorkstation(): Workstation {
  const [value, setValue] = useState(readWorkstation);
  useEffect(() => {
    const onChange = (e: Event) => setValue((e as CustomEvent<Workstation>).detail);
    window.addEventListener(EVENT, onChange);
    return () => window.removeEventListener(EVENT, onChange);
  }, []);
  return value;
}
