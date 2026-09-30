import { useEffect, useState } from "react";
import { VisionApi } from "../services/api";

// One fetch per page load, shared by every caller — the list only changes if
// the backend's detection model is swapped.
let cached: Promise<string[]> | null = null;

/** Class labels the backend's object detector can actually find (COCO's 80 for the stock model). */
export function useDetectableClasses(): { classes: string[]; error: string | null } {
  const [classes, setClasses] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    cached ??= VisionApi.classes().catch((err) => {
      cached = null; // retry on next mount instead of caching the failure
      throw err;
    });
    let active = true;
    cached
      .then((list) => active && setClasses(list))
      .catch((err: Error) => active && setError(err.message));
    return () => {
      active = false;
    };
  }, []);

  return { classes, error };
}
