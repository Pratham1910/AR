import type { CameraStatus } from "../hooks/useCamera";

interface Props {
  status: CameraStatus;
  deviceLabel: string | null;
  resolution: { width: number; height: number } | null;
  error: string | null;
  /** "Camera busy — retrying (2/4)…" while a busy camera is retried. */
  retryNote?: string | null;
  /** Another TVASTA tab holds the camera. */
  otherTabHasCamera?: boolean;
  onRetry?: () => void;
  onTakeOver?: () => void;
}

/**
 * Unambiguous camera connection feedback — a live video frame rendering is
 * itself evidence the camera works, but it's easy to miss a black/frozen
 * feed at a glance, so this makes the state explicit. On failure it offers
 * Retry (after closing the app that held the camera) and, when another
 * TVASTA tab has it, taking it over.
 */
export function CameraStatusBadge({
  status, deviceLabel, resolution, error, retryNote, otherTabHasCamera, onRetry, onTakeOver,
}: Props) {
  if (status === "streaming") {
    return (
      <p className="camera-status camera-status-ok">
        🟢 Camera connected{deviceLabel ? ` — ${deviceLabel}` : ""}
        {resolution ? ` (${resolution.width}×${resolution.height})` : ""}
      </p>
    );
  }
  if (status === "error") {
    return (
      <p className="camera-status camera-status-error">
        🔴 Camera not connected — {error ?? "unknown error"}
        {error?.includes("Permission") || error?.includes("NotAllowed") ? " (check browser camera permission)" : ""}
        {otherTabHasCamera && onTakeOver && (
          <button className="camera-retry" onClick={onTakeOver}>
            Use camera here
          </button>
        )}
        {onRetry && (
          <button className="camera-retry" onClick={onRetry}>
            Retry
          </button>
        )}
      </p>
    );
  }
  return <p className="camera-status camera-status-pending">🟡 {retryNote ?? "Requesting camera access…"}</p>;
}
