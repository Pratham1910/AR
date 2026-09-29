import type { CameraStatus } from "../hooks/useCamera";

interface Props {
  status: CameraStatus;
  deviceLabel: string | null;
  resolution: { width: number; height: number } | null;
  error: string | null;
}

/**
 * Unambiguous camera connection feedback — a live video frame rendering is
 * itself evidence the camera works, but it's easy to miss a black/frozen
 * feed at a glance, so this makes the state explicit.
 */
export function CameraStatusBadge({ status, deviceLabel, resolution, error }: Props) {
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
      </p>
    );
  }
  return <p className="camera-status camera-status-pending">🟡 Requesting camera access…</p>;
}
