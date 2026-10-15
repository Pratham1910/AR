import { useEffect, useState } from "react";
import { apiErrorMessage, CalibrationApi, type CalibrationStatus } from "../../services/api";

/**
 * Camera calibration with the live camera (backend app/api/calibration.py):
 * photograph a printed checkerboard 15-20 times, compute the camera's real
 * intrinsics and lens distortion, and use them instead of the assumed
 * field of view. Only frames where the board's corners are found are kept.
 */
export function CalibrationPanel({
  captureFrameJpeg,
  cameraLabel,
  onClose,
}: {
  captureFrameJpeg: () => Promise<Blob | null>;
  cameraLabel: string | null;
  onClose: () => void;
}) {
  const [status, setStatus] = useState<CalibrationStatus | null>(null);
  const [cols, setCols] = useState(9);
  const [rows, setRows] = useState(6);
  const [squareMm, setSquareMm] = useState(25);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    CalibrationApi.status()
      .then(setStatus)
      .catch((err) => setMessage(apiErrorMessage(err)));
  }, []);

  const run = async (action: () => Promise<void>) => {
    setBusy(true);
    setMessage(null);
    try {
      await action();
    } catch (err) {
      setMessage(apiErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const capture = () =>
    run(async () => {
      const blob = await captureFrameJpeg();
      if (!blob) throw new Error("No camera frame");
      const bytes = new Uint8Array(await blob.arrayBuffer());
      let binary = "";
      for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
      const result = await CalibrationApi.capture(btoa(binary), cols, rows);
      setStatus((s) => (s ? { ...s, images_captured: result.images_captured } : s));
      setMessage(result.found ? "Board found — frame kept." : "Board not found in this frame — hold it fully in view, flat, well lit.");
    });

  const captured = status?.images_captured ?? 0;

  return (
    <div className="calibration-panel">
      <div className="calibration-head">
        <strong>Camera calibration</strong>
        <span className="hint">
          {status?.calibrated
            ? `In use: ${status.camera_label ?? "unnamed camera"} · ${status.image_width}×${status.image_height} · ${
                status.rms_px != null ? `${status.rms_px.toFixed(2)} px RMS` : "—"
              } · lens bends corners by up to ${status.distortion_max_px?.toFixed(1) ?? "—"} px · ${status.created_at ?? ""}`
            : "Not calibrated — intrinsics are assumed from an 82° field of view."}
        </span>
        <button onClick={onClose}>Close</button>
      </div>
      <ol className="calibration-steps">
        <li>Print a checkerboard and measure one square. Count the inner corners (where four squares meet).</li>
        <li>Capture 15–20 frames: board tilted different ways, near and far, and in every part of the image — corners too.</li>
        <li>Compute. The result is used for this camera ({cameraLabel ?? "current camera"}) from the next frame on.</li>
      </ol>
      <div className="calibration-row">
        <label>
          Inner corners
          <input type="number" min={3} value={cols} onChange={(e) => setCols(Number(e.target.value))} />×
          <input type="number" min={3} value={rows} onChange={(e) => setRows(Number(e.target.value))} />
        </label>
        <label>
          Square (mm)
          <input type="number" min={1} step={0.1} value={squareMm} onChange={(e) => setSquareMm(Number(e.target.value))} />
        </label>
        <button onClick={() => void capture()} disabled={busy}>
          Capture frame ({captured})
        </button>
        <button
          onClick={() =>
            void run(async () => {
              const s = await CalibrationApi.run(cols, rows, squareMm / 1000, cameraLabel);
              setStatus(s);
              setMessage(`Calibrated: ${s.rms_px?.toFixed(2)} px RMS from ${s.images_used} frames.`);
            })
          }
          disabled={busy || captured < 10}
          title={captured < 10 ? "Capture at least 10 frames with the board found" : "Compute and use the calibration"}
        >
          Compute
        </button>
        <button onClick={() => void run(async () => setStatus(await CalibrationApi.clearImages()))} disabled={busy || !captured}>
          Clear frames
        </button>
        {status?.calibrated && (
          <button onClick={() => void run(async () => setStatus(await CalibrationApi.remove()))} disabled={busy}>
            Remove calibration
          </button>
        )}
      </div>
      {message && <p className="hint">{message}</p>}
      {status?.rms_px != null && status.rms_px > 1 && (
        <p className="warning">RMS above 1 px: recapture with sharper, better-spread frames for an accurate overlay.</p>
      )}
    </div>
  );
}
