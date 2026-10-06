import { useCallback, useEffect, useRef, useState } from "react";

export type CameraStatus = "requesting" | "streaming" | "error";

export interface CameraDeviceOption {
  deviceId: string;
  label: string;
}

/**
 * Wraps getUserMedia + a hidden canvas so callers can grab a base64 JPEG
 * snapshot on demand (Project.md #13 — camera-first MVP, no AR/Unity).
 *
 * `status` only reaches "streaming" once the browser has actually decoded a
 * frame (`loadedmetadata`, `videoWidth > 0`) — getUserMedia resolving just
 * means permission was granted and a track exists, not that video is
 * flowing. A black frame despite "streaming" usually means the *wrong*
 * camera got picked (laptops often expose an IR/depth camera alongside the
 * real webcam, or a virtual camera from other software) — that's what
 * `devices`/`selectDevice` are for.
 */
const SAVED_DEVICE_KEY = "tvasta.cameraDeviceId";
// Virtual cameras from other software: they fail with "Could not start video
// source" unless their app is running, so they're tried last when falling back.
const VIRTUAL_CAMERA = /droidcam|obs|virtual|meta quest|oculus|snap camera|manycam|xsplit|broadcast/i;

const readSavedDevice = (): string | null => {
  try {
    return localStorage.getItem(SAVED_DEVICE_KEY);
  } catch {
    return null;
  }
};
const saveDevice = (deviceId: string) => {
  try {
    localStorage.setItem(SAVED_DEVICE_KEY, deviceId);
  } catch {
    // private window / blocked storage: just won't be remembered
  }
};

const openStream = (deviceId?: string) =>
  navigator.mediaDevices.getUserMedia({
    video: deviceId ? { deviceId: { exact: deviceId }, width: 1280, height: 720 } : { width: 1280, height: 720 },
  });

const describeCameraError = (err: unknown): string => {
  const name = err instanceof DOMException ? err.name : "";
  if (name === "NotReadableError") {
    return (
      "Could not start this camera — it's in use by another app, or it's a virtual camera " +
      "(DroidCam, Meta Quest, OBS…) whose app isn't running. Pick another camera."
    );
  }
  if (name === "NotAllowedError") return "Camera permission was denied — allow it in the browser's site settings.";
  if (name === "OverconstrainedError") return "That camera is no longer available — pick another one.";
  return err instanceof Error ? err.message : String(err);
};

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [status, setStatus] = useState<CameraStatus>("requesting");
  const [error, setError] = useState<string | null>(null);
  const [deviceLabel, setDeviceLabel] = useState<string | null>(null);
  const [resolution, setResolution] = useState<{ width: number; height: number } | null>(null);
  const [devices, setDevices] = useState<CameraDeviceOption[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | null>(null);

  const stopCurrentStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  const refreshDeviceList = useCallback(async () => {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = all
        .filter((d) => d.kind === "videoinput")
        .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Camera ${i + 1}` }));
      setDevices(videoInputs);
      return videoInputs;
    } catch {
      return [];
    }
  }, []);

  /**
   * With `autoFallback` (on mount), a camera that fails to start is skipped
   * and the others are tried, real webcams before virtual ones — so a
   * default that changed to e.g. DroidCam doesn't leave the page with no
   * video. An explicit pick from the dropdown never falls back silently.
   */
  const startStream = useCallback(
    async (deviceId?: string, autoFallback = false) => {
      setStatus("requesting");
      setError(null);
      try {
        let stream: MediaStream;
        try {
          stream = await openStream(deviceId);
        } catch (firstError) {
          if (!autoFallback) throw firstError;
          const candidates = (await refreshDeviceList())
            .filter((d) => d.deviceId && d.deviceId !== deviceId)
            .sort((a, b) => Number(VIRTUAL_CAMERA.test(a.label)) - Number(VIRTUAL_CAMERA.test(b.label)));
          let opened: MediaStream | null = null;
          for (const candidate of candidates) {
            try {
              opened = await openStream(candidate.deviceId);
              break;
            } catch {
              // try the next camera
            }
          }
          if (!opened) throw firstError;
          stream = opened;
        }
        stopCurrentStream();
        streamRef.current = stream;

        const [track] = stream.getVideoTracks();
        const actualDeviceId = track?.getSettings().deviceId ?? deviceId ?? null;
        setDeviceLabel(track?.label || "camera");
        setSelectedDeviceId(actualDeviceId);
        if (actualDeviceId) saveDevice(actualDeviceId);

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          // Some browsers won't paint a frame without an explicit play()
          // call even with autoPlay set, especially after a stream swap.
          await videoRef.current.play().catch(() => undefined);
        }

        // Device labels are only populated once permission has been
        // granted at least once, so (re-)enumerate after a successful grant.
        await refreshDeviceList();
      } catch (err) {
        setError(describeCameraError(err));
        setStatus("error");
        // Device list (with labels) still matters here: it's how the user picks a working camera.
        await refreshDeviceList();
      }
    },
    [refreshDeviceList, stopCurrentStream]
  );

  useEffect(() => {
    const video = videoRef.current;
    const handleLoadedMetadata = () => {
      if (!video || video.videoWidth === 0) return;
      setResolution({ width: video.videoWidth, height: video.videoHeight });
      setStatus("streaming");
    };
    video?.addEventListener("loadedmetadata", handleLoadedMetadata);

    // Start on the last camera that actually streamed, not the OS default
    // (which other software, e.g. DroidCam, can silently take over).
    void startStream(readSavedDevice() ?? undefined, true);

    const handleDeviceChange = () => void refreshDeviceList();
    navigator.mediaDevices.addEventListener?.("devicechange", handleDeviceChange);

    return () => {
      video?.removeEventListener("loadedmetadata", handleLoadedMetadata);
      navigator.mediaDevices.removeEventListener?.("devicechange", handleDeviceChange);
      stopCurrentStream();
    };
    // Intentionally run once on mount; switching devices goes through
    // selectDevice() below rather than re-running this effect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectDevice = useCallback(
    (deviceId: string) => {
      void startStream(deviceId);
    },
    [startStream]
  );

  const captureFrameBase64 = useCallback((): string | null => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.videoWidth === 0) return null;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
    return dataUrl.split(",")[1] ?? null;
  }, []);

  // The current frame as JPEG bytes, encoded off the main thread (toBlob) —
  // for streaming every camera frame without stalling rendering.
  const captureFrameJpeg = useCallback((): Promise<Blob | null> => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.videoWidth === 0) return Promise.resolve(null);
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return Promise.resolve(null);
    ctx.drawImage(video, 0, 0);
    return new Promise((resolve) => canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.85));
  }, []);

  return {
    videoRef,
    canvasRef,
    status,
    ready: status === "streaming",
    error,
    deviceLabel,
    resolution,
    devices,
    selectedDeviceId,
    selectDevice,
    captureFrameBase64,
    captureFrameJpeg,
  };
}
