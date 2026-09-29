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

  const startStream = useCallback(
    async (deviceId?: string) => {
      setStatus("requesting");
      setError(null);
      try {
        const constraints: MediaStreamConstraints = {
          video: deviceId ? { deviceId: { exact: deviceId }, width: 1280, height: 720 } : { width: 1280, height: 720 },
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        stopCurrentStream();
        streamRef.current = stream;

        const [track] = stream.getVideoTracks();
        setDeviceLabel(track?.label || "camera");
        setSelectedDeviceId(track?.getSettings().deviceId ?? deviceId ?? null);

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
        setError(err instanceof Error ? err.message : String(err));
        setStatus("error");
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

    void startStream();

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
  };
}
