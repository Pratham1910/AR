import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Wraps getUserMedia + a hidden canvas so callers can grab a base64 JPEG
 * snapshot on demand (Project.md #13 — camera-first MVP, no AR/Unity).
 */
export function useCamera() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let stream: MediaStream | undefined;
    navigator.mediaDevices
      .getUserMedia({ video: { width: 1280, height: 720 } })
      .then((s) => {
        stream = s;
        if (videoRef.current) {
          videoRef.current.srcObject = s;
          setReady(true);
        }
      })
      .catch((err: Error) => setError(err.message));

    return () => {
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

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

  return { videoRef, canvasRef, ready, error, captureFrameBase64 };
}
