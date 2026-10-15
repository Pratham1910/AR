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
// A second drop within this window means the link can't sustain the camera,
// so reporting it beats reopening forever.
const RECOVERY_COOLDOWN_MS = 10_000;
// Virtual cameras from other software: they fail with "Could not start video
// source" unless their app is running, so they're tried last when falling back.
const VIRTUAL_CAMERA =
  /droidcam|obs|virtual|meta quest|oculus|snap camera|manycam|xsplit|broadcast|camo|camtwist|epoccam|iriun|ivcam|e2esoft|splitcam|reincubate/i;

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
const forgetDevice = () => {
  try {
    localStorage.removeItem(SAVED_DEVICE_KEY);
  } catch {
    // nothing was stored anyway
  }
};

/**
 * Modes tried in order, each capped with `max` so the browser cannot
 * negotiate anything larger.
 *
 * A bare `width: 1280` is only an *ideal* in the spec — no upper bound — and
 * high-spec webcams take that as licence to pick their top mode. The Logitech
 * BRIO advertises up to 4096x2160@30 and 1920x1080@60; at those the USB link
 * browns out on a 2.0 port, a shared hub, or next to other bus traffic, and
 * the camera drops out mid-session (a black frame, then the track ends).
 *
 * Nothing downstream wants those pixels anyway: the pose service crops to the
 * object and resizes, so 4K only costs USB bandwidth, JPEG encode time and
 * upload bytes. 720p30 is the target; the last rung is uncapped so a camera
 * with no mode at or below 640x480 still opens.
 */
const VIDEO_MODES: MediaTrackConstraints[] = [
  { width: { ideal: 1280, max: 1280 }, height: { ideal: 720, max: 720 }, frameRate: { ideal: 30, max: 30 } },
  { width: { ideal: 1280, max: 1280 }, height: { ideal: 720, max: 720 }, frameRate: { ideal: 15, max: 15 } },
  { width: { ideal: 640, max: 640 }, height: { ideal: 480, max: 480 }, frameRate: { ideal: 30, max: 30 } },
  {},
];

const openStream = async (deviceId?: string): Promise<MediaStream> => {
  const device = deviceId ? { deviceId: { exact: deviceId } } : {};
  let lastError: unknown;
  for (const mode of VIDEO_MODES) {
    try {
      return await navigator.mediaDevices.getUserMedia({ video: { ...device, ...mode } });
    } catch (err) {
      lastError = err;
      // Only a mode the camera can't do is worth retrying smaller. A denied
      // permission or a camera held by another app fails the same way at
      // every size, so stop and report it.
      if (!(err instanceof DOMException && err.name === "OverconstrainedError")) throw err;
    }
  }
  throw lastError;
};

/**
 * A USB camera that browns out ends (or mutes) its track without
 * getUserMedia ever rejecting — the page would sit at "streaming" over a
 * black video with no hint why. Returns an unsubscribe function.
 *
 * `ended` is final, so it reports straight away. A `mute` is not: cameras
 * mute briefly while re-exposing or switching modes, so that only counts as
 * lost if it is still muted a few seconds later — otherwise a normal hiccup
 * would tear down a working stream.
 */
const MUTE_GRACE_MS = 3000;

const watchTrack = (track: MediaStreamTrack, onLost: () => void) => {
  let muteTimer: ReturnType<typeof setTimeout> | undefined;
  const onEnded = () => onLost();
  const onMute = () => {
    clearTimeout(muteTimer);
    muteTimer = setTimeout(() => {
      if (track.muted && track.readyState !== "ended") onLost();
    }, MUTE_GRACE_MS);
  };
  const onUnmute = () => clearTimeout(muteTimer);
  track.addEventListener("ended", onEnded);
  track.addEventListener("mute", onMute);
  track.addEventListener("unmute", onUnmute);
  return () => {
    clearTimeout(muteTimer);
    track.removeEventListener("ended", onEnded);
    track.removeEventListener("mute", onMute);
    track.removeEventListener("unmute", onUnmute);
  };
};

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
  if (name === "NotFoundError") {
    return (
      "No camera found. If one is plugged in and works elsewhere, the browser is probably " +
      "holding a stale device list — quit it completely (check the tray) and reopen. " +
      "Windows must also list the camera under Device Manager > Cameras, not Imaging devices."
    );
  }
  return err instanceof Error ? err.message : String(err);
};

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const unwatchRef = useRef<(() => void) | null>(null);
  // Lets the dropped-track handler reopen the stream without startStream
  // having to depend on itself.
  const startStreamRef = useRef<((deviceId?: string, autoFallback?: boolean) => Promise<void>) | null>(null);
  // When the last dropped-track recovery was attempted, so a camera that
  // keeps browning out reports the problem instead of reconnecting forever.
  const lastRecoveryRef = useRef(0);

  const [status, setStatus] = useState<CameraStatus>("requesting");
  const [error, setError] = useState<string | null>(null);
  const [deviceLabel, setDeviceLabel] = useState<string | null>(null);
  const [resolution, setResolution] = useState<{ width: number; height: number } | null>(null);
  const [devices, setDevices] = useState<CameraDeviceOption[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | null>(null);

  const stopCurrentStream = useCallback(() => {
    unwatchRef.current?.();
    unwatchRef.current = null;
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
          // A remembered camera can stop existing under that id — unplugged,
          // or its device path changed (a driver/class change renames it), and
          // `deviceId: {exact}` then fails with NotFoundError. Drop the stale
          // id and let the browser pick, before trying each camera by name.
          forgetDevice();
          const candidates = (await refreshDeviceList())
            .filter((d) => d.deviceId && d.deviceId !== deviceId)
            .sort((a, b) => Number(VIRTUAL_CAMERA.test(a.label)) - Number(VIRTUAL_CAMERA.test(b.label)));
          let opened: MediaStream | null = null;
          // `undefined` first: whatever the browser considers the default,
          // which needs no id to match and is the likeliest to just work.
          for (const candidate of [undefined, ...candidates.map((c) => c.deviceId)]) {
            if (candidate === deviceId) continue; // already failed above
            try {
              opened = await openStream(candidate);
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

        // Reopen once if the camera drops off the bus; if it keeps happening,
        // say so rather than reconnecting in a loop behind a black video.
        if (track) {
          unwatchRef.current = watchTrack(track, () => {
            if (Date.now() - lastRecoveryRef.current > RECOVERY_COOLDOWN_MS) {
              lastRecoveryRef.current = Date.now();
              void startStreamRef.current?.(actualDeviceId ?? undefined, true);
              return;
            }
            setStatus("error");
            setError(
              "The camera keeps dropping off the USB bus. High-resolution webcams (e.g. the " +
                "Logitech BRIO) need more bandwidth than a hub or a USB 2.0 port can sustain — " +
                "plug it directly into a USB 3.0 port (usually blue) and unplug other cameras."
            );
          });
        }

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
  startStreamRef.current = startStream;

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

    // Swapping cameras used to refresh the dropdown but leave the page on its
    // old failure, so plugging a working camera into a page that started with
    // none still read "camera not found" until a manual reload. If nothing is
    // streaming when the device list changes, try again with what's there now.
    const handleDeviceChange = () => {
      void refreshDeviceList();
      const track = streamRef.current?.getVideoTracks()[0];
      if (track?.readyState !== "live") void startStreamRef.current?.(undefined, true);
    };
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
