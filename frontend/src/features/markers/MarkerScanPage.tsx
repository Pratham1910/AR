import { useEffect, useRef, useState } from "react";
import { CameraSelect } from "../../components/CameraSelect";
import { CameraStatusBadge } from "../../components/CameraStatusBadge";
import { useCamera } from "../../hooks/useCamera";
import { MarkersApi, Models3DApi, apiErrorMessage, resolveApiUrl } from "../../services/api";
import { ThreeViewer } from "../viewer3d/ThreeViewer";
import type {
  Asset,
  DetectedMarker,
  MarkerBinding,
  MarkerConfig,
  Model3DInfo,
  Product,
  ScanStatus,
} from "../../types";
import { drawScanOverlay, scanLabel, scanTone } from "./scanOverlay";

interface Props {
  assets: Asset[];
  /** Open the product in the 3D / AR Registration tab, locked onto this marker. */
  onOpenIn3D: (assetId: string, markerId: number | undefined) => void;
  /** Open the Procedure Inspection tab with this product's procedure selected. */
  onStartProcedure: (assetId: string, revisionId: string) => void;
}

const COLOR_KNOWN = "#00e676";
const COLOR_UNKNOWN = "#ffca28";
const COLOR_HELD = "#ff7043";

const markerColor = (marker: DetectedMarker) =>
  !marker.visible ? COLOR_HELD : marker.status === "known" ? COLOR_KNOWN : COLOR_UNKNOWN;

// The pipeline a scan walks through, in order; each lights up once passed.
const SCAN_STEPS: [string, (scan: ScanStatus) => boolean][] = [
  ["Marker", (s) => s.stages.marker],
  ["Product", (s) => s.stages.product],
  ["Verified", (s) => s.stages.verified],
  ["Position", (s) => s.stages.position],
  ["Steady", (s) => s.stages.steady],
  ["Confirmed", (s) => s.state === "CONFIRMED"],
];

// One scan session for the page's lifetime, not per visit to this tab: the
// backend keeps a confirmed product latched until it leaves the scan box, and
// that has to survive going to the 3D workspace and coming back.
const SESSION_ID = `m-${Math.random().toString(36).slice(2)}`;
// The last confirmation already acted on (shown / opened), so the same scan
// is never opened twice.
let handledConfirmation = 0;

// A webcam facing the user reads most naturally mirrored (move right, the
// picture moves right), so that is the default; remembered per browser.
const MIRROR_KEY = "tvasta.scanMirror";
const readMirror = (): boolean => {
  try {
    return localStorage.getItem(MIRROR_KEY) !== "0";
  } catch {
    return true;
  }
};

// How long "Confirmed" shows before the product's 3D model + procedure open.
const OPEN_AFTER_CONFIRM_MS = 1500;

/**
 * Automatic product scanner: point the camera at a product with its printed
 * marker and it confirms by itself, like a QR scanner. The marker says which
 * product it is, the object detector must see that product at the marker,
 * and the backend's scan gate waits for it to be well placed and held still
 * (POST /api/markers/scan). On confirmation the product's existing 3D model
 * and procedure open.
 *
 * Guidance ("Move LEFT", "Move CLOSER", ...) is computed by the backend from
 * each frame's marker corners and product box relative to the scan box; this
 * page only draws it (scanOverlay.ts). It is all relative to the picture —
 * nothing here is a distance in meters.
 */
export function MarkerScanPage({ assets, onOpenIn3D, onStartProcedure }: Props) {
  const {
    videoRef,
    canvasRef,
    ready,
    status: cameraStatus,
    error: cameraError,
    deviceLabel,
    resolution,
    devices,
    selectedDeviceId,
    selectDevice,
    captureFrameBase64,
  } = useCamera();
  const outlineCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const sessionIdRef = useRef(SESSION_ID);

  const [config, setConfig] = useState<MarkerConfig | null>(null);
  const [scanning, setScanning] = useState(true);
  const [markers, setMarkers] = useState<DetectedMarker[] | null>(null); // null = no frame scanned yet
  const [scan, setScan] = useState<ScanStatus | null>(null);
  const [autoOpen, setAutoOpen] = useState(true); // open the 3D workspace once confirmed
  const confirmed = scan?.state === "CONFIRMED";
  const confirmation = confirmed ? scan.confirmation : 0;
  const [mirrored, setMirrored] = useState(readMirror);
  // Read by the scan loop, which outlives any one render.
  const mirroredRef = useRef(mirrored);
  const confirmedRef = useRef(confirmed);
  useEffect(() => {
    mirroredRef.current = mirrored;
    confirmedRef.current = confirmed;
    try {
      localStorage.setItem(MIRROR_KEY, mirrored ? "1" : "0");
    } catch {
      // private window / blocked storage: just won't be remembered
    }
  }, [mirrored, confirmed]);
  const [apiError, setApiError] = useState<string | null>(null);

  // The product whose model + instructions are shown. Sticky: it stays after
  // its marker leaves the frame, so the marker needn't be held in view.
  const [focus, setFocus] = useState<{ assetId: string; markerId: number } | null>(null);
  const [product, setProduct] = useState<Product | null>(null);
  const [models, setModels] = useState<Model3DInfo[]>([]);

  const [bindings, setBindings] = useState<MarkerBinding[]>([]);
  const [newMarkerId, setNewMarkerId] = useState("");
  const [newAssetId, setNewAssetId] = useState("");
  const [bindingsOpen, setBindingsOpen] = useState(false);

  const refreshBindings = () =>
    MarkersApi.bindings()
      .then(setBindings)
      .catch((err) => setApiError(apiErrorMessage(err)));

  useEffect(() => {
    MarkersApi.config()
      .then(setConfig)
      .catch((err) => setApiError(apiErrorMessage(err)));
    void refreshBindings();
  }, []);

  const drawMarkers = (detected: DetectedMarker[], status: ScanStatus) => {
    const canvas = outlineCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || video.videoWidth === 0) return;
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    drawScanOverlay(canvas, detected, status, mirroredRef.current);
  };

  useEffect(() => {
    if (!scanning || !ready) return;
    let cancelled = false;

    // Self-rescheduling, like the AR overlay's live tracking: each scan is a
    // camera -> backend round trip, so the next starts when the last returns.
    const loop = async () => {
      while (!cancelled) {
        try {
          const frame = captureFrameBase64();
          if (frame) {
            const result = await MarkersApi.scan(sessionIdRef.current, frame, mirroredRef.current);
            if (cancelled) break;
            setMarkers(result.markers);
            setScan(result.scan);
            drawMarkers(result.markers, result.scan);
            setApiError(null);
          }
        } catch (err) {
          setApiError(apiErrorMessage(err));
          await new Promise((resolve) => setTimeout(resolve, 1000)); // don't hammer a backend that's down
        }
        // Once confirmed, frames are only needed to notice the product leaving.
        await new Promise((resolve) => setTimeout(resolve, confirmedRef.current ? 300 : 60));
      }
    };
    void loop();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scanning, ready]);

  // A confirmed scan shows its product here, then opens its 3D model +
  // procedure — once per confirmation. Coming back to this tab with the same
  // product still in the scan box is the same confirmation, so nothing reopens.
  const isNewConfirmation = confirmation > 0 && confirmation !== handledConfirmation;
  useEffect(() => {
    if (!confirmed || !scan?.product || scan.marker_id === null) return;
    const assetId = scan.product.asset_id;
    setFocus({ assetId, markerId: scan.marker_id });
    if (!isNewConfirmation) return;
    if (!autoOpen) {
      handledConfirmation = confirmation;
      return;
    }
    const timer = setTimeout(() => {
      handledConfirmation = confirmation;
      onOpenIn3D(assetId, undefined);
    }, OPEN_AFTER_CONFIRM_MS);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [confirmation, autoOpen]);
  useEffect(() => {
    if (!confirmed) setAutoOpen(true); // the next product opens by itself again
  }, [confirmed]);

  const scanAgain = async () => {
    await MarkersApi.endSession(sessionIdRef.current).catch(() => undefined);
    setScan(null);
    setMarkers(null);
    setFocus(null);
    setAutoOpen(true);
  };

  useEffect(() => {
    setProduct(null);
    setModels([]);
    if (!focus) return;
    let stale = false;
    Promise.all([MarkersApi.product(focus.assetId), Models3DApi.listForAsset(focus.assetId)])
      .then(([p, m]) => {
        if (stale) return;
        setProduct(p);
        setModels(m);
      })
      .catch((err) => setApiError(apiErrorMessage(err)));
    return () => {
      stale = true;
    };
  }, [focus?.assetId]);

  const addBinding = async () => {
    try {
      await MarkersApi.setBinding(Number(newMarkerId), newAssetId);
      setNewMarkerId("");
      setApiError(null);
      await refreshBindings();
    } catch (err) {
      setApiError(apiErrorMessage(err));
    }
  };

  const removeBinding = async (bindingId: string) => {
    try {
      await MarkersApi.removeBinding(bindingId);
      await refreshBindings();
    } catch (err) {
      setApiError(apiErrorMessage(err));
    }
  };

  const visible = markers?.filter((m) => m.visible) ?? [];
  const held = markers?.filter((m) => !m.visible) ?? [];
  const unknownCount = visible.filter((m) => m.status === "unknown").length;
  const model = models[0]; // newest first, same default as the 3D tab

  return (
    <div>
      <p className="hint">
        Point the camera at a product with its printed {config ? `${config.family} ${config.dictionary}` : ""} marker.
        It scans by itself once the product is verified, inside the frame and held still. Print markers with{" "}
        <code>python -m app.workers.generate_marker --id 0</code>.
      </p>

      <div style={{ position: "relative", width: "100%", maxWidth: 640 }}>
        {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          // Only the picture is flipped; the overlay canvas maps its own x (scanOverlay.ts).
          style={{ width: "100%", display: "block", transform: mirrored ? "scaleX(-1)" : undefined }}
        />
        <canvas
          ref={outlineCanvasRef}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
        />
        <canvas ref={canvasRef} style={{ display: "none" }} />
      </div>

      {ready && scan && (
        <div className={`scan-status scan-${scanTone(scan)}`}>
          <p className="scan-message">
            <span className="scan-label">{scanLabel(scan)}</span> {scan.message}
          </p>
          {scan.expected_class && (
            <p className="hint">
              Expected: {scan.product?.name} ({scan.expected_class}) · Detected:{" "}
              {scan.detected_class
                ? `${scan.detected_class} ${((scan.detected_confidence ?? 0) * 100).toFixed(0)}%`
                : "nothing at the marker"}
            </p>
          )}
          <ol className="scan-steps">
            {SCAN_STEPS.map(([label, passed]) => (
              <li key={label} className={passed(scan) ? "done" : ""}>
                {label}
              </li>
            ))}
          </ol>
          <div className="scan-progress">
            <div style={{ width: `${Math.round(scan.progress * 100)}%` }} />
          </div>
          {scan.geometry && !confirmed && (
            <p className="hint scan-geometry">
              offset {scan.geometry.offset_x >= 0 ? "+" : ""}
              {(scan.geometry.offset_x * 100).toFixed(0)}% / {scan.geometry.offset_y >= 0 ? "+" : ""}
              {(scan.geometry.offset_y * 100).toFixed(0)}% of the scan box · marker{" "}
              {(scan.geometry.marker_size * 100).toFixed(0)}% of frame
              {scan.geometry.product_fill !== null
                ? ` · product fills ${(scan.geometry.product_fill * 100).toFixed(0)}% of the box`
                : ""}{" "}
              · size {scan.geometry.distance.replace("_", " ")} · facing {(scan.geometry.squareness * 100).toFixed(0)}%
              {scan.geometry.speed !== null ? ` · speed ${scan.geometry.speed.toFixed(2)}` : ""}
            </p>
          )}
          {confirmed && (
            <div className="registration-controls">
              {isNewConfirmation && autoOpen ? (
                <>
                  <span className="hint">Opening its 3D model and procedure…</span>
                  <button onClick={() => setAutoOpen(false)}>Stay here</button>
                </>
              ) : (
                <>
                  <button
                    className="primary"
                    onClick={() => scan.product && onOpenIn3D(scan.product.asset_id, undefined)}
                  >
                    Open 3D model and procedure
                  </button>
                  <span className="hint">Take the product out of the scan box to scan another.</span>
                </>
              )}
              <button onClick={() => void scanAgain()}>Scan again</button>
            </div>
          )}
        </div>
      )}

      <CameraStatusBadge status={cameraStatus} deviceLabel={deviceLabel} resolution={resolution} error={cameraError} />
      <CameraSelect devices={devices} selectedDeviceId={selectedDeviceId} onSelect={selectDevice} />
      {cameraStatus === "error" && (
        <p className="warning">
          Scanning needs a webcam. Connect one (or allow camera access for this site) and pick it above; nothing is
          scanned until then.
        </p>
      )}

      <div className="registration-controls">
        <label>
          <input type="checkbox" checked={scanning} onChange={(e) => setScanning(e.target.checked)} />
          Live scan
        </label>
        <label title="Flip the preview left-right, like a mirror. Guidance directions follow what you see.">
          <input type="checkbox" checked={mirrored} onChange={(e) => setMirrored(e.target.checked)} />
          Mirror preview
        </label>
        {ready && markers && (
          <span className="marker-summary">
            {visible.length === 0 && held.length === 0 && "No marker in view"}
            {visible.length > 0 &&
              `${visible.length} marker${visible.length === 1 ? "" : "s"} in view` +
                (unknownCount > 0 ? ` (${unknownCount} unknown)` : "")}
            {held.length > 0 && `${visible.length > 0 ? ", " : ""}${held.length} temporarily out of frame`}
          </span>
        )}
      </div>

      {markers && markers.length > 0 && (
        <table className="marker-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Product</th>
              <th>Status</th>
              <th>Corners 0–3 (px)</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {markers.map((marker, index) => (
              <tr key={`${marker.marker_id}-${index}`}>
                <td>
                  <span className="marker-dot" style={{ background: markerColor(marker) }} />
                  {marker.marker_id}
                </td>
                <td>{marker.product?.name ?? <em>Unknown marker — no product bound</em>}</td>
                <td>{marker.visible ? "in view" : `out of frame ${(marker.ms_since_seen / 1000).toFixed(1)} s`}</td>
                <td className="marker-corners">
                  {marker.corners_px.map((c) => `(${c.x.toFixed(0)}, ${c.y.toFixed(0)})`).join(" ")}
                </td>
                <td>
                  {marker.product ? (
                    <button
                      disabled={focus?.assetId === marker.product.asset_id}
                      onClick={() => setFocus({ assetId: marker.product!.asset_id, markerId: marker.marker_id })}
                    >
                      {focus?.assetId === marker.product.asset_id ? "Showing" : "Show"}
                    </button>
                  ) : (
                    <button
                      onClick={() => {
                        setNewMarkerId(String(marker.marker_id));
                        setBindingsOpen(true);
                      }}
                    >
                      Bind…
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {apiError && <p className="error">{apiError}</p>}

      {focus && product && (
        <div className="product-panel">
          <h3>
            {product.name} <span className="hint">— marker ID {focus.markerId}</span>
          </h3>
          {product.description && <p className="hint">{product.description}</p>}

          <div className="product-panel-grid">
            <div>
              <strong>3D model</strong>
              {model ? (
                <>
                  <ThreeViewer modelUrl={resolveApiUrl(model.url)} height={300} />
                  <button onClick={() => onOpenIn3D(product.asset_id, focus.markerId)}>
                    Overlay on marker in AR Registration
                  </button>
                </>
              ) : (
                <>
                  <p className="hint">No 3D model registered for this product yet.</p>
                  <button onClick={() => onOpenIn3D(product.asset_id, undefined)}>Upload one in 3D / AR tab</button>
                </>
              )}
            </div>

            <div>
              <strong>Instructions</strong>
              {product.procedures.length === 0 && (
                <p className="hint">No published procedure for this product yet.</p>
              )}
              {product.procedures.map((procedure) => (
                <div key={procedure.procedure_id}>
                  <p>
                    {procedure.title} <span className="hint">rev {procedure.revision_label}</span>
                  </p>
                  <ol className="product-steps">
                    {procedure.steps.map((step) => (
                      <li key={step.step_id_str}>
                        {step.title}
                        <span className="hint">
                          {" "}
                          — {step.action_type}
                          {step.component ? ` ${step.component}` : ""} → {step.expected_state}
                        </span>
                      </li>
                    ))}
                  </ol>
                  <button onClick={() => onStartProcedure(product.asset_id, procedure.revision_id)}>
                    Run this procedure
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <details
        className="marker-bindings"
        open={bindingsOpen}
        onToggle={(e) => setBindingsOpen((e.target as HTMLDetailsElement).open)}
      >
        <summary>
          Marker → product mapping ({bindings.length}
          {config ? `, ${config.dictionary}` : ""})
        </summary>
        <table className="marker-table">
          <tbody>
            {bindings.map((binding) => (
              <tr key={binding.id}>
                <td>ID {binding.marker_id}</td>
                <td>{binding.asset_name}</td>
                <td>
                  <button className="danger" onClick={() => void removeBinding(binding.id)}>
                    Remove
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="registration-controls">
          <label>
            Marker ID{" "}
            <input
              type="number"
              min={0}
              max={config?.marker_count ? config.marker_count - 1 : undefined}
              value={newMarkerId}
              onChange={(e) => setNewMarkerId(e.target.value)}
              style={{ width: "5rem" }}
            />
          </label>
          <label>
            Product{" "}
            <select value={newAssetId} onChange={(e) => setNewAssetId(e.target.value)}>
              <option value="">Select an asset…</option>
              {assets.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.name}
                </option>
              ))}
            </select>
          </label>
          <button disabled={newMarkerId === "" || !newAssetId} onClick={() => void addBinding()}>
            Bind
          </button>
        </div>
        <p className="hint">Binding an ID that is already mapped replaces its product.</p>
      </details>
    </div>
  );
}
