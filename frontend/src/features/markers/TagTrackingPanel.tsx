import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { MarkerTagsApi, VisionApi, apiErrorMessage } from "../../services/api";
import type { PoseResponse, ScanProduct, ScanResponse, ScanState } from "../../types";
import { CORNER_SHORT } from "./corners";
import { drawScan, newSessionId, wait } from "./scanOverlay";

interface Props {
  captureFrame: () => string | null;
  stage: HTMLElement | null;
  /** The marker tab's "Start live tracking" is on: run the tracking loop. */
  active: boolean;
  /** The 3D model currently loaded in the AR view. */
  modelId: string;
  /** Places (or, with null, hides) the loaded model using the existing marker-pose alignment. */
  showMarkerPose: (pose: PoseResponse | null) => void;
  /** A product was confirmed whose model is not the one loaded — load it and resume. */
  onOtherProduct: (product: ScanProduct) => void;
}

const CHIP: Record<ScanState, "ok" | "warn" | "bad"> = {
  NO_MARKERS: "warn",
  UNKNOWN_MARKERS: "warn",
  INCOMPLETE: "warn",
  VERIFYING: "warn",
  OBJECT_NOT_FOUND: "warn",
  DUPLICATE_MARKERS: "bad",
  MIXED_PRODUCTS: "bad",
  MULTIPLE_PRODUCTS: "bad",
  LAYOUT_INVALID: "bad",
  CANNOT_VERIFY: "bad",
  WRONG_OBJECT: "bad",
  CONFIRMED: "ok",
};

const STATE_TEXT: Record<ScanState, string> = {
  NO_MARKERS: "No tags",
  UNKNOWN_MARKERS: "Unknown tags",
  DUPLICATE_MARKERS: "Duplicate tag",
  INCOMPLETE: "Tags missing",
  MIXED_PRODUCTS: "Mixed products",
  MULTIPLE_PRODUCTS: "Several products",
  LAYOUT_INVALID: "Wrong tag layout",
  CANNOT_VERIFY: "Cannot verify",
  WRONG_OBJECT: "Wrong object",
  OBJECT_NOT_FOUND: "Object not detected",
  VERIFYING: "Verifying…",
  CONFIRMED: "Tracking confirmed",
};

/**
 * Live tracking for the ArUco marker tab. Every frame goes through the strict
 * check (backend POST /api/marker-tags/scan): all four tags of one product,
 * on their registered corners, and YOLO finding that product's class inside
 * the area the four tags enclose — nowhere else. Only while that is confirmed
 * is the 3D model placed, with the tab's existing marker pose + alignment
 * (POST /api/vision/pose on the product's top-left tag). Anything less hides
 * the model and says why.
 *
 * One loop, one request in flight: the pose request is made inside the same
 * iteration, after the frame was confirmed.
 */
export function TagTrackingPanel({ captureFrame, stage, active, modelId, showMarkerPose, onOtherProduct }: Props) {
  const [result, setResult] = useState<ScanResponse | null>(null);
  const [pose, setPose] = useState<PoseResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  // The loop reads these through refs so a re-render never restarts it.
  const live = useRef({ captureFrame, modelId, showMarkerPose, onOtherProduct });
  live.current = { captureFrame, modelId, showMarkerPose, onOtherProduct };

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    const sessionId = newSessionId("track");
    setResult(null);
    setPose(null);
    setError(null);

    const loop = async () => {
      while (!cancelled) {
        const frame = live.current.captureFrame();
        if (!frame) {
          setError("The camera is not delivering frames.");
          live.current.showMarkerPose(null);
          await wait(500);
          continue;
        }
        try {
          const scan = await MarkerTagsApi.scan(sessionId, frame);
          if (cancelled) break;
          setResult(scan);
          setError(null);
          drawScan(canvasRef.current, {
            ...scan,
            regionBad: scan.state === "LAYOUT_INVALID" || scan.state === "MIXED_PRODUCTS",
          });

          const product = scan.confirmed ? scan.product : null;
          if (product && product.model_id && product.model_id !== live.current.modelId) {
            // Confirmed, but another product's model is on screen: load the right one first.
            live.current.onOtherProduct(product);
            break;
          }
          if (product && product.model_id) {
            const found = await VisionApi.estimatePose(frame, product.pose_marker_id);
            if (cancelled) break;
            setPose(found);
            live.current.showMarkerPose(found);
          } else {
            // Not confirmed (or nothing to overlay): the model must not be shown.
            setPose(null);
            live.current.showMarkerPose(null);
          }
        } catch (err) {
          if (cancelled) break;
          setError(apiErrorMessage(err));
          live.current.showMarkerPose(null);
          await wait(1000); // don't hammer a failing backend
        }
        await wait(30);
      }
    };
    void loop();

    return () => {
      cancelled = true;
      drawScan(canvasRef.current, null);
      live.current.showMarkerPose(null);
      void MarkerTagsApi.endScan(sessionId).catch(() => undefined);
    };
  }, [active]);

  const v = result?.verification ?? null;
  const seen = result ? result.tags.filter((t) => t.asset_id !== null && t.asset_id === result.candidate_asset_id).length : 0;

  return (
    <div className="card product-scan">
      {stage &&
        createPortal(
          <canvas
            ref={canvasRef}
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
          />,
          stage
        )}

      <div className="panel-title">
        Live tracking — four tags + object check
        {active && <span className="chip info">running</span>}
      </div>

      {!active && (
        <p className="hint">
          <strong>Start live tracking</strong> (above) tracks the physical product: all four of its tags must be in view
          on their registered corners, and the object detector must find the product's class inside the area they
          enclose. Only then is the 3D model placed on it.
        </p>
      )}

      {active && !result && !error && <p className="hint">Starting — looking for the four tags…</p>}

      {active && result && (
        <div className={`pose-status ${result.state === "CONFIRMED" ? "pose-found" : "pose-not-found"}`}>
          <p>
            <span className={`chip ${CHIP[result.state]}`}>{STATE_TEXT[result.state]}</span> {result.message}
          </p>

          <p className="scan-line">
            <span className="scan-key">Tags</span>
            {result.tags.length === 0 && <span className="hint">none detected</span>}
            {result.tags.map((tag) => (
              <span
                key={tag.marker_id}
                className={`chip ${tag.asset_id === null ? "bad" : tag.asset_id === result.candidate_asset_id ? "ok" : "warn"}`}
                title={tag.asset_name ? `${tag.asset_name}${tag.held ? " — not in this frame, seen a moment ago" : ""}` : "Not mapped to any product"}
              >
                {tag.marker_id}
                {tag.corner ? ` ${CORNER_SHORT[tag.corner]}` : " unmapped"}
                {tag.held ? " …" : ""}
              </span>
            ))}
            {result.candidate_asset_name && <span className="hint">{seen}/4</span>}
          </p>

          <p className="scan-line">
            <span className="scan-key">Object</span>
            {!v && <span className="hint">YOLO not run — it needs four matching tags to mark out where to look</span>}
            {v && (
              <>
                <span className={`chip ${v.status === "VERIFIED" ? "ok" : v.status === "NOT_FOUND" ? "warn" : "bad"}`}>
                  {v.status.replace("_", " ").toLowerCase()}
                </span>{" "}
                expected {v.expected_class ?? "—"} inside the tags
                {v.detected_class
                  ? `, sees ${v.detected_class} ${((v.confidence ?? 0) * 100).toFixed(0)}%${v.inside !== null ? ` (${(v.inside * 100).toFixed(0)}% of it inside)` : ""}`
                  : v.status === "NOT_FOUND"
                    ? ", sees nothing there"
                    : ""}
              </>
            )}
          </p>

          <p className="scan-line">
            <span className="scan-key">Product</span>
            {result.confirmed && result.product ? (
              <strong>{result.product.asset_name}</strong>
            ) : result.candidate_asset_name ? (
              <span className="hint">
                {result.candidate_asset_name} — not confirmed
                {result.state === "VERIFYING" ? ` (${result.stable_frames}/${result.required_frames} frames)` : ""}
              </span>
            ) : (
              <span className="hint">none</span>
            )}
          </p>

          {result.confirmed && result.product && (
            <p className="scan-line">
              <span className="scan-key">3D model</span>
              {!result.product.model_id ? (
                <span className="hint">no 3D model is registered for this product — nothing to overlay</span>
              ) : pose?.found ? (
                <span className="hint">
                  placed on tag {result.product.pose_marker_id} (top-left)
                  {pose.calibration_is_approximate ? " — approximate camera calibration" : ""}
                </span>
              ) : (
                <span className="hint">waiting for the top-left tag's pose</span>
              )}
            </p>
          )}
        </div>
      )}

      {error && <p className="error">{error}</p>}
    </div>
  );
}
