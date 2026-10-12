import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { MarkerTagsApi, apiErrorMessage } from "../../services/api";
import type { IdentifyResponse, IdentifyState, ScanProduct } from "../../types";
import { CORNER_SHORT } from "./corners";
import { ProductInlineView, type ProductView } from "./ProductInlineView";
import { drawScan, newSessionId, wait } from "./scanOverlay";

interface Props {
  captureFrame: () => string | null;
  ready: boolean; // the camera is delivering frames
  /** An element stacked over the video; the scan draws the tags it sees there. */
  stage: HTMLElement | null;
  /** Whether this scan's loop should run — owned by the parent, so it can never run alongside live tracking. */
  scanning: boolean;
  onStart: () => void;
  onStop: () => void;
  /** Opens the Procedure inspection tab — offered as an explicit button inside the procedure view only. */
  onOpenInspection?: (assetId: string, revisionId: string) => void;
}

const CHIP: Record<IdentifyState, "ok" | "warn" | "bad"> = {
  NO_MARKERS: "warn",
  IDENTIFYING: "warn",
  UNKNOWN_MARKERS: "bad",
  AMBIGUOUS: "bad",
  IDENTIFIED: "ok",
};

const STATE_TEXT: Record<IdentifyState, string> = {
  NO_MARKERS: "Looking for a tag",
  UNKNOWN_MARKERS: "Unknown tag",
  AMBIGUOUS: "More than one product",
  IDENTIFYING: "Confirming…",
  IDENTIFIED: "Product identified",
};

/**
 * Product scan: show ANY ONE of a product's ArUco tags and get that product's
 * 3D model and procedure. A lookup, nothing more (backend POST
 * /api/marker-tags/identify): it needs no four-tag set and runs no YOLO, so
 * it says which product a tag belongs to — not that the product is physically
 * there. Confirming the physical product is live tracking's job
 * (TagTrackingPanel), which shares no code path or state with this.
 *
 * An unmapped tag, or tags of two products at once, select nothing.
 *
 * "View 3D model" / "View procedure" show the product's model or procedure
 * right below this panel (ProductInlineView). Nothing navigates: the scanner,
 * its controls and the identified product stay on screen, and the two views
 * can be switched freely.
 */
export function ProductScanPanel({ captureFrame, ready, stage, scanning, onStart, onStop, onOpenInspection }: Props) {
  const [result, setResult] = useState<IdentifyResponse | null>(null);
  const [product, setProduct] = useState<ScanProduct | null>(null);
  // Which of the product's views is open below the panel (null: neither yet).
  const [view, setView] = useState<ProductView | null>(null);
  const inlineRef = useRef<HTMLDivElement | null>(null);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  // The loop reads these through refs so a re-render never restarts it.
  const captureRef = useRef(captureFrame);
  captureRef.current = captureFrame;
  const onStopRef = useRef(onStop);
  onStopRef.current = onStop;

  useEffect(() => {
    if (!scanning) return;
    let cancelled = false;
    const sessionId = newSessionId("identify");
    setResult(null);
    setProduct(null);
    setView(null); // a new scan is a new product; its views open again on request
    setError(null);

    // One request at a time, the next as soon as the last answers.
    const loop = async () => {
      while (!cancelled) {
        const frame = captureRef.current();
        if (!frame) {
          setError("The camera is not delivering frames.");
          await wait(500);
          continue;
        }
        try {
          const scan = await MarkerTagsApi.identify(sessionId, frame);
          if (cancelled) break;
          setResult(scan);
          setError(null);
          drawScan(canvasRef.current, scan);
          if (scan.identified && scan.product) {
            setProduct(scan.product);
            onStopRef.current(); // found: stop scanning, keep the result on screen
            break;
          }
        } catch (err) {
          if (cancelled) break;
          setError(apiErrorMessage(err));
          await wait(1000); // don't hammer a failing backend
        }
        await wait(80);
      }
    };
    void loop();

    return () => {
      cancelled = true;
      drawScan(canvasRef.current, null);
      void MarkerTagsApi.endIdentify(sessionId).catch(() => undefined);
    };
  }, [scanning]);

  const reset = () => {
    setProduct(null);
    setView(null);
    setResult(null);
    setError(null);
  };

  const open = (next: ProductView) => {
    setView(next);
    // Bring the view into sight once it has rendered — scrolling this page, not leaving it.
    window.setTimeout(() => inlineRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }), 60);
  };

  const panel = (
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
        Product scan — any one tag
        <span className="scan-buttons">
          {product && !scanning && <button onClick={reset}>Clear</button>}
          <button className={scanning ? "" : "primary"} onClick={scanning ? onStop : onStart} disabled={!ready && !scanning}>
            {scanning ? "■ Stop scan" : product ? "Scan another product" : "Start product scan"}
          </button>
        </span>
      </div>

      {!scanning && !result && !product && (
        <p className="hint">
          Show any one of a product's ArUco tags to open that product's 3D model or procedure. This only looks the tag
          up — to check that the physical product is really there, use <strong>Start live tracking</strong>, which needs
          all four tags and detects the object inside them.
          {!ready && " Waiting for the camera…"}
        </p>
      )}

      {scanning && !result && !error && <p className="hint">Scanning for a registered tag…</p>}

      {product && !scanning && (
        <div className="pose-status pose-found">
          <p>
            <span className="chip ok">Product identified</span> <strong>{product.asset_name}</strong>
            {result && result.tags.length > 0 && (
              <span className="hint">
                {" "}
                — from tag{" "}
                {result.tags
                  .filter((t) => t.asset_id === product.asset_id)
                  .map((t) => `${t.marker_id}${t.corner ? ` (${CORNER_SHORT[t.corner]})` : ""}`)
                  .join(", ")}
              </span>
            )}
          </p>
          <div className="scan-actions">
            <button className={view === "model" ? "primary" : ""} aria-pressed={view === "model"} onClick={() => open("model")}>
              View 3D model
            </button>
            <button
              className={view === "procedure" ? "primary" : ""}
              aria-pressed={view === "procedure"}
              onClick={() => open("procedure")}
            >
              View procedure
            </button>
            {view && <button onClick={() => setView(null)}>Hide</button>}
          </div>
          <p className="hint">
            Identified from its tag only — the physical product has not been checked. Use Start live tracking for that.
          </p>
        </div>
      )}

      {result && !product && (
        <div className={`pose-status ${result.state === "IDENTIFIED" ? "pose-found" : "pose-not-found"}`}>
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
                title={tag.asset_name ?? "Not mapped to any product"}
              >
                {tag.marker_id}
                {tag.asset_name ? ` → ${tag.asset_name}` : " unmapped"}
              </span>
            ))}
          </p>
        </div>
      )}

      {error && <p className="error">{error}</p>}
    </div>
  );

  // The card above is the scanner; the product's model / procedure opens directly beneath it.
  return (
    <>
      {panel}
      {product && view && (
        <div ref={inlineRef}>
          <ProductInlineView product={product} view={view} onOpenInspection={onOpenInspection} />
        </div>
      )}
    </>
  );
}
