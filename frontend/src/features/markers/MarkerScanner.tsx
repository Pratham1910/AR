import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { MarkerTagsApi, apiErrorMessage } from "../../services/api";
import type { IdentifyResponse, TagProduct, TagTrackResponse, TagTrackState } from "../../types";
import { CORNER_LABEL } from "./corners";
import { clearOverlay, drawIdentify, drawTrack } from "./overlay";
import { ProductPanel } from "./ProductPanel";

export interface MarkerPanelContext {
  captureFrame: () => string | null;
  ready: boolean; // the camera is delivering frames
  stage: HTMLElement | null; // layer over the video to draw on
  busy: boolean; // the tab's single-marker pose loop is running
  /** Four-tag scanning, switched by the view's "Start scanning" button. */
  scanning: boolean;
  onScanningChange: (on: boolean) => void;
}

interface Props extends MarkerPanelContext {
  onOpenInspection?: (assetId: string, revisionId: string) => void;
}

const STATE_TEXT: Record<TagTrackState, string> = {
  NO_MARKERS: "Looking for tags",
  UNKNOWN_MARKERS: "Unregistered tags",
  DUPLICATE_MARKERS: "Duplicate tag",
  INCOMPLETE: "Waiting for all four tags",
  MIXED_PRODUCTS: "Tags of different products",
  MULTIPLE_PRODUCTS: "Several products in view",
  LAYOUT_INVALID: "Tags in the wrong arrangement",
  OBJECT_NOT_FOUND: "Looking for the object between the tags",
  WRONG_OBJECT: "Wrong object between the tags",
  VERIFYING: "Object found — confirming",
  TRACKING: "Tracking",
};
const STATE_CHIP: Record<TagTrackState, "ok" | "warn" | "bad"> = {
  NO_MARKERS: "warn",
  UNKNOWN_MARKERS: "bad",
  DUPLICATE_MARKERS: "bad",
  INCOMPLETE: "warn",
  MIXED_PRODUCTS: "bad",
  MULTIPLE_PRODUCTS: "bad",
  LAYOUT_INVALID: "bad",
  OBJECT_NOT_FOUND: "warn",
  WRONG_OBJECT: "bad",
  VERIFYING: "warn",
  TRACKING: "ok",
};

const newSession = () => `tags-${Math.random().toString(36).slice(2)}`;
const pause = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const xy = (p: { x: number; y: number } | null) => (p ? `${Math.round(p.x)}, ${Math.round(p.y)}` : "—");

type Loop = "track" | "identify";

/**
 * The ArUco marker mode's product tags (docs/marker-tags.md).
 *
 * "Start scanning" (the view's main button) runs the four-tag scan, which
 * moves on by itself: all four tags read -> arrangement checked -> YOLO on
 * the region they enclose -> object confirmed -> tracking. Each step is shown
 * below the video. "Identify by one tag" is the separate lookup: any one tag
 * names its product.
 */
export function MarkerScanner({ captureFrame, ready, stage, busy, scanning, onScanningChange, onOpenInspection }: Props) {
  const [identifying, setIdentifying] = useState(false);
  const [scan, setScan] = useState<IdentifyResponse | null>(null);
  const [track, setTrack] = useState<TagTrackResponse | null>(null);
  const [product, setProduct] = useState<TagProduct | null>(null); // stays after scanning stops
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureRef = useRef(captureFrame);
  captureRef.current = captureFrame;

  const loop: Loop | null = scanning ? "track" : identifying ? "identify" : null;

  // The single-marker pose loop and the tag loops share the camera: one at a time.
  useEffect(() => {
    if (busy) setIdentifying(false);
  }, [busy]);
  useEffect(() => {
    if (scanning) setIdentifying(false);
  }, [scanning]);

  useEffect(() => {
    if (loop === null) return;
    let cancelled = false;
    const session = newSession();
    setError(null);
    setScan(null);
    setTrack(null);

    const run = async () => {
      while (!cancelled) {
        const frame = captureRef.current();
        if (!frame) {
          await pause(150);
          continue;
        }
        try {
          if (loop === "identify") {
            const r = await MarkerTagsApi.identify(session, frame);
            if (cancelled) break;
            setScan(r);
            if (canvasRef.current) drawIdentify(canvasRef.current, r);
            if (r.product) setProduct(r.product);
          } else {
            const r = await MarkerTagsApi.track(session, frame);
            if (cancelled) break;
            setTrack(r);
            if (canvasRef.current) drawTrack(canvasRef.current, r);
            if (r.product) setProduct(r.product);
          }
          setError(null);
        } catch (err) {
          if (cancelled) break;
          setError(apiErrorMessage(err));
          await pause(1000);
        }
        await pause(30);
      }
    };
    void run();

    return () => {
      cancelled = true;
      clearOverlay(canvasRef.current);
      const end = loop === "identify" ? MarkerTagsApi.endIdentify : MarkerTagsApi.endTrack;
      void end(session).catch(() => undefined);
    };
  }, [loop]);

  const hud =
    loop === "track" && track ? (
      <div className={`hud-pill tag-hud tag-hud-${STATE_CHIP[track.state]}`}>
        <span className="dot" />
        {STATE_TEXT[track.state]}
        {track.state === "VERIFYING" && ` ${track.stable_frames}/${track.required_frames}`}
        {track.asset_name && <span className="sub">{track.asset_name}</span>}
      </div>
    ) : loop === "identify" && scan ? (
      <div className={`hud-pill tag-hud tag-hud-${scan.state === "IDENTIFIED" ? "ok" : "warn"}`}>
        <span className="dot" />
        {scan.state === "IDENTIFIED" ? scan.candidate_asset_name : "Show any one tag"}
      </div>
    ) : null;

  const objectStep = (() => {
    if (!track || track.arrangement !== "valid") return { state: "", text: "Starts automatically once the four tags are valid" };
    const looking = track.expected_class ?? "any object";
    if (track.object && (track.state === "TRACKING" || track.state === "VERIFYING")) {
      const o = track.object;
      return {
        state: "ok",
        text: `${o.class_label} ${Math.round(o.confidence * 100)}% · ${Math.round(o.inside * 100)}% of its mask inside${o.held ? " · held" : ""}`,
      };
    }
    if (track.state === "WRONG_OBJECT") return { state: "bad", text: `Expected ${looking}; found ${track.object?.class_label ?? "something else"}` };
    return { state: "warn", text: `Searching inside the region for ${looking}${track.rejected.length ? ` (ignored outside: ${track.rejected.map((o) => o.class_label).join(", ")})` : ""}` };
  })();

  return (
    <div className="marker-scanner">
      {stage &&
        createPortal(
          <>
            <canvas ref={canvasRef} className="marker-overlay" />
            {hud && <div className="hud hud-top-left">{hud}</div>}
          </>,
          stage
        )}

      <div className="marker-scanner-head">
        <div className="panel-title">Product scan (four tags)</div>
        <button
          className={identifying ? "stop" : ""}
          disabled={(!ready || busy) && !identifying}
          onClick={() => {
            onScanningChange(false);
            setIdentifying((v) => !v);
          }}
          title="Any one registered tag names its product (no object detection)"
        >
          {identifying ? "■ Stop identifying" : "Identify by one tag"}
        </button>
      </div>

      {loop === null && (
        <p className="hint">
          Place the product's four registered tags around the object and press <strong>▶ Start scanning</strong> under the
          video. The tags are checked, then the object between them is detected and tracked — no further clicks.
          {busy && " (Stop the single-marker pose first.)"}
        </p>
      )}

      {loop === "track" && !track && <p className="hint">Starting…</p>}

      {loop === "track" && track && (
        <>
          <div className="scan-status">
            <span className={`chip ${STATE_CHIP[track.state]}`}>{STATE_TEXT[track.state]}</span>
            <span>{track.message}</span>
          </div>

          <ol className="scan-steps">
            <li className={track.corners.length && track.corners.every((c) => c.status !== "missing") ? "ok" : "warn"}>
              <strong>Four tags</strong>
              {track.corners.length === 0 ? (
                <span className="hint">
                  {track.tags.length === 0
                    ? "No tags in view."
                    : `Tags in view: ${track.tags.map((t) => t.marker_id).join(", ")} — none of one registered product.`}
                </span>
              ) : (
                <div className="corner-grid">
                  {track.corners.map((c) => (
                    <div key={c.corner} className={`corner-cell corner-${c.status}`}>
                      <span className="corner-name">{CORNER_LABEL[c.corner]}</span>
                      <span className="mono">id {c.marker_id}</span>
                      <span>{c.status === "seen" ? "✓ detected" : c.status === "estimated" ? "~ covered (estimated)" : "✕ missing"}</span>
                      <span className="hint mono">{xy(c.center_px)}</span>
                    </div>
                  ))}
                </div>
              )}
            </li>
            <li className={track.arrangement === "valid" ? "ok" : track.arrangement === "invalid" ? "bad" : "warn"}>
              <strong>Arrangement</strong>
              <span>
                {track.arrangement === "valid"
                  ? `Valid — ${track.asset_name}'s tags on their corners; scanning the region between them.`
                  : track.arrangement === "invalid"
                    ? `Invalid — ${track.arrangement_problem ?? "see above"}`
                    : "Waiting for all four tags."}
              </span>
            </li>
            <li className={objectStep.state}>
              <strong>Object</strong>
              <span>{objectStep.text}</span>
            </li>
            <li className={track.state === "TRACKING" ? "ok" : track.state === "VERIFYING" ? "warn" : ""}>
              <strong>Tracking</strong>
              <span>
                {track.state === "TRACKING"
                  ? `Tracking ${track.asset_name}`
                  : track.state === "VERIFYING"
                    ? `Confirming ${track.stable_frames}/${track.required_frames} frames`
                    : "Not yet"}
              </span>
            </li>
          </ol>

          <details className="scan-debug">
            <summary>Debug: every tag detected in the last frame</summary>
            {track.tags.length === 0 ? (
              <p className="hint">The detector read no tags in this frame.</p>
            ) : (
              <table className="tag-table">
                <thead>
                  <tr>
                    <th>Id</th>
                    <th>Registered to</th>
                    <th>Corner</th>
                    <th>Centre (px)</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {track.tags.map((t) => {
                    const cx = t.corners_px.reduce((a, p) => a + p.x, 0) / 4;
                    const cy = t.corners_px.reduce((a, p) => a + p.y, 0) / 4;
                    return (
                      <tr key={`${t.marker_id}-${t.estimated}`}>
                        <td className="mono">{t.marker_id}</td>
                        <td>{t.asset_name ?? <span className="hint">not registered</span>}</td>
                        <td>{t.corner ? CORNER_LABEL[t.corner] : "—"}</td>
                        <td className="mono">{xy({ x: cx, y: cy })}</td>
                        <td>{t.estimated ? "estimated" : "read"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
            <p className="hint">
              Frame {track.frame_width}×{track.frame_height} · tags {track.markers_ms.toFixed(0)} ms
              {track.yolo_ms !== null ? ` · YOLO ${track.yolo_ms.toFixed(0)} ms` : ""} · search area{" "}
              {track.search_rect ? track.search_rect.join(", ") : "—"}. The server logs each state change:{" "}
              <code>docker compose logs backend | grep four-tag</code>
            </p>
          </details>
        </>
      )}

      {loop === "identify" && scan && (
        <div className="scan-status">
          <span className={`chip ${scan.state === "IDENTIFIED" ? "ok" : scan.state === "IDENTIFYING" ? "warn" : scan.state === "NO_MARKERS" ? "" : "bad"}`}>
            {scan.state.replace("_", " ").toLowerCase()}
          </span>
          <span>{scan.message}</span>
        </div>
      )}

      {error && <p className="error">{error}</p>}

      {product && <ProductPanel product={product} onOpenInspection={onOpenInspection} onClose={() => setProduct(null)} />}
    </div>
  );
}
