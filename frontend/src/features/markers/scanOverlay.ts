import type { ScannedTag, ScanVerification, Vector2 } from "../../types";
import { CORNER_SHORT } from "./corners";

export const GREEN = "#00e676";
export const AMBER = "#ffb300";
export const RED = "#ff5252";

/** What one frame of either scan mode found, in the frame's own pixels. */
export interface ScanDrawing {
  frame_width: number;
  frame_height: number;
  tags: ScannedTag[];
  candidate_asset_id: string | null;
  /** Live tracking only: the area the four tags enclose. */
  region?: Vector2[] | null;
  regionBad?: boolean;
  /** Live tracking only: what YOLO found in that area (null: it did not run). */
  verification?: ScanVerification | null;
}

/**
 * Draws a scan result over the video: every tag (green = the candidate
 * product's, amber = another product's, red = not mapped; dashed = held from
 * a moment ago), and for live tracking the search region and YOLO's finding.
 * Passing null clears the canvas.
 */
export function drawScan(canvas: HTMLCanvasElement | null, scan: ScanDrawing | null): void {
  const ctx = canvas?.getContext("2d");
  if (!canvas || !ctx) return;
  if (!scan) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  if (canvas.width !== scan.frame_width || canvas.height !== scan.frame_height) {
    canvas.width = scan.frame_width;
    canvas.height = scan.frame_height;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const unit = Math.max(1, scan.frame_width / 640); // keep lines/text readable at any resolution

  const path = (points: Vector2[]) => {
    ctx.beginPath();
    points.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
    ctx.closePath();
  };
  const label = (text: string, x: number, y: number, color: string) => {
    ctx.font = `${Math.round(13 * unit)}px sans-serif`;
    const width = ctx.measureText(text).width + 8 * unit;
    ctx.fillStyle = "rgba(0, 0, 0, 0.65)";
    ctx.fillRect(x, y - 16 * unit, width, 19 * unit);
    ctx.fillStyle = color;
    ctx.fillText(text, x + 4 * unit, y - 2 * unit);
  };

  // The area the four tags enclose. When YOLO ran, it saw only this: the
  // rest of the picture is dimmed to show what was left out.
  const v = scan.verification ?? null;
  if (scan.region) {
    const searched = v !== null && v.status !== "NO_CLASS" && v.status !== "UNKNOWN_CLASS";
    if (searched) {
      ctx.fillStyle = "rgba(0, 0, 0, 0.45)";
      ctx.beginPath();
      ctx.rect(0, 0, canvas.width, canvas.height);
      scan.region.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
      ctx.closePath();
      ctx.fill("evenodd");
    }
    ctx.setLineDash([10 * unit, 6 * unit]);
    ctx.lineWidth = 2 * unit;
    ctx.strokeStyle = scan.regionBad ? RED : GREEN;
    path(scan.region);
    ctx.stroke();
    ctx.setLineDash([]);
    if (searched) {
      const left = Math.min(...scan.region.map((p) => p.x));
      const bottom = Math.max(...scan.region.map((p) => p.y));
      label("YOLO search area", left, Math.min(canvas.height - 4 * unit, bottom + 18 * unit), GREEN);
    }
  }

  if (v?.bbox) {
    const [x1, y1, x2, y2] = v.bbox;
    const color = v.status === "VERIFIED" ? GREEN : RED;
    ctx.lineWidth = 3 * unit;
    ctx.strokeStyle = color;
    if (v.polygon && v.polygon.length > 2) {
      path(v.polygon);
      ctx.stroke();
    } else {
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
    label(`${v.detected_class} ${((v.confidence ?? 0) * 100).toFixed(0)}%`, x1, Math.max(18 * unit, y1), color);
  }

  for (const tag of scan.tags) {
    const color = tag.asset_id === null ? RED : tag.asset_id === scan.candidate_asset_id ? GREEN : AMBER;
    ctx.lineWidth = 3 * unit;
    ctx.strokeStyle = color;
    ctx.setLineDash(tag.held ? [5 * unit, 5 * unit] : []);
    path(tag.corners_px);
    ctx.stroke();
    ctx.setLineDash([]);
    const top = Math.min(...tag.corners_px.map((p) => p.y));
    const left = Math.min(...tag.corners_px.map((p) => p.x));
    label(`${tag.marker_id}${tag.corner ? ` ${CORNER_SHORT[tag.corner]}` : " ?"}`, left, Math.max(18 * unit, top - 2 * unit), color);
  }
}

export const newSessionId = (prefix: string) =>
  `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

export const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
