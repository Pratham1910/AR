import type { IdentifyResponse, RegionObject, ScannedTag, TagTrackResponse, Vector2 } from "../../types";
import { CORNER_LABEL } from "./corners";

// Drawn over the live video, in the captured frame's own pixels (the canvas
// is sized to the frame and stretched over the video like the other AR overlays).

const GREEN = "#2ecc71";
const AMBER = "#f5a623";
const RED = "#ff4d4f";
const CYAN = "#22d3ee";

function prepare(canvas: HTMLCanvasElement, width: number, height: number): CanvasRenderingContext2D | null {
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.clearRect(0, 0, width, height);
  return ctx;
}

export function clearOverlay(canvas: HTMLCanvasElement | null) {
  canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
}

function path(ctx: CanvasRenderingContext2D, points: Vector2[]) {
  ctx.beginPath();
  points.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
  ctx.closePath();
}

function label(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string, scale: number) {
  ctx.font = `600 ${Math.round(15 * scale)}px sans-serif`;
  const w = ctx.measureText(text).width + 10 * scale;
  const h = 22 * scale;
  ctx.fillStyle = "rgba(0,0,0,0.65)";
  ctx.fillRect(x, y - h, w, h);
  ctx.fillStyle = color;
  ctx.fillText(text, x + 5 * scale, y - 6 * scale);
}

function drawTag(ctx: CanvasRenderingContext2D, tag: ScannedTag, color: string, scale: number) {
  ctx.lineWidth = 3 * scale;
  ctx.strokeStyle = color;
  ctx.setLineDash(tag.estimated ? [8 * scale, 6 * scale] : []);
  path(ctx, tag.corners_px);
  ctx.stroke();
  ctx.setLineDash([]);
  const top = tag.corners_px.reduce((a, b) => (b.y < a.y ? b : a));
  const text = tag.corner
    ? `${tag.marker_id} ${CORNER_LABEL[tag.corner]}${tag.estimated ? " (covered)" : ""}`
    : `${tag.marker_id} unknown`;
  label(ctx, text, top.x, top.y - 4 * scale, color, scale);
}

const scaleFor = (width: number) => Math.max(width / 1280, 0.6);

export function drawIdentify(canvas: HTMLCanvasElement, r: IdentifyResponse) {
  const ctx = prepare(canvas, r.frame_width, r.frame_height);
  if (!ctx) return;
  const scale = scaleFor(r.frame_width);
  for (const tag of r.tags) {
    const color = !tag.asset_id ? RED : tag.asset_id === r.candidate_asset_id && r.state !== "AMBIGUOUS" ? GREEN : AMBER;
    drawTag(ctx, tag, color, scale);
  }
}

function drawObject(ctx: CanvasRenderingContext2D, obj: RegionObject, color: string, text: string, scale: number, fill: boolean) {
  if (obj.polygon.length >= 3) {
    path(ctx, obj.polygon);
    if (fill) {
      ctx.globalAlpha = 0.28;
      ctx.fillStyle = color;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    ctx.lineWidth = 2.5 * scale;
    ctx.strokeStyle = color;
    ctx.setLineDash(fill ? [] : [6 * scale, 5 * scale]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  const [x1, y1, x2, y2] = obj.bbox;
  ctx.lineWidth = 1.5 * scale;
  ctx.strokeStyle = color;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  label(ctx, text, x1, y2 + 26 * scale, color, scale);
}

export function drawTrack(canvas: HTMLCanvasElement, r: TagTrackResponse) {
  const ctx = prepare(canvas, r.frame_width, r.frame_height);
  if (!ctx) return;
  const scale = scaleFor(r.frame_width);

  // Dim everything the detector was NOT shown.
  if (r.search_area) {
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.beginPath();
    ctx.rect(0, 0, r.frame_width, r.frame_height);
    r.search_area.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
    ctx.closePath();
    ctx.fill("evenodd");
  }

  if (r.region) {
    // The boundary through the four tags: cyan when it is the detection
    // region, red when the arrangement is wrong (drawn so it can be fixed).
    const valid = r.arrangement === "valid";
    const color = valid ? CYAN : RED;
    ctx.lineWidth = 3 * scale;
    ctx.strokeStyle = color;
    ctx.setLineDash(valid ? [] : [10 * scale, 6 * scale]);
    path(ctx, r.region);
    ctx.stroke();
    ctx.setLineDash([]);
    label(ctx, valid ? "Detection region" : "Invalid arrangement", r.region[0].x, r.region[0].y - 30 * scale, color, scale);
  }

  for (const tag of r.tags) {
    const color = !tag.asset_id ? RED : tag.asset_id === r.asset_id ? GREEN : AMBER;
    drawTag(ctx, tag, color, scale);
  }

  for (const obj of r.rejected) {
    drawObject(ctx, obj, RED, `${obj.class_label} — outside (${Math.round(obj.inside * 100)}% in)`, scale, false);
  }
  if (r.object) {
    const color = r.state === "TRACKING" ? GREEN : r.state === "VERIFYING" ? AMBER : RED;
    const text =
      `${r.object.class_label} ${Math.round(r.object.confidence * 100)}%` +
      ` · ${Math.round(r.object.inside * 100)}% inside` +
      (r.object.held ? " · held" : "");
    drawObject(ctx, r.object, color, text, scale, true);
  }
}
