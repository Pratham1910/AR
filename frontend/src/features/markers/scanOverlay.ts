import type { DetectedMarker, ScanStatus } from "../../types";

/*
 * Draws the scanner's overlay on the canvas that sits over the camera video:
 * scan box, product box, marker outline + corners, guidance and progress.
 *
 * Detections arrive in camera-image pixels. When the preview is mirrored the
 * <video> is flipped with CSS but this canvas is not (text would come out
 * backwards), so every x is mapped through displayX() instead. The backend is
 * told about the mirroring too, which is why its left/right hints already
 * match what is drawn here.
 */

const COLOR_OK = "#00e676";
const COLOR_WARN = "#ffca28";
const COLOR_BAD = "#ff5252";
const COLOR_HELD = "#ff7043";
const COLOR_IDLE = "rgba(255, 255, 255, 0.85)";

export type ScanLabel = "NOT READY" | "READY" | "VERIFYING" | "HOLD STEADY" | "SCANNING" | "CONFIRMED" | "MISMATCH";
export type ScanTone = "idle" | "warn" | "bad" | "ok";

/** The status shown to the user for a backend scan state. */
export function scanLabel(scan: ScanStatus): ScanLabel {
  switch (scan.state) {
    case "CONFIRMED":
    case "SCANNING":
    case "VERIFYING":
    case "MISMATCH":
      return scan.state;
    case "HOLD_STEADY":
      // Every check passes and the hold is starting, unless it is still moving.
      return scan.problem === "moving" ? "HOLD STEADY" : "READY";
    default:
      return "NOT READY";
  }
}

export function scanTone(scan: ScanStatus): ScanTone {
  if (scan.state === "MISMATCH") return "bad";
  if (scan.state === "SEARCHING") return "idle";
  return scan.stages.position ? "ok" : "warn";
}

const TONE_COLOR: Record<ScanTone, string> = { idle: COLOR_IDLE, warn: COLOR_WARN, bad: COLOR_BAD, ok: COLOR_OK };

/** Camera-image x -> where that column appears in the preview. */
export const displayX = (x: number, width: number, mirrored: boolean) => (mirrored ? width - x : x);

const markerColor = (marker: DetectedMarker) =>
  !marker.visible ? COLOR_HELD : marker.status === "known" ? COLOR_OK : COLOR_WARN;

function outlinedText(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string) {
  ctx.lineWidth = 5;
  ctx.lineJoin = "round"; // mitered joins spike out of the glyphs
  ctx.strokeStyle = "rgba(0, 0, 0, 0.8)";
  ctx.strokeText(text, x, y);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function arrow(ctx: CanvasRenderingContext2D, cx: number, cy: number, dx: number, dy: number, color: string) {
  const length = 70;
  const head = 26;
  const tipX = cx + dx * length;
  const tipY = cy + dy * length;
  ctx.strokeStyle = color;
  ctx.lineWidth = 10;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(cx - dx * length, cy - dy * length);
  ctx.lineTo(tipX, tipY);
  // Head: two strokes back from the tip, either side of the shaft.
  ctx.moveTo(tipX - dx * head - dy * head, tipY - dy * head + dx * head);
  ctx.lineTo(tipX, tipY);
  ctx.lineTo(tipX - dx * head + dy * head, tipY - dy * head - dx * head);
  ctx.stroke();
  ctx.lineCap = "butt";
}

// Preview-space direction each hint asks the product to travel in.
const HINT_DIRECTION: Record<string, [number, number]> = {
  move_left: [-1, 0],
  move_right: [1, 0],
  move_up: [0, -1],
  move_down: [0, 1],
};

export function drawScanOverlay(
  canvas: HTMLCanvasElement,
  markers: DetectedMarker[],
  scan: ScanStatus,
  mirrored: boolean
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  const X = (x: number) => displayX(x, W, mirrored);
  const tone = TONE_COLOR[scanTone(scan)];
  ctx.clearRect(0, 0, W, H);
  ctx.setLineDash([]);

  // --- Scan box: dim everything outside it, bracket its corners. It is
  // centered, so mirroring leaves it where it is.
  const [ax1, ay1, ax2, ay2] = scan.scan_area;
  ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
  ctx.fillRect(0, 0, W, H);
  ctx.clearRect(ax1, ay1, ax2 - ax1, ay2 - ay1);
  ctx.strokeStyle = tone;
  ctx.globalAlpha = 0.45;
  ctx.lineWidth = 2;
  ctx.strokeRect(ax1, ay1, ax2 - ax1, ay2 - ay1);
  ctx.globalAlpha = 1;
  ctx.lineWidth = 8;
  const bracket = 56;
  for (const [x, y, sx, sy] of [
    [ax1, ay1, 1, 1],
    [ax2, ay1, -1, 1],
    [ax2, ay2, -1, -1],
    [ax1, ay2, 1, -1],
  ]) {
    ctx.beginPath();
    ctx.moveTo(x + sx * bracket, y);
    ctx.lineTo(x, y);
    ctx.lineTo(x, y + sy * bracket);
    ctx.stroke();
  }
  const targetX = (ax1 + ax2) / 2;
  const targetY = (ay1 + ay2) / 2;

  // --- What the object detector saw at the marker.
  if (scan.product_bbox && scan.detected_class) {
    const [bx1, by1, bx2, by2] = scan.product_bbox;
    const left = Math.min(X(bx1), X(bx2));
    const boxColor = scan.state === "MISMATCH" ? COLOR_BAD : scan.stages.verified ? COLOR_OK : COLOR_WARN;
    ctx.strokeStyle = boxColor;
    ctx.lineWidth = 3;
    ctx.setLineDash(scan.stages.verified || scan.state === "MISMATCH" ? [] : [10, 8]); // dashed = not sure yet
    ctx.strokeRect(left, by1, bx2 - bx1, by2 - by1);
    ctx.setLineDash([]);
    ctx.font = "bold 20px sans-serif";
    outlinedText(
      ctx,
      `${scan.detected_class} ${((scan.detected_confidence ?? 0) * 100).toFixed(0)}%`,
      left + 6,
      Math.max(22, by1 - 8),
      boxColor
    );

    // Centering error: from the product's center to where it should be.
    if (scan.geometry?.product_center && scan.state === "POSITIONING") {
      ctx.strokeStyle = COLOR_WARN;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 6]);
      ctx.beginPath();
      ctx.moveTo(X(scan.geometry.product_center.x), scan.geometry.product_center.y);
      ctx.lineTo(targetX, targetY);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // --- Markers: boundary, four numbered corners, id + product.
  for (const marker of markers) {
    const color = markerColor(marker);
    const corners = marker.corners_px.map((pt) => ({ x: X(pt.x), y: pt.y }));

    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.setLineDash(marker.visible ? [] : [10, 8]); // dashed = held, not in this frame
    ctx.beginPath();
    corners.forEach((pt, i) => (i === 0 ? ctx.moveTo(pt.x, pt.y) : ctx.lineTo(pt.x, pt.y)));
    ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]);

    // Corner 0 is the printed pattern's own top-left, so the numbers turn with the marker.
    ctx.font = "16px sans-serif";
    corners.forEach((pt, i) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillText(String(i), pt.x + 8, pt.y - 8);
    });

    const label = `ID ${marker.marker_id} · ${marker.product?.name ?? "unknown"}${marker.visible ? "" : " (out of frame)"}`;
    ctx.font = "bold 22px sans-serif";
    const textWidth = ctx.measureText(label).width;
    const bottom = corners.reduce((a, b) => (b.y > a.y ? b : a));
    const x = Math.max(4, Math.min(X(marker.center_px.x) - textWidth / 2, W - textWidth - 4));
    outlinedText(ctx, label, x, Math.min(H - 8, bottom.y + 30), color);
  }

  // --- Which way to move, as an arrow in the preview's own directions.
  if (scan.state === "POSITIONING") {
    let dx = 0;
    let dy = 0;
    for (const hint of scan.hints) {
      dx += HINT_DIRECTION[hint]?.[0] ?? 0;
      dy += HINT_DIRECTION[hint]?.[1] ?? 0;
    }
    const length = Math.hypot(dx, dy);
    if (length > 0) arrow(ctx, targetX, targetY, dx / length, dy / length, COLOR_WARN);
  }

  // --- Hold progress ring, and the tick once confirmed.
  if (scan.progress > 0) {
    const radius = 64;
    ctx.lineWidth = 12;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
    ctx.beginPath();
    ctx.arc(targetX, targetY, radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.strokeStyle = COLOR_OK;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.arc(targetX, targetY, radius, -Math.PI / 2, -Math.PI / 2 + scan.progress * Math.PI * 2);
    ctx.stroke();
    if (scan.state === "CONFIRMED") {
      ctx.beginPath();
      ctx.moveTo(targetX - 28, targetY + 2);
      ctx.lineTo(targetX - 8, targetY + 22);
      ctx.lineTo(targetX + 30, targetY - 20);
      ctx.stroke();
    }
    ctx.lineCap = "butt";
  }

  // --- Status above the box, guidance inside its bottom edge.
  ctx.textAlign = "center";
  ctx.font = "bold 24px sans-serif";
  outlinedText(ctx, scan.state === "SEARCHING" ? "SCAN PRODUCT" : scanLabel(scan), targetX, Math.max(28, ay1 - 14), tone);
  ctx.font = "bold 40px sans-serif";
  let size = 40;
  while (ctx.measureText(scan.message).width > W - 40 && size > 18) {
    size -= 2;
    ctx.font = `bold ${size}px sans-serif`;
  }
  outlinedText(ctx, scan.message, W / 2, ay2 - 22, "#ffffff");
  ctx.textAlign = "start";
}
