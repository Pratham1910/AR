import { useState, type PointerEvent as ReactPointerEvent, type RefObject } from "react";

type Box = [number, number, number, number]; // x1, y1, x2, y2 in the camera frame's pixels

interface Props {
  videoRef: RefObject<HTMLVideoElement>;
  box: Box | null;
  onBox: (box: Box) => void;
}

/**
 * Drag a rectangle on the video to mark the object to track — finding it
 * needs no detector class and no 3D model. The box is reported in the camera
 * frame's own pixels (the video may be shown scaled). Dragging again redraws it.
 */
export function BoxSelector({ videoRef, box, onBox }: Props) {
  const [drag, setDrag] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);

  const scale = () => {
    const video = videoRef.current;
    return video && video.videoWidth ? video.videoWidth / video.clientWidth : 1;
  };
  const local = (e: ReactPointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const onDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    const p = local(e);
    setDrag({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
  };
  const onMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!drag) return;
    const p = local(e);
    setDrag({ ...drag, x1: p.x, y1: p.y });
  };
  const onUp = () => {
    if (!drag) return;
    const s = scale();
    const x1 = Math.min(drag.x0, drag.x1);
    const y1 = Math.min(drag.y0, drag.y1);
    const x2 = Math.max(drag.x0, drag.x1);
    const y2 = Math.max(drag.y0, drag.y1);
    setDrag(null);
    if (x2 - x1 >= 10 && y2 - y1 >= 10) onBox([x1 * s, y1 * s, x2 * s, y2 * s]); // ignore clicks
  };

  // What to draw: the rectangle being dragged, else the chosen box (frame px -> screen px).
  const s = scale();
  const shown = drag
    ? {
        left: Math.min(drag.x0, drag.x1),
        top: Math.min(drag.y0, drag.y1),
        width: Math.abs(drag.x1 - drag.x0),
        height: Math.abs(drag.y1 - drag.y0),
      }
    : box
      ? { left: box[0] / s, top: box[1] / s, width: (box[2] - box[0]) / s, height: (box[3] - box[1]) / s }
      : null;

  return (
    <div className="box-selector" onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp}>
      {shown && <div className="box-selector-rect" style={shown} />}
    </div>
  );
}
