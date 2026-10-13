import type { TagCorner } from "../../types";

// Grid order: how the corners sit around a product (and on the printed sheet).
export const CORNERS_GRID: TagCorner[] = ["top_left", "top_right", "bottom_left", "bottom_right"];
// Walking around the product, as the backend's region vertices come back.
export const CORNERS_AROUND: TagCorner[] = ["top_left", "top_right", "bottom_right", "bottom_left"];

export const CORNER_LABEL: Record<TagCorner, string> = {
  top_left: "Top-left",
  top_right: "Top-right",
  bottom_right: "Bottom-right",
  bottom_left: "Bottom-left",
};

export const cornerList = (corners: TagCorner[]) => corners.map((c) => CORNER_LABEL[c].toLowerCase()).join(", ");

// Opens the browser's print dialog for one image (a tag or a four-tag sheet)
// at its natural size, so it prints at 100% with its white margin.
export function printImage(url: string, title: string) {
  const win = window.open("", "_blank", "width=900,height=900");
  if (!win) return;
  win.document.title = title;
  const img = win.document.createElement("img");
  img.style.maxWidth = "100%";
  img.onload = () => {
    win.focus();
    win.print();
  };
  img.src = url;
  win.document.body.style.margin = "0";
  win.document.body.appendChild(img);
}
