import type { TagCorner } from "../../types";

// Reading order, which is also how the tags sit on the product.
export const CORNERS: TagCorner[] = ["top_left", "top_right", "bottom_left", "bottom_right"];

export const CORNER_LABEL: Record<TagCorner, string> = {
  top_left: "Top-Left",
  top_right: "Top-Right",
  bottom_left: "Bottom-Left",
  bottom_right: "Bottom-Right",
};

export const CORNER_SHORT: Record<TagCorner, string> = {
  top_left: "TL",
  top_right: "TR",
  bottom_left: "BL",
  bottom_right: "BR",
};
