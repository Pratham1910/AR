"""
Where the product is relative to the scan box, and what the user should do
about it — recomputed from every frame's detections.

Inputs are the marker's four corners (always) and the object detector's box
for the product (when it found one). Everything is relative: sizes are shares
of the frame or of the scan box, "distance" is only how big things look.
Nothing here is in centimeters or true degrees; that needs a calibrated or
depth camera.

Coordinates: all measurements are in camera-image pixels (x right, y down).
Only the left/right *hints* depend on how the picture is shown: on a mirrored
preview the image's right is the viewer's left, so `mirrored` swaps them.
A hint always names the direction the product should travel in the preview
the user is looking at.
"""

import math
from dataclasses import dataclass

from app.services.markers.detector import MarkerDetection

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass(frozen=True)
class ScanGateConfig:
    # Scan box, centered, as a share of the frame.
    area_width: float = 0.6
    area_height: float = 0.8
    # Allowed center offset, as a share of the scan box's width / height.
    center_tolerance: float = 0.15
    # Marker side / frame short side. Below = looks too far, above = too close.
    min_marker_size: float = 0.06
    max_marker_size: float = 0.5
    # Product box / scan box (the larger of the width and height ratios).
    # Above max = too close. At or above `marker_size_waived_fill` the
    # product already fills the box, so a small marker is no longer "too far".
    max_product_fill: float = 1.0
    marker_size_waived_fill: float = 0.6
    # Orientation: shortest / longest marker side (1 = facing the camera), and
    # in-plane rotation of the marker. 180 = any rotation is fine, since a
    # marker may be stuck on a product at any angle.
    min_squareness: float = 0.6
    max_roll_deg: float = 180.0
    # A product box this close to the frame edge (share of the frame) is cut off.
    frame_edge_margin: float = 0.01
    # Stability and confirmation.
    hold_ms: float = 800.0  # how long it must stay valid and still
    max_speed: float = 0.8  # marker sides per second: above = not steady
    verification_grace_ms: float = 400.0  # a match survives the detector missing this long
    rearm_ms: float = 1000.0  # confirmed product gone from the scan box this long = ready for the next


@dataclass(frozen=True)
class Positioning:
    scan_area: Box
    target_center: tuple[float, float]
    marker_center: tuple[float, float]
    product_center: tuple[float, float] | None
    # Offset of the product (or, without one, the marker) from the scan box's
    # center, in scan-box widths / heights. Image coordinates: +x = right in
    # the camera image, +y = down.
    offset_x: float
    offset_y: float
    marker_size: float  # marker side / frame short side
    product_fill: float | None  # product box / scan box
    distance: str  # "too_far" | "ok" | "too_close" — from apparent size only
    squareness: float
    roll_deg: float
    inside_area: bool  # marker (and product box) entirely inside the scan box
    complete: bool  # product box not cut off by the frame edge
    hints: tuple[str, ...]  # what to change, as seen in the preview; empty = well placed


def scan_area(frame_width: int, frame_height: int, config: ScanGateConfig) -> Box:
    margin_x = frame_width * (1 - config.area_width) / 2
    margin_y = frame_height * (1 - config.area_height) / 2
    return (margin_x, margin_y, frame_width - margin_x, frame_height - margin_y)


def marker_sides(marker: MarkerDetection) -> list[float]:
    c = marker.corners_px
    return [math.dist(c[i], c[i - 1]) for i in range(4)]


def _inside(box: Box, x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def analyze_position(
    marker: MarkerDetection,
    product_box: Box | None,
    frame_width: int,
    frame_height: int,
    config: ScanGateConfig,
    mirrored: bool = False,
) -> Positioning:
    area = scan_area(frame_width, frame_height, config)
    area_w, area_h = area[2] - area[0], area[3] - area[1]
    target = ((area[0] + area[2]) / 2, (area[1] + area[3]) / 2)

    sides = marker_sides(marker)
    marker_size = (sum(sides) / 4) / min(frame_width, frame_height)
    squareness = min(sides) / max(sides)
    (x0, y0), (x1, y1) = marker.corners_px[0], marker.corners_px[1]
    roll_deg = math.degrees(math.atan2(y1 - y0, x1 - x0))  # 0 = the marker's top edge is level

    product_center = product_fill = None
    complete = True
    if product_box is not None:
        product_center = ((product_box[0] + product_box[2]) / 2, (product_box[1] + product_box[3]) / 2)
        product_fill = max((product_box[2] - product_box[0]) / area_w, (product_box[3] - product_box[1]) / area_h)
        edge_x, edge_y = frame_width * config.frame_edge_margin, frame_height * config.frame_edge_margin
        complete = (
            product_box[0] > edge_x
            and product_box[1] > edge_y
            and product_box[2] < frame_width - edge_x
            and product_box[3] < frame_height - edge_y
        )

    # The physical product is what should sit in the scan box; the marker stands in until it is found.
    reference = product_center or marker.center_px
    offset_x = (reference[0] - target[0]) / area_w
    offset_y = (reference[1] - target[1]) / area_h

    if marker_size > config.max_marker_size or (product_fill is not None and product_fill > config.max_product_fill):
        distance = "too_close"
    elif marker_size < config.min_marker_size and (product_fill is None or product_fill < config.marker_size_waived_fill):
        distance = "too_far"
    else:
        distance = "ok"

    inside_area = all(_inside(area, x, y) for x, y in marker.corners_px) and (
        product_box is None
        or (_inside(area, product_box[0], product_box[1]) and _inside(area, product_box[2], product_box[3]))
    )

    hints: list[str] = []
    if not complete:
        # A cut-off box has a meaningless center and size, so nothing else is said about it.
        hints.append("show_complete_product")
    else:
        if distance != "ok":
            hints.append("move_farther" if distance == "too_close" else "move_closer")
        if abs(offset_x) > config.center_tolerance:
            # Right of the target in the image -> must travel left in the image;
            # on a mirrored preview that is the viewer's right.
            image_direction = "left" if offset_x > 0 else "right"
            if mirrored:
                image_direction = "right" if image_direction == "left" else "left"
            hints.append(f"move_{image_direction}")
        if abs(offset_y) > config.center_tolerance:
            hints.append("move_up" if offset_y > 0 else "move_down")
        if not hints and not inside_area:
            hints.append("center")
    if squareness < config.min_squareness or abs(roll_deg) > config.max_roll_deg:
        hints.append("straighten")

    return Positioning(
        scan_area=area,
        target_center=target,
        marker_center=marker.center_px,
        product_center=product_center,
        offset_x=offset_x,
        offset_y=offset_y,
        marker_size=marker_size,
        product_fill=product_fill,
        distance=distance,
        squareness=squareness,
        roll_deg=roll_deg,
        inside_area=inside_area,
        complete=complete,
        hints=tuple(hints),
    )
