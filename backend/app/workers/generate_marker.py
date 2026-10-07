"""
Generates a printable ArUco marker image for the Phase 5 registration test
(Project.md #24). Print it at a known physical size, measure the printed
square's side length in meters, and set ARUCO_MARKER_LENGTH_M in .env to
match — pose accuracy depends entirely on that measurement being correct.

Run with:  python -m app.workers.generate_marker --id 0 --size-px 600
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from app.core.config import get_settings
from app.services.markers.aruco_detector import aruco_dictionary_names

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "images"


def render_marker(dictionary_name: str, marker_id: int, size_px: int) -> np.ndarray:
    """
    The marker (size_px square) surrounded by a white quiet zone two cells
    wide. The quiet zone is part of the marker: the detector finds a marker
    by its black border against a lighter surround, so the bare pattern shown
    on a dark screen, or cut out along its edge, is never detected.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    quiet_zone_px = 2 * size_px // (dictionary.markerSize + 2)  # pattern cells + its 1-cell black border
    return cv2.copyMakeBorder(marker, *[quiet_zone_px] * 4, cv2.BORDER_CONSTANT, value=255)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, default=0, help="Marker id to encode")
    parser.add_argument("--size-px", type=int, default=600, help="Side of the black marker square in pixels")
    parser.add_argument("--dictionary", default=settings.aruco_dictionary, choices=aruco_dictionary_names())
    args = parser.parse_args()

    marker_image = render_marker(args.dictionary, args.id, args.size_px)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"aruco_{args.dictionary}_id{args.id}.png"
    cv2.imwrite(str(output_path), marker_image)
    print(f"Wrote {output_path}")
    print(
        f"Keep the white border when printing or showing it on a screen. For AR "
        f"registration, measure the black square's side length in meters and set "
        f"ARUCO_MARKER_LENGTH_M in .env to match (currently {settings.aruco_marker_length_m} m)."
    )


if __name__ == "__main__":
    main()
