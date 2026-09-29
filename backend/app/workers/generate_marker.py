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

from app.core.config import get_settings

_DICTIONARY_NAMES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "images"


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, default=0, help="Marker id to encode")
    parser.add_argument("--size-px", type=int, default=600, help="Output image size in pixels (square)")
    parser.add_argument("--dictionary", default=settings.aruco_dictionary, choices=list(_DICTIONARY_NAMES))
    args = parser.parse_args()

    dictionary = cv2.aruco.getPredefinedDictionary(_DICTIONARY_NAMES[args.dictionary])
    marker_image = cv2.aruco.generateImageMarker(dictionary, args.id, args.size_px)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"aruco_{args.dictionary}_id{args.id}.png"
    cv2.imwrite(str(output_path), marker_image)
    print(f"Wrote {output_path}")
    print(
        f"Print at a known size and measure the printed marker's side length "
        f"in meters, then set ARUCO_MARKER_LENGTH_M in .env to match "
        f"(currently {settings.aruco_marker_length_m} m)."
    )


if __name__ == "__main__":
    main()
