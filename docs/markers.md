# Marker → product scanning

The **Marker Scan** tab identifies products from printed fiducial markers in
the live webcam feed and leads to each product's existing 3D model and
procedure.

```
webcam frame ─→ MarkerDetector ─→ marker ids + pixel corners
                                        │
                     marker_bindings (id → Asset)
                                        │
                 Asset ─→ Model3D (3D viewer / AR Registration)
                       └→ Procedure → published revision → Steps
```

## Layers

- **Detection** — `app/services/markers/`. `MarkerDetector` is the interface
  (`detect(frame) -> [MarkerDetection(marker_id, corners_px)]`);
  `ArucoDetector` implements it on `cv2.aruco`. It knows nothing about
  products or pose. `ArucoPoseEstimator` (`/api/vision/pose`) uses the same
  detector for its own detection step.
- **Presence** — `presence.py` keeps a marker reported (`visible: false`, last
  known corners) for `MARKER_HOLD_MS` after it leaves the frame, per
  `session_id`.
- **Mapping** — the `marker_bindings` table (`app/models/marker.py`):
  `(family, dictionary, marker_id) → asset`. A marker id with no row is
  reported with `status: "unknown"`.
- **API** — `app/api/markers.py`:
  `POST /api/markers/detect`, `GET|PUT /api/markers/bindings`,
  `DELETE /api/markers/bindings/{id}`, `GET /api/markers/products/{asset_id}`,
  `GET /api/markers/config`.

## Configuration

| `.env` | Default | |
|---|---|---|
| `MARKER_FAMILY` | `aruco` | Which detector `factory.py` builds |
| `ARUCO_DICTIONARY` | `DICT_4X4_50` | Any `cv2.aruco` `DICT_*` name |
| `MARKER_HOLD_MS` | `1500` | How long a marker is held after leaving the frame |

Bindings are keyed by dictionary, so after changing `ARUCO_DICTIONARY` the
ids need binding again for the new dictionary.

## Adding markers and products

Open **Marker Scan → Marker → product mapping**, enter the id and pick the
asset, or `PUT /api/markers/bindings {"marker_id": 2, "asset_id": "<uuid>"}`.
No code changes. `python -m app.workers.seed_demo` binds id 0 → `PUMP-001`
and id 1 → `BOTTLE-001`. Print a marker with
`python -m app.workers.generate_marker --id 2`.

## Switching to AprilTag

Write `AprilTagDetector(MarkerDetector)` with `family = "apriltag"`, register
it in `app/services/markers/factory.py`, and set `MARKER_FAMILY=apriltag`. The
bindings table, API and UI are unchanged. (OpenCV's ArUco module can also
already read AprilTag patterns: `ARUCO_DICTIONARY=DICT_APRILTAG_36h11`.)

## What this does not do

Scanning reports marker ids and corners **in image pixels only** — no
distance or X/Y/Z. Placing the 3D model on the marker in metric space is the
AR Registration tab (`/api/vision/pose`), which without a real camera
calibration uses an assumed field of view and says so
(`calibration_is_approximate`); see `docs/pose.md`.

## Automatic scan (marker + product verification)

`POST /api/markers/scan`, called with every frame, runs:

```
ArucoDetector ─→ marker id + corners ─→ marker_bindings ─→ product + expected class
                                                                  │
YOLO (object detector) ─→ objects in the frame ─→ product_verifier ─→ match / mismatch / no product
                                                                  │
                                 scan_gate: placement, stillness, hold ─→ CONFIRMED
```

- **Identity** comes only from the marker. The expected detector class is the
  detection class of the product's newest 3D model.
- **Verification** (`app/services/scan/product_verifier.py`): the object whose
  box contains the marker (or is within one marker side of it) must be of the
  expected class. Another class there is a **mismatch** and the scan is
  rejected. A product whose class the model does not have is reported as
  *cannot verify* and never confirms.
- **Positioning** (`app/services/scan/positioning.py`): every frame, from the
  marker's corners and the product's box, measures the offset from the scan
  box's center, the marker's and product's apparent size, tilt, and whether
  everything is inside the box, and turns that into hints: `move_left`,
  `move_right`, `move_up`, `move_down`, `move_closer`, `move_farther`,
  `center`, `straighten`, `show_complete_product`. The product's box is what
  gets centered; the marker may sit anywhere on it. The numbers are returned
  in `scan.geometry`. All of it is relative to the picture — "too far" means
  "looks too small"; there are no centimeters or true angles.
- **Mirrored preview**: measurements are in camera-image coordinates. The
  request's `mirrored` flag says the user sees the frame flipped, and only
  swaps `move_left` / `move_right`, so a hint always names the direction the
  product should travel in the preview being looked at. The Marker Scan tab
  mirrors by default (toggle: "Mirror preview") and draws its overlay through
  the same mapping (`frontend/src/features/markers/scanOverlay.ts`).
- **Scan gate** (`app/services/scan/scan_gate.py`): one known marker, product
  verified, no positioning hints, moving slower than `SCAN_MAX_SPEED`, for
  `SCAN_HOLD_MS`. Any lapse restarts the hold. Once `CONFIRMED` it stays so,
  with the same `scan.confirmation` number, until that marker has been out of
  the scan box for `SCAN_REARM_MS`; then the next product can be scanned.

`scan.state` is one of `SEARCHING`, `MARKER_DETECTED`, `VERIFYING` (expected
product seen, below the required confidence), `MISMATCH`, `POSITIONING`,
`HOLD_STEADY`, `SCANNING`, `CONFIRMED`; `scan.stages` says how far the frame
got (marker, product, verified, position, steady), `scan.hints` what to
change, and `scan.message` is the guidance text. The UI shows these as NOT
READY / VERIFYING / MISMATCH / READY / HOLD STEADY / SCANNING / CONFIRMED.

The verification model is the stock COCO segmentation model unless
`PRODUCT_MODEL_PATH` points at custom YOLO weights. COCO knows `bottle`, `cup`
and similar, not industrial equipment: to scan such a product, train a model
with a class for it, set `PRODUCT_MODEL_PATH`, and give the product's 3D model
that detection class. Thresholds are the `scan_*` settings in
`app/core/config.py`.
