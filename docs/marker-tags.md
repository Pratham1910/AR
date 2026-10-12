# ArUco product tags (four corners + YOLO verification)

A product is identified by **four printed ArUco tags, one per corner**, and
the identification is accepted only if the object detector also sees that
product's own class at the tags.

```
camera frame
   │  ArUco detector (the one /api/vision/pose uses)
   ▼
tag ids + pixel corners ──► marker_tags table: id → (product, corner)
   │
   ▼  identification.py   one product, all four tags, corners in registered order, plausible layout
candidate product + region (outline around the four tags)
   │  everything outside the region is blanked; YOLO (yolov8n-seg, the AR
   │  object finder) runs on that crop only — and only at this point
   ▼  verification.py     product's detection class found inside the region
verified this frame ──► same result for 3 consecutive frames ──► CONFIRMED
   │
   ▼
Asset → newest Model3D (+ its Vishwa procedure) and the asset's inspection procedures
```

## Using it

### 1. Give the product a detection class

The class YOLO must see is the product's **detection class**, set when a 3D
model is uploaded or later in *3D & AR workspace → Model settings*. Without
one, the tags are recognized but the product is never confirmed
(`CANNOT_VERIFY`). It must be a class the detector knows
(`GET /api/vision/classes`; the stock model has the 80 COCO classes).

### 2. Generate and map tags — navbar → **ArUco tags**

1. Pick the product.
2. **Suggest free ids** fills in four unused ids (or type your own).
3. **Save mapping**. Clashes are refused: an id already on another product,
   the same id on two corners, an id outside the dictionary.
4. Download each tag, or **all four as one sheet**, as PNG.
5. Print at 100% with the white margin intact. Fix each tag to the corner
   named under it, upright as printed, all on one face so the camera can see
   the four together. They can be on the product's corners or on a mat or
   fixture around it.

The table below the form lists every tagged product with its ids, YOLO class
and model; *Edit* loads a set back into the form, *Remove* deletes it.

### 3. Use the tags — **3D & AR workspace → AR tracking → ArUco marker**

Open any asset that has a 3D model to get to the camera and choose the
**ArUco marker** tab. It has two separate modes; starting one stops the other.

| | **Start product scan** | **Start live tracking** |
|---|---|---|
| Purpose | Open a product's 3D model or procedure | Track the physical product and overlay its model |
| Needs | **Any one** registered tag | **All four** tags of one product, on their corners |
| YOLO | Not used | Must find the product's class **inside** the four tags |
| Backend | `POST /api/marker-tags/identify` | `POST /api/marker-tags/scan`, then `POST /api/vision/pose` |
| Proves | Which product the tag belongs to | That the product is physically there |

**Product scan.** Show one tag. After two agreeing frames the product's name
appears with **View 3D model** and **View procedure**. Both open **in place,
directly below the scan panel** — nothing navigates, and the scanner, its
controls and the identified product stay on screen:

- *View 3D model* shows the product's newest model in the application's own
  viewer (`ThreeViewer`: orbit, zoom, pan, click a part).
- *View procedure* shows the procedure attached to that model in the
  application's own player (`ProcedurePanel`), together with the model it
  animates. A product's inspection procedures are listed there too; running
  one needs its own camera view, so that opens the Procedure inspection tab,
  only when its button is pressed.
- The two can be switched freely (the model is not reloaded), and *Hide*
  closes the view. A product with no model or no procedure gets a message
  saying so.

The inline view (`ProductInlineView.tsx`) has its own `ProcedureHost`, so
playing a step there never moves the model in the AR view above. An unmapped
tag selects nothing; tags of two products at once are reported as ambiguous
rather than picked between. This is a lookup only — it says nothing about
what is in front of the camera.

**Live tracking.** Every frame goes through the strict check below. Only
while it is `CONFIRMED` is the 3D model placed, using the tab's existing
marker pose and *Model alignment* on the product's top-left tag; anything
less hides the model and shows why. If the confirmed product is not the one
whose model is loaded, its model is loaded and tracking resumes (and
re-confirms) automatically. The video shows each tag (green: the candidate
product's, amber: another product's, red: unmapped; dashed: held from a
moment ago), the area the four tags enclose, and what YOLO found. While YOLO
is running, everything outside the tags is dimmed and the enclosed area is
labelled *YOLO search area* — that is literally all the detector is given.

The two modes share the camera feed and the tag mapping, and nothing else:
separate endpoints, separate sessions and thresholds
(`MARKER_TAG_IDENTIFY_FRAMES` vs `MARKER_TAG_CONFIRM_FRAMES`). Loosening the
product scan cannot loosen tracking; `test_what_satisfies_the_lookup_never_satisfies_tracking`
pins that.

Live tracking states:

| State | Meaning |
|---|---|
| `NO_MARKERS` / `UNKNOWN_MARKERS` | Nothing in view / tags not assigned to any product |
| `INCOMPLETE` | Some of one product's tags; the missing corners are named |
| `MIXED_PRODUCTS` | Tags of different products with no complete set, or a foreign tag inside a set |
| `MULTIPLE_PRODUCTS` | Two or more complete sets in view — show one at a time |
| `DUPLICATE_MARKERS` | The same tag id twice in the frame |
| `LAYOUT_INVALID` | All four tags, but not on their registered corners / implausible geometry |
| `CANNOT_VERIFY` | Product has no detection class, or one the detector doesn't know |
| `WRONG_OBJECT` | A different class is inside the tags, the expected one is not |
| `OBJECT_NOT_FOUND` | Nothing detected inside the tags (an object elsewhere in view is not looked at) |
| `VERIFYING` | Tags and YOLO agree; counting consecutive frames |
| `CONFIRMED` | Product identified |

Only `CONFIRMED` places the model. Product scan states: `NO_MARKERS`,
`UNKNOWN_MARKERS`, `AMBIGUOUS`, `IDENTIFYING`, `IDENTIFIED`.

## How it decides

- **Identity** needs all four tags of one product. Tags of two products are
  never combined; a complete set is still accepted with another product's tag
  elsewhere in view, but not with one inside its four corners.
- **Corner arrangement**: walking top-left → top-right → bottom-right →
  bottom-left must turn clockwise at every corner. That is true at any
  rotation or tilt and false if two tags are swapped or the set is mirrored.
- **Spatial consistency** (ratios, so distance doesn't matter): tag sizes
  within 2.5× of each other, opposite edges within 2.5×, shortest edge at
  least 1.5 tag sides.
- **Detection region**: the convex outline around all 16 corners of the
  four tags (`identification.enclosed_region`) — from the tags' outer edges
  inward, following the product's rotation and tilt.
- **YOLO sees only that region** (`verification.crop_to_region`): the frame
  is cropped to the region's bounding rectangle and every pixel outside the
  region itself is set to neutral gray before the model runs. Detections are
  then mapped back to frame coordinates. An object anywhere else in the
  camera view is never detected, so it can neither verify nor contradict.
- **Inside test**: independently of the crop, at least 60% of a detected
  object's box must lie inside the region (`MARKER_TAG_MIN_INSIDE`). A large
  object that merely contains the tagged area does not count.
- **Robustness**: detection runs on the frame and on a contrast-equalized
  copy (shadow, uneven light); a tag lost briefly is held for 600 ms and
  flagged; confirmation needs 3 consecutive verified frames.

Settings (`app/core/config.py`): `MARKER_TAG_HOLD_MS`,
`MARKER_TAG_CONFIRM_FRAMES`, `MARKER_TAG_MAX_SIZE_RATIO`,
`MARKER_TAG_MAX_OPPOSITE_SIDE_RATIO`, `MARKER_TAG_MIN_SIDE_IN_TAGS`,
`MARKER_TAG_MIN_INSIDE`. Tags use `ARUCO_DICTIONARY`.

## Code

| Piece | Where |
|---|---|
| Table `marker_tags` (id → product corner; unique per dictionary+id and per product+corner) | `app/models/marker_tag.py`, migration `0005` |
| Tag detection (two-pass) and printable PNGs | `app/services/markers/tags.py` |
| Product identification and layout check | `app/services/markers/identification.py` |
| YOLO verification | `app/services/markers/verification.py` |
| Held tags and confirmation streak | `app/services/markers/scan_session.py` |
| Mapping rules | `app/services/markers/registry.py` |
| API | `app/api/marker_tags.py`, schemas `app/schemas/marker_tag.py` |
| Product lookup by one tag | `lookup_product_by_tag` in `app/services/markers/identification.py` |
| Management page | `frontend/src/features/markers/MarkerTagsPage.tsx` |
| Marker tab: coordinator, product scan, inline model/procedure view, live tracking, drawing | `MarkerTab.tsx`, `ProductScanPanel.tsx`, `ProductInlineView.tsx`, `TagTrackingPanel.tsx`, `scanOverlay.ts` in the same folder |
| Tests | `tests/backend/test_marker_tags.py` |

### API (`/api/marker-tags`)

| Method | Path | |
|---|---|---|
| GET | `/config` | Dictionary, size, used ids |
| GET | `/suggest?asset_id=` | Four free ids (409 if fewer than four are left) |
| GET | `/sets` | All tagged products |
| PUT | `/sets/{asset_id}` | Assign or replace a product's four tags (400 invalid, 409 clash) |
| DELETE | `/sets/{asset_id}` | Remove a product's tags |
| GET | `/markers/{id}.png?size_px=` | One tag |
| GET | `/sets/{asset_id}/sheet.png?size_px=` | The four tags on one page |
| POST | `/identify` | Product scan, one frame: any one registered tag → product |
| DELETE | `/identify/{session_id}` | Forget a product scan's streak |
| POST | `/scan` | Live tracking, one frame: `{image_base64, session_id}` → state, tags, region, verification, product |
| DELETE | `/scan/{session_id}` | Forget a tracking session's held tags and streak |

Changes to existing code are additive: `ArucoPoseEstimator.detect_markers()`
and two helpers in `app/api/vision.py` expose the existing detector and
segmenter; `RegistrationOverlay` gained three optional props (`initialMode`,
`initialLiveTracking`, `markerPanel`); with `markerPanel` given, the marker
tab's "Start live tracking" is run by that panel instead of the built-in
single-marker loop ("Detect & align" still uses the built-in one). No
existing endpoint or response changed.

## Limitations

- **Identification is 2D.** It works in image pixels and needs no camera
  calibration and no per-product calibration. It does not measure distance.
- **3D pose** in the marker tab is still the existing single-marker solve
  (`/api/vision/pose`), now pointed at the product's top-left tag. Its
  accuracy depends on `ARUCO_MARKER_LENGTH_M` matching the printed tag size
  and on a real camera calibration (`CAMERA_CALIBRATION_PATH`); without one
  the response is flagged `calibration_is_approximate`. The model sits on
  that tag until aligned once with the tab's *Model alignment* controls. A
  pose from all four tags would need their physical spacing per product,
  which is not stored.
- **The product must sit inside its tags.** Because the detector only sees
  the enclosed area, put the tags at or beyond the product's extremities (on
  its outer corners, or on a mat/fixture around it). A product that extends
  well past its tags is cut off in what YOLO sees and may not be recognized.
  The region is 2D: anything visible *through* it in the image — an object
  behind or in front of the product — is inside it as far as the camera can
  tell.
- **YOLO classes**: the stock model knows COCO classes (bottle, cup, …), not
  industrial parts. Such a product needs custom weights
  (`SEGMENTATION_MODEL_NAME`) with a class for it. Several products sharing
  one class are told apart by their tags only.
- **Occlusion**: a tag must be fully visible to be read. A missing tag is
  bridged for 600 ms, never guessed; three tags do not identify a product.
- **Capacity**: `DICT_4X4_50` has 50 ids = 12 products. A larger dictionary
  (`DICT_6X6_250`) gives 62; mappings are per dictionary, so switching means
  re-mapping and re-printing.
- **Viewing side**: seen mirrored or from behind (tags on glass), the
  layout check rejects the set.
- **Access**: the mapping endpoints are unauthenticated, like the asset and
  model endpoints; changes are written to the audit log without a user.
- **Scan state is in server memory** (as AR sessions are): single worker, and
  a reload restarts any confirmation in progress.
- To reach the scan you must first open an asset that has a 3D model, since
  the camera view lives inside the AR tracking page.
- Verified with synthetic frames and a real photo through the real detector;
  **not yet tried with printed tags and a physical camera** (this machine has
  no camera device).
