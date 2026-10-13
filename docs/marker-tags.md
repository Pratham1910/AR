# ArUco product tags

A product carries **four printed ArUco tags, one on each corner, placed around
the physical object**. The tags are used in two separate ways:

| | **Identify by one tag** | **Four-tag scan** |
|---|---|---|
| Needs in view | Any **one** of the product's tags | All **four** tags, on their registered corners |
| Answers | Which product is this? | Which object is physically between the tags, and where? |
| Object detection | None | YOLO segmentation, restricted to the area the tags enclose |
| Endpoint | `POST /api/marker-tags/identify` | `POST /api/marker-tags/track` |
| Started by | *Identify by one tag* | **▶ Start scanning** |
| Result | Product → **View 3D model** / **View procedure** | Region, the object's mask and box, tracking state, and the product with the same actions |

They share the tag mapping and the camera and nothing else: separate
endpoints, sessions and thresholds.

```
camera frame (browser camera, AR tracking → ArUco marker)
   │  ArUco detector (the same one /api/vision/pose uses), plain + contrast-equalized pass
   ▼
tag ids + their 4 pixel corners ──► marker_tags table: id → (product, corner)
   │
   ├─ one mapped tag ──────────────────────────────► product (scan)
   │
   └─ all four of one product, clockwise TL→TR→BR→BL, enclosing a real area
        │  region = quadrilateral through each tag's outermost corner
        ▼
      YOLO segmentation (the AR modes' yolov8n-seg) on region + context band;
      everything farther away is blanked to gray before the model runs
        │  each mask measured against the quadrilateral: ≥ 80 % inside to count
        ▼
      object inside the tags ──► 3 agreeing frames ──► TRACKING
        (a covered tag is re-projected from the visible ones for 1.5 s;
         a missed detection keeps the last object for 1.5 s)
```

## Using it

### 1. Map and print the tags — navbar → **ArUco tags**

1. Pick the product (an existing asset; upload a 3D model or create a
   procedure for it elsewhere as usual).
2. **Suggest free ids** fills four unused ids; or type your own. Each corner
   shows a preview of its tag.
3. **Save mapping**. Refused with a message: an id already on another
   product, the same id on two corners, an id outside the dictionary.
4. **Sheet** shows the four tags in their corner positions: **Download sheet**,
   **Print sheet**, or download each tag. *Single tag* prints any id.
5. Print at 100 %. Cut each tag out with its white margin (the detector needs
   it), keep it upright as printed, and fix it to the corner named under it,
   **around** the object, all four facing the camera.

The table lists every tagged product with its ids, the object class tracking
will look for, and what it can open; **Edit** loads a set into the form,
**Delete** removes it. Mappings are rows in `marker_tags`, so they survive
restarts; every change is written to the audit log.

### 2. The object class (optional)

Four-tag tracking looks for the product's **detection class** — set when its
3D model is uploaded or in *3D & AR workspace → Model settings*. With one set
(e.g. `bottle`), only that class between the tags counts; any other object
there is reported as `WRONG_OBJECT`. Without one, the most prominent object
between the tags is taken and its class reported.

### 3. Use the tags — **3D & AR workspace → AR tracking → ArUco marker**

Open any asset that has a 3D model (the camera view lives in the AR tracking
page) and choose **ArUco marker**:

- **▶ Start scanning** (the main button under the video) runs the four-tag
  scan. One click is all it takes: every frame the tags are read, their
  arrangement is checked, and as soon as it is valid the object detector runs
  on the region between them; after three agreeing frames the state becomes
  *Tracking* by itself. The panel under the video shows each step — the four
  corners with their registered id, ✓ detected / ~ covered / ✕ missing and
  pixel position; whether the arrangement is valid (and why not); what the
  detector found inside; and the tracking state — and a pill on the video
  shows the current state.
- **Identify by one tag** (in that panel) is the separate lookup: any one
  registered tag names its product.
- **Single-marker pose** and **Detect & align** are the 6DoF pose of one
  marker (`/api/vision/pose`, first marker only). They are *not* the product
  scan; before this change the pose loop was the mode's only "Start" button,
  which is why a scan could run for minutes without ever looking for four
  tags.

On the video: the boundary through the four tags (cyan when it is the
detection region, red dashed when the arrangement is wrong), each tag
(green: this product's; amber: another product's; red: unmapped; dashed:
covered, position estimated), the object's segmentation mask and box with the
share of it inside the region, and objects seen but rejected (red dashed,
"outside"). Everything the detector was not given is dimmed.

**Debugging without being able to see the screen:** the panel's *Debug*
section lists every tag read in the last frame (id, product, corner, centre in
pixels) and the frame size and timings, and the server logs one line per
state change:

```bash
docker compose logs backend | grep four-tag
# four-tag tags-…: INCOMPLETE | tags read [0, 1, 3] | BOTTLE-001: 3 of 4 tags in view — missing bottom-right.
```

The product card offers **View 3D model** (the product's newest model, in the
application's `ThreeViewer`) and **View procedure** (the procedure attached to
its model, in the application's `ProcedurePanel` with the model it animates),
both opened below the scanner without leaving the page. The product's
inspection procedures are listed there with **Run inspection**, which opens the
Procedure inspection tab with that procedure selected (it needs its own camera
view). An action the product cannot offer is disabled and the card says why.

Only one of scanning, identifying and the single-marker pose runs at a time;
starting one stops the others.

### Tracking states

| State | Meaning |
|---|---|
| `NO_MARKERS` / `UNKNOWN_MARKERS` | No tags / tags not mapped to any product |
| `DUPLICATE_MARKERS` | A mapped id appears twice in the frame |
| `INCOMPLETE` | Some of one product's tags; the missing corners are named |
| `MIXED_PRODUCTS` | Tags of several products without a complete set, or another product's tag inside the set |
| `MULTIPLE_PRODUCTS` | Two complete sets in view |
| `LAYOUT_INVALID` | Four tags, but not on their registered corners, or bunched together with no room for an object |
| `OBJECT_NOT_FOUND` | Valid region, nothing detected inside it |
| `WRONG_OBJECT` | Something other than the product's detection class is inside |
| `VERIFYING` | Object found inside; counting consecutive frames |
| `TRACKING` | Tags and object agree; product card shown |

Product scan states: `NO_MARKERS`, `UNKNOWN_MARKERS`, `AMBIGUOUS` (tags of two
products), `IDENTIFYING`, `IDENTIFIED`.

## How the region and the object are found

- **Region.** The four tags' centres must run clockwise top-left → top-right →
  bottom-right → bottom-left (true at any rotation or perspective, false if two
  tags are swapped or the set is mirrored), and enclose at least six tag
  areas. The region's vertices are each tag's **outermost detected corner**,
  so it is a true quadrilateral that follows perspective — not the bounding
  box of the tags — and the tags lie inside it
  (`services/markers/geometry.py`).
- **What YOLO sees.** The region scaled up by 1.5 about its centre (a band
  of a quarter of its size on each side), cropped to its bounding rectangle,
  with everything outside that blanked to the gray YOLO pads with
  (`region_detection.masked_crop`). Objects farther away are never seen.
  The band is there so an object that only *reaches into* the region is seen
  whole and can be rejected; blanking exactly at the edge would cut it down to
  the sliver inside, which would then look entirely inside.
- **Which object counts.** Each detection's segmentation mask is rasterized
  against the quadrilateral; it counts only if at least 80 % of it is inside
  (`MARKER_TAG_MIN_MASK_INSIDE`). Everything else is returned in `rejected`.
  Coordinates are mapped back to the full frame by the crop's offset — a pure
  translation, so masks line up with the video exactly.
- **No perspective rectification.** Warping the region to a rectangle would
  distort the object into a shape the COCO-trained model has not seen; the
  mask-against-quadrilateral test already handles perspective.
- **Continuity** (`services/markers/tracking.py`). Tracking starts only after
  3 consecutive frames in which the four tags and the object agree, and only
  from four tags actually read. While tracking, a covered tag is re-projected:
  the homography from the visible tags' corners at the last full read to their
  corners now is applied to the covered tag's last corners. That needs at
  least 2 visible tags and lasts at most 1.5 s after the last full read. A
  frame in which YOLO misses the object keeps the last object for 1.5 s,
  flagged `held`.

Settings (`app/core/config.py`): `MARKER_TAG_IDENTIFY_FRAMES` (2),
`MARKER_TAG_CONFIRM_FRAMES` (3), `MARKER_TAG_HOLD_MS` (1500),
`MARKER_TAG_MIN_VISIBLE_FOR_HOLD` (2), `MARKER_TAG_MIN_MASK_INSIDE` (0.8),
`MARKER_TAG_CONTEXT` (0.5), `MARKER_TAG_MIN_REGION_IN_TAGS` (6). Tags use
`ARUCO_DICTIONARY` (default `DICT_4X4_50`).

## Code

| Piece | Where |
|---|---|
| Table `marker_tags` (unique per dictionary + id, and per product + corner) | `app/models/marker_tag.py`, migration `0005` |
| `MarkerCorner` | `app/models/enums.py` |
| Reading tags (two passes, duplicates) and printable PNGs | `app/services/markers/tags.py` |
| Region geometry, inside tests | `app/services/markers/geometry.py` |
| One-tag lookup, four-tag set validation | `app/services/markers/identification.py` |
| Object inside the region | `app/services/markers/region_detection.py` |
| Confirmation, covered tags, held object | `app/services/markers/tracking.py` |
| Mapping rules | `app/services/markers/mapping.py` |
| API + schemas | `app/api/marker_tags.py`, `app/schemas/marker_tag.py` |
| Detector / segmenter access | `ArucoPoseEstimator.detect_markers`, `detect_aruco_markers` and `segment_objects` in `app/api/vision.py` |
| Management page | `frontend/src/features/markers/MarkerTagsPage.tsx` |
| Scanner, product card, overlay | `MarkerScanner.tsx`, `ProductPanel.tsx`, `overlay.ts` (same folder), plugged into `RegistrationOverlay` via its `markerPanel` prop |
| Tests | `tests/backend/test_marker_tags.py` |

### API (`/api/marker-tags`)

| Method | Path | |
|---|---|---|
| GET | `/config` | Dictionary, its size, supported dictionaries, used ids, thresholds |
| GET | `/suggest?asset_id=` | Four free ids (409 if fewer are left) |
| GET | `/sets` | All tagged products |
| PUT | `/sets/{asset_id}` | Assign or replace a product's four tags (400 invalid, 409 clash) |
| DELETE | `/sets/{asset_id}` | Remove a product's tags |
| GET | `/markers/{id}.png?size_px=&download=` | One tag (inline, or as a download) |
| GET | `/sets/{asset_id}/sheet.png?size_px=&download=` | The product's four tags on one page |
| POST | `/identify` | `{session_id, image_base64}` → product scan state and product |
| DELETE | `/identify/{session_id}` | End a scan session |
| POST | `/track` | `{session_id, image_base64}` → tags, region, search area, object, rejected objects, state, product |
| DELETE | `/track/{session_id}` | End a tracking session |

## Limitations

- **No 6DoF object pose.** Four-tag tracking identifies the product and
  locates and segments the object in the image (2D). It does not estimate the
  object's 3D position or orientation, and nothing here places the 3D model
  on the object. For a pose use *Detect & align* (one marker), Model-based
  (CAD) or Markerless tracking.
- **The object must sit between its tags.** Put them at or beyond the
  object's outer corners. A part of the object outside the region counts
  against it (80 % must be inside), so tags placed too tightly can reject the
  real object. The region is 2D: anything visible *through* it — in front of
  or behind the product — is inside it as far as the camera can tell.
- **Occlusion.** A tag must be fully visible to be read. While tracking, up
  to two covered tags are bridged for 1.5 s from the others; tracking never
  starts from fewer than four read tags. The re-projection assumes the tags
  move together (a rigid product); it drifts if the product bends or the
  camera moves a lot while a tag is covered.
- **Perspective.** Steep viewing angles shrink the tags until the detector
  cannot read them; the region itself handles perspective. Seen mirrored or
  from behind (tags on glass), the corner order check rejects the set.
- **Segmentation.** The stock `yolov8n-seg.pt` knows the 80 COCO classes
  (bottle, cup, person, …), not industrial parts; such products need custom
  weights (`SEGMENTATION_MODEL_NAME`). An object YOLO cannot segment is
  `OBJECT_NOT_FOUND` even if it is there. Masks are approximate at the edges.
- **Capacity.** `DICT_4X4_50` has 50 ids = 12 products. Mappings are per
  dictionary: switching `ARUCO_DICTIONARY` means re-mapping and re-printing.
- **Sessions are in server memory** like the other AR sessions: single uvicorn
  worker, and a reload restarts any confirmation in progress.
- **Access**: the mapping endpoints are unauthenticated, like the asset and
  model endpoints beside them; changes are audited without a user.
- **Verified with**: unit tests on synthetic frames read by the real ArUco
  detector; API tests against PostgreSQL; the real YOLO model on a photo with
  four tags around one of several people; and the full UI in Chromium with a
  fake camera playing that photo. **Not yet verified with printed tags in
  front of a physical camera** — this machine has no camera.
