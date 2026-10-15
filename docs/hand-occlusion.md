# Hands in view (tracking under occlusion)

A hand holding or passing the object must never move its pose, and a hand
over a component must never make it "missing". Every frame carries a hand
mask; tracking and inspection leave those pixels out.

## The hand mask (`backend/app/services/vision/hand_mask.py`)

Union of two sources (`HAND_MASK_METHOD`, default `both`):

| Source | Strength | Weakness |
|---|---|---|
| MediaPipe Hands (landmarks → palm + finger mask) | exact hand shape, never spills onto the object | can miss a hand gripping an object seen from behind |
| COCO YOLO "person" masks (existing model, own instance) | catches arms and gripping hands | spilled onto half the held bottle in a real frame |

The union errs towards "too much is hand", which every consumer handles
safely. Masks are computed on a worker thread at most every 100 ms
(`HAND_MASK_INTERVAL_MS`); a frame uses the newest mask up to 400 ms old,
grown for the hand's motion since.

## Tracking (`flow_tracker.py`, `trackers.py`, `ar_session.py`)

- The detected outline has hand pixels cut out before locking (`without_hand`):
  the hand is never part of the object's contour, its flow points or the
  silhouette used to pick the first pose.
- Optical flow never seeds on a hand and drops points that land on one before
  fitting the motion; hidden points count as occlusion, not failure.
- MegaPose corrections are skipped while a hand covers > 30 % of the object
  and get the frame with hand pixels painted grey otherwise.
- Confidence = geometric overlap × optical-flow support (share of visible
  points agreeing), never the detector's score. `quality` in every response
  breaks it down.
- New state `TRACKING_OCCLUDED`: flow loses the object while a hand covers
  ≥ 20 % of it → the last pose is held (not filtered, not predicted) for up to
  1.5 s, and flow resumes from visible pixels as soon as there are enough. After
  that it is `LOST` and the detector re-acquires as before.
- `tracking_status`: TRACKING / TRACKING_OCCLUDED / REACQUIRING / TRACKING_LOST
  (plus SEARCHING / INITIALIZING). The frontend fades the overlay and stops
  pose prediction in any state but healthy TRACKING.

## Inspection

A component with > 15 % of its visible pixels under the hand is OCCLUDED for
that frame (`INSPECTION_MAX_HAND_COVER`), and evidence is only taken from
pixels that are neither hidden nor hand. Inspection also needs a healthy
TRACKING pose, so a held (occluded) pose never counts as an observation.

## Debug

AR page → "Show hands" draws the mask; the tracked outline is labelled with the
share covered; the HUD shows the hold time while occluded.

## Setup

MediaPipe is optional (without it: "person" masks only). Install without its
declared dependencies — they add a second OpenCV that clashes with ours:

    pip install --no-deps mediapipe==1.1.0 "absl-py~=2.3" "flatbuffers~=25.9" "sounddevice~=0.5"

and put `hand_landmarker.task` in `models/mediapipe/` (URL in `backend/requirements.txt`).
