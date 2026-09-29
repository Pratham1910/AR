Absolutely. Below is the **complete master prompt** you can give to an AI coding agent such as Claude Code, Cursor, ZCode, Codex, or another coding agent to build the project.

I’ve incorporated the latest direction: **industrial AR/QA platform, multimodal 3D + image + video, State → Action → Validation → State, camera-first MVP, no Unity in Phase 1, and Unity/AR later as a client rather than the core system.**

# Master Implementation Prompt — Industrial AR/MR Procedure & QA Platform

```text
You are the lead software architect, computer-vision engineer, AI/ML engineer,
3D engineer, backend engineer, and technical-product engineer responsible for
building an industrial maintenance, procedure execution, inspection, training,
and quality-assurance platform.

The target product is conceptually similar to an industrial AR platform such as
Diota / DELMIA Augmented Experience, but the architecture must be our own and
must focus heavily on:

- executable maintenance procedures
- computer-vision verification
- 3D digital twins
- image and video references
- physical-vs-digital state comparison
- deterministic QA
- procedure traceability
- evidence collection
- S1000D integration
- connected tools
- offline-first operation
- future AR/MR execution

IMPORTANT:

DO NOT begin by building a Unity application.

The first working product must be a desktop/web camera-based multimodal
procedure and QA system.

Unity/AR will be added later as another runtime client.

============================================================
1. PRODUCT VISION
============================================================

Build a system where an industrial maintenance procedure is not merely a PDF,
manual, video, or static 3D animation.

Instead, a procedure becomes an executable digital workflow:

    EXPECTED STATE
          ↓
        ACTION
          ↓
    PHYSICAL OBSERVATION
          ↓
       VALIDATION
          ↓
    OBSERVED STATE
          ↓
       NEXT STEP

The platform must connect:

    Engineering data
          ↓
    CAD / 3D
          ↓
    Images
          ↓
    Videos
          ↓
    Technical publications / S1000D
          ↓
    Executable procedure
          ↓
    Physical equipment
          ↓
    Computer vision
          ↓
    QA validation
          ↓
    Evidence
          ↓
    Traceable inspection record

Long-term digital thread:

    CAD ↔ BOM ↔ S1000D ↔ Procedure ↔ Physical Asset ↔ QA Record


============================================================
2. CORE PRODUCT REQUIREMENT
============================================================

The platform must support THREE representations of the same physical asset:

1. 3D
2. Image
3. Video

These are not independent features.

They represent different views/evidence of the same Asset, Part,
Procedure, State, Action, or Requirement.

Example:

A PCB removal procedure may have:

    3D model:
        PCB.glb

    Images:
        PCB installed
        PCB removed
        connector close-up
        accepted inspection image

    Video:
        expert technician removing PCB

All of these must be linked to the same semantic entities.

The system must NOT assume that CAD/3D is always available.

If CAD is available:
    use it.

If CAD is unavailable:
    images and video can still drive the procedure.

If only images are available:
    the system must still support visual inspection workflows.

If only video is available:
    the system must be able to extract/candidate state transitions,
    with human approval before becoming an official procedure.


============================================================
3. MOST IMPORTANT DESIGN PRINCIPLE
============================================================

Separate these four concepts:

A. PERCEPTION
B. TRACKING
C. MEASUREMENT
D. QA DECISION

Do not combine them into one AI model.

Example:

YOLO answers:

    "What object is present?"

Tracking answers:

    "Where is that object over time?"

Pose estimation answers:

    "Where is the object in 3D and how is it oriented?"

Measurement answers:

    "What is the physical distance / gap / deviation?"

QA engine answers:

    "Does this satisfy the engineering requirement?"

These must remain separate architectural layers.


============================================================
4. TARGET USE CASE
============================================================

Initial demonstration assembly:

Use a manageable mechanical/electronic assembly such as:

    Water pump
    housing
    removable cover
    screws
    PCB
    connectors
    fan
    mechanical components

The exact object can change, but the architecture must remain generic.

Example procedure:

STEP 1
    Identify assembly

STEP 2
    Remove/open cover

STEP 3
    Verify screws removed

STEP 4
    Disconnect connector

STEP 5
    Remove PCB

STEP 6
    Inspect PCB

STEP 7
    Reinstall PCB

STEP 8
    Verify final state

Each step must contain:

    target
    action
    expected starting state
    expected ending state
    validation rules
    optional 3D animation
    reference images
    optional reference video
    evidence requirements


============================================================
5. PROCEDURE MODEL
============================================================

The central procedure representation must be:

    STATE → ACTION → VALIDATION → STATE

Example:

    COVER_CLOSED
          ↓
    REMOVE_SCREWS
          ↓
    VALIDATE_SCREWS_REMOVED
          ↓
    SCREWS_REMOVED
          ↓
    OPEN_COVER
          ↓
    VALIDATE_COVER_OPEN
          ↓
    COVER_OPEN


Do NOT model procedures simply as:

    Step 1
    Step 2
    Step 3

Instead, represent them as a state/action graph.

This allows:

- branching procedures
- optional steps
- failure states
- recovery states
- alternate procedures
- starting in the middle of a procedure
- resuming interrupted procedures
- conditional inspection
- troubleshooting
- future fault diagnosis


============================================================
6. PROCEDURE JSON MODEL
============================================================

Create a platform-independent procedure format.

Example:

{
  "procedureId": "RADAR-MAINT-001",
  "revision": "A",
  "title": "PCB Removal Procedure",

  "assetId": "RADAR-001",

  "states": [
    {
      "id": "STATE-COVER-CLOSED",
      "name": "Cover Closed"
    },
    {
      "id": "STATE-SCREWS-REMOVED",
      "name": "Screws Removed"
    },
    {
      "id": "STATE-COVER-OPEN",
      "name": "Cover Open"
    },
    {
      "id": "STATE-PCB-REMOVED",
      "name": "PCB Removed"
    }
  ],

  "steps": [

    {
      "id": "STEP-001",

      "title": "Remove Cover Screws",

      "startingState": "STATE-COVER-CLOSED",

      "action": {
        "type": "REMOVE"
      },

      "target": {
        "componentId": "SCREW-GROUP-001"
      },

      "expectedState": "STATE-SCREWS-REMOVED",

      "validation": {
        "methods": [
          "OBJECT_DETECTION",
          "STATE_CLASSIFICATION"
        ]
      }
    },

    {
      "id": "STEP-002",

      "title": "Open Cover",

      "startingState": "STATE-SCREWS-REMOVED",

      "action": {
        "type": "OPEN"
      },

      "target": {
        "componentId": "COVER-001"
      },

      "expectedState": "STATE-COVER-OPEN",

      "validation": {
        "methods": [
          "POSE",
          "STATE_CLASSIFICATION"
        ]
      }
    },

    {
      "id": "STEP-003",

      "title": "Remove PCB",

      "startingState": "STATE-COVER-OPEN",

      "action": {
        "type": "REMOVE"
      },

      "target": {
        "componentId": "PCB-001"
      },

      "expectedState": "STATE-PCB-REMOVED",

      "validation": {
        "methods": [
          "OBJECT_DETECTION",
          "TRACKING",
          "POSE",
          "STATE_CLASSIFICATION"
        ]
      }
    }

  ]
}


============================================================
7. DATA MODEL
============================================================

Design the database around these primary entities:

Asset
Part
Assembly
Component
BOMItem
Procedure
ProcedureRevision
Step
State
Action
Requirement
ValidationRule
Observation
Measurement
Evidence
InspectionRun
InspectionResult
User
Role
ReferenceImage
ReferenceVideo
Model3D
Animation
Tool
ToolMeasurement
Fault
S1000DReference


Relationships:

Asset
 ├── Assembly
 ├── BOM
 ├── Parts
 ├── Procedures
 ├── Images
 ├── Videos
 ├── 3D Models
 └── QA Records


Procedure
 ├── Revision
 ├── Steps
 ├── States
 ├── Requirements
 └── References


Step
 ├── Target Components
 ├── Action
 ├── Expected State
 ├── Validation Rules
 ├── Animation
 ├── Images
 ├── Videos
 └── Evidence Requirements


InspectionRun
 ├── Asset
 ├── Procedure
 ├── Procedure Revision
 ├── Operator
 ├── Step Results
 ├── Observations
 ├── Measurements
 ├── Evidence
 └── Final Result


============================================================
8. QA RESULT MODEL
============================================================

Never use only:

    PASS
    FAIL

The system must support:

    PASS
    FAIL
    UNCERTAIN
    NOT_EVALUATED
    MANUAL_REVIEW

Example:

{
  "stepId": "STEP-003",

  "expectedState": "PCB_REMOVED",

  "observedState": "PCB_REMOVED",

  "confidence": 0.96,

  "result": "PASS",

  "validation": {
    "objectDetected": true,
    "trackingStable": true,
    "poseValid": true,
    "stateMatched": true
  },

  "evidence": [
    "frame_00123.jpg",
    "frame_00124.jpg"
  ]
}


IMPORTANT:

AI confidence must NOT automatically equal QA approval.

The QA engine must apply deterministic rules.

For example:

    detection confidence > 0.80
    AND
    state confidence > 0.90
    AND
    pose error < tolerance
    AND
    required component present

→ PASS

Otherwise:

    FAIL

or

    MANUAL_REVIEW / UNCERTAIN

depending on the configured rule.


============================================================
9. TECHNOLOGY STACK
============================================================

PHASE 1 STACK:

Frontend:

    React
    TypeScript
    Vite
    Three.js
    React Three Fiber if useful

Backend:

    Python
    FastAPI
    Pydantic
    WebSocket

Computer Vision:

    OpenCV
    PyTorch
    Ultralytics YOLO
    segmentation
    tracking

Tracking:

    ByteTrack
    BoT-SORT

Pose:

    OpenCV
    solvePnP
    camera calibration
    Open3D

Video:

    OpenCV
    FFmpeg

Database:

    PostgreSQL

Object storage:

    MinIO
    or S3-compatible storage

3D:

    GLB
    glTF
    Three.js

Infrastructure:

    Docker
    Docker Compose

Future inference optimization:

    ONNX
    TensorRT

DO NOT introduce TensorRT during the first prototype unless required.


============================================================
10. WHY FASTAPI
============================================================

The computer vision and ML stack is primarily Python.

Therefore use:

    FastAPI

for the initial backend.

It should expose services such as:

    POST /vision/detect
    POST /vision/segment
    POST /vision/track
    POST /vision/pose
    POST /vision/state

    POST /procedure
    GET  /procedure/{id}

    POST /inspection/start
    POST /inspection/{id}/step
    POST /inspection/{id}/validate

    POST /evidence

    WS /inspection/{id}/stream

Node/Express can be introduced later for enterprise/product services if needed,
but do not create unnecessary microservices in the MVP.


============================================================
11. SYSTEM ARCHITECTURE
============================================================

Implement:

                         WEB CLIENT
                              │
                   React + TypeScript
                              │
             ┌────────────────┴────────────────┐
             │                                 │
       PROCEDURE UI                       QA UI
             │                                 │
             └────────────────┬────────────────┘
                              │
                           FastAPI
                              │
      ┌───────────────────────┼───────────────────────┐
      │                       │                       │
      ▼                       ▼                       ▼
  3D ENGINE              VISION ENGINE          VIDEO ENGINE
  Three.js               YOLO                    OpenCV
  GLB/glTF               Segmentation            FFmpeg
  Animation              Pose                    Tracking
  Transform              Detection               State extraction
      │                       │                       │
      └───────────────────────┼───────────────────────┘
                              │
                         QA ENGINE
                              │
                State comparison
                Rule evaluation
                Tolerance
                Evidence
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
            PostgreSQL                  MinIO
```

Keep the architecture modular.

============================================================
12. PROJECT STRUCTURE
=====================

Create:

industrial-qa-platform/

```
frontend/
    src/
        components/
        pages/
        features/
            procedures/
            inspection/
            assets/
            qa/
            viewer3d/
        services/
        types/
        hooks/
        stores/

backend/
    app/
        main.py

        api/
            procedures.py
            inspection.py
            assets.py
            vision.py
            evidence.py

        core/
            config.py
            database.py

        models/
            asset.py
            component.py
            procedure.py
            step.py
            state.py
            inspection.py
            evidence.py

        schemas/
            procedure.py
            inspection.py
            vision.py

        services/
            procedure_engine/
            qa_engine/
            vision/
            tracking/
            pose/
            state_detection/
            evidence/

        workers/

vision/
    detection/
    segmentation/
    tracking/
    pose/
    calibration/
    state/

video/
    extraction/
    preprocessing/
    event_detection/

threejs/
    loaders/
    animations/
    registration/

data/
    assets/
    images/
    videos/
    models/
    procedures/

models/
    yolo/

database/
    migrations/

docker/

tests/

docs/
```

Do not put all functionality into one giant Python file.

============================================================
13. PHASE 1 — CAMERA QA MVP
============================

This is the first actual development phase.

DO NOT build AR.

DO NOT build Unity.

DO NOT build a full LLM system.

DO NOT build S1000D integration yet.

DO NOT build complex enterprise infrastructure.

Goal:

Prove that a camera can observe a physical assembly and determine whether
a procedure state has been achieved.

Example:

Physical assembly:

```
COVER CLOSED
```

Camera observes:

```
COVER CLOSED
```

Expected:

```
COVER CLOSED
```

Result:

```
PASS
```

Then technician performs:

```
OPEN COVER
```

Camera observes:

```
COVER OPEN
```

Expected:

```
COVER OPEN
```

Result:

```
PASS
```

============================================================
14. COMPUTER VISION PIPELINE
============================

Implement:

Camera
↓
Frame acquisition
↓
Preprocessing
↓
Object detection
↓
Segmentation
↓
Object identification
↓
Tracking
↓
State estimation
↓
Optional pose estimation
↓
QA validation

Do not assume YOLO alone solves everything.

============================================================
15. YOLO
========

Use YOLO for:

* component detection
* object localization
* optional segmentation
* optional pose/keypoints

Example classes:

```
housing
cover
pcb
connector
screw
fan
cable
```

The model must be custom-trainable.

Do not hard-code the solution to one assembly.

Create configuration-driven class definitions.

============================================================
16. SEGMENTATION
================

Where bounding boxes are insufficient, use segmentation.

Example:

Bounding box:

```
PCB occupies rectangle.
```

Segmentation:

```
actual PCB pixels.
```

Segmentation is important for future:

* contour inspection
* edge detection
* surface comparison
* geometric validation
* object overlap
* occlusion reasoning

============================================================
17. TRACKING
============

Use:

```
ByteTrack
or
BoT-SORT
```

Pipeline:

Initial detection
↓
Object ID
↓
Tracking
↓
Frame-to-frame position
↓
Periodic detection
↓
Recovery when tracking is lost

Each physical component must receive a temporary tracking ID.

Example:

```
PCB-001
trackerId = 17
```

The tracker must tolerate:

* movement
* rotation
* temporary occlusion
* partial visibility
* lighting variation

============================================================
18. STATE DETECTION
===================

State detection is more important than generic action recognition for MVP.

Example:

```
COVER_CLOSED
COVER_OPEN
PCB_INSTALLED
PCB_REMOVED
CONNECTOR_CONNECTED
CONNECTOR_DISCONNECTED
SCREWS_PRESENT
SCREWS_REMOVED
```

Do not immediately build a huge action-recognition model.

Instead detect observable states.

Example:

Video:

```
Frame 1:
    Cover closed

Frame 50:
    Cover partially open

Frame 100:
    Cover open
```

System infers:

```
COVER_CLOSED
   →
COVER_OPEN
```

============================================================
19. ACTION VS STATE
===================

The system should distinguish:

ACTION:

```
user removes PCB
```

STATE:

```
PCB is removed
```

The final QA decision should generally be based on the resulting state.

For example:

Do not require the system to perfectly recognize:

```
"user performed the exact PCB removal gesture"
```

if it can reliably determine:

```
PCB was installed
   →
PCB is now removed
```

This makes the first system substantially more robust.

============================================================
20. 3D MODEL PIPELINE
=====================

3D is NOT the first dependency.

But the architecture must support it from the beginning.

Import:

```
GLB
glTF
```

Example hierarchy:

RADAR_ASSEMBLY
├── Housing
├── Upper_Cover
├── Screw_01
├── Screw_02
├── Screw_03
├── Screw_04
├── PCB
│   ├── Connector_J1
│   ├── Connector_J2
│   ├── IC_01
│   ├── Capacitor_01
│   └── Resistor_01
└── Fan

Every engineering object should ideally have:

```
componentId
partNumber
bomId
cadNodeId
s1000dReference
revision
instanceId
```

Do not rely only on the visible mesh name.

============================================================
21. 3D PROCEDURE ANIMATION
==========================

Each procedure step can contain an animation.

Example:

PCB installed:

```
[PCB inside housing]
```

Removal animation:

```
PCB moves upward
PCB moves outward
PCB becomes detached
```

The animation represents:

```
EXPECTED ACTION / EXPECTED TRANSITION
```

It does NOT mean the 3D model itself has proven that the physical
action happened.

The physical camera observation remains authoritative for physical QA.

============================================================
22. THREE.JS PROCEDURE VIEWER
=============================

Build a web-based 3D viewer.

Features:

* load GLB
* orbit
* zoom
* pan
* select component
* highlight component
* isolate component
* hide/show component
* play animation
* pause animation
* reset animation
* step forward
* step backward
* show procedure step
* show target component
* show expected state

Example UI:

---

## Procedure: PCB Removal

[3D VIEW]

```
   ┌──────────────────┐
   │                  │
   │   3D Assembly    │
   │                  │
   │      PCB         │
   │       ↑          │
   │       ↑          │
   └──────────────────┘
```

Step 3 / 7

Remove PCB

Target:
PCB-001

Expected state:
PCB_REMOVED

[ Play Animation ]

[ Previous ] [ Next ]

---

============================================================
23. PHYSICAL ↔ 3D REGISTRATION
===============================

Eventually the system must register the digital model against the physical
object.

Concept:

Physical camera
↓
Detect known component
↓
Estimate pose
↓
Calculate transform
↓
Apply transform to 3D model
↓
3D model overlays physical object

The transformation is:

```
Translation:
    X Y Z

Rotation:
    Roll Pitch Yaw
```

This is a 6DoF pose.

============================================================
24. INITIAL REGISTRATION METHOD
===============================

For the first prototype, it is acceptable to use:

```
AprilTag
or
ArUco
```

to simplify registration.

This is NOT the final product requirement.

Purpose:

```
prove the coordinate-system architecture.
```

Later replace/supplement markers with:

```
markerless pose estimation
feature matching
learned pose estimation
CAD-based registration
```

Do not build a complicated markerless system before proving the architecture.

============================================================
25. OPENCV POSE ESTIMATION
==========================

Use:

```
camera calibration
intrinsic matrix
distortion coefficients
solvePnP
```

where appropriate.

Concept:

```
known 3D points
      +
detected 2D points
      ↓
   solvePnP
      ↓
rotation + translation
```

Store calibration.

Do not repeatedly assume an arbitrary camera matrix.

============================================================
26. COORDINATE SYSTEM
=====================

Define explicit coordinate systems:

```
CAD coordinate system
Asset coordinate system
Camera coordinate system
World coordinate system
Tool coordinate system
```

Create transformation utilities.

Example:

```
T_world_asset
T_camera_asset
T_world_camera
```

Do not scatter raw XYZ transformations throughout the code.

Create a dedicated transformation module.

============================================================
27. IMAGE SUPPORT
=================

Images must be first-class entities.

Support:

* reference photographs
* inspection photographs
* annotated images
* drawings
* accepted images
* multiple viewpoints
* close-up component images
* defect images

Each image can be associated with:

```
asset
component
procedure
step
state
requirement
inspection run
```

Example:

PCB-001
├── front_reference.jpg
├── connector_reference.jpg
├── accepted_state.jpg
└── defect_example.jpg

============================================================
28. VIDEO SUPPORT
=================

Video must also be first-class.

Support:

* expert demonstration videos
* maintenance videos
* inspection videos
* recorded inspection sessions
* evidence videos

Pipeline:

Video
↓
Frame extraction
↓
Detection
↓
Tracking
↓
State estimation
↓
Temporal transition detection
↓
Candidate procedure

Example:

Video:

```
Cover closed
   ↓
screws removed
   ↓
cover opened
   ↓
PCB removed
```

System generates:

```
candidate procedure
```

But:

IMPORTANT:

Candidate procedures generated from video must require human review
before being published as production procedures.

============================================================
29. VIDEO-TO-PROCEDURE PIPELINE
===============================

Implement later:

```
Expert video
      ↓
Object detection
      ↓
Tracking
      ↓
State recognition
      ↓
Temporal segmentation
      ↓
Candidate steps
      ↓
Human author review
      ↓
Approved procedure
```

Example output:

{
"candidateSteps": [
{
"startState": "COVER_CLOSED",
"action": "REMOVE_SCREWS",
"endState": "SCREWS_REMOVED"
},
{
"startState": "SCREWS_REMOVED",
"action": "OPEN_COVER",
"endState": "COVER_OPEN"
}
]
}

============================================================
30. QA ENGINE
=============

Build a deterministic QA engine.

Input:

```
expected state
observations
measurements
confidence
tolerance
requirements
```

Output:

```
PASS
FAIL
UNCERTAIN
MANUAL_REVIEW
```

Example:

Requirement:

```
PCB must be completely removed.
```

Observation:

```
PCB detected = false
PCB installed = false
PCB removed = true
```

Result:

```
PASS
```

Another:

Requirement:

```
connector must be disconnected.
```

Observation:

```
connector connected = true
```

Result:

```
FAIL
```

============================================================
31. TOLERANCES
==============

Support tolerance rules.

Example:

```
required distance = 5.0 mm
tolerance = ±0.5 mm
```

Observed:

```
5.2 mm
```

Result:

```
PASS
```

Observed:

```
6.1 mm
```

Result:

```
FAIL
```

But do not claim millimeter accuracy from a normal RGB camera without
appropriate calibration and geometry/depth assumptions.

For physical measurement use:

```
calibrated camera
stereo
depth camera
structured light
3D reconstruction
known geometry
```

============================================================
32. GAP / CLEARANCE INSPECTION
==============================

The platform must eventually support QA requirements such as:

* gap between two parts
* distance between components
* alignment
* angle
* connector position
* screw presence
* screw orientation
* surface deviation
* missing components
* incorrect assembly
* wrong component
* incomplete insertion

Example:

Requirement:

```
Gap between Part A and Part B:
2.0 mm ± 0.2 mm
```

System:

```
detect Part A
detect Part B
estimate geometry
calculate distance
compare against tolerance
```

Do not use pixel distance as millimeters without calibration.

============================================================
33. INSPECTION EVIDENCE
=======================

Every QA decision must be traceable.

Store:

```
timestamp
asset
procedure
revision
step
operator
camera
model version
detection confidence
state
measurements
result
images
video references
manual override
comments
```

Example:

InspectionRun
|
+-- Step 001
|      |
|      +-- PASS
|      +-- image.jpg
|
+-- Step 002
|      |
|      +-- PASS
|
+-- Step 003
|
+-- FAIL
+-- evidence.jpg
+-- measurement.json

============================================================
34. MODEL VERSIONING
====================

QA results must record which AI model was used.

Example:

```
detector:
    yolo-pcb-v3

modelVersion:
    3.1.0

stateModel:
    pcb-state-v2

procedureRevision:
    A
```

This is essential for traceability.

============================================================
35. OFFLINE-FIRST
=================

The target environment may be:

```
industrial network
intranet
defense environment
disconnected/offline workstation
```

Therefore:

The core workflow must operate locally.

Do not require:

```
cloud AI
cloud database
internet
external API
```

for basic inspection.

Models should be locally deployable.

Storage should work locally.

Backend should work locally.

Inference should work locally.

============================================================
36. AI / LLM ROLE
=================

Do NOT make an LLM responsible for deterministic safety-critical QA.

LLM/VLM may later assist with:

* technical document understanding
* procedure extraction
* image explanation
* video summarization
* fault diagnosis assistance
* semantic search
* S1000D interpretation
* operator assistance

But:

```
approved technical data
      ↓
   retrieval
      ↓
   AI answer
      ↓
source citation
```

AI must not invent maintenance instructions.

For PASS/FAIL:

```
deterministic QA engine
```

must remain authoritative.

============================================================
37. S1000D INTEGRATION
======================

S1000D is a future integration layer.

Do not make the runtime depend directly on raw S1000D XML.

Instead:

```
S1000D
   ↓
semantic extraction
   ↓
normalized technical data
   ↓
executable procedure
   ↓
3D / image / video mapping
```

Map S1000D references to:

```
Asset
Component
Procedure
Step
Requirement
Illustration
Warning
Caution
Note
```

Example:

S1000D DMC
↓
Procedure
↓
Step
↓
Component
↓
3D node
↓
Physical component
↓
QA validation

============================================================
38. DIGITAL THREAD
==================

Maintain stable identifiers.

Example:

Part:

```
PCB-001
```

CAD:

```
CAD_NODE_123
```

BOM:

```
BOM-00452
```

S1000D:

```
DMC-XX-XX-XX-...
```

Procedure:

```
PROC-PCB-REMOVE
```

Physical asset:

```
ASSET-001
```

QA:

```
INSPECTION-2026-0001
```

The same component must remain traceable across all representations.

============================================================
39. AUTHORING SYSTEM
====================

Build a web-based Procedure Authoring Studio.

Layout:

---

PROCEDURE AUTHORING STUDIO

---

LEFT:
Procedure steps

```
Step 01
Step 02
Step 03
Step 04
Step 05
```

CENTER:
3D viewport / image / video viewer

RIGHT:
Properties

```
Step:
Target:
Action:
Expected State:
Validation:
Tolerance:
Evidence:
```

BOTTOM:
Timeline / animation

---

Buttons:

```
Save
Preview
Validate
Publish
```

============================================================
40. AUTHORING WORKFLOW
======================

Author:

1. Create Asset
2. Upload model/image/video
3. Define components
4. Create procedure
5. Create states
6. Create step
7. Select target component
8. Define action
9. Define expected state
10. Attach animation
11. Attach reference image/video
12. Define validation rule
13. Define tolerance if required
14. Preview
15. Validate procedure
16. Publish revision

============================================================
41. RUNTIME WORKFLOW
====================

Operator starts:

```
InspectionRun
```

System:

```
loads procedure
loads revision
loads required assets
```

Step:

```
show instruction

show 3D animation if available

show reference image if useful

show video if useful

activate camera

detect target

track target

observe state

compare expected vs observed

show result
```

If PASS:

```
allow next step
```

If FAIL:

```
show reason
capture evidence
optionally allow retry
```

If UNCERTAIN:

```
request additional observation
or manual review
```

============================================================
42. USER INTERFACE
==================

The operator should clearly see:

```
Current step
Expected state
Physical observation
Validation status
Confidence
Evidence
Next action
```

Example:

---

STEP 4 / 8

Disconnect Connector J2

EXPECTED:

```
Connector disconnected
```

OBSERVED:

```
Connector disconnected
```

VISION:

```
Detection: 97%
State: 94%
```

VALIDATION:

```
✓ PASS
```

[ Continue ]

---

============================================================
43. FAILURE UX
==============

Never simply display:

```
FAIL
```

Instead show:

```
FAIL
```

Reason:

```
Connector J2 still appears connected.
```

Evidence:

```
[captured image]
```

Suggested action:

```
Disconnect connector J2 and retry.
```

The explanation must come from the validation data.

Do not hallucinate failure reasons.

============================================================
44. MANUAL OVERRIDE
===================

Industrial workflows may require human review.

Support:

```
PASS
FAIL
MANUAL_REVIEW
```

A qualified user can override an uncertain result.

Store:

```
original AI result
human decision
user
timestamp
reason
```

Never overwrite the original AI observation.

============================================================
45. SECURITY / AUDIT
====================

Eventually support:

```
RBAC
```

Roles:

```
Admin
Engineer
Technical Author
QA Inspector
Maintainer
Operator
Reviewer
```

Audit:

```
who created procedure
who modified procedure
who published revision
who executed inspection
who changed result
who approved manual override
```

============================================================
46. TOOL INTEGRATION
====================

Later support connected tools.

Architecture:

Physical Tool
↓
Manufacturer Adapter
↓
Tool Gateway
↓
Normalized Measurement
↓
Procedure Engine
↓
QA Engine

Normalized measurement:

{
"toolId": "TORQUE-001",
"measurementType": "TORQUE",
"value": 41.3,
"unit": "Nm",
"timestamp": "...",
"status": "VALID"
}

Support future protocols such as:

```
USB
Serial
Bluetooth
CAN
Ethernet
TCP/IP
WebSocket
```

Do not implement every protocol in the MVP.

============================================================
47. AR/MR FUTURE
================

Unity is a FUTURE CLIENT.

Architecture:

```
             BACKEND
                │
    ┌───────────┴────────────┐
    │                        │
 Web Client              Unity Client
    │                        │
Desktop QA               AR/MR
    │                        │
    └───────────┬────────────┘
                │
         Same Procedure
         Same QA Engine
         Same Data
```

Unity should NOT contain the business logic.

Unity should consume:

```
procedures
states
assets
transforms
validation results
```

This prevents the product from becoming a Unity-only application.

============================================================
48. FUTURE UNITY STACK
======================

When AR phase begins:

```
Unity 6 LTS
C#
URP
OpenXR
XR Interaction Toolkit
Meta XR where required
Android Build Support
```

Potential hardware:

```
Meta Quest
industrial AR headsets
tablets
phones
future optical-see-through devices
```

============================================================
49. FUTURE AR FLOW
==================

Camera/headset

```
↓
```

Detect equipment

```
↓
```

Estimate pose

```
↓
```

Register digital twin

```
↓
```

Overlay:

```
arrows
labels
highlighted components
procedure instructions
animations
warnings
measurements
```

Technician performs action.

Vision/tool system validates action.

Next instruction appears.

============================================================
50. PROCEDURE STARTING IN THE MIDDLE
====================================

The system must NOT assume the technician always starts at Step 1.

Example:

Procedure:

```
1. Open cover
2. Disconnect connector
3. Remove PCB
4. Inspect PCB
5. Replace PCB
```

Technician opens the application when PCB is already removed.

System should:

```
observe current state
    ↓
determine likely procedure state
    ↓
allow procedure resume from matching state
```

However, do not silently assume the state.

Show:

```
Detected current state:
PCB_REMOVED

Confidence:
95%

Resume from:
Step 4
```

For uncertain state:

```
request confirmation.
```

============================================================
51. PROCEDURE GRAPH
===================

Do not hard-code linear execution.

Support graph:

```
                ┌── Inspection A
                │
```

STATE A → ACTION → STATE B
│
└── Inspection B
↓
STATE C

Future support:

```
branching
conditional paths
optional steps
recovery steps
fault diagnosis
alternate configurations
```

============================================================
52. FAULT DIAGNOSIS
===================

Future architecture:

Observed symptoms
↓
Possible faults
↓
Diagnostic procedure
↓
Inspection
↓
Measurement
↓
Decision
↓
Repair procedure

Example:

Symptom:

```
fan not rotating
```

System:

```
check connector
check power
check fan
inspect PCB
```

This should be driven by approved technical data and deterministic
diagnostic logic, with AI assisting retrieval/explanation rather than
inventing procedures.

============================================================
53. PERFORMANCE REQUIREMENTS
============================

MVP should aim for:

Camera:

```
720p or 1080p
```

Inference:

```
near-real-time where hardware permits
```

Tracking:

```
stable frame-to-frame tracking
```

UI:

```
responsive
```

Procedure transitions:

```
deterministic
```

Evidence:

```
automatically captured
```

Do not optimize prematurely.

First establish correctness.

Then optimize:

```
ONNX
TensorRT
GPU inference
batching
model quantization
frame skipping
```

============================================================
54. TESTING
===========

Create automated tests for:

Procedure engine
State machine
QA rules
Tolerance calculations
Coordinate transforms
API
Database
Evidence handling

Vision tests:

```
detection accuracy
segmentation accuracy
tracking stability
state classification
pose error
```

Create recorded test videos.

The same video must be replayable for regression testing.

============================================================
55. DATASET PIPELINE
====================

Create:

```
dataset/
    images/
    labels/
    videos/
    annotations/
```

Use CVAT or equivalent annotation tooling.

Annotations should support:

```
bounding boxes
segmentation masks
object IDs
states
keyframes
```

Training:

```
dataset
   ↓
train/validation/test
   ↓
YOLO
   ↓
model version
   ↓
evaluation
   ↓
deployment
```

Never silently replace a production model.

============================================================
56. MODEL EVALUATION
====================

Record:

```
precision
recall
mAP
confusion matrix
tracking metrics
state classification accuracy
```

But QA suitability must be evaluated at the procedure level too.

Example:

Even if object detection is 95% accurate, ask:

```
How often does the complete procedure validation correctly
identify PASS/FAIL?
```

Procedure-level metrics are important.

============================================================
57. OBSERVABILITY
=================

Log:

```
inference time
FPS
detection count
tracking state
pose status
QA result
API latency
model version
```

Provide developer/debug mode showing:

```
bounding boxes
segmentation
IDs
confidence
state
FPS
pose
transforms
```

This mode is extremely important during development.

============================================================
58. DEVELOPMENT PHASES
======================

PHASE 0
DATA MODEL

Implement:

```
Asset
Component
State
Procedure
Step
Requirement
Evidence
```

PHASE 1
CAMERA QA

Implement:

```
camera input
YOLO
segmentation
tracking
state recognition
deterministic QA
```

PHASE 2
TRACKING

Implement:

```
ByteTrack/BoT-SORT
object identity
lost/recovery tracking
```

PHASE 3
VIDEO PROCEDURE EXTRACTION

Implement:

```
video ingestion
frame extraction
state transitions
candidate procedure generation
human approval
```

PHASE 4
3D

Implement:

```
GLB/glTF
Three.js
component hierarchy
animation
procedure visualization
```

PHASE 5
3D ↔ PHYSICAL

Implement:

```
calibration
pose
solvePnP
registration
coordinate transforms
```

PHASE 6
DEPTH / METROLOGY

Implement:

```
depth camera
Open3D
3D geometry
gap measurements
clearance
deviation
```

PHASE 7
AR/MR

Implement:

```
Unity
OpenXR
AR/MR runtime
digital twin overlay
```

PHASE 8
ENTERPRISE

Implement:

```
PostgreSQL production architecture
RBAC
audit
S1000D
PLM
MES
QMS
connected tools
offline synchronization
```

============================================================
59. MVP DEFINITION OF DONE
==========================

Do not claim MVP completion until the following works end-to-end:

A physical assembly is placed in front of a camera.

The system detects the relevant components.

The system tracks the components.

The system identifies the current physical state.

A procedure is loaded.

The UI displays the expected state.

The operator performs the physical action.

The camera observes the resulting state.

The QA engine compares expected vs observed.

The system returns:

```
PASS
FAIL
or
UNCERTAIN
```

Evidence is captured.

The result is stored.

The next procedure step can be executed.

The same procedure can be replayed using recorded video.

No Unity is required for this MVP.

============================================================
60. FIRST DEMO
==============

Build exactly this demonstration first:

Assembly:

```
Water Pump / small mechanical-electronic assembly
```

Components:

```
Housing
Cover
Screws
PCB
Connector
Fan
```

Procedure:

STEP 1

Expected:

```
COVER_CLOSED
```

Camera:

```
detect housing
detect cover
```

Result:

```
PASS
```

STEP 2

Action:

```
Remove screws
```

Expected:

```
SCREWS_REMOVED
```

Camera:

```
detect screw state
```

Result:

```
PASS / FAIL
```

STEP 3

Action:

```
Open cover
```

Expected:

```
COVER_OPEN
```

Camera:

```
estimate cover state
```

Result:

```
PASS
```

STEP 4

Action:

```
Disconnect connector
```

Expected:

```
CONNECTOR_DISCONNECTED
```

Camera:

```
detect connector state
```

Result:

```
PASS
```

STEP 5

Action:

```
Remove PCB
```

Expected:

```
PCB_REMOVED
```

Camera:

```
detect PCB
track PCB
```

Result:

```
PASS
```

STEP 6

Action:

```
Inspect PCB
```

Camera:

```
capture image
```

QA:

```
compare against inspection requirements
```

STEP 7

Action:

```
Reinstall PCB
```

Expected:

```
PCB_INSTALLED
```

STEP 8

Action:

```
Close cover
```

Expected:

```
COVER_CLOSED
```

============================================================
61. WHAT NOT TO DO
==================

DO NOT:

* build Unity first
* build a giant monolithic application
* assume YOLO solves pose
* assume detection equals QA
* assume RGB camera gives millimeter measurements
* make an LLM responsible for PASS/FAIL
* make video recognition responsible for exact procedure execution
* hard-code one product
* hard-code one camera
* hard-code one model
* hard-code one procedure
* put business logic inside Unity
* require cloud services for the core workflow
* create unnecessary microservices
* optimize TensorRT before correctness is established
* implement every industrial protocol in the MVP
* make raw S1000D XML the runtime procedure representation

============================================================
62. ENGINEERING PRINCIPLES
==========================

Follow:

```
modular architecture
typed interfaces
configuration-driven behavior
stable IDs
versioned procedures
versioned AI models
deterministic QA
testability
offline operation
auditability
```

Avoid:

```
magic numbers
hard-coded paths
global mutable state
giant classes
giant files
duplicated business logic
hidden AI decisions
```

============================================================
63. API DESIGN
==============

Implement APIs approximately like:

Assets:

```
GET /api/assets
POST /api/assets
GET /api/assets/{id}
```

Components:

```
GET /api/assets/{id}/components
POST /api/components
```

Procedures:

```
GET /api/procedures
POST /api/procedures
GET /api/procedures/{id}
POST /api/procedures/{id}/publish
```

Steps:

```
GET /api/procedures/{id}/steps
```

Inspection:

```
POST /api/inspection/start
GET /api/inspection/{id}
POST /api/inspection/{id}/step/{stepId}/observe
POST /api/inspection/{id}/step/{stepId}/validate
POST /api/inspection/{id}/complete
```

Vision:

```
POST /api/vision/detect
POST /api/vision/state
POST /api/vision/pose
```

Evidence:

```
POST /api/evidence
GET /api/evidence/{id}
```

============================================================
64. WEBSOCKET
=============

Use WebSocket for real-time inspection status.

Example:

Client:

```
camera frame
```

Backend:

```
detection
tracking
state
```

WebSocket response:

{
"type": "inspection_update",

"stepId": "STEP-004",

"objects": [
{
"id": "PCB-001",
"trackerId": 12,
"state": "REMOVED",
"confidence": 0.96
}
],

"validation": {
"result": "PASS"
}
}

============================================================
65. DATABASE
============

Use PostgreSQL.

At minimum create tables:

```
assets
components
procedures
procedure_revisions
states
actions
steps
requirements
validation_rules
reference_images
reference_videos
models_3d
animations
inspection_runs
inspection_steps
observations
measurements
evidence
users
audit_logs
```

Use foreign keys.

Use UUIDs or stable IDs.

Use revision fields.

Do not store large videos/images directly in PostgreSQL.

Use object storage.

============================================================
66. OBJECT STORAGE
==================

Use MinIO during development.

Buckets:

```
assets
models
images
videos
evidence
datasets
model-artifacts
```

Store metadata in PostgreSQL.

Store binary files in MinIO.

============================================================
67. DOCKER
==========

Provide:

```
docker-compose.yml
```

Services:

```
postgres
minio
backend
frontend
```

Do not containerize GPU inference unnecessarily during the first local
development iteration if that complicates debugging.

Make local Python execution possible.

============================================================
68. CONFIGURATION
=================

Use environment variables.

Example:

DATABASE_URL
MINIO_ENDPOINT
MINIO_ACCESS_KEY
MINIO_SECRET_KEY
MODEL_PATH
CAMERA_INDEX
CONFIDENCE_THRESHOLD
TRACKING_ENABLED
POSE_ENABLED

Do not hard-code secrets.

============================================================
69. DOCUMENTATION
=================

Create:

```
README.md

docs/
    architecture.md
    data-model.md
    procedure-engine.md
    vision-pipeline.md
    tracking.md
    pose.md
    qa-engine.md
    video-pipeline.md
    3d.md
    deployment.md
    dataset.md
    roadmap.md
```

Every major subsystem must have documentation.

============================================================
70. IMPLEMENTATION STRATEGY
===========================

DO NOT attempt to implement everything at once.

Work incrementally.

First:

```
backend skeleton
database
frontend skeleton
```

Then:

```
camera feed
```

Then:

```
YOLO detection
```

Then:

```
tracking
```

Then:

```
state detection
```

Then:

```
QA engine
```

Then:

```
procedure UI
```

Then:

```
evidence
```

Then:

```
3D
```

Then:

```
pose
```

Then:

```
registration
```

Every phase must produce a runnable result.

============================================================
71. CODING AGENT BEHAVIOR
=========================

Before writing large amounts of code:

1. Inspect the repository.
2. Determine existing technology.
3. Do not destroy existing functionality.
4. Create an architecture plan.
5. Identify reusable modules.
6. Implement incrementally.
7. Run tests.
8. Fix errors.
9. Update documentation.
10. Report exactly what was implemented.

Do not blindly rewrite the repository.

============================================================
72. WHEN REQUIREMENTS ARE AMBIGUOUS
===================================

Choose the smallest architecture that satisfies the requirement.

Do not introduce unnecessary complexity.

Prefer:

```
simple working implementation
```

over:

```
theoretically perfect architecture.
```

However, do not create shortcuts that prevent later:

```
3D
AR
S1000D
tracking
tool integration
enterprise deployment.
```

============================================================
73. CODE QUALITY
================

Use:

```
Python type hints
Pydantic models
TypeScript types
clear interfaces
modular services
error handling
structured logging
```

Every important business object should have a typed representation.

============================================================
74. IMPORTANT SEPARATION
========================

The following layers must remain independent:

PROCEDURE ENGINE

knows:

```
expected state
action
transition
```

VISION ENGINE

knows:

```
objects
detections
tracking
state observations
```

POSE ENGINE

knows:

```
position
rotation
coordinate transforms
```

MEASUREMENT ENGINE

knows:

```
geometry
distances
tolerances
```

QA ENGINE

knows:

```
requirements
validation rules
PASS/FAIL logic
```

EVIDENCE ENGINE

knows:

```
images
video
timestamps
observations
```

3D ENGINE

knows:

```
models
animation
visualization
```

No subsystem should secretly perform another subsystem's responsibility.

============================================================
75. FUTURE DIGITAL TWIN
=======================

The 3D model eventually becomes a digital twin representation.

But distinguish:

```
CAD geometry
visual twin
physical asset
observed state
```

Example:

Digital:

```
PCB-001
state = INSTALLED
```

Physical:

```
PCB-001
observed state = REMOVED
```

QA:

```
expected = REMOVED
observed = REMOVED
```

Result:

```
PASS
```

This relationship is central to the product.

============================================================
76. FINAL ARCHITECTURAL MODEL
=============================

The final platform should evolve toward:

```
         ENGINEERING DATA
                │
      ┌─────────┼─────────┐
      │         │         │
     CAD       BOM     S1000D
      │         │         │
      └─────────┼─────────┘
                ↓
         DIGITAL THREAD
                ↓
      PROCEDURE AUTHORING
                ↓
      EXECUTABLE PROCEDURE
                ↓
    ┌───────────┼────────────┐
    │           │            │
   3D         IMAGE        VIDEO
    │           │            │
    └───────────┼────────────┘
                ↓
          PHYSICAL ASSET
                ↓
            CAMERA
                ↓
          COMPUTER VISION
                ↓
      ┌─────────┼──────────┐
      │         │          │
   Detect    Track       Pose
      │         │          │
      └─────────┼──────────┘
                ↓
           STATE ENGINE
                ↓
         MEASUREMENT ENGINE
                ↓
            QA ENGINE
                ↓
          PASS / FAIL
                ↓
            EVIDENCE
                ↓
          QA RECORD
                ↓
          DIGITAL THREAD
```

============================================================
77. FINAL PRODUCT GOAL
======================

The final product should enable this workflow:

Engineer creates procedure
↓
Procedure linked to CAD / images / videos / technical data
↓
Procedure becomes executable
↓
Technician starts procedure
↓
System identifies physical equipment
↓
System determines current state
↓
System displays expected action
↓
Technician performs action
↓
Camera observes physical result
↓
Vision detects components
↓
Tracking maintains identity
↓
Pose determines spatial relationship
↓
Measurement determines geometry where required
↓
QA engine validates requirement
↓
PASS / FAIL / UNCERTAIN
↓
Evidence captured
↓
Next step
↓
Procedure completed
↓
Traceable QA record

============================================================
78. FIRST IMPLEMENTATION TASK
=============================

Do NOT immediately implement the entire system.

Start by creating:

1. Repository architecture
2. PostgreSQL schema
3. FastAPI backend
4. React frontend
5. Procedure JSON schema
6. Procedure engine
7. Camera capture module
8. YOLO detection module
9. Tracking module
10. Basic state engine
11. Deterministic QA engine
12. Evidence capture
13. One complete 5–8 step demo procedure

The first success criterion is:

CAMERA → DETECTION → STATE → QA → EVIDENCE

After that is stable:

```
add 3D
```

Then:

```
add pose
```

Then:

```
add physical ↔ 3D registration
```

Then:

```
add measurement
```

Then:

```
add AR/MR.
```

============================================================
79. REQUIRED FIRST RESPONSE FROM THE CODING AGENT
=================================================

Before modifying the repository, respond with:

A. Current repository analysis

B. Existing technology stack

C. Proposed architecture

D. Directory structure

E. Database schema

F. API design

G. Phase 1 implementation plan

H. Exact files that will be created/modified

I. Dependencies required

J. How the first end-to-end demo will work

Then wait for implementation approval if the environment requires it.

Do not silently make destructive changes.

============================================================
80. SUCCESS CRITERION
=====================

The project is successful when it demonstrates:

```
Physical object
      ↓
   Camera
      ↓
   YOLO
      ↓
  Tracking
      ↓
State detection
      ↓
```

Procedure engine
↓
Expected state
↓
QA engine
↓
PASS / FAIL / UNCERTAIN
↓
Evidence

while maintaining an architecture capable of eventually supporting:

```
3D
CAD
digital twins
images
videos
6DoF
registration
metrology
AR/MR
S1000D
connected tools
fault diagnosis
offline AI
enterprise integration.
```

IMPORTANT FINAL RULE:

Build the smallest working version of the complete architecture first.

Do not build a huge demo with disconnected features.

Every new feature must connect to the central model:

```
ASSET
   ↓
COMPONENT
   ↓
STATE
   ↓
PROCEDURE
   ↓
ACTION
   ↓
OBSERVATION
   ↓
VALIDATION
   ↓
EVIDENCE
   ↓
QA RECORD
```

This model is the core intellectual architecture of the platform.

````

### One important architectural point

The biggest change from the earlier Unity-oriented approach is this:

```text
             WRONG APPROACH

                 UNITY
                   ↓
             Everything else
````

Instead, build:

```text
                 CORE PLATFORM
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   Procedure       Vision         QA
     Engine        Engine        Engine
        │             │             │
        └─────────────┼─────────────┘
                      │
                 DATA MODEL
                      │
       ┌──────────────┼──────────────┐
       │              │              │
      3D            Image          Video
       │              │              │
       └──────────────┼──────────────┘
                      │
                Runtime Clients
                 /           \
              Web            Unity
                              ↓
                         Future AR/MR
```

That makes the **procedure/state/QA engine the product**, while Three.js, computer vision, video processing and eventually Unity are clients/capabilities around it.

The first truly meaningful milestone is therefore not “I loaded a 3D model.” It is:

> **A real camera observes a physical component, determines its state, compares that state against an executable procedure requirement, produces PASS/FAIL/UNCERTAIN, and stores evidence.**

Then 3D registration, digital twins and AR become much easier to add without rebuilding the foundation.
