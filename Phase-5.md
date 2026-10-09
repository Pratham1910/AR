I want to add a NEW QUALITY INSPECTION / MISSING-COMPONENT DETECTION
layer to my existing AR digital-twin system.

The current system already does:

    Camera
      ↓
    Object detection
      ↓
    Initial 6DoF pose
      ↓
    Continuous object tracking
      ↓
    3D model registration
      ↓
    3D model follows the physical object

DO NOT BREAK THIS EXISTING FUNCTIONALITY.

The next requirement is:

    COMPARE THE PHYSICAL OBJECT AGAINST THE REGISTERED 3D MODEL

and identify components that exist in the 3D reference but are missing
from the physical object.

============================================================
MAIN EXAMPLE
============

Suppose the reference 3D model contains:

    BODY
    CAP
    HANDLE
    SCREWS
    CONNECTOR

But the real physical object contains:

    BODY
    HANDLE
    SCREWS
    CONNECTOR

and the CAP is physically missing.

The system should detect:

    CAP = EXPECTED
    CAP = NOT OBSERVED

and highlight the missing CAP in the camera view.

The result should visually indicate:

    🔴 CAP MISSING

or highlight the expected cap location in a strong warning color.

============================================================
VERY IMPORTANT
==============

This is NOT simply:

    compare 3D bounding box vs camera bounding box

and it is NOT:

    compare the entire image pixel-by-pixel.

We need COMPONENT-LEVEL COMPARISON.

The 3D model must become the authoritative reference geometry.

============================================================
3D MODEL PREPROCESSING
======================

Inspect the 3D model format and determine whether it already contains:

    separate meshes
    separate objects
    material IDs
    part names
    hierarchy
    assemblies
    subassemblies

For example:

    Cup
      ├── Body
      ├── Cap
      ├── Handle
      ├── Screw_01
      ├── Screw_02
      └── Connector

If the model is currently a single mesh, design a preprocessing step
that allows important components to be separated or assigned component
IDs.

Every inspectable component should have a unique ID.

Example:

    component_id = 1
    component_name = "CAP"

    component_id = 2
    component_name = "HANDLE"

    component_id = 3
    component_name = "SCREW_01"

============================================================
COMPONENT REGISTRY
==================

Create a component registry.

Example:

components = [
    {
        id: "CAP",
        required: true,
        mesh: "cap_mesh",
        inspection_type: "presence",
        tolerance: ...
    },
    {
        id: "HANDLE",
        required: true,
        mesh: "handle_mesh",
        inspection_type: "presence"
    },
    {
        id: "SCREW_01",
        required: true,
        mesh: "screw_01",
        inspection_type: "presence"
    }
]

This should become the foundation for future QA inspections.

============================================================
EXPECTED COMPONENT GEOMETRY
===========================

For every component, the system must know its expected geometry in
object coordinates.

For example:

    CAP

    position relative to object:
        X
        Y
        Z

    orientation:
        Rx
        Ry
        Rz

    geometry:
        mesh

The complete object's current 6DoF pose is already known from the
tracking system.

Therefore:

    Object Pose
         +
    Component Local Pose
         ↓
    Component Camera Pose

This lets us calculate exactly where the CAP SHOULD appear in the
camera.

============================================================
PROJECT THE 3D COMPONENTS
=========================

For every required component:

    3D component mesh
          ↓
    transform using tracked object pose
          ↓
    project into camera
          ↓
    obtain expected 2D region

For example:

    3D CAP
       ↓
    camera projection
       ↓
    expected CAP region

The system now knows:

    "If the physical object is correct,
     the CAP should appear approximately HERE."

============================================================
REAL OBJECT OBSERVATION
=======================

Now inspect the actual camera image.

Do NOT immediately assume:

    expected region empty = missing component

because the component may be:

    hidden
    occluded
    outside camera view
    blocked by another object
    poorly illuminated
    motion blurred
    partially visible

The system must distinguish:

    MISSING
    PRESENT
    PARTIALLY_VISIBLE
    OCCLUDED
    UNKNOWN

============================================================
COMPONENT PRESENCE DETECTION
============================

For each expected component:

    EXPECTED COMPONENT
            ↓
    projected 2D region
            ↓
    inspect camera evidence
            ↓
    calculate presence score
            ↓
    classify

Example:

    CAP

    expected = TRUE
    observed = FALSE
    visibility = HIGH

    RESULT:

        CAP MISSING

But:

    expected = TRUE
    observed = FALSE
    visibility = LOW

    RESULT:

        UNKNOWN / OCCLUDED

Do NOT report a component as missing if the camera cannot actually
verify its location.

============================================================
MASK-BASED COMPARISON
=====================

Use the existing object segmentation.

We already know:

    physical object mask

Now generate:

    expected component mask

from the registered 3D model.

For example:

    Real object mask:
        M_real

    Expected CAP mask:
        M_cap_expected

Use these masks to determine whether the expected component has
corresponding visual evidence.

However, do not rely solely on mask overlap.

Use multiple signals.

============================================================
MULTI-SIGNAL COMPONENT VERIFICATION
===================================

For each component, calculate:

1. Expected projected area
2. Expected projected location
3. Real object evidence
4. Appearance / texture evidence
5. Edge / contour evidence
6. Depth evidence if available
7. Component-specific detector confidence
8. Occlusion visibility
9. Geometric consistency

Combine these into a component presence score.

Conceptually:

    presence_score =
        geometry_score
        +
        visual_score
        +
        depth_score
        +
        edge_score

Use configurable weights.

============================================================
MISSING COMPONENT EXAMPLE
=========================

Reference:

            CAP
        ┌─────────┐
        │         │
        └─────────┘
        ┌─────────┐
        │         │
        │  BODY   │
        │         │
        └─────────┘

Physical object:

        ┌─────────┐
        │         │
        │  BODY   │
        │         │
        └─────────┘

The system knows from the 3D model:

    CAP should be here.

The camera shows:

    CAP region has no corresponding physical geometry.

Therefore:

    CAP = MISSING

Highlight the projected expected CAP region.

For example:

    RED outline
    RED transparent fill
    "CAP MISSING"

============================================================
VISUAL HIGHLIGHTING
===================

Add a QA visualization mode.

For missing components:

    RED

For present components:

    GREEN

For partially visible components:

    YELLOW

For uncertain/occluded components:

    ORANGE

Example:

    GREEN  = verified present
    RED    = missing
    YELLOW = partially present
    ORANGE = cannot verify

The expected 3D geometry can be rendered in the warning color at the
location where the missing component SHOULD be.

Example:

    physical object
          +
    translucent red CAP
          ↓
    "CAP MISSING"

This makes the problem immediately visible to the operator.

============================================================
IMPORTANT: DO NOT JUST HIGHLIGHT EMPTY SPACE
============================================

The system must not simply say:

    "There is no pixel here."

It must reason:

    1. The object is correctly registered.
    2. The reference model says CAP should exist here.
    3. This region is sufficiently visible.
    4. The physical-object observation does not contain the expected
       component.
    5. Therefore CAP is likely missing.

This distinction is critical for industrial QA.

============================================================
OCCLUSION HANDLING
==================

A component should only be declared missing if it is actually
observable.

For example:

    3D CAP exists
    but another physical component blocks it.

Do NOT report:

    CAP MISSING

Instead:

    CAP = OCCLUDED / UNKNOWN

If depth is available, use depth comparison to determine whether the
expected component is hidden behind another physical surface.

If depth is unavailable, use visibility reasoning based on the
projected geometry and camera viewpoint.

============================================================
PARTIAL MISSING COMPONENT
=========================

Support partial component failures.

Example:

    Reference CAP:
       ██████████

    Physical CAP:
       ██████

The system should not only support:

    PRESENT
    MISSING

but also:

    PARTIAL

Example:

    CAP
    Expected area = 100%
    Observed area = 58%

Result:

    CAP = PARTIALLY PRESENT

Highlight the missing portion where possible.

============================================================
COMPONENT-LEVEL DIFFERENCE MAP
==============================

Generate a difference map between:

    expected 3D projection

and:

    observed physical object

The difference should be component-aware.

Example:

    Expected:
       CAP
       HANDLE
       SCREW_01
       SCREW_02

    Observed:
       HANDLE
       SCREW_01
       SCREW_02

Difference:

       CAP

Therefore:

       MISSING COMPONENT = CAP

============================================================
DO NOT USE ONLY GLOBAL SILHOUETTE
=================================

A global silhouette comparison is insufficient.

Example:

A missing screw may have almost no effect on the outer silhouette.

The system must still detect:

    SCREW_01 MISSING

Therefore inspect internal/component geometry, not just the object's
outer contour.

============================================================
SMALL COMPONENTS
================

The system must support very small components such as:

    screws
    bolts
    washers
    clips
    connectors
    wires
    switches
    covers
    caps

Do not assume that a component must significantly change the global
object silhouette.

For small components, use:

    projected 3D component region
    high-resolution crop
    feature matching
    local segmentation
    component-specific detector
    depth if available

depending on the component.

============================================================
COMPONENT DETECTOR ARCHITECTURE
===============================

Do not run a heavy detector for every component on every frame.

Use the known 3D pose to calculate where each component should be.

Then inspect only the relevant ROI.

Pipeline:

    tracked object
         ↓
    3D reference model
         ↓
    component registry
         ↓
    project component
         ↓
    ROI
         ↓
    local verification
         ↓
    component status

This should be significantly more efficient.

============================================================
TRACKING + QA
=============

The existing tracking system remains responsible for:

    Object pose
    X/Y/Z
    Rotation
    Tracking

The new QA layer uses that pose.

Architecture:

                 CAMERA
                    │
                    ▼
             OBJECT DETECTION
                    │
                    ▼
             6DoF TRACKING
                    │
                    ▼
              OBJECT POSE
                    │
           ┌────────┴────────┐
           │                 │
           ▼                 ▼
      3D RENDERING      QA ENGINE
                             │
                             ▼
                    COMPONENT PROJECTION
                             │
                             ▼
                    REAL vs EXPECTED
                             │
                             ▼
                    COMPONENT STATUS
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
           PRESENT         MISSING       OCCLUDED
              │              │              │
              ▼              ▼              ▼
           GREEN           RED           ORANGE


### The key idea

Your existing system is now:

**"Where is the object?"**

The new system becomes:

**"What should be on this object, and is it actually there?"**

So the architecture evolves to:


┌─────────────────────────────┐
│       QA INSPECTION         │
├─────────────────────────────┤
│ ✓ BODY       PRESENT        │
│ ✗ CAP        MISSING        │
│ ✓ HANDLE     PRESENT        │
│ ✓ SCREW 01   PRESENT        │
├─────────────────────────────┤
│ RESULT: FAILED              │
└─────────────────────────────┘

```text
             3D MODEL
                │
       ┌────────┼────────┐
       ▼        ▼        ▼
      CAP     HANDLE    SCREW
       │        │        │
       └────────┼────────┘
                │
          Expected state
                │
                ▼
       ┌────────────────┐
       │ 6DoF REGISTERED│
       │  WITH REAL OBJ │
       └───────┬────────┘
               │
               ▼
        CAMERA OBSERVATION
               │
               ▼
       COMPONENT COMPARISON
               │
       ┌───────┼────────┐
       ▼       ▼        ▼
    PRESENT  MISSING  OCCLUDED
       │       │        │
      🟢      🔴       🟠
```
