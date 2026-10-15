"""
Component-level missing-part inspection on a tracked object (Phase 6):

    registry      which GLB parts are inspectable components, under stable IDs
    projection    where each component is in the camera image at the tracked pose
    visibility    whether it can be judged in this frame (frame edge, size, CAD
                  occlusion, blur, exposure, tracking quality)
    verifiers     is it there? — pluggable checks on the component's own pixels
    decision      per-component state machine with temporal confirmation
    session       runs the above on live AR frames at a configurable rate

Reuses the AR session's pose (T_camera_object) and camera calibration; it
never estimates a pose of its own.
"""
