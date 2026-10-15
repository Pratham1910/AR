"""
Centralized, typed configuration.

All environment-dependent values must be read through this module — do not
scatter os.environ / os.getenv calls through the codebase (Project.md #68, #62).
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # backend/.env wherever the server is started from (then a .env in the
    # current folder, which wins — as before).
    model_config = SettingsConfigDict(
        env_file=(Path(__file__).resolve().parents[2] / ".env", ".env"), extra="ignore", protected_namespaces=()
    )

    # Database
    database_url: str = "postgresql+psycopg2://tvasta:tvasta@localhost:5432/tvasta"

    # Object storage (MinIO / S3-compatible)
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "tvasta"
    minio_secret_key: str = "tvasta12345"
    minio_secure: bool = False
    minio_evidence_bucket: str = "evidence"

    # Vision
    model_path: Optional[str] = None
    camera_index: int = 0
    confidence_threshold: float = 0.80
    tracking_enabled: bool = False
    pose_enabled: bool = False

    # QA engine thresholds (Project.md #8 / #30) — deterministic, not AI-decided.
    qa_detection_confidence_threshold: float = 0.80
    qa_state_confidence_threshold: float = 0.90

    # Pose / registration (Project.md #24, #25, #26) — Phase 5.
    camera_calibration_path: Optional[str] = None
    # Without a calibration file, the lens is modelled from this horizontal
    # field of view. 82 = Logitech BRIO at its default 90 deg (diagonal)
    # setting in Logi Tune, at 16:9; set 70 for its 78 deg setting, 58 for 65.
    camera_assumed_hfov_deg: float = 82.0
    aruco_dictionary: str = "DICT_4X4_50"
    aruco_marker_length_m: float = 0.05

    # Static file mount for served GLB/glTF models (Project.md #20, Phase 4).
    models_3d_dir: str = "../data/models"
    # FBX uploads are converted to GLB with headless Blender. Unset = look on
    # PATH, then the newest C:/Program Files/Blender Foundation/Blender */.
    blender_path: Optional[str] = None

    # Live object-outline visualization (Project.md #16, #57) — a debug/demo
    # view, not the QA-critical detector. Defaults to a stock COCO-pretrained
    # model (auto-downloaded by ultralytics on first use) since COCO already
    # has a "bottle" class; swap to a custom-trained model via this setting
    # once real assembly components need outlining.
    segmentation_model_name: str = "yolov8n-seg.pt"
    segmentation_confidence_threshold: float = 0.4

    # Feature/keypoint ("image target") tracking (Project.md #24's markerless
    # upgrade) — registered reference planes are persisted here, one file
    # pair (.npz + .json) per asset_id.
    reference_images_dir: str = "../data/reference_images"
    # Raw camera frames saved from the AR page ("Save frame") for offline
    # analysis/calibration, e.g. labelled cap-on / cap-off. Not committed.
    debug_frames_dir: str = "../data/debug_frames"
    # Per-part presence calibrations (e.g. what a bottle's cap region looks
    # like on vs off), one JSON per model part. Not committed.
    part_calibration_dir: str = "../data/part_calibration"
    # Recorded camera clips (raw frames + capture times) for replaying the AR
    # pipeline as a benchmark (app/workers/replay_clip.py). Not committed.
    clips_dir: str = "../data/clips"
    # Procedures authored in Vishwa (.procedure.json), one per 3D model.
    procedures_dir: str = "../data/procedures"

    # Component-level missing-part inspection (app/services/component_inspection).
    # Optional mapping of GLB nodes to component IDs, one JSON per model
    # ({model_id}.json; see registry.py). Without one, named parts and nodes
    # whose names say what they are (Screw_01) are the components.
    component_registry_dir: str = "../data/component_registry"
    # The learned "good state" appearance of each model's components. Not committed.
    component_reference_dir: str = "../data/component_reference"
    # One JSON-lines evidence log per inspection run. Not committed.
    inspection_log_dir: str = "../data/inspection_logs"
    # Saved inspection reports (JSON + evidence JPEG). Not committed.
    inspection_reports_dir: str = "../data/inspection_reports"
    inspection_interval_s: float = 0.25  # inspection passes at most this often
    # Run passes on a worker thread so tracking never waits for them (tests turn this off).
    inspection_async: bool = True
    inspection_confirm_missing_after: int = 6  # valid "absent" observations in a row
    inspection_confirm_present_after: int = 3
    inspection_clear_missing_after: int = 4  # valid "present" observations to clear a confirmed missing
    inspection_stale_after_s: float = 5.0
    inspection_viewpoint_reset_deg: float = 30.0
    inspection_min_area_px: int = 40  # smaller in the image than this: too small to judge
    inspection_min_sharpness: float = 8.0  # variance of the Laplacian around a component
    inspection_min_cad_visible: float = 0.3  # share of a component that must be the nearest CAD surface
    # The learned-reference check: within this many spreads of the good state
    # is present, beyond `absent_z` absent, in between undecided; and it
    # stands down more than `view_deg` from any viewpoint it learned from.
    inspection_reference_present_z: float = 3.0
    inspection_reference_absent_z: float = 6.0
    inspection_reference_view_deg: float = 35.0
    # A custom-trained component detector (YOLO weights) and how its class
    # names map to component types, as JSON, e.g. {"m4_screw": "screw"}.
    # Unset: no detector check (the stock COCO model has no such classes).
    component_detector_weights: Optional[str] = None
    component_detector_classes: Optional[str] = None

    # Model-based (CAD) 6DoF pose — MegaPose, served by pose_service/ running
    # in WSL2 with CUDA (Windows localhost is forwarded into WSL2).
    # 127.0.0.1, not "localhost": on Windows a new connection to "localhost"
    # tries IPv6 first and costs ~2s before falling back (measured 2017ms vs
    # 19ms), which hit every time the connection to the pose service idled.
    pose_service_url: str = "http://127.0.0.1:8765"
    pose_service_timeout_s: float = 30.0  # first lock + first-ever mesh registration can take seconds
    # Refiner iterations when tracking from the previous frame's pose (the
    # first lock uses MegaPose's default of 5). Measured on an RTX 4090: 2
    # iterations ~220ms at the same ~3-4mm accuracy as 5 (~380ms), since
    # frame-to-frame motion starts the refiner close to the answer.
    model_pose_track_iterations: int = 2
    # Hold the model still while optical flow says the object hasn't moved
    # more than this (px) since the last MegaPose pose; 0 = refine every frame.
    model_pose_still_motion_px: float = 1.5
    # How far each background MegaPose correction pulls the flow-tracked pose
    # (0..1): partial pulls average out the few-degree error of single refines.
    model_pose_correction_gain_translation: float = 0.5
    model_pose_correction_gain_rotation: float = 0.3

    # Hands in view (Phase 7, app/services/vision/hand_mask.py). A hand mask
    # keeps hand pixels out of the tracked object's features, outline, pose
    # refinement and component checks. "both" = MediaPipe Hands landmarks
    # plus the segmentation model's "person" masks (union: errs towards
    # treating too much as hand, the safe side); "mediapipe"; "person"; "off".
    hand_mask_method: str = "both"
    hand_landmarker_model: str = "../models/mediapipe/hand_landmarker.task"
    hand_mask_interval_ms: float = 100.0  # masks are computed at most this often (off the tracking thread)
    hand_mask_max_age_ms: float = 400.0  # an older mask isn't used
    hand_mask_margin_px: int = 6  # grown by this around the hand
    hand_person_confidence: float = 0.35  # segmentation "person" regions at least this confident
    # A hand covering at least this share of the object's outline when flow
    # loses it means "occluded": the last pose is held for up to
    # occlusion_hold_ms (TRACKING_OCCLUDED) instead of declaring it lost.
    tracking_occlusion_min: float = 0.2
    tracking_occlusion_hold_ms: float = 1500.0
    # MegaPose corrections are skipped while a hand covers more than this of the object.
    tracking_refine_max_occlusion: float = 0.3
    # Below this share of agreeing optical-flow points, confidence is scaled down.
    tracking_flow_support_full: float = 0.5
    # Component inspection: a hand over more than this share of a component's
    # visible pixels makes it OCCLUDED for that frame, never missing.
    inspection_max_hand_cover: float = 0.15
    # Model-based mode finds the object from its CAD model alone (pose
    # service /detect: FastSAM regions matched against renders of the mesh),
    # no object class needed. A region must match the renders at least this
    # well (cosine similarity of DINOv2 descriptors) to count as the object.
    # 0.35: a dark, plain HDD case scored 0.36-0.40 appearance with shape 0.82-0.85
    # (combined ~0.31) — the shape and combined checks, and the working-distance
    # check on the pose, are what keep wrong regions out.
    cad_detect_min_score: float = 0.35
    # ...and appearance x outline match (shape) at least this. Ranked and
    # accepted on both: appearance alone let small background patches outscore
    # a plain white PC case. A wrong region is still caught by the pose step
    # (its silhouette must fit, ar_model_lost_confidence).
    cad_detect_min_combined: float = 0.30

    # Live AR state machine (app/services/tracking/ar_session.py): detect
    # once, track continuously, re-detect only when tracking is lost.
    ar_detect_interval_ms: float = 150.0  # detector rate cap while SEARCHING/RECOVERING
    ar_grace_frames: int = 2  # LOST frames holding the last pose before RECOVERING (detector)
    ar_lost_timeout_ms: float = 1500.0  # not re-acquired this long after loss -> hide model, SEARCHING
    # Pose filter (app/services/tracking/pose_filter.py): translation =
    # constant-velocity Kalman, rotation = Kalman-gain slerp. Defaults from a
    # sweep: still jitter 4.0->2.3mm / 3.0->2.0deg, ~1mm lag at 0.2 m/s,
    # 30deg turn followed within 200ms. Raise process noise = more responsive.
    ar_filter_enabled: bool = True
    ar_filter_translation_process_noise: float = 0.3  # m/s^2
    ar_filter_translation_measurement_noise: float = 0.004  # m
    ar_filter_rotation_process_noise_deg: float = 40.0  # deg/s
    ar_filter_rotation_measurement_noise_deg: float = 3.0  # deg
    # Confidence thresholds are per tracker, since their scales differ:
    # optical flow = share of points agreeing with the fitted motion (1.0
    # when stationary); model-based = overlap (IoU) of the posed model's
    # image box with the object's box (YOLO, then optical flow) — measured on
    # a real frame: 0.96 for a correct pose, 0.19 for a wrong sideways one.
    ar_flow_good_confidence: float = 0.7
    ar_flow_lost_confidence: float = 0.4
    ar_model_good_confidence: float = 0.7
    ar_model_lost_confidence: float = 0.5
    # Working distance: a first lock (or re-lock) that puts the object nearer
    # or farther than this from the camera is rejected. A plain box can be
    # "found" in any small bright rectangle (a ceiling light) and its pose then
    # fits that rectangle by being far away; this stops such locks.
    ar_min_object_distance_m: float = 0.15
    ar_max_object_distance_m: float = 2.0
    # 3D edge alignment (app/services/pose/edge_align.py): fit the model's own
    # 3D edges (crease edges + silhouette) onto the camera image's edges at
    # every lock and every background correction — corrects the tilt outline
    # matching can't see. Models above `ar_edge_max_faces` triangles skip it
    # (hidden-edge removal rasterises the mesh).
    ar_edge_align_enabled: bool = True
    # Orientation (Phase 8): symmetric models' twin poses are told apart by
    # the pose features in the component mapping file ("poseFeatures"); see
    # app/services/pose/disambiguation.py. Symmetry: allowed mean surface
    # mismatch, as a share of the model's size, for a turn to count as one.
    ar_orientation_enabled: bool = True
    ar_symmetry_tolerance: float = 0.01
    ar_orientation_min_evidence: float = 0.35
    ar_orientation_min_margin: float = 0.15
    # A background correction turning the pose more than this must be backed
    # by the 3D edges (then adopted outright), else it's rejected.
    ar_correction_verify_deg: float = 15.0
    # Click-to-select (ARFrameRequest.select_point): segments under the click
    # between these shares of the image are tried, smallest first, at most
    # this many (each costs one MegaPose full search, ~1-2 s).
    # Guide view (detect_by "guide"): lock once the model's edges at the guide
    # pose fit the image at least this well (edge_align score / support), in
    # this many checks in a row. On the user's upright case, lined-up starts
    # (up to 12 deg / 3 cm off) reached support 0.65-0.70; the unaligned
    # default outline scored 0.64 overall (its blank background inflates
    # coverage) but support 0.41 — support is what separates them.
    ar_guide_min_score: float = 0.58
    ar_guide_min_support: float = 0.55
    ar_guide_confirmations: int = 2
    ar_select_min_area: float = 0.003
    ar_select_max_area: float = 0.6
    ar_select_max_candidates: int = 4
    ar_edge_crease_deg: float = 30.0
    ar_edge_max_points: int = 6000  # edge samples over the whole model (only visible ones are fitted)
    ar_edge_max_faces: int = 60000
    ar_edge_tilt_starts_deg: float = 8.0  # at a lock, also start from the pose tilted +- this much
    ar_edge_gain_translation: float = 0.8  # how far an edge-confirmed correction pulls the pose
    ar_edge_gain_rotation: float = 0.8
    # Fixed-station mode (app/services/tracking/station.py): a locked pose per
    # model, saved here; a part that shifts more than this (px) since locking
    # counts as moved. Not committed.
    station_locks_dir: str = "../data/station_locks"
    station_max_shift_px: float = 4.0
    # Per-model station settings (resting side, ...), one JSON per model. They
    # live on the server so every browser and session uses them.
    station_settings_dir: str = "../data/station_settings"

    # Auth / RBAC (Project.md #45, Phase 8). Deliberately minimal for now —
    # email-only login, no password/SSO (not requested, and a materially
    # larger scope). MUST be overridden via .env for any real deployment;
    # the default here only exists so local dev works without extra setup.
    jwt_secret_key: str = "dev-only-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expires_minutes: int = 480


@lru_cache
def get_settings() -> Settings:
    return Settings()
