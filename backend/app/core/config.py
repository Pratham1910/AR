"""
Centralized, typed configuration.

All environment-dependent values must be read through this module — do not
scatter os.environ / os.getenv calls through the codebase (Project.md #68, #62).
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

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
