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


@lru_cache
def get_settings() -> Settings:
    return Settings()
