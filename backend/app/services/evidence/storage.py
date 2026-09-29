"""
Evidence storage (Project.md #33, #65, #66): binary evidence (frames, video
clips) goes to MinIO/S3, never into PostgreSQL — only the storage_key and
metadata live in the DB (app/models/evidence.py).
"""

import io
import uuid
from datetime import datetime, timezone

from minio import Minio

from app.core.config import get_settings


class EvidenceStore:
    def __init__(self, client: Minio, bucket: str):
        self._client = client
        self._bucket = bucket
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    def put_frame(self, inspection_step_id: uuid.UUID, data: bytes, content_type: str = "image/jpeg") -> str:
        """Uploads one evidence frame and returns its storage key."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        key = f"{inspection_step_id}/{timestamp}-{uuid.uuid4().hex[:8]}.jpg"
        self._client.put_object(
            self._bucket,
            key,
            data=io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return key

    def get_frame_url(self, key: str, expires_seconds: int = 3600) -> str:
        from datetime import timedelta

        return self._client.presigned_get_object(self._bucket, key, expires=timedelta(seconds=expires_seconds))


def build_evidence_store() -> EvidenceStore:
    settings = get_settings()
    client = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    return EvidenceStore(client, settings.minio_evidence_bucket)
