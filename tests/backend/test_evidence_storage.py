"""
Evidence links must be signed for the address the *browser* uses.

In docker-compose the backend reaches MinIO as "minio:9000", a name that only
resolves inside the compose network. A presigned URL carries its host in the
signature, so it has to be produced for the public address from the start —
and without contacting that address, which the backend may not be able to
reach. No MinIO server is needed here.
"""

from urllib.parse import urlsplit

from minio import Minio

from app.core.config import Settings
from app.services.evidence import storage
from app.services.evidence.storage import EvidenceStore


class _InternalClient:
    """Stands in for the client that talks to MinIO over the compose network."""

    def __init__(self) -> None:
        self.created: list[str] = []

    def bucket_exists(self, bucket: str) -> bool:
        return False

    def make_bucket(self, bucket: str) -> None:
        self.created.append(bucket)

    def presigned_get_object(self, bucket: str, key: str, expires) -> str:  # noqa: ARG002
        return f"http://minio:9000/{bucket}/{key}?signed-for=minio"


def test_without_a_public_endpoint_links_come_from_the_one_client():
    client = _InternalClient()
    store = EvidenceStore(client, "evidence")
    assert client.created == ["evidence"]
    assert urlsplit(store.get_frame_url("step/frame.jpg")).netloc == "minio:9000"


def test_public_endpoint_signs_links_for_the_browser_without_network_access():
    # Port 9 (discard) on localhost: any attempt to connect would fail, so a
    # URL coming back proves presigning made no request.
    presign = Minio("localhost:9", access_key="tvasta", secret_key="tvasta12345", secure=False, region="us-east-1")
    store = EvidenceStore(_InternalClient(), "evidence", presign_client=presign)

    url = urlsplit(store.get_frame_url("step/frame.jpg"))

    assert (url.scheme, url.netloc, url.path) == ("http", "localhost:9", "/evidence/step/frame.jpg")
    assert "X-Amz-Signature=" in url.query


def _built_clients(monkeypatch, **settings) -> list[tuple[str, dict]]:
    built: list[tuple[str, dict]] = []

    class _RecordingMinio(_InternalClient):
        def __init__(self, endpoint: str, **kwargs) -> None:
            super().__init__()
            built.append((endpoint, kwargs))

    monkeypatch.setattr(storage, "Minio", _RecordingMinio)
    monkeypatch.setattr(storage, "get_settings", lambda: Settings(_env_file=None, **settings))
    storage.build_evidence_store()
    return built


def test_build_uses_a_second_client_only_when_the_public_endpoint_differs(monkeypatch):
    assert [e for e, _ in _built_clients(monkeypatch, minio_endpoint="localhost:9000")] == ["localhost:9000"]
    assert [
        e
        for e, _ in _built_clients(monkeypatch, minio_endpoint="localhost:9000", minio_public_endpoint="localhost:9000")
    ] == ["localhost:9000"]

    built = _built_clients(monkeypatch, minio_endpoint="minio:9000", minio_public_endpoint="files.example:9000")
    assert [e for e, _ in built] == ["minio:9000", "files.example:9000"]
    assert built[1][1]["region"] == "us-east-1"  # what keeps presigning offline
