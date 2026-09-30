import json

import httpx
import numpy as np
import pytest

from app.services.pose.model_pose_client import ModelPoseClient, PoseServiceUnavailable


def _pose(z: float) -> list[list[float]]:
    t = np.eye(4)
    t[2, 3] = z
    return t.tolist()


class FakeService:
    """Stands in for pose_service: records requests, returns queued responses."""

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if request.url.path == "/objects":
            return httpx.Response(200, json={"label": body["label"], "extents_m": [0.1, 0.1, 0.1], "vertex_count": 3})
        return httpx.Response(200, json=self.responses.pop(0))


def _client(service: FakeService) -> ModelPoseClient:
    return ModelPoseClient("http://pose", 5.0, transport=httpx.MockTransport(service))


def _estimate(client: ModelPoseClient, bbox=(10.0, 10.0, 50.0, 50.0)):
    return client.estimate("m1", "s1", b"jpeg", np.eye(3), list(bbox) if bbox else None, min_score=0.5, track_iterations=2)


def test_first_frame_does_full_search_then_tracks_from_previous_pose():
    service = FakeService(
        [
            {"found": True, "pose": _pose(0.4), "score": 0.9, "mode": "coarse+refine", "elapsed_ms": 900},
            {"found": True, "pose": _pose(0.41), "score": 0.95, "mode": "refine", "elapsed_ms": 200},
        ]
    )
    client = _client(service)

    first = _estimate(client)
    assert first.found and first.mode == "coarse+refine"
    assert service.requests[0][1]["bbox"] == [10.0, 10.0, 50.0, 50.0]
    assert service.requests[0][1]["prev_pose"] is None

    second = _estimate(client, bbox=None)
    assert second.found
    sent = service.requests[1][1]
    assert sent["bbox"] is None
    assert sent["prev_pose"] == _pose(0.4)  # refines from the last good pose
    assert sent["n_refiner_iterations"] == 2


def test_low_score_drops_the_track_so_next_frame_needs_a_fresh_detection():
    service = FakeService([{"found": True, "pose": _pose(0.4), "score": 0.2, "mode": "coarse+refine", "elapsed_ms": 900}])
    client = _client(service)

    result = _estimate(client)
    assert not result.found and result.t_camera_object is None
    assert not client.has_track("m1", "s1")
    # No track and no detection box -> nothing to do, no request sent.
    assert _estimate(client, bbox=None).mode == "no_detection"
    assert len(service.requests) == 1


@pytest.mark.parametrize("z", [-0.3, 0.0, 7.0])
def test_implausible_depth_is_rejected_even_with_high_score(z):
    service = FakeService([{"found": True, "pose": _pose(z), "score": 1.0, "mode": "coarse+refine", "elapsed_ms": 900}])
    client = _client(service)
    assert not _estimate(client).found
    assert not client.has_track("m1", "s1")


def test_mesh_is_reregistered_when_scale_changes_and_tracks_reset():
    service = FakeService([{"found": True, "pose": _pose(0.4), "score": 0.9, "mode": "coarse+refine", "elapsed_ms": 900}])
    client = _client(service)

    client.ensure_registered("m1", b"glb", 0.01)
    client.ensure_registered("m1", b"glb", 0.01)  # same scale: no second upload
    _estimate(client)
    assert client.has_track("m1", "s1")

    client.ensure_registered("m1", b"glb", 0.0137)  # corrected real-world size
    assert [path for path, _ in service.requests].count("/objects") == 2
    assert not client.has_track("m1", "s1")  # old pose was solved against the wrong-sized mesh


def test_forget_drops_mesh_and_tracks_and_tolerates_a_stopped_service():
    service = FakeService([{"found": True, "pose": _pose(0.4), "score": 0.9, "mode": "coarse+refine", "elapsed_ms": 900}])
    deletes: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            deletes.append(request.url.path)
            return httpx.Response(200, json={"deleted": True})
        return service(request)

    client = ModelPoseClient("http://pose", 5.0, transport=httpx.MockTransport(handler))
    client.ensure_registered("m1", b"glb", 0.01)
    _estimate(client)
    client.forget("m1")
    assert deletes == ["/objects/m1"]
    assert not client.has_track("m1", "s1")

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    ModelPoseClient("http://pose", 5.0, transport=httpx.MockTransport(refuse)).forget("m1")  # no exception


def test_unreachable_service_raises_clear_error():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = ModelPoseClient("http://pose", 5.0, transport=httpx.MockTransport(refuse))
    with pytest.raises(PoseServiceUnavailable, match="not reachable"):
        _estimate(client)
