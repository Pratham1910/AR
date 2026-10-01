import json

import httpx
import numpy as np
import pytest

from app.schemas.vision import BoundingBox, SegmentedObject
from app.services.pose.calibration import CameraCalibration
from app.services.pose.model_pose_client import ModelPoseClient, PoseServiceUnavailable
from app.services.tracking.trackers import Frame, MegaPoseTracker


def _pose(z: float) -> list[list[float]]:
    t = np.eye(4)
    t[2, 3] = z
    return t.tolist()


class FakeService:
    """Stands in for pose_service: records requests, returns queued responses."""

    def __init__(self, responses: list[dict] | None = None):
        self.responses = list(responses or [])
        self.requests: list[tuple[str, str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.requests.append((request.method, request.url.path, body))
        if request.method == "DELETE":
            return httpx.Response(200, json={"deleted": True})
        if request.url.path == "/objects":
            return httpx.Response(200, json={"label": body["label"], "extents_m": [0.1, 0.1, 0.1], "vertex_count": 3})
        return httpx.Response(200, json=self.responses.pop(0))


def _client(service: FakeService) -> ModelPoseClient:
    return ModelPoseClient("http://pose", 5.0, transport=httpx.MockTransport(service))


def test_full_search_sends_the_detector_box_and_returns_pose_and_score():
    service = FakeService([{"found": True, "pose": _pose(0.4), "score": 0.9, "mode": "coarse+refine", "elapsed_ms": 900}])
    result = _client(service).full_search("m1", b"jpeg", np.eye(3), [10.0, 10.0, 50.0, 50.0])
    _, path, body = service.requests[0]
    assert path == "/estimate" and body["bbox"] == [10.0, 10.0, 50.0, 50.0] and "prev_pose" not in body
    assert result.score == 0.9 and result.t_camera_object[2, 3] == 0.4


def test_refine_sends_previous_pose_and_iterations_and_no_box():
    service = FakeService([{"found": True, "pose": _pose(0.41), "score": 0.95, "mode": "refine", "elapsed_ms": 200}])
    _client(service).refine("m1", b"jpeg", np.eye(3), np.asarray(_pose(0.4)), iterations=2)
    body = service.requests[0][2]
    assert body["prev_pose"] == _pose(0.4) and body["n_refiner_iterations"] == 2 and "bbox" not in body


@pytest.mark.parametrize("z", [-0.3, 0.0, 7.0])
def test_implausible_depth_is_a_failed_solve_whatever_its_score(z):
    service = FakeService([{"found": True, "pose": _pose(z), "score": 1.0, "mode": "coarse+refine", "elapsed_ms": 900}])
    result = _client(service).full_search("m1", b"jpeg", np.eye(3), [1, 1, 5, 5])
    assert result.t_camera_object is None and result.score == 0.0


def test_mesh_is_reregistered_only_when_scale_changes_and_read_lazily():
    service = FakeService()
    client = _client(service)
    reads = []

    def read():
        reads.append(1)
        return b"glb"

    client.ensure_registered("m1", read, 0.01)
    client.ensure_registered("m1", read, 0.01)  # same scale: no second upload, file not re-read
    client.ensure_registered("m1", read, 0.0137)  # corrected real-world size
    assert [p for _, p, _ in service.requests].count("/objects") == 2
    assert len(reads) == 2


def test_forget_deletes_mesh_and_tolerates_a_stopped_service():
    service = FakeService()
    client = _client(service)
    client.ensure_registered("m1", lambda: b"glb", 0.01)
    client.forget("m1")
    assert ("DELETE", "/objects/m1", {}) in service.requests
    client.ensure_registered("m1", lambda: b"glb", 0.01)  # forgotten -> registers again
    assert [p for _, p, _ in service.requests].count("/objects") == 2

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    ModelPoseClient("http://pose", 5.0, transport=httpx.MockTransport(refuse)).forget("m1")  # no exception


def test_unreachable_service_raises_clear_error():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = ModelPoseClient("http://pose", 5.0, transport=httpx.MockTransport(refuse))
    with pytest.raises(PoseServiceUnavailable, match="not reachable"):
        client.full_search("m1", b"jpeg", np.eye(3), [1, 1, 5, 5])


def test_service_error_detail_is_surfaced():
    def gpu_down(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "No CUDA GPU visible inside WSL"})

    client = ModelPoseClient("http://pose", 5.0, transport=httpx.MockTransport(gpu_down))
    with pytest.raises(PoseServiceUnavailable, match="503: No CUDA GPU visible inside WSL"):
        client.full_search("m1", b"jpeg", np.eye(3), [1, 1, 5, 5])


def test_megapose_tracker_refines_from_the_last_accepted_pose_only():
    service = FakeService(
        [
            {"found": True, "pose": _pose(0.40), "score": 0.9, "mode": "coarse+refine", "elapsed_ms": 900},
            {"found": True, "pose": _pose(0.90), "score": 0.1, "mode": "refine", "elapsed_ms": 200},
            {"found": True, "pose": _pose(0.41), "score": 0.9, "mode": "refine", "elapsed_ms": 200},
        ]
    )
    tracker = MegaPoseTracker(_client(service), "m1", refine_iterations=2)
    frame = Frame(np.zeros((480, 640, 3), dtype=np.uint8), CameraCalibration.approximate(640, 480))
    detection = SegmentedObject(class_label="cup", confidence=0.8, bbox=BoundingBox(x1=1, y1=1, x2=5, y2=5), polygon=[])

    first = tracker.initialize(frame, detection)
    assert first.confidence == 0.9 and first.position == pytest.approx((0.0, 0.0, -0.40))
    tracker.commit(first)  # the session accepted it
    tracker.update(frame)  # low score: the session would NOT commit this one
    tracker.update(frame)
    assert service.requests[1][2]["prev_pose"] == _pose(0.40)
    assert service.requests[2][2]["prev_pose"] == _pose(0.40)  # not the rejected 0.90
