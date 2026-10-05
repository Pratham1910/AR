import json

import httpx
import numpy as np
import pytest

from app.schemas.vision import BoundingBox, SegmentedObject, Vector2
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
            offset = [0.0, 0.05, 0.0] if body.get("node_names") else [0.0, 0.0, 0.0]
            return httpx.Response(
                200, json={"label": body["label"], "extents_m": [0.1, 0.1, 0.1], "vertex_count": 3, "offset_m": offset}
            )
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


def test_part_registration_sends_node_names_and_returns_the_parts_offset():
    service = FakeService()
    client = _client(service)
    assert client.ensure_registered("m1", lambda: b"glb", 0.01) == pytest.approx([0, 0, 0])
    offset = client.ensure_registered("m1__node0", lambda: b"glb", 0.01, ["Cylinder"])
    assert offset == pytest.approx([0.0, 0.05, 0.0])
    assert service.requests[-1][2]["node_names"] == ["Cylinder"]
    # Cached: same part again is no new upload, but still gives the offset.
    assert client.ensure_registered("m1__node0", lambda: b"glb", 0.01, ["Cylinder"]) == pytest.approx([0, 0.05, 0])
    assert [p for _, p, _ in service.requests].count("/objects") == 2
    client.forget("m1")  # deleting the model also drops its part registrations
    assert {p for m, p, _ in service.requests if m == "DELETE"} == {"/objects/m1", "/objects/m1__node0"}


def test_part_pose_is_converted_to_the_assembly_pose_and_refined_in_the_part_frame():
    """Body tracked, assembly rendered. The body mesh's center is 5 cm above
    the assembly's center (offset (0, 0.05, 0)). MegaPose returns the body
    rotated 90deg about the camera's Z at (0, 0, 0.5): x_cam = R (p - offset) + t,
    so the assembly origin is at t - R @ offset = (0.05, 0, 0.5) (R maps +Y to -X)."""
    rz90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    body = np.eye(4)
    body[:3, :3] = rz90
    body[:3, 3] = [0.0, 0.0, 0.5]
    service = FakeService(
        [
            {"found": True, "pose": body.tolist(), "score": 0.9, "mode": "coarse+refine", "elapsed_ms": 900},
            {"found": True, "pose": body.tolist(), "score": 0.9, "mode": "refine", "elapsed_ms": 200},
        ]
    )
    tracker = MegaPoseTracker(_client(service), "m1__node0", refine_iterations=2, part_offset=np.array([0.0, 0.05, 0.0]))
    frame = Frame(np.zeros((480, 640, 3), dtype=np.uint8), CameraCalibration.approximate(640, 480))
    detection = SegmentedObject(class_label="bottle", confidence=0.8, bbox=BoundingBox(x1=1, y1=1, x2=5, y2=5), polygon=[])

    m = tracker.initialize(frame, detection)
    assert m.extra["t_camera_object"][:3, 3] == pytest.approx([0.05, 0.0, 0.5])  # assembly, OpenCV frame
    assert m.position == pytest.approx((0.05, 0.0, -0.5))  # renderer frame: C = diag(1, -1, -1)
    tracker.commit(m)
    tracker.update(frame)
    assert service.requests[-1][2]["prev_pose"] == body.tolist()  # refines the BODY's pose, not the assembly's


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


def _textured_frame() -> Frame:
    rng = np.random.default_rng(0)
    gray = rng.integers(100, 140, (480, 640)).astype(np.uint8)
    gray[160:320, 240:360] = np.kron(rng.integers(0, 255, (16, 12)), np.ones((10, 10))).astype(np.uint8)
    return Frame(np.dstack([gray] * 3), CameraCalibration.approximate(640, 480))


@pytest.mark.parametrize(
    "projected, score, expected",
    [
        ([240, 160, 360, 320], 0.11, 1.0),  # model covers the object exactly; low appearance score is ignored
        # Model lying sideways across it: 6000 px^2 overlap / (19200 + 10000 - 6000) union.
        ([200, 215, 400, 265], 0.41, 6000 / 23200),
    ],
)
def test_megapose_confidence_is_overlap_with_the_object_not_appearance_score(projected, score, expected):
    service = FakeService(
        [{"found": True, "pose": _pose(0.4), "score": score, "mode": "coarse+refine", "elapsed_ms": 900, "projected_bbox": projected}]
    )
    tracker = MegaPoseTracker(_client(service), "m1", refine_iterations=2)
    detection = SegmentedObject(class_label="cup", confidence=0.8, bbox=BoundingBox(x1=240, y1=160, x2=360, y2=320), polygon=[])
    m = tracker.initialize(_textured_frame(), detection)
    assert m.confidence == pytest.approx(expected, abs=0.02)
    assert m.extra["confidence_source"] == "overlap" and m.extra["pose_score"] == score


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


def test_first_lock_sends_the_outline_and_is_judged_by_silhouette_overlap():
    """An upside-down bottle has the same box as an upright one, so box overlap
    can't reject it; the service picks the pose by silhouette and the lock's
    confidence is that silhouette overlap."""
    service = FakeService(
        [
            {
                "found": True, "pose": _pose(0.4), "score": 0.9, "mode": "coarse+refine", "elapsed_ms": 900,
                "projected_bbox": [240, 160, 360, 320], "silhouette_iou": 0.17, "candidate_ious": [0.17, 0.12],
            }
        ]
    )
    tracker = MegaPoseTracker(_client(service), "m1", refine_iterations=2)
    outline = [Vector2(x=240, y=160), Vector2(x=360, y=160), Vector2(x=300, y=320)]
    detection = SegmentedObject(class_label="cup", confidence=0.8, bbox=BoundingBox(x1=240, y1=160, x2=360, y2=320), polygon=outline)
    m = tracker.initialize(_textured_frame(), detection)
    assert service.requests[0][2]["mask_polygon"] == [[240.0, 160.0], [360.0, 160.0], [300.0, 320.0]]
    assert m.confidence == pytest.approx(0.17)  # box overlap alone would have said 1.0
    assert m.extra["confidence_source"] == "silhouette"


def test_without_an_outline_the_lock_keeps_box_overlap():
    service = FakeService([{"found": True, "pose": _pose(0.4), "score": 0.9, "mode": "coarse+refine", "elapsed_ms": 900}])
    _client(service).full_search("m1", b"jpeg", np.eye(3), [10.0, 10.0, 50.0, 50.0], None)
    assert "mask_polygon" not in service.requests[0][2]


def _shifted(frame: Frame, dx: int) -> Frame:
    return Frame(np.roll(frame.bgr, dx, axis=1), frame.calibration)


def test_a_still_object_holds_its_pose_without_calling_megapose():
    """Refining a still object every frame made its rotation random-walk (each
    refine starts from the last one); while flow sees no motion the pose is held."""
    refined = {"found": True, "pose": _pose(0.4), "score": 0.9, "mode": "coarse+refine", "elapsed_ms": 900,
               "projected_bbox": [240, 160, 360, 320]}
    service = FakeService([refined, {**refined, "mode": "refine", "pose": _pose(0.42)}])
    tracker = MegaPoseTracker(
        _client(service), "m1", refine_iterations=2, still_motion_px=1.5, async_refine=False,
        correction_gain_translation=1.0, correction_gain_rotation=1.0,
    )
    frame = _textured_frame()
    detection = SegmentedObject(class_label="cup", confidence=0.8, bbox=BoundingBox(x1=240, y1=160, x2=360, y2=320), polygon=[])
    tracker.commit(tracker.initialize(frame, detection))

    for _ in range(3):
        held = tracker.update(frame)
        tracker.commit(held)
    assert len(service.requests) == 1  # only the first lock reached the pose service
    assert held.extra["held"] and held.position == pytest.approx((0.0, 0.0, -0.40))
    assert held.confidence == pytest.approx(1.0, abs=0.02)  # flow box still on the projected model

    moved = tracker.update(_shifted(frame, 8))  # the object really moved: refine again
    assert len(service.requests) == 2 and not moved.extra.get("held")
    assert moved.position == pytest.approx((0.0, 0.0, -0.42))


def test_holding_can_be_switched_off():
    refined = {"found": True, "pose": _pose(0.4), "score": 0.9, "mode": "coarse+refine", "elapsed_ms": 900,
               "projected_bbox": [240, 160, 360, 320]}
    service = FakeService([refined, {**refined, "mode": "refine"}])
    tracker = MegaPoseTracker(_client(service), "m1", refine_iterations=2, still_motion_px=0.0, async_refine=False)
    frame = _textured_frame()
    detection = SegmentedObject(class_label="cup", confidence=0.8, bbox=BoundingBox(x1=240, y1=160, x2=360, y2=320), polygon=[])
    tracker.commit(tracker.initialize(frame, detection))
    tracker.update(frame)
    assert len(service.requests) == 2
