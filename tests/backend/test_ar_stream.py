"""The live tracking stream: newest frame wins, poses carry their frame's capture time."""

import json
import struct
import threading
import time

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api import vision
from app.main import app

CONFIG = {"session_id": "s1", "mode": "model", "model_id": "m1", "detect_by": "model"}


class _Reply:
    def __init__(self, capture_ms: float, image: bytes):
        self.capture_ms, self.image = capture_ms, image

    def model_dump_json(self) -> str:
        return json.dumps({"capture_ms": self.capture_ms, "image": self.image.decode()})


def _frame(capture_ms: float, image: bytes) -> bytes:
    return struct.pack("<d", capture_ms) + image


def test_frames_arriving_while_one_is_processed_are_dropped_except_the_newest(monkeypatch):
    processed: list[float] = []
    first_started = threading.Event()

    def slow_step(config, encoded, started, capture_ms):
        processed.append(capture_ms)
        first_started.set()
        time.sleep(0.3)  # a slow tracker: the camera keeps producing frames meanwhile
        return _Reply(capture_ms, encoded)

    monkeypatch.setattr(vision, "_stream_step", slow_step)
    with TestClient(app).websocket_connect("/api/vision/ar-session/stream") as ws:
        ws.send_text(json.dumps(CONFIG))
        ws.send_bytes(_frame(0.0, b"f0"))
        assert first_started.wait(2)
        for i in range(1, 5):  # four frames arrive during the slow step
            ws.send_bytes(_frame(i * 33.0, f"f{i}".encode()))
        first = json.loads(ws.receive_text())
        second = json.loads(ws.receive_text())
    assert first == {"capture_ms": 0.0, "image": "f0"}
    assert second == {"capture_ms": 132.0, "image": "f4"}  # 1-3 skipped: never behind the camera
    assert processed == [0.0, 132.0]


def test_errors_come_back_as_messages_and_the_stream_continues(monkeypatch):
    calls = []

    def step(config, encoded, started, capture_ms):
        calls.append(capture_ms)
        if len(calls) == 1:
            raise HTTPException(status_code=503, detail="pose service down")
        return _Reply(capture_ms, encoded)

    monkeypatch.setattr(vision, "_stream_step", step)
    with TestClient(app).websocket_connect("/api/vision/ar-session/stream") as ws:
        ws.send_text(json.dumps(CONFIG))
        ws.send_bytes(_frame(1.0, b"a"))
        error = json.loads(ws.receive_text())
        ws.send_bytes(_frame(2.0, b"b"))
        ok = json.loads(ws.receive_text())
    assert error == {"error": "pose service down", "status": 503, "capture_ms": 1.0}
    assert ok["capture_ms"] == 2.0


def test_frames_before_the_configuration_are_ignored(monkeypatch):
    seen = []
    monkeypatch.setattr(vision, "_stream_step", lambda c, e, s, t: seen.append(t) or _Reply(t, e))
    with TestClient(app).websocket_connect("/api/vision/ar-session/stream") as ws:
        ws.send_bytes(_frame(1.0, b"early"))
        ws.send_text(json.dumps(CONFIG))
        ws.send_bytes(_frame(2.0, b"x"))
        assert json.loads(ws.receive_text())["capture_ms"] == 2.0
    assert seen == [2.0]


def test_an_invalid_configuration_is_reported_not_silently_fatal(monkeypatch):
    monkeypatch.setattr(vision, "_stream_step", lambda c, e, s, t: _Reply(t, e))
    with TestClient(app).websocket_connect("/api/vision/ar-session/stream") as ws:
        ws.send_text(json.dumps({**CONFIG, "detect_by": "telepathy"}))
        error = json.loads(ws.receive_text())
        assert error["status"] == 422 and "Invalid tracking configuration" in error["error"]
        ws.send_text(json.dumps(CONFIG))  # the stream is still usable
        ws.send_bytes(_frame(5.0, b"ok"))
        assert json.loads(ws.receive_text())["capture_ms"] == 5.0
