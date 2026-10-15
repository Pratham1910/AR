"""Inspection reports: saved from a live inspection, reviewed afterwards (api/inspection_reports.py, reports.py)."""

import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import component_inspection, inspection_reports
from app.core.database import get_db
from app.main import app
from app.services.component_inspection.reports import ReportStore, evidence_image

from test_component_inspection_api import FakeSession, frame, model  # noqa: F401  (fixture)


@pytest.fixture
def client(model, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(inspection_reports._settings, "inspection_reports_dir", str(tmp_path / "reports"))
    db = type("DB", (), {"get": lambda self, cls, key: model if key == model.id else None})()
    app.dependency_overrides[get_db] = lambda: db
    yield TestClient(app)
    app.dependency_overrides.pop(get_db, None)


def test_a_report_records_the_live_inspection_and_its_evidence(client, model):  # noqa: F811
    session = FakeSession()
    for k in range(8):
        frame(model, session, k, "learn")
    for k in range(8, 16):
        frame(model, session, k, "inspect", removed={"SCREW_004"})
    response = client.post("/api/inspection-reports", json={"session_id": "s1", "station": "Cell A", "operator": "PN"})
    assert response.status_code == 201
    report = response.json()
    assert report["verdict"] == "FAIL" and report["station"] == "Cell A" and report["evidence"]
    screw = next(c for c in report["components"] if c["id"] == "SCREW_004")
    assert screw["status"] == "MISSING_CONFIRMED" and "consecutive" in screw["reason"]
    assert client.get(f"/api/inspection-reports/{report['id']}/evidence.jpg").headers["content-type"] == "image/jpeg"
    listed = client.get("/api/inspection-reports").json()
    assert [r["id"] for r in listed] == [report["id"]] and "components" not in listed[0]


def test_review_sets_note_and_disposition_but_never_the_findings(client, model):  # noqa: F811
    session = FakeSession()
    for k in range(3):
        frame(model, session, k, "inspect")
    report = client.post("/api/inspection-reports", json={"session_id": "s1"}).json()
    reviewed = client.patch(f"/api/inspection-reports/{report['id']}",
                            json={"note": "cap on the bench", "disposition": "rework", "reviewer": "QA"}).json()
    assert reviewed["note"] == "cap on the bench" and reviewed["disposition"] == "rework" and reviewed["disposition_by"] == "QA"
    assert reviewed["components"] == report["components"] and reviewed["verdict"] == report["verdict"]
    assert client.patch(f"/api/inspection-reports/{report['id']}", json={"disposition": "maybe"}).status_code == 422


def test_no_inspection_no_report(client):
    response = client.post("/api/inspection-reports", json={"session_id": "nobody"})
    assert response.status_code == 404 and "start one first" in response.json()["detail"]


def test_report_ids_cannot_reach_other_files(tmp_path):
    store = ReportStore(tmp_path)
    assert store.get("../secrets") is None and store.get("abc") is None


def test_evidence_draws_outlines_in_status_colours():
    image = np.zeros((100, 200, 3), np.uint8)
    jpeg = evidence_image(image, [{"id": "CAP_001", "status": "MISSING_CONFIRMED", "outline_px": [[10, 10], [60, 10], [60, 60], [10, 60]]}])
    import cv2

    drawn = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert drawn[10, 35, 2] > 200  # red outline (BGR)


def test_system_status_reports_each_service(client, monkeypatch):
    monkeypatch.setattr(inspection_reports._settings, "pose_service_url", "http://127.0.0.1:9")  # nothing listens
    body = client.get("/api/system/status").json()
    assert body["camera"]["assumed_hfov_deg"] > 0 and not body["pose_service"]["ok"]
    assert "not reachable" in body["pose_service"]["error"]
