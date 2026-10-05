"""
Replay a recorded clip through the live AR pipeline and score it.

    python -m app.workers.replay_clip <clip> --model <model-id> [--note "what changed"]

Pass 1 feeds the clip's frames to /api/vision/ar-session/frame (in-process,
the same code as the live view) at the recorded pace, always taking the
newest frame that has arrived — a slow tracker skips frames, as it would
live. Pass 2 then measures how well the pose that was SHOWN on each sampled
frame overlays the real object (silhouette IoU against the object's outline
found by its 3D model), so the extra GPU work can't distort pass 1's timing.

Results go to data/clips/<clip>/runs/<time>.json and are compared with the
previous run, so every tracking change can be judged on the same footage.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.api.vision import assembly_geometry, find_object_by_model, register_model_mesh
from app.core.config import get_settings
from app.main import app
from app.services.pose.calibration import load_calibration
from app.services.tracking.replay import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    ReplayRecord,
    clip_dir,
    load_clip,
    next_frame,
    summarize,
    threejs_pose_to_cv,
)
from app.services.vision.part_presence import project_part_mask

settings = get_settings()


def replay(client: TestClient, frames, body: dict) -> list[ReplayRecord]:
    records: list[ReplayRecord] = []
    session = f"replay-{time.time()}"
    started = time.perf_counter()
    position = -1
    while True:
        elapsed_ms = (time.perf_counter() - started) * 1000
        position = next_frame(frames, elapsed_ms, position)
        if position is None:
            break
        frame = frames[position]
        wait = frame.t_ms - elapsed_ms
        if wait > 0:
            time.sleep(wait / 1000)  # the camera hasn't produced it yet
        sent = time.perf_counter()
        response = client.post(
            "/api/vision/ar-session/frame",
            json={**body, "session_id": session, "image_base64": base64.b64encode(frame.path.read_bytes()).decode()},
        )
        if response.status_code != 200:
            raise SystemExit(f"frame {frame.index}: {response.status_code} {response.text[:300]}")
        d = response.json()
        records.append(
            ReplayRecord(
                clip_index=frame.index,
                t_ms=frame.t_ms,
                latency_ms=(time.perf_counter() - sent) * 1000,
                state=d["state"],
                detector_ran=d["detector_ran"],
                position=tuple(d["position"][k] for k in "xyz") if d["visible"] and d["position"] else None,
                quaternion=tuple(d["quaternion"][k] for k in "xyzw") if d["visible"] and d["quaternion"] else None,
            )
        )
    return records


def measure_overlap(records: list[ReplayRecord], frames, model: dict, every: int) -> None:
    by_index = {f.index: f for f in frames}
    glb = Path(settings.models_3d_dir) / model["storage_key"]
    register_model_mesh(model["id"], glb.read_bytes, model["scale"])
    meshes, center = assembly_geometry(glb)
    shown = [r for r in records if r.state == "TRACKING" and r.position is not None]
    for record in shown[::every]:
        path = by_index[record.clip_index].path
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        obj = find_object_by_model(path.read_bytes(), model["id"], model["name"])
        if obj is None or len(obj.polygon) < 3:
            continue
        real = np.zeros((h, w), np.uint8)
        cv2.fillPoly(real, [np.round([[p.x, p.y] for p in obj.polygon]).astype(np.int32)], 1)
        k = load_calibration(settings.camera_calibration_path, w, h).camera_matrix
        pose = threejs_pose_to_cv(record.position, record.quaternion)
        drawn = np.zeros((h, w), bool)
        for mesh in meshes.values():
            drawn |= project_part_mask(mesh, center, model["scale"], pose, k, (h, w))
        real = real.astype(bool)
        record.overlap = float((real & drawn).sum() / max((real | drawn).sum(), 1))


def compare(summary: dict, previous: dict | None) -> None:
    print(f"\n{'metric':22s} {'this run':>10s} {'previous':>10s}")
    for key, value in summary.items():
        before = previous.get(key) if previous else None
        mark = ""
        if isinstance(value, (int, float)) and isinstance(before, (int, float)) and value != before:
            better = (value > before) if key in HIGHER_IS_BETTER else (value < before) if key in LOWER_IS_BETTER else None
            mark = "" if better is None else ("  better" if better else "  WORSE")
        print(f"{key:22s} {str(value):>10s} {str(before) if before is not None else '-':>10s}{mark}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("clip")
    parser.add_argument("--model", required=True, help="Model3D id")
    parser.add_argument("--detect-by", choices=["model", "class"], default="model")
    parser.add_argument("--class-label", default=None)
    parser.add_argument("--track-part", type=int, default=None)
    parser.add_argument("--overlap-every", type=int, default=5, help="measure overlay accuracy on every Nth tracked frame")
    parser.add_argument("--note", default="", help="what changed since the last run")
    args = parser.parse_args()

    directory = clip_dir(Path(settings.clips_dir), args.clip)
    frames = load_clip(directory)
    if not frames:
        raise SystemExit(f"No frames in {directory}")
    client = TestClient(app)
    model = next((m for m in client.get("/api/models3d").json() if m["id"] == args.model), None)
    if model is None:
        raise SystemExit(f"No model {args.model}")

    body = {"mode": "model", "model_id": args.model, "detect_by": args.detect_by, "track_part": args.track_part}
    if args.class_label:
        body["class_label"] = args.class_label
    print(f"Replaying {args.clip}: {len(frames)} frames, {frames[-1].t_ms / 1000:.1f} s, live pace…")
    records = replay(client, frames, body)
    print(f"Measuring overlay accuracy on every {args.overlap_every}th tracked frame…")
    measure_overlap(records, frames, model, args.overlap_every)
    summary = summarize(records, len(frames), frames[-1].t_ms / 1000)

    runs = directory / "runs"
    runs.mkdir(exist_ok=True)
    previous_files = sorted(runs.glob("*.json"))
    previous = json.loads(previous_files[-1].read_text())["summary"] if previous_files else None
    out = runs / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(
        json.dumps(
            {
                "note": args.note,
                "config": vars(args),
                "summary": summary,
                "frames": [vars(r) for r in records],
            },
            indent=1,
        )
    )
    compare(summary, previous)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
