"""
Repeatable orientation / alignment evaluation (Phase 8, sections A and J).

Runs the real tracker code (MegaPoseTracker: lock -> 3D edge fit ->
orientation decision; background corrections -> flip guard) on rendered
scenes of a symmetric box with a circle near one corner, with ground truth.
MegaPose itself is simulated, because what matters here is what the rest of
the pipeline does with its answers: for a symmetric box MegaPose returns
whichever symmetric twin it lands on (here: a random allowed twin, seeded),
plus a few degrees / millimetres of error.

"before" = the orientation logic switched off (the pipeline as it was);
"after"  = symmetries + the configured circle feature.

    python -m app.eval.pose_eval            # writes docs/phase8/eval_results.{json,md}

Real-camera behaviour (lighting, BRIO noise, real MegaPose) is NOT measured
here — see docs/phase8/report.md for what was measured live and what not.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.eval.synthetic_scene import K, SHAPE, box_mesh, corner_circle, look_at, project, render, turned
from app.schemas.vision import BoundingBox, SegmentedObject
from app.services.pose.calibration import CameraCalibration
from app.services.pose.edge_align import EdgeModel
from app.services.pose.model_pose_client import PoseServiceResult
from app.services.pose.pose_features import PoseFeature, observe_features, with_feature_edges
from app.services.pose.edge_align import edge_image
from app.services.pose.symmetry import find_symmetries, rotation_angle_deg, twins
from app.services.tracking.trackers import Frame, MegaPoseTracker, resting_side_ok

V, F = box_mesh()
MARK = corner_circle()
FEATURE = PoseFeature("UR_CIRCLE", "circle", MARK.center, MARK.normal, MARK.radius)
SYM = find_symmetries(V, F)
OUT_DIR = Path(__file__).resolve().parents[3] / "docs" / "phase8"


@dataclass
class Scene:
    name: str
    truth: np.ndarray
    marks: list = field(default_factory=lambda: [MARK])
    hand: str | None = None  # "circle" | "corner"
    contrast: float = 1.0
    noise: int = 3


def scenes() -> list[Scene]:
    base = look_at((0.03, 0.30, 0.20))
    out = [Scene(f"turned {d} deg on the desk", turned(base, (0, 1, 0), d)) for d in (0, 90, 180, 270)]
    for elevation in (35, 55, 75):
        e = np.radians(elevation)
        out.append(Scene(f"camera {elevation} deg above the desk", look_at((0.0, 0.32 * np.sin(e), 0.32 * np.cos(e)))))
    for distance in (0.25, 0.45):
        out.append(Scene(f"distance {distance:.2f} m", look_at(np.array([0.03, 0.30, 0.20]) * distance / 0.36)))
    out.append(Scene("moved to the left of the frame", look_at((0.13, 0.30, 0.20), target=(0.10, 0.0, 0.0))))
    out.append(Scene("moved to the right of the frame", look_at((-0.09, 0.30, 0.20), target=(-0.12, 0.0, 0.0))))
    out.append(Scene("tilted 12 deg", turned(base, (1, 0, 0), 12)))
    out.append(Scene("hand over the circle", base, hand="circle"))
    out.append(Scene("hand over a corner (circle visible)", base, hand="corner"))
    out.append(Scene("low contrast, noisy", base, contrast=0.4, noise=6))
    out.append(Scene("plain box (no circle)", base, marks=[]))
    return out


def _hand_box(scene: Scene) -> tuple[int, int, int, int] | None:
    if scene.hand == "circle":
        c = project(MARK.center[None], scene.truth)[0].astype(int)
        return c[0] - 45, c[1] - 45, c[0] + 45, c[1] + 45
    if scene.hand == "corner":
        px = project(V, scene.truth)
        c = px[np.argmax(px[:, 0] + px[:, 1])].astype(int)  # the lower-right corner
        return c[0] - 70, c[1] - 70, c[0] + 30, c[1] + 30
    return None


def _frame(scene: Scene, t=None):
    gray, mask = render(V, F, scene.truth if t is None else t, scene.marks, hand=_hand_box(scene),
                        contrast=scene.contrast, noise=scene.noise)
    calibration = CameraCalibration.approximate(SHAPE[1], SHAPE[0])
    calibration.camera_matrix = K.copy()
    return Frame(np.dstack([gray] * 3), calibration, hand_mask=mask), gray, mask


def _detection(t):
    px = project(V, t)
    (x1, y1), (x2, y2) = px.min(axis=0), px.max(axis=0)
    return SegmentedObject(class_label="box", confidence=0.9, bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2), polygon=[])


def _megapose_answer(truth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random allowed twin of the truth, a few degrees / mm off — MegaPose on a symmetric box."""
    options = [t for t in twins(truth, SYM) if resting_side_ok(t, "camera")]
    pose = options[int(rng.integers(len(options)))].copy()
    pose[:3, :3] = cv2.Rodrigues(np.radians(rng.normal(0, 2.0, 3)))[0] @ pose[:3, :3]
    pose[:3, 3] += rng.normal(0, 0.003, 3)
    return pose


class _FakeMegaPose:
    def __init__(self, pose):
        self.pose = pose
        self.refine_pose = pose

    def full_search(self, label, jpeg, k, bbox, outline=None):
        return PoseServiceResult(self.pose, 0.9, "coarse+refine", 0.0, tuple(bbox), 0.95)

    def refine(self, label, jpeg, k, prev_pose, iterations):
        px = project(V, self.refine_pose)
        return PoseServiceResult(self.refine_pose, 0.9, "refine", 0.0, (*px.min(axis=0), *px.max(axis=0)))


def _tracker(service, orientation: bool) -> MegaPoseTracker:
    """The tracker with the production defaults; "before" also without the
    large-correction check (Phase 8), i.e. corrections are blended as they were."""
    features = [FEATURE] if orientation else []
    geometry = EdgeModel.from_mesh(V, F)
    return MegaPoseTracker(
        service, "box", refine_iterations=2, edge_model=with_feature_edges(geometry, features), geometry_model=geometry,
        async_refine=False, resting="camera", symmetries=SYM if orientation else None, pose_features=features,
        correction_verify_deg=15.0 if orientation else float("inf"),
    )


def _verdict(t: np.ndarray, truth: np.ndarray) -> str:
    angle = rotation_angle_deg(t, truth)
    shift = float(np.linalg.norm(t[:3, 3] - truth[:3, 3]))
    if angle < 5.0 and shift < 0.01:
        return "correct"
    return "flipped" if angle > 90 else "off"


def evaluate_locks(trials: int = 6, seed: int = 0) -> dict:
    """Every scene locked `trials` times (each with a fresh random MegaPose twin), before and after."""
    rng_master = np.random.default_rng(seed)
    rows = []
    for scene in scenes():
        frame, gray, mask = _frame(scene)
        answers = [_megapose_answer(scene.truth, rng_master) for _ in range(trials)]
        row = {"scene": scene.name}
        for label, orientation in (("before", False), ("after", True)):
            verdicts, statuses, latencies, residuals, wrong_inspectable = [], [], [], [], 0
            for answer in answers:
                tracker = _tracker(_FakeMegaPose(answer), orientation)
                started = time.perf_counter()
                m = tracker.initialize(frame, _detection(scene.truth))
                latencies.append((time.perf_counter() - started) * 1000)
                t = m.extra.get("t_camera_object")
                locked = t is not None and m.confidence > 0
                verdict = _verdict(t, scene.truth) if locked else "refused"
                verdicts.append(verdict)
                status = (m.extra.get("orientation") or {}).get("status", "not checked")
                statuses.append(status)
                # Inspection would run on this pose (Phase 8 gate: not on an unconfirmed orientation).
                inspectable = locked and not (status == "ambiguous")
                if inspectable and verdict != "correct":
                    wrong_inspectable += 1
                feature = next(iter((m.extra.get("orientation") or {}).get("features") or []), None)
                if locked and verdict == "correct":
                    if feature and feature.get("residual_px") is not None:
                        residuals.append(feature["residual_px"])
                    elif scene.marks:
                        # Before: no feature check — measure the circle's error directly at the pose.
                        obs = observe_features([FEATURE], t, K, gray, edge_image(gray, exclude=mask))[0]
                        if obs.residual_px is not None:
                            residuals.append(obs.residual_px)
            row[label] = {
                "correct": verdicts.count("correct"), "flipped": verdicts.count("flipped"), "off": verdicts.count("off"),
                "refused": verdicts.count("refused"), "trials": trials,
                "confirmed": statuses.count("confirmed"), "ambiguous": statuses.count("ambiguous"),
                "wrong_pose_inspectable": wrong_inspectable,
                "feature_residual_px_mean": float(np.mean(residuals)) if residuals else None,
                "latency_ms_mean": float(np.mean(latencies)), "latency_ms_p95": float(np.percentile(latencies, 95)),
            }
        rows.append(row)
    return {"locks": rows}


def evaluate_tracking(corrections: int = 12, seed: int = 1) -> dict:
    """A tracked, still box; MegaPose corrections return random twins. How often does the shown pose flip?"""
    rng = np.random.default_rng(seed)
    scene = scenes()[0]
    frame, _, _ = _frame(scene)
    answers = [_megapose_answer(scene.truth, rng) for _ in range(corrections)]
    out = {}
    for label, orientation in (("before", False), ("after", True)):
        service = _FakeMegaPose(scene.truth)
        tracker = _tracker(service, orientation)
        tracker.commit(tracker.initialize(frame, _detection(scene.truth)))
        flips, worst, previous, timings = 0, 0.0, scene.truth, []
        for answer in answers:
            service.refine_pose = answer
            tracker._needs_settle = True
            tracker.update(frame)
            started = time.perf_counter()
            m = tracker.update(frame)
            timings.append((time.perf_counter() - started) * 1000)
            shown = m.extra["t_camera_object"]
            if rotation_angle_deg(shown, previous) > 90:
                flips += 1
            worst = max(worst, rotation_angle_deg(shown, scene.truth))
            previous = shown
        out[label] = {"corrections": corrections, "shown_pose_flips": flips, "worst_error_deg": worst,
                      "guard": dict(tracker.guard_counts), "update_ms_mean": float(np.mean(timings))}
    # A real turnaround (the operator turns the box round): the corrections are the truth's new orientation.
    out["operator_turns_box_round"] = _turnaround()
    return out


def _turnaround() -> dict:
    scene = scenes()[0]
    turned_truth = turned(scene.truth, (0, 1, 0), 180)
    out = {}
    for label, orientation in (("before", False), ("after", True)):
        service = _FakeMegaPose(scene.truth)
        tracker = _tracker(service, orientation)
        frame, _, _ = _frame(scene)
        tracker.commit(tracker.initialize(frame, _detection(scene.truth)))
        frame2, _, _ = _frame(scene, turned_truth)
        service.refine_pose = turned_truth
        tracker._needs_settle = True
        tracker.update(frame2)
        m = tracker.update(frame2)
        out[label] = {"error_deg_after_one_correction": rotation_angle_deg(m.extra["t_camera_object"], turned_truth)}
    return out


def _markdown(results: dict) -> str:
    lines = ["# Phase 8 evaluation (synthetic scenes, real tracker code)", "",
             "Each scene locked 6 times; MegaPose simulated as a random symmetric twin of the truth (+-2 deg, +-3 mm).",
             "`correct` = within 5 deg and 1 cm of ground truth; `flipped` = more than 90 deg off.",
             "`wrong pose inspectable` = locks where inspection would have judged components on a wrong pose.", "",
             "| Scene | before: correct / flipped | after: correct / flipped / refused | after: confirmed / ambiguous | wrong pose inspectable before -> after | circle error px (after) | lock ms p95 before -> after |",
             "|---|---|---|---|---|---|---|"]
    for row in results["locks"]:
        b, a = row["before"], row["after"]
        res = f"{a['feature_residual_px_mean']:.2f}" if a["feature_residual_px_mean"] is not None else "—"
        lines.append(
            f"| {row['scene']} | {b['correct']} / {b['flipped']} | {a['correct']} / {a['flipped']} / {a['refused']} | "
            f"{a['confirmed']} / {a['ambiguous']} | {b['wrong_pose_inspectable']} -> {a['wrong_pose_inspectable']} | {res} | "
            f"{b['latency_ms_p95']:.0f} -> {a['latency_ms_p95']:.0f} |"
        )
    t = results["tracking"]
    lines += ["", "## Tracking: 12 background corrections, each a random twin (box held still)", "",
              "| | shown-pose flips | worst error deg | guard counters | update ms (mean) |", "|---|---|---|---|---|"]
    for label in ("before", "after"):
        r = t[label]
        lines.append(f"| {label} | {r['shown_pose_flips']} | {r['worst_error_deg']:.1f} | {r['guard']} | {r['update_ms_mean']:.0f} |")
    turn = t["operator_turns_box_round"]
    lines += ["", "## Operator turns the box round (legitimate 180 deg change, circle visible)", "",
              f"- before: {turn['before']['error_deg_after_one_correction']:.1f} deg off after one correction",
              f"- after: {turn['after']['error_deg_after_one_correction']:.1f} deg off after one correction"]
    return "\n".join(lines) + "\n"


def main() -> None:
    results = evaluate_locks()
    results["tracking"] = evaluate_tracking()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT_DIR / "eval_results.md").write_text(_markdown(results), encoding="utf-8")
    print(_markdown(results))


if __name__ == "__main__":
    main()
