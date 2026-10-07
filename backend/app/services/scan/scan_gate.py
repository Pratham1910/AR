"""
Decides when a product scan is good enough to confirm on its own, the way a
QR scanner does: one known marker, the right product behind it, well placed
in the scan box (positioning.py), and held still for a moment.

The gate itself only sequences those checks and keeps the hold timer; the
identity check is product_verifier.py and the geometry is positioning.py.
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum

from app.services.markers.detector import MarkerDetection
from app.services.scan.positioning import Positioning, ScanGateConfig, analyze_position, marker_sides, scan_area
from app.services.scan.product_verifier import Verification, VerificationStatus


class ScanState(str, Enum):
    SEARCHING = "SEARCHING"  # no marker
    MARKER_DETECTED = "MARKER_DETECTED"  # marker seen, product not established yet
    VERIFYING = "VERIFYING"  # the expected product seems to be there, not confidently yet
    MISMATCH = "MISMATCH"  # product at the marker is not the one the marker names
    POSITIONING = "POSITIONING"  # verified; needs moving into place
    HOLD_STEADY = "HOLD_STEADY"  # in place; moving too much to start / just started
    SCANNING = "SCANNING"  # in place and still; hold timer running
    CONFIRMED = "CONFIRMED"  # latched until the product leaves the scan box (or reset)


@dataclass(frozen=True)
class ScanObservation:
    """What one frame showed. `marker` is set only when exactly one marker is visible."""

    frame_width: int
    frame_height: int
    marker_count: int
    marker: MarkerDetection | None = None
    known: bool = False  # the marker is bound to a product
    verification: Verification | None = None
    mirrored: bool = False  # the user sees the frame flipped left-right


@dataclass(frozen=True)
class ScanResult:
    state: ScanState
    problem: str | None = None  # why it is not progressing, e.g. "unknown_marker"
    hints: tuple[str, ...] = ()  # e.g. ("move_closer", "move_left"), as seen in the preview
    progress: float = 0.0  # 0..1 through the hold
    marker_id: int | None = None
    verification: Verification | None = None
    positioning: Positioning | None = None  # measured whenever a single marker is in view
    speed: float | None = None  # marker sides per second since the previous frame
    confirmation: int = 0  # how many scans this gate has confirmed; a new number = a new scan
    # How far through the pipeline this frame got, for a step display.
    stages: dict[str, bool] = field(default_factory=dict)


class ScanGate:
    """
    One per camera stream. Feed it every frame's observation. After
    CONFIRMED it stays there — no repeated scans of the same product — until
    that marker has been out of the scan box for `rearm_ms`, or reset().
    """

    def __init__(self, config: ScanGateConfig | None = None):
        self.config = config or ScanGateConfig()
        self._confirmations = 0
        self.reset()

    @property
    def confirmed(self) -> bool:
        return self._confirmed is not None

    def reset(self) -> None:
        self._confirmed: ScanResult | None = None
        self._confirmed_seen_at = 0.0  # when the confirmed marker was last in the scan box
        self._steady_since: float | None = None
        self._last: tuple[float, int, tuple[float, float]] | None = None  # time, marker id, center
        self._last_match: tuple[float, int, Verification] | None = None  # time, marker id, verification

    def update(self, observation: ScanObservation, now_ms: float | None = None) -> ScanResult:
        now = time.monotonic() * 1000.0 if now_ms is None else now_ms
        if self._confirmed is not None:
            if self._confirmed_marker_in_area(observation):
                self._confirmed_seen_at = now
            if now - self._confirmed_seen_at <= self.config.rearm_ms:
                return self._confirmed
            self.reset()  # it left: ready for the next product

        result = self._evaluate(observation, now)
        if result.state not in (ScanState.HOLD_STEADY, ScanState.SCANNING, ScanState.CONFIRMED):
            self._steady_since = None
        if result.state is ScanState.CONFIRMED:
            self._confirmed, self._confirmed_seen_at = result, now
        return result

    def _confirmed_marker_in_area(self, obs: ScanObservation) -> bool:
        marker = obs.marker
        if marker is None or self._confirmed is None or marker.marker_id != self._confirmed.marker_id:
            return False
        x1, y1, x2, y2 = scan_area(obs.frame_width, obs.frame_height, self.config)
        return x1 <= marker.center_px[0] <= x2 and y1 <= marker.center_px[1] <= y2

    def _evaluate(self, obs: ScanObservation, now: float) -> ScanResult:
        stages = {"marker": False, "product": False, "verified": False, "position": False, "steady": False}
        marker = obs.marker
        previous, self._last = self._last, (None if marker is None else (now, marker.marker_id, marker.center_px))

        if obs.marker_count == 0:
            return ScanResult(ScanState.SEARCHING, problem="no_marker", stages=stages)
        stages["marker"] = True
        if obs.marker_count > 1 or marker is None:
            return ScanResult(ScanState.MARKER_DETECTED, problem="multiple_markers", stages=stages)

        verification = self._with_grace(marker.marker_id, obs.verification, now) if obs.known else None
        seen = verification.detection if verification is not None else None
        positioning = analyze_position(
            marker, seen.bbox if seen else None, obs.frame_width, obs.frame_height, self.config, obs.mirrored
        )
        speed = None
        if previous is not None and previous[1] == marker.marker_id and now > previous[0]:
            side = sum(marker_sides(marker)) / 4
            speed = (math.dist(marker.center_px, previous[2]) / side) / ((now - previous[0]) / 1000.0)
        common = dict(
            marker_id=marker.marker_id, verification=verification, positioning=positioning, speed=speed, stages=stages
        )

        if not obs.known:
            return ScanResult(ScanState.MARKER_DETECTED, problem="unknown_marker", **common)
        if verification is None or verification.status is VerificationStatus.UNVERIFIABLE:
            return ScanResult(ScanState.MARKER_DETECTED, problem="unverifiable", **common)
        # A product that looks too far away is the likeliest reason the detector
        # can't see it (or isn't sure), so say that rather than just waiting.
        closer = ("move_closer",) if positioning.distance == "too_far" else ()
        if verification.status is VerificationStatus.NO_PRODUCT:
            return ScanResult(ScanState.MARKER_DETECTED, problem="no_product", hints=closer, **common)
        stages["product"] = True
        if verification.status is VerificationStatus.UNCERTAIN:
            return ScanResult(ScanState.VERIFYING, problem="uncertain", hints=closer, **common)
        if verification.status is VerificationStatus.MISMATCH:
            return ScanResult(ScanState.MISMATCH, problem="mismatch", **common)
        stages["verified"] = True

        if positioning.hints:
            return ScanResult(ScanState.POSITIONING, hints=positioning.hints, **common)
        stages["position"] = True

        if previous is not None and previous[1] != marker.marker_id:
            self._steady_since = None  # a different marker: its hold starts from scratch
        if speed is not None and speed > self.config.max_speed:
            self._steady_since = None
            return ScanResult(ScanState.HOLD_STEADY, problem="moving", hints=("hold_steady",), **common)
        stages["steady"] = True

        if self._steady_since is None:
            self._steady_since = now
            return ScanResult(ScanState.HOLD_STEADY, hints=("hold_steady",), **common)
        progress = min(1.0, (now - self._steady_since) / self.config.hold_ms)
        if progress >= 1.0:
            self._confirmations += 1
            return ScanResult(ScanState.CONFIRMED, progress=1.0, confirmation=self._confirmations, **common)
        return ScanResult(ScanState.SCANNING, hints=("hold_steady",), progress=progress, **common)

    def _with_grace(self, marker_id: int, verification: Verification | None, now: float) -> Verification | None:
        """A detector that blinks for a frame shouldn't restart the hold; a mismatch is never excused."""
        if verification is not None and verification.status is VerificationStatus.MATCH:
            self._last_match = (now, marker_id, verification)
        elif (
            verification is not None
            and verification.status in (VerificationStatus.NO_PRODUCT, VerificationStatus.UNCERTAIN)
            and self._last_match is not None
            and self._last_match[1] == marker_id
            and now - self._last_match[0] <= self.config.verification_grace_ms
        ):
            return self._last_match[2]
        return verification
