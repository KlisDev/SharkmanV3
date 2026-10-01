"""Conservative, meter-only post-press feedback and latency learning.

Visual centering is experimental: a frozen bar is a geometric observation,
not proof that the game registered a critical hit. No pixels enter profiles.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from .config import CalibrationProfile, TimingConfig
from .models import FireAction, MeterObservation

MODEL_PATH = Path(__file__).resolve().parent / "assets" / "adaptive" / "feedback_model.json"
# Live recordings on 2026-09-26 and 2026-09-27 yielded no reliable same-target
# frozen positions: the meter changed or vanished before feedback could settle.
# Keep the prototype code for offline investigation, but never enable learning
# or altered firing from a saved profile until a different signal is validated.
LIVE_ADAPTIVE_FEEDBACK_SUPPORTED = False
CENTER_FEEDBACK_VERSION = "visual-center-v1"
CENTER_DEADBAND_FRACTION = 0.10
CENTER_MAX_POSITION = 0.90
MIN_FEEDBACK_CONFIDENCE = 0.80
FREEZE_POSITION_SPREAD_PX = 1.5
FREEZE_ZONE_SPREAD_PX = 1.5
MIN_LEARNING_VELOCITY_PX_S = 50.0
EVIDENCE_LIMIT = 128
LEARNING_BATCH = 6


@dataclass(frozen=True)
class FeedbackModel:
    version: str
    critical_center: float
    critical_edge: float
    holdout_count: int
    holdout_accuracy: float

    @property
    def validated(self) -> bool:
        return (
            self.holdout_count >= 6
            and self.holdout_accuracy >= 0.95
            and -0.5 <= self.critical_center <= 0.5
            and 0.05 <= self.critical_edge < 1.0
        )

    def classify(self, position: float) -> str:
        if not math.isfinite(position):
            return "unknown"
        distance = abs(position - self.critical_center)
        # Leave an uncertainty band at both boundaries rather than silently
        # turning a noisy crop into a miss or an apparently perfect hit.
        if distance <= self.critical_edge - 0.03:
            return "critical"
        if abs(position) > 1.08:
            return "miss"
        if abs(position) <= 0.96 and distance >= self.critical_edge + 0.03:
            return "edge"
        return "unknown"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackModel:
        model = cls(
            version=str(data["version"]),
            critical_center=float(data["critical_center"]),
            critical_edge=float(data["critical_edge"]),
            holdout_count=int(data["holdout_count"]),
            holdout_accuracy=float(data["holdout_accuracy"]),
        )
        if not model.validated:
            raise ValueError("feedback classifier has not passed held-out validation")
        return model


def load_feedback_model(path: Path = MODEL_PATH) -> FeedbackModel | None:
    try:
        return FeedbackModel.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def normalized_landing(marker: float, zone: tuple[float, float]) -> float:
    half_width = (zone[1] - zone[0]) / 2.0
    if half_width <= 0:
        return math.nan
    return (marker - (zone[0] + zone[1]) / 2.0) / half_width


@dataclass(frozen=True)
class FeedbackResult:
    outcome: str
    reason: str
    marker: float | None = None
    zone: tuple[float, float] | None = None
    normalized_position: float | None = None
    velocity: float = 0.0
    trial_offset_ms: float = 0.0


@dataclass
class _Pending:
    action: FireAction
    began_at: float
    trial_offset_ms: float
    source_zone: tuple[float, float] | None = None
    samples: list[MeterObservation] = field(default_factory=list)
    last_at: float | None = None


class FeedbackCollector:
    """Return a result only for a confirmed, stable post-input gauge."""

    def __init__(self, timing: TimingConfig) -> None:
        self.timing = timing
        self.pending: _Pending | None = None

    def begin(
        self, action: FireAction, pressed_at: float, *, trial_offset_ms: float = 0,
        source_zone: tuple[float, float] | None = None,
    ) -> None:
        self.pending = _Pending(action, pressed_at, trial_offset_ms, source_zone)

    def cancel(self) -> None:
        self.pending = None

    def _finish(self, result: FeedbackResult) -> FeedbackResult:
        self.pending = None
        return result

    def observe(self, observation: MeterObservation, *, focused: bool) -> FeedbackResult | None:
        pending = self.pending
        if pending is None:
            return None
        age_ms = (observation.timestamp - pending.began_at) * 1000.0
        if not focused:
            return self._finish(FeedbackResult("unknown", "Focus lost after Space."))
        if age_ms > self.timing.feedback_timeout_ms:
            return self._finish(FeedbackResult("unknown", "No stable post-hit gauge before timeout."))
        if age_ms < self.timing.feedback_start_ms:
            return None
        if pending.last_at is not None and (
            observation.timestamp - pending.last_at
        ) * 1000.0 > self.timing.feedback_max_frame_gap_ms:
            return self._finish(FeedbackResult("unknown", "Feedback capture gap was too long."))
        pending.last_at = observation.timestamp
        if not observation.complete or observation.confidence < MIN_FEEDBACK_CONFIDENCE:
            return self._finish(FeedbackResult("unknown", "Meter was hidden or uncertain after Space."))
        if pending.source_zone is not None and observation.zone_bounds is not None:
            source_height = pending.source_zone[1] - pending.source_zone[0] + 1
            current_height = observation.zone_bounds[1] - observation.zone_bounds[0] + 1
            source_center = sum(pending.source_zone) / 2.0
            current_center = sum(observation.zone_bounds) / 2.0
            if (
                abs(current_height - source_height) > max(3.0, source_height * 0.2)
                or abs(current_center - source_center) > max(3.0, source_height * 0.2)
            ):
                return self._finish(FeedbackResult("unknown", "Target changed before feedback settled."))
        pending.samples.append(observation)
        # 240 Hz needs at least 13 frames to establish a 50 ms freeze.
        if len(pending.samples) > 64:
            pending.samples.pop(0)
        if len(pending.samples) < 3:
            return None
        markers = [item.marker_center for item in pending.samples if item.marker_center is not None]
        lows = [item.zone_bounds[0] for item in pending.samples if item.zone_bounds is not None]
        highs = [item.zone_bounds[1] for item in pending.samples if item.zone_bounds is not None]
        if (
            max(markers) - min(markers) > FREEZE_POSITION_SPREAD_PX
            or max(lows) - min(lows) > FREEZE_ZONE_SPREAD_PX
            or max(highs) - min(highs) > FREEZE_ZONE_SPREAD_PX
        ):
            pending.samples[:] = pending.samples[-2:]
            return None
        if (pending.samples[0].timestamp - pending.began_at) * 1000.0 > self.timing.feedback_max_settle_ms:
            return self._finish(FeedbackResult("unknown", "Marker settled too late to attribute to Space."))
        elapsed_ms = (pending.samples[-1].timestamp - pending.samples[0].timestamp) * 1000.0
        if elapsed_ms < self.timing.feedback_freeze_confirm_ms:
            return None
        marker = float(median(markers))
        zone = (float(median(lows)), float(median(highs)))
        position = normalized_landing(marker, zone)
        if not math.isfinite(position):
            return self._finish(FeedbackResult("unknown", "Post-hit green bounds were invalid."))
        return self._finish(FeedbackResult(
            "measured", "Stable post-press marker measured; game hit grade is unknown.", marker, zone,
            position, pending.action.velocity, pending.trial_offset_ms,
        ))


def propose_center_learning(
    profile: CalibrationProfile, result: FeedbackResult, timing: TimingConfig,
) -> tuple[CalibrationProfile, float | None]:
    """Move saved latency toward the visual green center, never infer hit grade.

    Six eligible normal prompts supply a robust median signed error. Small
    errors produce small changes, with a center deadband and a 3 ms cap.
    Persistence remains the caller's responsibility before applying a change.
    """
    candidate = CalibrationProfile.from_dict(profile.to_dict())
    state = candidate.adaptive
    old_latency = candidate.timing.input_latency_ms
    signature = candidate.calibration_signature()
    if state.model_version != CENTER_FEEDBACK_VERSION or state.calibration_signature != signature:
        state.clear(old_latency)
        state.enabled = profile.adaptive.enabled
        state.model_version = CENTER_FEEDBACK_VERSION
        state.calibration_signature = signature
    if state.baseline_latency_ms is None:
        state.baseline_latency_ms = old_latency
    state.last_outcome = "unknown"
    state.last_reason = result.reason
    position = result.normalized_position
    if (
        result.outcome != "measured" or result.marker is None or result.zone is None
        or position is None or not math.isfinite(position)
        or abs(position) > CENTER_MAX_POSITION
        or abs(result.velocity) < MIN_LEARNING_VELOCITY_PX_S
        or result.trial_offset_ms != 0
    ):
        if result.outcome == "measured":
            state.last_reason = "Frozen position was outside reliable green bounds, too slow, or a trial. No learning."
        return candidate, None
    error_ms = (result.marker - sum(result.zone) / 2.0) / result.velocity * 1000.0
    if not math.isfinite(error_ms) or abs(error_ms) > 250:
        state.last_reason = "Visual timing error was implausible. No learning."
        return candidate, None
    state.last_outcome = "centered" if abs(position) <= CENTER_DEADBAND_FRACTION else "off-center"
    state.last_reason = f"Frozen marker {position:+.2f} half-band widths from center; game hit grade unknown."
    state.evidence.append({
        "position": round(position, 4), "error_ms": round(error_ms, 3),
        "velocity": round(result.velocity, 2),
    })
    state.evidence[:] = state.evidence[-EVIDENCE_LIMIT:]
    state.sample_count += 1
    state.samples_since_adjustment += 1
    if state.sample_count < LEARNING_BATCH or state.sample_count % LEARNING_BATCH:
        return candidate, None
    batch = state.evidence[-LEARNING_BATCH:]
    median_position = float(median(float(item["position"]) for item in batch))
    if abs(median_position) <= CENTER_DEADBAND_FRACTION:
        state.last_reason += " Center deadband; no adjustment."
        return candidate, None
    correction = float(median(float(item["error_ms"]) for item in batch))
    # Scale by both time error and *visual* distance. A slow marker 0.11
    # half-band widths from center must not receive the full 3 ms correction
    # merely because that small pixel offset represents a long travel time.
    positional_cap = timing.adaptive_max_step_ms * min(1.0, abs(median_position) / 0.5)
    step = max(-positional_cap, min(positional_cap, correction * 0.5))
    baseline = state.baseline_latency_ms
    low = max(0.0, baseline - timing.adaptive_max_drift_ms)
    high = min(500.0, baseline + timing.adaptive_max_drift_ms)
    new_latency = round(max(low, min(high, old_latency + step)), 2)
    if abs(new_latency - old_latency) < 0.1:
        state.last_reason += " Correction below 0.1 ms; no adjustment."
        return candidate, None
    state.previous_latency_ms = old_latency
    state.samples_since_adjustment = 0
    candidate.timing_overrides = {**candidate.timing_overrides, "input_latency_ms": new_latency}
    state.last_reason += f" Saved latency {old_latency:g} → {new_latency:g} ms."
    return candidate, new_latency


def propose_learning(
    profile: CalibrationProfile, result: FeedbackResult,
    model: FeedbackModel, timing: TimingConfig,
) -> tuple[CalibrationProfile, float | None]:
    """Pure proposal: caller must persist successfully before applying it."""
    candidate = CalibrationProfile.from_dict(profile.to_dict())
    state = candidate.adaptive
    old_latency = candidate.timing.input_latency_ms
    if state.baseline_latency_ms is None:
        state.baseline_latency_ms = old_latency
    if state.model_version != model.version or state.calibration_signature != candidate.calibration_signature():
        state.clear(old_latency)
        state.enabled = profile.adaptive.enabled
        state.model_version = model.version
        state.calibration_signature = candidate.calibration_signature()
    state.last_outcome = result.outcome
    state.last_reason = result.reason
    if result.outcome == "unknown" or result.marker is None or result.zone is None:
        return candidate, None
    outcome = model.classify(result.normalized_position if result.normalized_position is not None else math.nan)
    state.last_outcome = outcome
    if outcome == "unknown" or abs(result.velocity) < MIN_LEARNING_VELOCITY_PX_S:
        state.last_reason = "Landing was ambiguous or marker speed was too low."
        return candidate, None
    target_y = (result.zone[0] + result.zone[1]) / 2.0 + model.critical_center * (result.zone[1] - result.zone[0]) / 2.0
    error_ms = (result.marker - target_y) / result.velocity * 1000.0
    if not math.isfinite(error_ms) or abs(error_ms) > 250:
        state.last_reason = "Landing error was implausible."
        return candidate, None
    state.last_reason = f"{outcome.capitalize()} landing at {result.normalized_position:+.2f} zone widths."
    state.evidence.append({
        "outcome": outcome, "position": round(result.normalized_position, 4),
        "error_ms": round(error_ms, 3), "trial_ms": result.trial_offset_ms,
        "velocity": round(result.velocity, 2),
    })
    state.evidence[:] = state.evidence[-EVIDENCE_LIMIT:]
    state.sample_count += 1
    if abs(result.trial_offset_ms) > 0:
        state.trial_sample_count += 1
    state.samples_since_adjustment += 1
    if state.sample_count < LEARNING_BATCH or state.sample_count % LEARNING_BATCH:
        return candidate, None
    batch = state.evidence[-LEARNING_BATCH:]
    trials = [item for item in batch if abs(float(item.get("trial_ms", 0))) > 0]
    if trials:
        # A trial is only adopted after comparable offsets in both directions.
        # Never fold an exploratory result into an ordinary correction batch.
        positive = [item for item in trials if float(item["trial_ms"]) > 0]
        negative = [item for item in trials if float(item["trial_ms"]) < 0]
        if len(positive) < 2 or len(negative) < 2:
            state.last_reason += " Waiting for balanced trial results."
            return candidate, None
        def trial_quality(items: list[dict[str, Any]]) -> float:
            return float(median(
                abs(float(item["position"]) - model.critical_center)
                + (1.0 if item["outcome"] == "miss" else 0.0)
                for item in items
            ))
        plus_quality = trial_quality(positive)
        minus_quality = trial_quality(negative)
        baseline = state.evidence[-2 * LEARNING_BATCH:-LEARNING_BATCH]
        if len(baseline) < LEARNING_BATCH or any(float(item.get("trial_ms", 0)) != 0 for item in baseline):
            state.last_reason += " No comparable unmodified baseline."
            return candidate, None
        baseline_quality = trial_quality(baseline)
        if abs(plus_quality - minus_quality) < 0.08:
            state.last_reason += " Trial directions were not distinguishable."
            return candidate, None
        better = positive if plus_quality < minus_quality else negative
        worse = negative if better is positive else positive
        if trial_quality(better) > baseline_quality - 0.08:
            state.last_reason += " Neither trial improved the unmodified baseline."
            return candidate, None
        if sum(item["outcome"] == "miss" for item in better) > sum(item["outcome"] == "miss" for item in worse):
            state.last_reason += " Better-position trial had more misses; no change."
            return candidate, None
        step = float(median(float(item["trial_ms"]) for item in better))
        step = max(-timing.adaptive_max_step_ms, min(timing.adaptive_max_step_ms, step))
    else:
        corrections = [float(item["error_ms"]) for item in batch]
        correction = float(median(corrections))
        if abs(correction) < 1.0:
            return candidate, None
        step = max(-timing.adaptive_max_step_ms, min(timing.adaptive_max_step_ms, correction * 0.5))
    baseline = state.baseline_latency_ms
    low = max(0.0, baseline - timing.adaptive_max_drift_ms)
    high = min(500.0, baseline + timing.adaptive_max_drift_ms)
    new_latency = round(max(low, min(high, old_latency + step)), 2)
    if abs(new_latency - old_latency) < 0.5:
        return candidate, None
    state.previous_latency_ms = old_latency
    state.samples_since_adjustment = 0
    candidate.timing_overrides = {**candidate.timing_overrides, "input_latency_ms": new_latency}
    state.last_reason += f" Latency {old_latency:g} → {new_latency:g} ms."
    return candidate, new_latency
