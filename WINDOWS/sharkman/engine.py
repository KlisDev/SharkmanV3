from __future__ import annotations

import math
from collections import deque

import numpy as np

from .config import CalibrationProfile, TimingConfig
from .models import EngineSnapshot, EngineState, FireAction, MeterObservation


class TimingEngine:
    """Deterministic one-prompt/one-action meter state machine."""

    # The FINISH target occupies roughly 13.5% of the meter in the supplied
    # recording. Size alone is not unique, so rearming also requires a partial
    # disappearance and strong growth from the target that was just fired.
    FINISH_ZONE_MIN_FRACTION = 0.12
    FINISH_ZONE_GROWTH_RATIO = 1.35
    RETRY_DEPARTURE_TRACK_FRACTION = 0.25

    def __init__(self, profile: CalibrationProfile) -> None:
        self.profile = profile
        self.timing: TimingConfig = profile.timing
        self.state = EngineState.IDLE
        self._history: deque[tuple[float, float]] = deque()
        self._zone_history: deque[tuple[float, float]] = deque()
        self._acquire_since: float | None = None
        self._lost_since: float | None = None
        self._gone_since: float | None = None
        self._meter_since: float | None = None
        self._fired_at: float | None = None
        self._last_action_at = -math.inf
        self._last_zone: tuple[int, int] | None = None
        self._last_zone_at = -math.inf
        self._scheduled_fire_at: float | None = None
        self._velocity = 0.0
        self._predicted: float | None = None
        self._previous_marker: float | None = None
        self._last_marker: float | None = None
        self._last_marker_at: float | None = None
        self._last_motion_at: float | None = None
        self._motion_sign = 0
        self._direction_streak = 0
        self._departed_since_fire = False
        self._zone_velocity = 0.0
        self._fired_zone_fraction: float | None = None
        self._last_zone_fraction = 0.0
        self._finish_since: float | None = None
        self._adaptive_prompt_kind = "normal"
        self.action_count = 0
        self.prompt_count = 0
        self.focused = False

    def reset(self) -> None:
        self.state = EngineState.IDLE
        self._history.clear()
        self._zone_history.clear()
        self._acquire_since = None
        self._lost_since = None
        self._gone_since = None
        self._meter_since = None
        self._fired_at = None
        self._last_zone = None
        self._last_zone_at = -math.inf
        self._scheduled_fire_at = None
        self._velocity = 0.0
        self._predicted = None
        self._previous_marker = None
        self._last_marker = None
        self._last_marker_at = None
        self._last_motion_at = None
        self._motion_sign = 0
        self._direction_streak = 0
        self._departed_since_fire = False
        self._zone_velocity = 0.0
        self._fired_zone_fraction = None
        self._last_zone_fraction = 0.0
        self._finish_since = None
        self._adaptive_prompt_kind = "normal"

    @property
    def adaptive_prompt_kind(self) -> str:
        """Broad re-entry prompts are excluded from post-hit learning."""
        return self._adaptive_prompt_kind

    def capture_unavailable(self) -> None:
        """Invalidate motion without inventing an absence or unlocking a fired meter."""
        self._scheduled_fire_at = None
        self._history.clear()
        self._zone_history.clear()
        self._last_zone = None
        self._last_zone_at = -math.inf
        self._velocity = self._zone_velocity = 0.0
        self._predicted = self._previous_marker = self._last_marker = None
        self._last_marker_at = None
        self._direction_streak = self._motion_sign = 0
        self._last_motion_at = None
        self._gone_since = self._lost_since = self._finish_since = None
        self._departed_since_fire = False
        if self.state == EngineState.ACQUIRING:
            self.reset()
        elif self.state == EngineState.ARMED:
            self.state = EngineState.TRACKING

    def _valid(self, observation: MeterObservation) -> bool:
        return observation.complete and observation.confidence >= self.profile.detection.required_confidence

    def _trim_history(self, now: float) -> None:
        cutoff = now - self.timing.velocity_window_ms / 1000.0
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def _push_marker(self, observation: MeterObservation) -> None:
        if observation.marker_center is None:
            return
        self._previous_marker = self._history[-1][1] if self._history else None
        if self._previous_marker is not None:
            delta = observation.marker_center - self._previous_marker
            if abs(delta) >= 2.0:
                self._last_motion_at = observation.timestamp
                sign = 1 if delta > 0 else -1
                if sign != self._motion_sign:
                    # Keep the last position but discard velocity from the old
                    # direction. A second moving sample must confirm a bounce.
                    last = self._history[-1]
                    self._history.clear()
                    self._history.append(last)
                    self._motion_sign = sign
                    self._direction_streak = 1
                else:
                    self._direction_streak += 1
        self._history.append((observation.timestamp, observation.marker_center))
        self._last_marker = observation.marker_center
        self._last_marker_at = observation.timestamp
        self._trim_history(observation.timestamp)
        self._velocity = self._regression_velocity() if self._direction_streak >= 2 else 0.0

    def _push_zone(self, observation: MeterObservation) -> None:
        if observation.zone_bounds is None or not self._valid(observation):
            return
        center = (observation.zone_bounds[0] + observation.zone_bounds[1]) / 2.0
        now = observation.timestamp
        if self._zone_history and self._zone_history[-1][0] == now:
            return
        if self._zone_history and (
            now - self._zone_history[-1][0] > self.timing.meter_loss_grace_ms / 1000.0
            or abs(center - self._zone_history[-1][1]) > 8.0
        ):
            self._zone_history.clear()
        self._zone_history.append((now, center))
        cutoff = now - self.timing.velocity_window_ms / 1000.0
        while self._zone_history and self._zone_history[0][0] < cutoff:
            self._zone_history.popleft()
        if len(self._zone_history) < 3:
            self._zone_velocity = 0.0
            return
        times = np.asarray([item[0] for item in self._zone_history], dtype=np.float64)
        values = np.asarray([item[1] for item in self._zone_history], dtype=np.float64)
        times -= times.mean()
        denominator = float(np.dot(times, times))
        speed = float(np.dot(times, values - values.mean()) / denominator) if denominator > 1e-9 else 0.0
        limit = self.timing.zone_velocity_limit_px_s
        self._zone_velocity = min(limit, max(-limit, speed))

    def _regression_velocity(self) -> float:
        if len(self._history) < 2:
            return 0.0
        times = np.asarray([item[0] for item in self._history], dtype=np.float64)
        values = np.asarray([item[1] for item in self._history], dtype=np.float64)
        times -= times.mean()
        denominator = float(np.dot(times, times))
        return float(np.dot(times, values - values.mean()) / denominator) if denominator > 1e-9 else 0.0

    def _safe_zone(self, zone: tuple[int, int]) -> tuple[float, float]:
        low, high = map(float, zone)
        inset = max(0.0, (high - low) * self.profile.detection.safe_zone_inset)
        safe_low, safe_high = low + inset, high - inset
        if safe_low > safe_high:
            return low, high
        return safe_low, safe_high

    @staticmethod
    def _target_eta(position: float, velocity: float, target: float) -> float:
        delta = target - position
        if abs(delta) <= 1e-6:
            return 0.0
        if delta * velocity > 0.0 and abs(velocity) > 1e-6:
            return delta / velocity
        return math.inf

    @staticmethod
    def _zone_fraction(observation: MeterObservation) -> float:
        if observation.zone_bounds is None or observation.track_bounds is None:
            return 0.0
        zone_height = observation.zone_bounds[1] - observation.zone_bounds[0] + 1
        track_height = observation.track_bounds[1] - observation.track_bounds[0] + 1
        return zone_height / track_height if track_height > 0 else 0.0

    def _far_from_zone(self, observation: MeterObservation) -> bool:
        if observation.marker_center is None or observation.zone_bounds is None or observation.track_bounds is None:
            return False
        zone_center = sum(observation.zone_bounds) / 2.0
        track_height = observation.track_bounds[1] - observation.track_bounds[0] + 1
        return abs(observation.marker_center - zone_center) >= track_height * self.RETRY_DEPARTURE_TRACK_FRACTION

    def _commit_action(
        self,
        timestamp: float,
        marker: float,
        predicted: float,
        safe_zone: tuple[float, float],
        reason: str,
    ) -> FireAction:
        action = FireAction(timestamp, marker, predicted, safe_zone, self._velocity, reason)
        self._last_action_at = timestamp
        self._fired_at = timestamp
        self.action_count += 1
        self.state = EngineState.FIRED
        self._scheduled_fire_at = None
        self._departed_since_fire = False
        return action

    def fire_scheduled(self, now: float) -> FireAction | None:
        """Commit a predicted center crossing without waiting for another frame."""
        due = self._scheduled_fire_at
        if due is not None and now - due > self.timing.precision_wait_threshold_ms / 1000.0:
            self._scheduled_fire_at = None
            return None
        if (
            due is None
            or now < due
            or now - self._last_action_at < self.timing.minimum_input_interval_ms / 1000.0
            or self.state not in (EngineState.TRACKING, EngineState.ARMED)
            or self._last_marker is None
            or self._last_marker_at is None
            or self._last_zone is None
            or now - self._last_marker_at > self.timing.zone_cache_ms / 1000.0
            or now - self._last_zone_at > self.timing.zone_cache_ms / 1000.0
        ):
            return None
        safe_zone = self._safe_zone(self._last_zone)
        elapsed = max(0.0, now - self._last_marker_at)
        marker = self._last_marker + self._velocity * elapsed
        latency = self.timing.input_latency_ms / 1000.0
        predicted = marker + self._velocity * latency
        zone_shift = self._zone_velocity * (elapsed + latency)
        projected_zone = (safe_zone[0] + zone_shift, safe_zone[1] + zone_shift)
        if not (projected_zone[0] <= predicted <= projected_zone[1]):
            self._scheduled_fire_at = None
            return None
        self._fired_zone_fraction = self._last_zone_fraction
        return self._commit_action(now, marker, predicted, projected_zone, "sub-frame center crossing")

    def _fire_if_due(self, observation: MeterObservation, zone: tuple[int, int]) -> FireAction | None:
        assert observation.marker_center is not None
        now = observation.timestamp
        marker = observation.marker_center
        safe_zone = self._safe_zone(zone)
        self._last_zone_fraction = self._zone_fraction(observation)
        latency = self.timing.input_latency_ms / 1000.0
        predicted = marker + self._velocity * latency
        self._predicted = predicted
        center_target = (safe_zone[0] + safe_zone[1]) / 2.0
        relative_velocity = self._velocity - self._zone_velocity
        center_eta = self._target_eta(marker, relative_velocity, center_target)
        lookahead = self.timing.lookahead_ms / 1000.0
        if self._direction_streak >= 2 and center_eta <= lookahead:
            center_fire_at = now + center_eta - latency
            self._scheduled_fire_at = center_fire_at if center_fire_at > now else None
        else:
            self._scheduled_fire_at = None

        predicted_zone = (
            safe_zone[0] + self._zone_velocity * latency,
            safe_zone[1] + self._zone_velocity * latency,
        )
        predicted_inside = predicted_zone[0] <= predicted <= predicted_zone[1]
        meter_age = 0.0 if self._meter_since is None else now - self._meter_since
        track_height = (
            observation.track_bounds[1] - observation.track_bounds[0] + 1
            if observation.track_bounds else 0
        )
        # Only genuinely broad targets need the startup guard. A 7% target can
        # already have a sub-frame crossing at high speed and must be allowed
        # to schedule immediately.
        wide_zone = track_height > 0 and (zone[1] - zone[0] + 1) / track_height >= 0.10
        startup_guard = self.timing.wide_zone_settle_ms / 1000.0
        if wide_zone and meter_age < startup_guard:
            self._scheduled_fire_at = None
            return None
        interval_ok = now - self._last_action_at >= self.timing.minimum_input_interval_ms / 1000.0
        if interval_ok and self._direction_streak >= 2 and predicted_inside:
            self._fired_zone_fraction = self._zone_fraction(observation)
            return self._commit_action(now, marker, predicted, predicted_zone, "latency-compensated crossing")
        return None

    def update(self, observation: MeterObservation, focused: bool) -> FireAction | None:
        now = observation.timestamp
        self.focused = focused
        valid = self._valid(observation)

        if not focused:
            if self.state not in (EngineState.IDLE, EngineState.WAIT_GONE):
                self.state = EngineState.WAIT_GONE
                self._gone_since = None
                self._scheduled_fire_at = None
                self._history.clear()
                self._zone_history.clear()
                self._zone_velocity = 0.0
                self._fired_zone_fraction = None
            return None

        if self.state == EngineState.IDLE:
            if not valid:
                return None
            self.state = EngineState.ACQUIRING
            self._acquire_since = now
            self._meter_since = now
            self._history.clear()
            self._push_marker(observation)
            self._push_zone(observation)
            self._last_zone = observation.zone_bounds
            self._last_zone_at = now
            return None

        if self.state == EngineState.ACQUIRING:
            if not valid:
                self.reset()
                self.focused = focused
                return None
            self._push_marker(observation)
            self._push_zone(observation)
            self._last_zone = observation.zone_bounds
            self._last_zone_at = now
            acquire_since = now if self._acquire_since is None else self._acquire_since
            if now - acquire_since < self.timing.acquire_confirm_ms / 1000.0:
                return None
            self.prompt_count += 1
            self.state = EngineState.TRACKING

        if self.state in (EngineState.TRACKING, EngineState.ARMED):
            if valid and not self._history and self._last_marker_at is None:
                # Capture downtime is unknown, not evidence of a frozen marker.
                self._last_motion_at = now
            if valid and observation.zone_bounds is not None:
                self._last_zone = observation.zone_bounds
                self._last_zone_at = now
                self._push_zone(observation)
            zone = self._last_zone
            zone_fresh = zone is not None and now - self._last_zone_at <= self.timing.zone_cache_ms / 1000.0
            if observation.present:
                self._lost_since = None
            elif self._lost_since is None:
                self._lost_since = now
            if self._lost_since is not None and now - self._lost_since > self.timing.meter_loss_grace_ms / 1000.0:
                self.reset()
                self.focused = focused
                return None
            if (
                observation.marker_center is not None
                and (not self._history or self._history[-1][0] != observation.timestamp)
            ):
                self._push_marker(observation)
            last_activity = self._last_motion_at if self._last_motion_at is not None else self._meter_since
            if last_activity is not None and now - last_activity > self.timing.max_meter_ms / 1000.0:
                self.state = EngineState.WAIT_GONE
                self._gone_since = None
                self._scheduled_fire_at = None
                return None
            if not valid or not zone_fresh or zone is None or observation.marker_center is None:
                # A predicted press is valid only while the newest capture
                # still confirms this exact meter. Never fire from a cache
                # after an occluded or low-confidence frame.
                self._scheduled_fire_at = None
                return None
            self.state = EngineState.ARMED
            return self._fire_if_due(observation, zone)

        if self.state == EngineState.FIRED:
            fired_at = now if self._fired_at is None else self._fired_at
            if now - fired_at >= self.timing.post_fire_lockout_ms / 1000.0:
                self.state = EngineState.WAIT_GONE
                self._gone_since = None
            return None

        if self.state == EngineState.WAIT_GONE:
            if observation.present:
                fraction = self._zone_fraction(observation)
                fired_fraction = self._fired_zone_fraction or 0.0
                fired_at = self._fired_at
                finish_window_open = (
                    fired_at is not None
                    and 0 <= now - fired_at <= self.timing.finish_transition_window_ms / 1000.0
                    and self._fired_zone_fraction is not None
                    and fired_fraction <= 0.08
                )
                partial_gap_confirmed = (
                    self._finish_since is not None
                    or (
                        self._gone_since is not None
                        and now - self._gone_since >= self.timing.finish_confirm_ms / 1000.0
                    )
                )
                finish_like = (
                    valid
                    and finish_window_open
                    and partial_gap_confirmed
                    and fraction >= self.FINISH_ZONE_MIN_FRACTION
                    and fraction >= fired_fraction * self.FINISH_ZONE_GROWTH_RATIO
                )
                if finish_like:
                    if self._finish_since is None:
                        self._finish_since = now
                        self._history.clear()
                        self._zone_history.clear()
                        self._zone_velocity = 0.0
                    self._push_marker(observation)
                    self._push_zone(observation)
                    if now - self._finish_since >= self.timing.finish_confirm_ms / 1000.0:
                        fired_at = self._fired_at if self._fired_at is not None else now
                        prompt_gap_ms = (now - fired_at) * 1000.0
                        self._adaptive_prompt_kind = (
                            "finish"
                            if prompt_gap_ms <= self.timing.finish_transition_window_ms
                            and fired_fraction <= 0.08
                            and fraction >= 0.12
                            and fraction >= fired_fraction * 1.5
                            else "ambiguous"
                        )
                        self.prompt_count += 1
                        self.state = EngineState.TRACKING
                        self._meter_since = self._finish_since
                        self._fired_at = None
                        self._last_zone = observation.zone_bounds
                        self._last_zone_at = now
                        return self._fire_if_due(observation, observation.zone_bounds)
                else:
                    self._finish_since = None
                    if valid and observation.marker_center is not None:
                        prior_sign = self._motion_sign
                        prior_streak = self._direction_streak
                        self._push_marker(observation)
                        self._push_zone(observation)
                        far = self._far_from_zone(observation)
                        if far:
                            self._departed_since_fire = True
                        bounced = (
                            prior_sign != 0
                            and self._motion_sign != prior_sign
                            and prior_streak >= 2
                        )
                        fired_at = self._fired_at if self._fired_at is not None else now
                        lockout_elapsed = now - fired_at >= self.timing.post_fire_lockout_ms / 1000.0
                        retry_delay_elapsed = now - fired_at >= self.timing.same_gauge_retry_ms / 1000.0
                        if self._departed_since_fire and far and bounced and lockout_elapsed and retry_delay_elapsed:
                            # The previous press did not clear the gauge. This
                            # is a new traversal, not a duplicate on the same
                            # pass, so allow one fresh prediction.
                            self.prompt_count += 1
                            self._adaptive_prompt_kind = "normal"
                            self.state = EngineState.TRACKING
                            self._meter_since = now
                            self._fired_at = None
                            self._fired_zone_fraction = None
                            self._departed_since_fire = False
                            self._last_zone = observation.zone_bounds
                            self._last_zone_at = now
                            self._scheduled_fire_at = None
                # The game's FINISH transition can briefly show a narrow or
                # incomplete target before the broad one. Preserve the
                # preceding disappearance through those transition frames;
                # clearing it here used to force a 3.5 s same-gauge retry.
                if not finish_window_open:
                    self._gone_since = None
                return None
            if self._gone_since is None:
                self._gone_since = now
                self._finish_since = None
                return None
            if now - self._gone_since >= self.timing.rearm_invisible_ms / 1000.0:
                self.reset()
                self.focused = focused
            return None
        return None

    def snapshot(self) -> EngineSnapshot:
        return EngineSnapshot(
            state=self.state,
            velocity=self._velocity,
            predicted_center=self._predicted,
            scheduled_fire_at=self._scheduled_fire_at,
            action_count=self.action_count,
            prompt_count=self.prompt_count,
            focused=self.focused,
        )
