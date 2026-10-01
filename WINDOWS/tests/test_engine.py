from __future__ import annotations

from sharkman.engine import TimingEngine
from sharkman.models import EngineState, MeterObservation

from .helpers import calibrated_profile


def observed(
    t: float,
    marker: float | None,
    *,
    present: bool = True,
    zone: tuple[int, int] = (100, 140),
    track: tuple[int, int] = (0, 239),
) -> MeterObservation:
    return MeterObservation(
        timestamp=t,
        present=present,
        confidence=1.0 if present else 0.0,
        marker_center=marker,
        marker_span=None if marker is None else (int(marker - 2), int(marker + 2)),
        zone_bounds=zone if present else None,
        track_bounds=track if present else None,
    )


def test_one_action_until_meter_has_disappeared_long_enough() -> None:
    profile = calibrated_profile()
    profile.timing_overrides = {"wide_zone_settle_ms": 0}
    engine = TimingEngine(profile)
    actions = []
    for t, marker in ((0.0, 40), (0.02, 60), (0.04, 75), (0.06, 90), (0.08, 105)):
        action = engine.update(observed(t, marker), focused=True)
        if action:
            actions.append(action)
    assert len(actions) == 1
    assert engine.state in (EngineState.FIRED, EngineState.WAIT_GONE)
    for t in (0.12, 0.18, 0.30):
        assert engine.update(observed(t, 115), focused=True) is None
    assert len(actions) == 1

    # Disappearance must persist for the configured rearm interval.
    engine.update(observed(0.32, None, present=False), focused=True)
    engine.update(observed(0.50, None, present=False), focused=True)
    assert engine.state == EngineState.IDLE
    second = []
    for t, marker in ((0.60, 50), (0.62, 70), (0.64, 90)):
        action = engine.update(observed(t, marker), focused=True)
        if action:
            second.append(action)
    assert len(second) == 1


def test_focus_loss_blocks_and_requires_meter_to_clear() -> None:
    profile = calibrated_profile()
    engine = TimingEngine(profile)
    engine.update(observed(1.0, 60), focused=True)
    assert engine.update(observed(1.02, 110), focused=False) is None
    assert engine.state == EngineState.WAIT_GONE
    assert engine.update(observed(1.04, 120), focused=True) is None


def test_late_inter_frame_crossing_does_not_send_an_instant_press() -> None:
    profile = calibrated_profile()
    engine = TimingEngine(profile)
    assert engine.update(observed(2.0, 160), focused=True) is None
    assert engine.update(observed(2.02, 80), focused=True) is None
    assert engine.update(observed(2.04, 40), focused=True) is None
    assert engine.action_count == 0


def test_reported_instant_fire_traces_are_suppressed() -> None:
    # Sanitized marker/target coordinates from 0917(2).mp4, around the two
    # visibly late instant inputs. No screenshot or other game data is stored.
    traces = (
        ((271.5, (282, 298)), (299.0, (282, 291)), (340.0, (282, 299))),
        ((219.5, (242, 253)), (228.0, (242, 253)), (263.5, (246, 257)), (286.0, (246, 257))),
    )
    for trace in traces:
        engine = TimingEngine(calibrated_profile())
        for frame, (marker, zone) in enumerate(trace):
            assert engine.update(observed(frame / 30.0, marker, zone=zone), True) is None
        assert engine.action_count == 0


def test_wide_zone_guard_blocks_only_the_speculative_startup_press() -> None:
    profile = calibrated_profile()
    engine = TimingEngine(profile)

    assert engine.update(observed(3.00, 350, zone=(100, 180)), True) is None
    assert engine.update(observed(3.02, 300, zone=(100, 180)), True) is None

    action = engine.update(observed(3.12, 160, zone=(100, 180)), True)

    assert action is not None
    assert action.timestamp == 3.12


def test_sub_frame_schedule_targets_the_center_between_capture_frames() -> None:
    profile = calibrated_profile()
    engine = TimingEngine(profile)

    assert engine.update(observed(4.00, 40, zone=(100, 106)), True) is None
    assert engine.update(observed(4.02, 50, zone=(100, 106)), True) is None
    assert engine.update(observed(4.04, 60, zone=(100, 106)), True) is None

    scheduled_at = engine.snapshot().scheduled_fire_at
    assert scheduled_at is not None
    action = engine.fire_scheduled(scheduled_at)

    assert action is not None
    assert action.reason == "sub-frame center crossing"
    assert abs(action.predicted_center - 103.0) < 0.01


def test_saved_adaptive_toggle_does_not_change_firing_behavior() -> None:
    for enabled in (False, True):
        profile = calibrated_profile()
        profile.adaptive.enabled = enabled
        profile.timing_overrides = {"wide_zone_settle_ms": 0}
        engine = TimingEngine(profile)
        for stamp, marker in ((0.00, 80), (0.02, 84), (0.04, 88)):
            assert engine.update(observed(stamp, marker, zone=(100, 140)), True) is None
        action = engine.update(observed(0.10, 100, zone=(100, 140)), True)
        assert action is not None and action.reason == "latency-compensated crossing"


def test_default_wait_covers_a_crossing_20_ms_after_the_last_frame() -> None:
    profile = calibrated_profile()
    profile.timing_overrides = {"input_latency_ms": 0}
    engine = TimingEngine(profile)
    for t, marker in ((0.00, 40), (0.02, 55), (0.04, 70)):
        assert engine.update(observed(t, marker, zone=(83, 87)), True) is None
    due = engine.snapshot().scheduled_fire_at
    assert due is not None
    assert 0.016 < due - 0.04 <= engine.timing.precision_wait_threshold_ms / 1000.0
    assert engine.fire_scheduled(due) is not None


def test_scheduled_press_is_cancelled_by_a_bad_capture() -> None:
    engine = TimingEngine(calibrated_profile())
    for t, marker in ((4.00, 40), (4.02, 50), (4.04, 60)):
        engine.update(observed(t, marker, zone=(100, 106)), True)
    due = engine.snapshot().scheduled_fire_at
    assert due is not None
    engine.update(observed(4.05, None, present=False), True)
    assert engine.snapshot().scheduled_fire_at is None
    assert engine.fire_scheduled(due) is None
    assert engine.action_count == 0


def test_stale_scheduled_press_cannot_fire_after_its_window() -> None:
    engine = TimingEngine(calibrated_profile())
    for t, marker in ((4.00, 40), (4.02, 50), (4.04, 60)):
        engine.update(observed(t, marker, zone=(100, 106)), True)
    due = engine.snapshot().scheduled_fire_at
    assert due is not None
    assert engine.fire_scheduled(due + engine.timing.precision_wait_threshold_ms / 1000 + 0.001) is None
    assert engine.action_count == 0


def test_visible_gauge_retries_only_after_delay_and_a_full_bounce() -> None:
    profile = calibrated_profile()
    profile.timing_overrides = {
        "input_latency_ms": 0,
        "minimum_input_interval_ms": 0,
        "post_fire_lockout_ms": 20,
        "same_gauge_retry_ms": 500,
    }
    engine = TimingEngine(profile)
    for t, marker in ((0.00, 40), (0.02, 70)):
        assert engine.update(observed(t, marker, zone=(100, 106)), True) is None
    assert engine.update(observed(0.04, 103, zone=(100, 106)), True) is not None

    # The first bounce is well outside the zone, but is too soon to be a
    # retry; it could still be the game's post-hit animation.
    for t, marker in (
        (0.06, 130), (0.08, 210), (0.10, 240), (0.12, 250),
        (0.14, 240), (0.16, 200), (0.20, 100), (0.22, 40),
        (0.24, 30), (0.26, 40), (0.30, 100), (0.54, 250),
    ):
        assert engine.update(observed(t, marker, zone=(100, 106)), True) is None
    assert engine.state == EngineState.WAIT_GONE
    assert engine.action_count == 1

    assert engine.update(observed(0.56, 240, zone=(100, 106)), True) is None
    assert engine.state == EngineState.TRACKING
    for t, marker in ((0.58, 220), (0.60, 180), (0.62, 140)):
        assert engine.update(observed(t, marker, zone=(100, 106)), True) is None
    assert engine.update(observed(0.64, 103, zone=(100, 106)), True) is not None
    assert engine.action_count == 2


def test_active_meter_does_not_expire_while_marker_keeps_moving() -> None:
    profile = calibrated_profile()
    profile.timing_overrides = {"max_meter_ms": 500}
    engine = TimingEngine(profile)
    for step in range(12):
        t = step * 0.1
        marker = 140 + (step % 3) * 30
        engine.update(observed(t, marker, zone=(0, 10)), True)
    assert engine.state == EngineState.ARMED
    assert engine.action_count == 0


def test_frozen_meter_still_times_out_without_input() -> None:
    profile = calibrated_profile()
    profile.timing_overrides = {"max_meter_ms": 500}
    engine = TimingEngine(profile)
    for t in (0.0, 0.1, 0.6):
        assert engine.update(observed(t, 150, zone=(0, 10)), True) is None
    assert engine.state == EngineState.WAIT_GONE
    assert engine.action_count == 0


def test_direction_reversal_discards_stale_velocity_before_rearming_prediction() -> None:
    profile = calibrated_profile()
    engine = TimingEngine(profile)

    for timestamp, marker in ((4.00, 180), (4.02, 200), (4.04, 220)):
        engine.update(observed(timestamp, marker, zone=(20, 40)), True)
    assert engine.snapshot().velocity > 0

    engine.update(observed(4.06, 210, zone=(20, 40)), True)
    assert engine.snapshot().velocity == 0

    engine.update(observed(4.08, 195, zone=(20, 40)), True)
    assert engine.snapshot().velocity < 0


def test_large_finish_transition_rearms_without_meter_disappearance() -> None:
    profile = calibrated_profile()
    profile.timing_overrides = {
        "post_fire_lockout_ms": 20,
        "finish_confirm_ms": 100,
        "input_latency_ms": 0,
        "minimum_input_interval_ms": 0,
    }
    engine = TimingEngine(profile)

    assert engine.update(observed(5.00, 40, zone=(100, 106)), True) is None
    assert engine.update(observed(5.01, 70, zone=(100, 106)), True) is None
    first = engine.update(observed(5.02, 103, zone=(100, 106)), True)
    assert first is not None
    assert engine.update(observed(5.05, 130, zone=(100, 106)), True) is None
    assert engine.state == EngineState.WAIT_GONE

    # A normal lingering target does not bypass duplicate protection.
    assert engine.update(observed(5.08, 150, zone=(100, 106)), True) is None
    assert engine.state == EngineState.WAIT_GONE

    # FINISH briefly clears the prior target, but not for the full ordinary
    # rearm interval. Its subsequent full-size target is confirmed separately.
    assert engine.update(observed(5.10, None, present=False), True) is None
    assert engine.update(observed(5.22, 170, zone=(100, 140)), True) is None
    assert engine.update(observed(5.27, 145, zone=(100, 140)), True) is None
    finish = engine.update(observed(5.33, 120, zone=(100, 140)), True)
    assert finish is not None
    assert engine.adaptive_prompt_kind == "finish"
    assert engine.prompt_count == 2
    assert engine.action_count == 2


def test_partial_finish_animation_does_not_erase_disappearance_gap() -> None:
    # Shape adapted from the 2026-09-26 run: the old tiny target vanishes,
    # a sub-threshold partial target appears, then the full FINISH target.
    profile = calibrated_profile()
    profile.timing_overrides = {
        "post_fire_lockout_ms": 20,
        "finish_confirm_ms": 100,
        "input_latency_ms": 0,
        "minimum_input_interval_ms": 0,
        "same_gauge_retry_ms": 3500,
    }
    engine = TimingEngine(profile)
    def seen(t, marker, zone=(100, 106)):
        return observed(t, marker, zone=zone, track=(0, 564))
    assert engine.update(seen(5.00, 40), True) is None
    assert engine.update(seen(5.01, 70), True) is None
    assert engine.update(seen(5.02, 103), True) is not None
    engine.update(seen(5.05, 130), True)
    engine.update(observed(5.10, None, present=False), True)
    engine.update(observed(5.18, None, present=False), True)
    assert engine.update(seen(5.19, 185, (100, 146)), True) is None
    assert engine.update(seen(5.22, 170, (100, 193)), True) is None
    assert engine.update(seen(5.28, 145, (100, 193)), True) is None
    finish = engine.update(seen(5.34, 120, (100, 193)), True)
    assert finish is not None
    assert engine.adaptive_prompt_kind == "finish"
    assert finish.timestamp - 5.02 < 0.5


def test_finish_gap_just_under_confirmation_survives_to_next_frame() -> None:
    profile = calibrated_profile()
    profile.timing_overrides = {
        "post_fire_lockout_ms": 20, "finish_confirm_ms": 100,
        "input_latency_ms": 0, "minimum_input_interval_ms": 0,
    }
    engine = TimingEngine(profile)
    def seen(t, marker, zone=(100, 106)):
        return observed(t, marker, zone=zone, track=(0, 564))
    for t, marker in ((6.00, 40), (6.01, 70), (6.02, 103)):
        engine.update(seen(t, marker), True)
    assert engine.action_count == 1
    engine.update(seen(6.05, 130), True)
    engine.update(observed(6.10, None, present=False), True)
    assert engine.update(seen(6.199, 180, (100, 193)), True) is None
    assert engine.update(seen(6.23, 165, (100, 193)), True) is None
    assert engine.update(seen(6.28, 145, (100, 193)), True) is None
    finish = engine.update(seen(6.35, 120, (100, 193)), True)
    assert finish is not None
    assert engine.adaptive_prompt_kind == "finish"


def test_large_target_without_disappearance_does_not_rearm_early() -> None:
    profile = calibrated_profile()
    profile.timing_overrides = {
        "post_fire_lockout_ms": 20, "input_latency_ms": 0,
        "minimum_input_interval_ms": 0,
    }
    engine = TimingEngine(profile)
    def seen(t, marker, zone=(100, 106)):
        return observed(t, marker, zone=zone, track=(0, 564))
    for t, marker in ((7.00, 40), (7.01, 70), (7.02, 103)):
        engine.update(seen(t, marker), True)
    assert engine.action_count == 1
    for t, marker in ((7.06, 130), (7.12, 160), (7.20, 170), (7.32, 120)):
        assert engine.update(seen(t, marker, (100, 193)), True) is None
    assert engine.state == EngineState.WAIT_GONE
    assert engine.action_count == 1


def test_tiny_target_after_partial_gap_is_not_mistaken_for_finish() -> None:
    profile = calibrated_profile()
    profile.timing_overrides = {
        "post_fire_lockout_ms": 20, "input_latency_ms": 0,
        "minimum_input_interval_ms": 0,
    }
    engine = TimingEngine(profile)
    def seen(t, marker, zone=(100, 106)):
        return observed(t, marker, zone=zone, track=(0, 564))
    for t, marker in ((8.00, 40), (8.01, 70), (8.02, 103)):
        engine.update(seen(t, marker), True)
    assert engine.action_count == 1
    engine.update(seen(8.05, 130), True)
    engine.update(observed(8.10, None, present=False), True)
    for t, marker in ((8.22, 180), (8.28, 150), (8.34, 120)):
        assert engine.update(seen(t, marker, (100, 113)), True) is None
    assert engine.state == EngineState.WAIT_GONE
    assert engine.action_count == 1
