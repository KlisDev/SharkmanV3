from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from sharkman.adaptive import (
    FeedbackCollector,
    FeedbackModel,
    FeedbackResult,
    load_feedback_model,
    normalized_landing,
    propose_center_learning,
    propose_learning,
)
from sharkman.config import CalibrationProfile, ProfileStore
from sharkman.models import FireAction, MeterObservation
from sharkman.runner import LiveRunner

from .helpers import calibrated_profile

MODEL = FeedbackModel("synthetic-v1", 0.0, 0.30, 12, 1.0)


def frame(t: float, marker: float = 55, *, zone=(40, 60), confidence=1.0, present=True) -> MeterObservation:
    return MeterObservation(
        t, present, confidence, marker if present else None,
        zone_bounds=zone if present else None, track_bounds=(0, 100) if present else None,
    )


def action(velocity: float = 100.0) -> FireAction:
    return FireAction(0, 45, 50, (40, 60), velocity, "test")


def test_feedback_model_requires_heldout_validation(tmp_path) -> None:
    path = tmp_path / "model.json"
    assert load_feedback_model(path) is None
    path.write_text(json.dumps({**MODEL.to_dict(), "holdout_accuracy": 0.8}), encoding="utf-8")
    assert load_feedback_model(path) is None
    path.write_text(json.dumps(MODEL.to_dict()), encoding="utf-8")
    assert load_feedback_model(path) == MODEL
    assert MODEL.classify(0.0) == "critical"
    assert MODEL.classify(0.6) == "edge"
    assert MODEL.classify(1.2) == "miss"
    assert MODEL.classify(1.0) == "unknown"
    assert normalized_landing(55, (40, 60)) == 0.5


@pytest.mark.parametrize("step", [1 / 30, 1 / 60, 1 / 120, 1 / 240])
def test_frozen_feedback_at_common_capture_rates(step: float) -> None:
    profile = calibrated_profile()
    collector = FeedbackCollector(profile.timing)
    collector.begin(action(), 0.0)
    result = None
    for index in range(1, 50):
        result = collector.observe(frame(index * step), focused=True)
        if result:
            break
    assert result is not None
    assert result.outcome == "measured"
    assert result.marker == 55
    assert result.normalized_position == 0.5


def test_uncertain_feedback_never_becomes_a_miss() -> None:
    profile = calibrated_profile()
    collector = FeedbackCollector(profile.timing)
    collector.begin(action(), 0.0)
    result = collector.observe(frame(0.04, confidence=0.5), focused=True)
    assert result is not None and result.outcome == "unknown"

    collector.begin(action(), 2.0)
    assert collector.observe(frame(2.04), focused=True) is None
    result = collector.observe(frame(2.20), focused=True)
    assert result is not None and result.outcome == "unknown"

    collector.begin(action(), 3.0)
    result = collector.observe(frame(3.05), focused=False)
    assert result is not None and result.outcome == "unknown"


def test_moving_or_moving_zone_is_not_scored() -> None:
    collector = FeedbackCollector(calibrated_profile().timing)
    collector.begin(action(), 0)
    for index in range(1, 10):
        result = collector.observe(frame(index / 60, marker=40 + index * 3), focused=True)
        assert result is None
    collector.cancel()
    assert collector.pending is None

    collector.begin(action(), 0)
    for index in range(1, 10):
        result = collector.observe(frame(index / 60, zone=(40 + index * 2, 60 + index * 2)), focused=True)
        assert result is None


def test_new_large_target_cannot_be_scored_as_previous_hit() -> None:
    collector = FeedbackCollector(calibrated_profile().timing)
    collector.begin(action(), 0.0, source_zone=(40, 60))
    assert collector.observe(frame(0.04), focused=True) is None
    result = collector.observe(frame(0.06, zone=(20, 70)), focused=True)
    assert result is not None
    assert result.outcome == "unknown"
    assert "changed" in result.reason

    collector.begin(action(), 1.0, source_zone=(40, 60))
    result = collector.observe(frame(1.04, zone=(46, 66)), focused=True)
    assert result is not None and result.outcome == "unknown"


def test_late_freeze_cannot_steer_center() -> None:
    collector = FeedbackCollector(calibrated_profile().timing)
    collector.begin(action(), 0.0, source_zone=(40, 60))
    result = None
    for stamp in (0.20, 0.22, 0.24):
        result = collector.observe(frame(stamp), focused=True)
        if result:
            break
    assert result is not None and result.outcome == "unknown"
    assert "late" in result.reason


@pytest.mark.parametrize(
    ("marker", "velocity", "expected"),
    [(55, 100, 33), (45, 100, 27), (45, -100, 33)],
)
def test_visual_centering_corrects_signed_latency(marker: float, velocity: float, expected: float) -> None:
    profile = calibrated_profile()
    profile.adaptive.enabled = True
    result = FeedbackResult("measured", "Frozen.", marker, (40, 60),
                            normalized_landing(marker, (40, 60)), velocity)
    for _ in range(5):
        profile, changed = propose_center_learning(profile, result, profile.timing)
        assert changed is None
    profile, changed = propose_center_learning(profile, result, profile.timing)
    assert changed == expected
    assert profile.timing.input_latency_ms == expected
    assert profile.adaptive.last_outcome == "off-center"
    assert "grade unknown" in profile.adaptive.last_reason


def test_center_deadband_and_smaller_error_reduce_adjustment() -> None:
    centered = calibrated_profile()
    centered.adaptive.enabled = True
    near = calibrated_profile()
    near.adaptive.enabled = True
    far = calibrated_profile()
    far.adaptive.enabled = True
    for _ in range(6):
        centered, centered_change = propose_center_learning(
            centered, FeedbackResult("measured", "Frozen.", 50.5, (40, 60), 0.05, 1000), centered.timing,
        )
        near, near_change = propose_center_learning(
            near, FeedbackResult("measured", "Frozen.", 52, (40, 60), 0.2, 1000), near.timing,
        )
        far, far_change = propose_center_learning(
            far, FeedbackResult("measured", "Frozen.", 55, (40, 60), 0.5, 1000), far.timing,
        )
    assert centered_change is None
    assert near_change is not None and far_change is not None
    assert 30 < near_change < far_change <= 33


def test_small_visual_error_cannot_trigger_full_step_at_low_speed() -> None:
    profile = calibrated_profile()
    profile.adaptive.enabled = True
    result = FeedbackResult("measured", "Frozen.", 51.1, (40, 60), 0.11, 50)
    for _ in range(6):
        profile, changed = propose_center_learning(profile, result, profile.timing)
    assert changed is not None
    assert 30 < changed < 31


def test_uncertain_or_off_band_visual_positions_never_adjust() -> None:
    profile = calibrated_profile()
    profile.adaptive.enabled = True
    for _ in range(12):
        for result in (
            FeedbackResult("unknown", "Gauge disappeared."),
            FeedbackResult("measured", "Outside.", 62, (40, 60), 1.2, 100),
            FeedbackResult("measured", "Slow.", 55, (40, 60), 0.5, 20),
        ):
            profile, changed = propose_center_learning(profile, result, profile.timing)
            assert changed is None
    assert profile.adaptive.sample_count == 0
    assert profile.timing.input_latency_ms == 30


@pytest.mark.parametrize(
    ("marker", "velocity", "expected"),
    [(55, 100, 33), (45, 100, 27), (45, -100, 33)],
)
def test_signed_landing_error_adjusts_latency_correctly(marker: float, velocity: float, expected: float) -> None:
    profile = calibrated_profile()
    profile.adaptive.enabled = True
    for _ in range(5):
        result = FeedbackResult("measured", "Frozen.", marker, (40, 60), normalized_landing(marker, (40, 60)), velocity)
        profile, changed = propose_learning(profile, result, MODEL, profile.timing)
        assert changed is None
    profile, changed = propose_learning(profile, result, MODEL, profile.timing)
    assert changed == expected
    assert profile.timing.input_latency_ms == expected
    assert profile.adaptive.previous_latency_ms == 30
    assert len(profile.adaptive.evidence) == 6


def test_unknown_and_low_speed_never_adjust_or_add_evidence() -> None:
    profile = calibrated_profile()
    profile.adaptive.enabled = True
    for result in (
        FeedbackResult("unknown", "Occluded."),
        FeedbackResult("measured", "Too slow.", 55, (40, 60), 0.5, 10),
    ):
        profile, changed = propose_learning(profile, result, MODEL, profile.timing)
        assert changed is None
    assert not profile.adaptive.evidence


def test_balanced_trials_commit_only_the_better_direction() -> None:
    profile = calibrated_profile()
    profile.adaptive.enabled = True
    profile.adaptive.model_version = MODEL.version
    profile.adaptive.calibration_signature = profile.calibration_signature()
    profile.adaptive.baseline_latency_ms = 30
    profile.adaptive.evidence = [
        {"outcome": "critical", "position": 0.25, "error_ms": 0.5,
         "trial_ms": 0.0, "velocity": 100.0}
        for _ in range(6)
    ]
    profile.adaptive.sample_count = 6
    profile.adaptive.samples_since_adjustment = 6
    for index in range(6):
        trial = 2.0 if index % 2 == 0 else -2.0
        position = 0.1 if trial > 0 else 0.5
        marker = 50 + position * 10
        result = FeedbackResult("measured", "Frozen.", marker, (40, 60), position, 100, trial)
        profile, changed = propose_learning(profile, result, MODEL, profile.timing)
        if index < 5:
            assert changed is None
    assert changed == 32.0


def test_one_sided_trials_never_change_saved_latency() -> None:
    profile = calibrated_profile()
    profile.adaptive.enabled = True
    profile.adaptive.model_version = MODEL.version
    profile.adaptive.calibration_signature = profile.calibration_signature()
    profile.adaptive.baseline_latency_ms = 30
    profile.adaptive.evidence = [
        {"outcome": "critical", "position": 0.25, "error_ms": 0.5,
         "trial_ms": 0.0, "velocity": 100.0}
        for _ in range(6)
    ]
    profile.adaptive.sample_count = 6
    profile.adaptive.samples_since_adjustment = 6
    for _ in range(6):
        result = FeedbackResult("measured", "Frozen.", 51, (40, 60), 0.1, 100, 2.0)
        profile, changed = propose_learning(profile, result, MODEL, profile.timing)
    assert changed is None
    assert profile.timing.input_latency_ms == 30


def test_trial_must_improve_unmodified_baseline() -> None:
    profile = calibrated_profile()
    profile.adaptive.enabled = True
    profile.adaptive.model_version = MODEL.version
    profile.adaptive.calibration_signature = profile.calibration_signature()
    profile.adaptive.baseline_latency_ms = 30
    profile.adaptive.evidence = [
        {"outcome": "critical", "position": 0.01, "error_ms": 0.5,
         "trial_ms": 0.0, "velocity": 100.0}
        for _ in range(6)
    ]
    profile.adaptive.sample_count = 6
    profile.adaptive.samples_since_adjustment = 6
    for index in range(6):
        trial = 2.0 if index % 2 == 0 else -2.0
        position = 0.1 if trial > 0 else 0.5
        result = FeedbackResult("measured", "Frozen.", 50 + position * 10,
                                (40, 60), position, 100, trial)
        profile, changed = propose_learning(profile, result, MODEL, profile.timing)
    assert changed is None
    assert profile.timing.input_latency_ms == 30


def test_numeric_evidence_is_bounded_while_batch_counter_continues() -> None:
    profile = calibrated_profile()
    profile.adaptive.enabled = True
    result = FeedbackResult("measured", "Frozen.", 50, (40, 60), 0.0, 100)
    for _ in range(132):
        profile, changed = propose_learning(profile, result, MODEL, profile.timing)
        assert changed is None
    assert len(profile.adaptive.evidence) == 128
    assert profile.adaptive.sample_count == 132
    restored = CalibrationProfile.from_dict(profile.to_dict())
    assert restored.adaptive.sample_count == 132


def test_profile_migration_isolation_and_save_without_activating(tmp_path) -> None:
    store = ProfileStore(tmp_path)
    first = calibrated_profile("first")
    second = calibrated_profile("second")
    store.save(first)
    store.save(second)
    store.set_active(first.profile_id)
    second.adaptive.enabled = True
    second.adaptive.evidence.append({"outcome": "edge", "error_ms": 5})
    store.save(second, activate=False)
    assert store.active_id() == first.profile_id
    restored = next(item for item in store.list() if item.profile_id == second.profile_id)
    assert restored.adaptive.enabled
    assert len(restored.adaptive.evidence) == 1

    old = CalibrationProfile.from_dict({**second.to_dict(), "schema_version": 2})
    assert not old.adaptive.enabled and not old.adaptive.evidence
    exported = tmp_path / "export.json"
    store.export_profile(second, exported)
    imported = store.import_profile(exported)
    assert not imported.adaptive.enabled and not imported.adaptive.evidence


def test_runner_centering_requires_windows_live_profile_and_store(monkeypatch, tmp_path) -> None:
    from sharkman import runner as runner_module

    monkeypatch.setattr(runner_module, "sys", SimpleNamespace(platform="win32"))
    profile = calibrated_profile()
    profile.adaptive.enabled = True
    store = ProfileStore(tmp_path)
    args = (None, profile, lambda *_: None, lambda *_: None)
    assert not LiveRunner(*args, dry_run=True, profile_store=store).adaptive_ready
    assert not LiveRunner(*args, dry_run=False).adaptive_ready
    assert not LiveRunner(*args, dry_run=False, profile_store=store).adaptive_ready
    monkeypatch.setattr(runner_module, "LIVE_ADAPTIVE_FEEDBACK_SUPPORTED", True)
    assert LiveRunner(*args, dry_run=False, profile_store=store).adaptive_ready
    profile.adaptive.enabled = False
    assert not LiveRunner(*args, dry_run=False, profile_store=store).adaptive_ready
    profile.adaptive.enabled = True
    monkeypatch.setattr(runner_module, "sys", SimpleNamespace(platform="linux"))
    assert not LiveRunner(*args, dry_run=False, profile_store=store).adaptive_ready


def test_failed_feedback_save_disables_learning_without_timing_change(monkeypatch) -> None:
    from sharkman import runner as runner_module

    monkeypatch.setattr(runner_module, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(runner_module, "LIVE_ADAPTIVE_FEEDBACK_SUPPORTED", True)

    class FailingStore:
        def save(self, _profile, *, activate: bool = True):
            raise OSError("disk full")

    profile = calibrated_profile()
    profile.adaptive.enabled = True
    logs: list[str] = []
    runner = LiveRunner(None, profile, lambda *_: None, logs.append,
                        dry_run=False, profile_store=FailingStore())
    runner.engine.prompt_count = 1
    runner._feedback_prompt_count = 1
    runner.feedback.begin(action(), 0.0, source_zone=(40, 60))
    for stamp in (0.04, 0.06, 0.10):
        runner._record_feedback(frame(stamp), focused=True)
    assert not runner.adaptive_ready
    assert profile.timing.input_latency_ms == 30
    assert not profile.adaptive.evidence
    assert any("disk full" in line for line in logs)


def test_windows_center_feedback_auto_saves_only_after_six_prompts(monkeypatch, tmp_path) -> None:
    from sharkman import runner as runner_module

    monkeypatch.setattr(runner_module, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(runner_module, "LIVE_ADAPTIVE_FEEDBACK_SUPPORTED", True)
    store = ProfileStore(tmp_path)
    profile = calibrated_profile()
    profile.adaptive.enabled = True
    store.save(profile)
    runner = LiveRunner(None, profile, lambda *_: None, lambda *_: None,
                        dry_run=False, profile_store=store)
    assert runner.adaptive_ready
    for prompt in range(1, 7):
        runner.engine.prompt_count = prompt
        runner._feedback_prompt_count = prompt
        runner.feedback.begin(action(), float(prompt), source_zone=(40, 60))
        for delay in (0.04, 0.06, 0.10):
            runner._record_feedback(frame(prompt + delay), focused=True)
        assert profile.timing.input_latency_ms == (33 if prompt == 6 else 30)
    assert store.active().timing.input_latency_ms == 33
    assert store.active().adaptive.sample_count == 6
    assert runner.engine.timing.input_latency_ms == 33


def test_stopped_or_unfocused_center_feedback_cannot_adjust(monkeypatch, tmp_path) -> None:
    from sharkman import runner as runner_module

    monkeypatch.setattr(runner_module, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(runner_module, "LIVE_ADAPTIVE_FEEDBACK_SUPPORTED", True)
    store = ProfileStore(tmp_path)
    profile = calibrated_profile()
    profile.adaptive.enabled = True
    store.save(profile)
    runner = LiveRunner(None, profile, lambda *_: None, lambda *_: None,
                        dry_run=False, profile_store=store)
    runner.engine.prompt_count = 1
    runner._feedback_prompt_count = 1
    runner.feedback.begin(action(), 0, source_zone=(40, 60))
    runner._record_feedback(frame(0.04), focused=False)
    assert profile.adaptive.sample_count == 0
    runner.feedback.begin(action(), 1, source_zone=(40, 60))
    runner.stop()
    for stamp in (1.04, 1.06, 1.10):
        runner._record_feedback(frame(stamp), focused=True)
    assert store.active().adaptive.sample_count == 0
