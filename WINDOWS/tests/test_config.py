from __future__ import annotations

import json

import pytest

from sharkman.config import TIMING_BY_NAME, CalibrationProfile, ProfileStore, TimingConfig
from sharkman.models import ColorSample, NormalizedRect, WindowFingerprint

from .helpers import calibrated_profile


def test_timing_only_persists_real_overrides() -> None:
    timing = TimingConfig()
    assert timing.overrides() == {}
    timing.set("input_latency_ms", TIMING_BY_NAME["input_latency_ms"].default + 5)
    assert timing.overrides() == {
        "input_latency_ms": TIMING_BY_NAME["input_latency_ms"].default + 5
    }
    timing.set("input_latency_ms", TIMING_BY_NAME["input_latency_ms"].default)
    assert timing.overrides() == {}


def test_new_profile_requires_an_explicit_meter_region() -> None:
    profile = CalibrationProfile(name="new")
    assert not profile.meter_roi.valid
    assert "Draw a valid meter region." in profile.calibration_errors()


def test_timing_rejects_out_of_range_values() -> None:
    timing = TimingConfig()
    with pytest.raises(ValueError):
        timing.set("key_hold_ms", 0)


def test_profile_store_round_trip_and_active_profile(tmp_path) -> None:
    store = ProfileStore(tmp_path)
    profile = calibrated_profile("My 1080p profile")
    profile.timing_overrides = {"input_latency_ms": 70}
    profile.mark_validated()
    path = store.save(profile)
    assert path.exists()
    restored = store.active()
    assert restored.profile_id == profile.profile_id
    assert restored.samples["green"][0].rgb == (35, 245, 30)
    assert restored.timing.input_latency_ms == 70
    assert restored.validation_current
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["timing_overrides"] == {"input_latency_ms": 70.0}


def test_profile_geometry_mismatch_disarms() -> None:
    profile = calibrated_profile()
    errors = profile.geometry_errors(WindowFingerprint(1920, 1080, 120, "fullscreen"))
    assert len(errors) == 3


def test_schema_v1_migration_retains_calibration_but_requires_new_test() -> None:
    original = calibrated_profile()
    payload = original.to_dict()
    payload["schema_version"] = 1
    payload.pop("validation_signature", None)
    payload.pop("validated_at", None)

    migrated = CalibrationProfile.from_dict(payload)

    assert migrated.schema_version == 3
    assert not migrated.adaptive.enabled
    assert migrated.meter_roi == original.meter_roi
    assert migrated.samples == original.samples
    assert migrated.fingerprint == original.fingerprint
    assert not migrated.validation_current
    assert "Test detection successfully before running." in migrated.calibration_errors()


def test_validation_signature_tracks_every_calibration_input() -> None:
    profile = calibrated_profile()
    profile.mark_validated()
    assert profile.validation_current

    profile.samples["green"].append(ColorSample((20, 230, 20), 40))
    assert not profile.validation_current

    profile.samples["green"].pop()
    assert profile.validation_current

    original = profile.samples["marker"][0]
    profile.samples["marker"][0] = ColorSample(original.rgb, original.tolerance + 1)
    assert not profile.validation_current
    profile.samples["marker"][0] = original
    assert profile.validation_current

    profile.meter_roi = NormalizedRect(0.01, 0.0, 0.99, 1.0)
    assert not profile.validation_current


def test_calibration_requires_roi_and_all_click_sample_groups() -> None:
    profile = calibrated_profile()
    profile.meter_roi = NormalizedRect(0.0, 0.0, 0.0, 0.0)
    profile.samples["track"] = []

    errors = profile.calibration_input_errors()

    assert "Draw a valid meter region." in errors
    assert "Capture at least one track color sample." in errors


def test_switching_named_profiles_changes_the_active_profile(tmp_path) -> None:
    store = ProfileStore(tmp_path)
    first = calibrated_profile("First")
    second = calibrated_profile("Second")
    store.save(first)
    store.save(second)

    store.set_active(first.profile_id)
    assert store.active().name == "First"
    store.set_active(second.profile_id)
    assert store.active().name == "Second"
