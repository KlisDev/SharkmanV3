from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import ColorSample, NormalizedRect, WindowFingerprint

SCHEMA_VERSION = 3
SAMPLE_GROUPS = ("green", "marker", "track")


@dataclass(frozen=True)
class TimingFieldSpec:
    name: str
    group: str
    default: float
    minimum: float
    maximum: float
    unit: str
    label: str
    help: str
    integer: bool = False


# This registry is deliberately the single inventory of all runtime waits,
# rates, debounces and timeouts. The engine, GUI, persistence and tests consume
# it directly; adding a hidden cooldown elsewhere is a bug.
TIMING_FIELDS: tuple[TimingFieldSpec, ...] = (
    TimingFieldSpec("startup_delay_ms", "Startup", 1000, 0, 10000, "ms", "Start delay", "Time to refocus Roblox after pressing Start."),
    TimingFieldSpec("scan_interval_ms", "Capture", 0, 0, 100, "ms", "Scan delay (0 = fastest)", "0 is recommended and captures continuously. Higher values wait longer between frames and reduce precision."),
    TimingFieldSpec("focus_poll_ms", "Capture", 50, 10, 1000, "ms", "Focus re-check", "How often foreground ownership is checked."),
    TimingFieldSpec("mac_capture_hz", "Mac capture", 240, 30, 240, "Hz", "Mac capture ceiling", "Requested ScreenCaptureKit rate, not a guarantee of delivered frames.", integer=True),
    TimingFieldSpec("mac_frame_wait_ms", "Mac capture", 100, 10, 500, "ms", "Mac fresh-frame wait", "Maximum wait for a new complete frame; unavailable frames never trigger input."),
    TimingFieldSpec("mac_frame_age_ms", "Mac capture", 50, 1, 100, "ms", "Mac maximum frame age", "Reject frames older than this; never reuse a cached frame as a new observation."),
    TimingFieldSpec("mac_capture_start_ms", "Mac capture", 2000, 100, 10000, "ms", "Mac capture startup", "Timeout for window discovery and stream startup."),
    TimingFieldSpec("mac_capture_stop_ms", "Mac capture", 1000, 100, 5000, "ms", "Mac capture shutdown", "Timeout for stopping a native capture stream."),
    TimingFieldSpec("acquire_confirm_ms", "Meter lifecycle", 0, 0, 500, "ms", "Extra meter confirmation", "Additional time after the mandatory two-frame lifecycle confirmation."),
    TimingFieldSpec("meter_loss_grace_ms", "Meter lifecycle", 90, 0, 1000, "ms", "Detection-loss grace", "Ignore a brief hidden or damaged meter."),
    TimingFieldSpec("zone_cache_ms", "Meter lifecycle", 80, 0, 1000, "ms", "Green-zone cache", "Retain the last zone during a brief color miss."),
    TimingFieldSpec("rearm_invisible_ms", "Meter lifecycle", 150, 0, 2000, "ms", "Invisible time before rearm", "A fired prompt must disappear for this long."),
    TimingFieldSpec("max_meter_ms", "Meter lifecycle", 3000, 100, 15000, "ms", "Maximum marker inactivity", "Abandon a frozen meter after this long without meaningful marker movement. An actively sweeping challenge can last longer."),
    TimingFieldSpec("wide_zone_settle_ms", "Meter lifecycle", 100, 0, 500, "ms", "Wide-zone startup guard", "Briefly blocks prediction while a newly appearing broad target is still animating. Tiny hard-stage targets are not delayed."),
    TimingFieldSpec("finish_confirm_ms", "Meter lifecycle", 100, 0, 500, "ms", "FINISH transition confirmation", "Minimum brief disappearance and full-size target confirmation used to distinguish FINISH from the ordinary success animation."),
    TimingFieldSpec("velocity_window_ms", "Prediction", 80, 20, 1000, "ms", "Velocity history", "Recent marker history used by regression. A shorter window follows acceleration and direction changes on harder stages."),
    TimingFieldSpec("zone_velocity_limit_px_s", "Prediction", 250, 0, 2000, "px/s", "Maximum target movement", "Limit implausible green-target speed estimates caused by detection noise."),
    TimingFieldSpec("input_latency_ms", "Prediction", 30, 0, 500, "ms", "Capture + input latency", "Initial estimate between captured pixels and effective Space input. This varies by PC and game performance; verify with a live test before adjusting."),
    TimingFieldSpec("lookahead_ms", "Prediction", 400, 0, 1000, "ms", "Scheduling lookahead", "Maximum future crossing considered for a pending fire."),
    TimingFieldSpec("precision_wait_threshold_ms", "Prediction", 24, 0, 100, "ms", "Sub-frame precision wait", "Maximum time to wait for a predicted crossing without another capture. Covers a narrow target that can pass between video frames; longer waits increase motion-model uncertainty."),
    TimingFieldSpec("key_hold_ms", "Input", 12, 1, 250, "ms", "Space hold", "Duration between Space key-down and key-up."),
    TimingFieldSpec("minimum_input_interval_ms", "Input", 250, 0, 3000, "ms", "Minimum input interval", "Independent final guard against duplicate input."),
    TimingFieldSpec("post_fire_lockout_ms", "Input", 100, 0, 2000, "ms", "Post-fire lockout", "Time before the engine begins waiting for disappearance."),
    TimingFieldSpec("same_gauge_retry_ms", "Input", 3500, 500, 10000, "ms", "Same-gauge retry delay", "If the gauge stays visible after a press, wait this long before allowing a new full sweep. Prevents duplicates during the post-hit animation."),
    TimingFieldSpec("shutdown_join_ms", "Input", 2000, 100, 10000, "ms", "Shutdown join timeout", "Maximum time the GUI waits for the capture thread to exit cleanly."),
    TimingFieldSpec("calibration_capture_delay_ms", "Interface", 250, 0, 2000, "ms", "Calibration capture delay", "Time for the app to hide before taking a screenshot."),
    TimingFieldSpec("diagnostic_refresh_ms", "Interface", 50, 16, 1000, "ms", "Diagnostic refresh", "Maximum GUI/overlay update rate."),
    TimingFieldSpec("inspector_redraw_ms", "Interface", 16, 1, 100, "ms", "Zoom redraw coalescing", "Delay used to combine rapid zoom and pan events into one image-inspector render."),
    TimingFieldSpec("feedback_start_ms", "Adaptive feedback", 25, 0, 250, "ms", "Post-press observation delay", "Wait briefly after Space before looking for a frozen marker."),
    TimingFieldSpec("feedback_freeze_confirm_ms", "Adaptive feedback", 50, 16, 300, "ms", "Frozen-bar confirmation", "Minimum time over at least three stable frames before measuring the marker."),
    TimingFieldSpec("feedback_timeout_ms", "Adaptive feedback", 400, 100, 3000, "ms", "Feedback timeout", "Give up without learning if a stable post-press gauge is not observed."),
    TimingFieldSpec("feedback_max_settle_ms", "Adaptive feedback", 180, 50, 500, "ms", "Latest trusted freeze", "Reject a marker that first settles too long after Space; it may belong to an animation rather than the press."),
    TimingFieldSpec("feedback_max_frame_gap_ms", "Adaptive feedback", 100, 20, 500, "ms", "Maximum feedback frame gap", "Reject a result when capture paused or frames were delayed."),
    TimingFieldSpec("finish_transition_window_ms", "Adaptive feedback", 3000, 500, 3000, "ms", "FINISH transition window", "A rapidly appearing broad target after a small one is excluded from learning."),
    TimingFieldSpec("adaptive_max_step_ms", "Adaptive feedback", 3, 0.5, 3, "ms", "Maximum learned step", "Largest permanent latency change from one evidence batch."),
    TimingFieldSpec("adaptive_max_drift_ms", "Adaptive feedback", 60, 5, 200, "ms", "Maximum learned drift", "Maximum automatic distance from the last manually chosen latency."),
)
TIMING_BY_NAME = {item.name: item for item in TIMING_FIELDS}


class TimingConfig:
    def __init__(self, overrides: dict[str, float] | None = None) -> None:
        self._values = {spec.name: spec.default for spec in TIMING_FIELDS}
        for name, value in (overrides or {}).items():
            if name in TIMING_BY_NAME:
                self.set(name, value)

    def set(self, name: str, value: float) -> None:
        spec = TIMING_BY_NAME[name]
        number = int(value) if spec.integer else float(value)
        if not spec.minimum <= number <= spec.maximum:
            raise ValueError(f"{spec.label} must be between {spec.minimum:g} and {spec.maximum:g} {spec.unit}")
        self._values[name] = number

    def __getattr__(self, name: str) -> float:
        if name in self._values:
            return self._values[name]
        raise AttributeError(name)

    def as_dict(self) -> dict[str, float]:
        return dict(self._values)

    def overrides(self) -> dict[str, float]:
        return {
            name: value
            for name, value in self._values.items()
            if value != TIMING_BY_NAME[name].default
        }


@dataclass
class DetectionConfig:
    safe_zone_inset: float = 0.15
    required_confidence: float = 0.62
    min_track_height_fraction: float = 0.42
    # A real meter fills almost the entire narrow ROI with its saturated
    # red/yellow/green gradient. Scenery can contain the same colors, but does
    # not produce this near-solid vertical strip.
    min_colored_fraction: float = 0.90
    min_marker_width_fraction: float = 0.52
    marker_darkness_ratio: float = 0.42
    min_zone_height_px: int = 3
    max_zone_height_fraction: float = 0.42

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not 0 <= self.safe_zone_inset < 0.5:
            errors.append("Safe-zone inset must be between 0 and 0.49.")
        if not 0 < self.required_confidence <= 1:
            errors.append("Required confidence must be between 0 and 1.")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DetectionConfig:
        known = cls().__dict__
        return cls(**{key: value for key, value in (data or {}).items() if key in known})


@dataclass
class AdaptiveState:
    enabled: bool = False
    baseline_latency_ms: float | None = None
    previous_latency_ms: float | None = None
    model_version: str = ""
    calibration_signature: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    sample_count: int = 0
    trial_sample_count: int = 0
    samples_since_adjustment: int = 0
    last_outcome: str = "unknown"
    last_reason: str = "No scored hit yet."

    def clear(self, latency_ms: float, *, disable: bool = False) -> None:
        self.baseline_latency_ms = float(latency_ms)
        self.previous_latency_ms = None
        self.model_version = ""
        self.calibration_signature = ""
        self.evidence.clear()
        self.sample_count = 0
        self.trial_sample_count = 0
        self.samples_since_adjustment = 0
        self.last_outcome = "unknown"
        self.last_reason = "Learning reset."
        if disable:
            self.enabled = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "baseline_latency_ms": self.baseline_latency_ms,
            "previous_latency_ms": self.previous_latency_ms,
            "model_version": self.model_version,
            "calibration_signature": self.calibration_signature,
            "evidence": self.evidence[-128:],
            "sample_count": self.sample_count,
            "trial_sample_count": self.trial_sample_count,
            "samples_since_adjustment": self.samples_since_adjustment,
            "last_outcome": self.last_outcome,
            "last_reason": self.last_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AdaptiveState:
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            baseline_latency_ms=(None if data.get("baseline_latency_ms") is None else float(data["baseline_latency_ms"])),
            previous_latency_ms=(None if data.get("previous_latency_ms") is None else float(data["previous_latency_ms"])),
            model_version=str(data.get("model_version", "")),
            calibration_signature=str(data.get("calibration_signature", "")),
            evidence=[item for item in data.get("evidence", [])[-128:] if isinstance(item, dict)],
            sample_count=max(0, int(data.get("sample_count", len(data.get("evidence", []))))),
            trial_sample_count=max(0, int(data.get("trial_sample_count", 0))),
            samples_since_adjustment=max(0, int(data.get("samples_since_adjustment", 0))),
            last_outcome=str(data.get("last_outcome", "unknown")),
            last_reason=str(data.get("last_reason", "")),
        )


@dataclass
class CalibrationProfile:
    name: str
    profile_id: str = field(default_factory=lambda: str(uuid4()))
    schema_version: int = SCHEMA_VERSION
    platform: str = field(default_factory=lambda: sys.platform)
    window_title: str = "Roblox"
    fingerprint: WindowFingerprint = field(default_factory=WindowFingerprint)
    # A new profile deliberately has no usable region. Calibration must be an
    # explicit user gesture; a plausible-looking default rectangle could arm
    # against unrelated pixels.
    meter_roi: NormalizedRect = field(default_factory=lambda: NormalizedRect(0.0, 0.0, 0.0, 0.0))
    samples: dict[str, list[ColorSample]] = field(default_factory=lambda: {key: [] for key in SAMPLE_GROUPS})
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    timing_overrides: dict[str, float] = field(default_factory=dict)
    adaptive: AdaptiveState = field(default_factory=AdaptiveState)
    validation_signature: str = ""
    validated_at: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def timing(self) -> TimingConfig:
        return TimingConfig(self.timing_overrides)

    def calibration_input_errors(self) -> list[str]:
        errors = self.detection.validate()
        if not self.meter_roi.valid:
            errors.append("Draw a valid meter region.")
        for group in SAMPLE_GROUPS:
            if not self.samples.get(group):
                errors.append(f"Capture at least one {group} color sample.")
        if self.fingerprint.width <= 0 or self.fingerprint.height <= 0:
            errors.append("Capture the Roblox window once to record its size.")
        return errors

    def calibration_payload(self) -> dict[str, Any]:
        """Return only the values whose change requires re-validation."""
        return {
            "platform": self.platform,
            "fingerprint": self.fingerprint.to_dict(),
            "meter_roi": self.meter_roi.to_dict(),
            "samples": {
                key: [sample.to_dict() for sample in self.samples.get(key, [])]
                for key in SAMPLE_GROUPS
            },
            "detection": self.detection.to_dict(),
        }

    def calibration_signature(self) -> str:
        packed = json.dumps(
            self.calibration_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return sha256(packed).hexdigest()

    @property
    def validation_current(self) -> bool:
        return bool(
            not self.calibration_input_errors()
            and self.validation_signature
            and self.validation_signature == self.calibration_signature()
        )

    def mark_validated(self) -> None:
        errors = self.calibration_input_errors()
        if errors:
            raise ValueError("Cannot validate incomplete calibration: " + " ".join(errors))
        self.validation_signature = self.calibration_signature()
        self.validated_at = datetime.now(UTC).isoformat()

    def invalidate_validation(self) -> None:
        self.validation_signature = ""
        self.validated_at = None

    def calibration_errors(self) -> list[str]:
        errors = self.calibration_input_errors()
        if not errors and not self.validation_current:
            errors.append("Test detection successfully before running.")
        return errors

    def geometry_errors(self, fingerprint: WindowFingerprint) -> list[str]:
        errors: list[str] = []
        expected = self.fingerprint
        if (expected.width, expected.height) != (fingerprint.width, fingerprint.height):
            errors.append(
                f"Profile is {expected.width}×{expected.height}; Roblox is {fingerprint.width}×{fingerprint.height}."
            )
        if expected.dpi != fingerprint.dpi:
            errors.append(f"Profile DPI is {expected.dpi}; Roblox DPI is {fingerprint.dpi}.")
        if expected.display_mode != fingerprint.display_mode:
            errors.append(
                f"Profile mode is {expected.display_mode}; Roblox is {fingerprint.display_mode}."
            )
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "name": self.name,
            "platform": self.platform,
            "window_title": self.window_title,
            "fingerprint": self.fingerprint.to_dict(),
            "meter_roi": self.meter_roi.to_dict(),
            "samples": {
                key: [sample.to_dict() for sample in self.samples.get(key, [])]
                for key in SAMPLE_GROUPS
            },
            "detection": self.detection.to_dict(),
            "timing_overrides": self.timing.overrides(),
            "adaptive": self.adaptive.to_dict(),
            "validation_signature": self.validation_signature,
            "validated_at": self.validated_at,
            "created_at": self.created_at,
            "updated_at": datetime.now(UTC).isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationProfile:
        version = int(data.get("schema_version", 1))
        if version > SCHEMA_VERSION:
            raise ValueError(f"profile schema {version} is newer than this app supports")
        samples = {
            key: [ColorSample.from_dict(item) for item in data.get("samples", {}).get(key, [])]
            for key in SAMPLE_GROUPS
        }
        # Schema-v1 profiles keep every useful calibration value, but are
        # deliberately unvalidated until the new guided detector test passes.
        validation_signature = (
            str(data.get("validation_signature", "")) if version >= 2 else ""
        )
        validated_at = data.get("validated_at") if version >= 2 else None
        return cls(
            schema_version=SCHEMA_VERSION,
            profile_id=str(data.get("profile_id") or uuid4()),
            name=str(data.get("name", "Profile")),
            platform=str(data.get("platform", sys.platform)),
            window_title=str(data.get("window_title", "Roblox")),
            fingerprint=WindowFingerprint.from_dict(data.get("fingerprint")),
            meter_roi=NormalizedRect.from_dict(
                data.get("meter_roi", NormalizedRect(0.0, 0.0, 0.0, 0.0).to_dict())
            ),
            samples=samples,
            detection=DetectionConfig.from_dict(data.get("detection")),
            timing_overrides=TimingConfig(data.get("timing_overrides", {})).overrides(),
            adaptive=AdaptiveState.from_dict(data.get("adaptive") if version >= 3 else None),
            validation_signature=validation_signature,
            validated_at=None if validated_at is None else str(validated_at),
            created_at=str(data.get("created_at", datetime.now(UTC).isoformat())),
            updated_at=str(data.get("updated_at", datetime.now(UTC).isoformat())),
        )


def default_data_root() -> Path:
    override = os.environ.get("SHARKMAN_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "SharkmanV3"
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "SharkmanV3"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "sharkman-v3"


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


class ProfileStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or default_data_root()
        self.profiles_dir = self.root / "profiles"
        self.settings_path = self.root / "settings.json"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slug(profile: CalibrationProfile) -> str:
        readable = re.sub(r"[^a-z0-9]+", "-", profile.name.lower()).strip("-") or "profile"
        return f"{readable}-{profile.profile_id[:8]}.json"

    def list(self) -> list[CalibrationProfile]:
        profiles: list[CalibrationProfile] = []
        for path in sorted(self.profiles_dir.glob("*.json")):
            try:
                profiles.append(CalibrationProfile.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return sorted(profiles, key=lambda item: item.name.casefold())

    def save(self, profile: CalibrationProfile, *, activate: bool = True) -> Path:
        profile.updated_at = datetime.now(UTC).isoformat()
        existing = self._path_for_id(profile.profile_id)
        path = existing or self.profiles_dir / self._slug(profile)
        _atomic_json(path, profile.to_dict())
        if activate:
            self.set_active(profile.profile_id)
        return path

    def delete(self, profile_id: str) -> None:
        path = self._path_for_id(profile_id)
        if path:
            path.unlink()
        if self.active_id() == profile_id:
            self.set_active(None)

    def _path_for_id(self, profile_id: str) -> Path | None:
        for path in self.profiles_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("profile_id") == profile_id:
                    return path
            except (OSError, json.JSONDecodeError):
                continue
        return None

    def active_id(self) -> str | None:
        try:
            return json.loads(self.settings_path.read_text(encoding="utf-8")).get("active_profile_id")
        except (OSError, json.JSONDecodeError):
            return None

    def set_active(self, profile_id: str | None) -> None:
        _atomic_json(self.settings_path, {"active_profile_id": profile_id})

    def active(self) -> CalibrationProfile:
        profiles = self.list()
        active_id = self.active_id()
        for profile in profiles:
            if profile.profile_id == active_id:
                return profile
        if profiles:
            self.set_active(profiles[0].profile_id)
            return profiles[0]
        profile = CalibrationProfile(name="Default")
        self.save(profile)
        return profile

    def import_profile(self, path: Path) -> CalibrationProfile:
        profile = CalibrationProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
        profile.profile_id = str(uuid4())
        profile.name = f"{profile.name} (imported)"
        profile.adaptive.clear(profile.timing.input_latency_ms, disable=True)
        self.save(profile)
        return profile

    def export_profile(self, profile: CalibrationProfile, path: Path) -> None:
        _atomic_json(path, profile.to_dict())


def timing_groups() -> Iterable[tuple[str, tuple[TimingFieldSpec, ...]]]:
    groups: dict[str, list[TimingFieldSpec]] = {}
    for spec in TIMING_FIELDS:
        groups.setdefault(spec.group, []).append(spec)
    return ((name, tuple(items)) for name, items in groups.items())
