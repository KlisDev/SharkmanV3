from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable

from .adaptive import (
    LIVE_ADAPTIVE_FEEDBACK_SUPPORTED,
    FeedbackCollector,
    propose_center_learning,
)
from .backends.base import BackendError, CaptureUnavailable, PlatformBackend
from .config import CalibrationProfile, ProfileStore
from .diagnostics import DiagnosticSession
from .engine import TimingEngine
from .models import EngineSnapshot, FireAction, MeterObservation, WindowInfo
from .vision import MeterDetector

UpdateCallback = Callable[[MeterObservation, EngineSnapshot, FireAction | None, WindowInfo], None]
LogCallback = Callable[[str], None]


class LiveRunner:
    def __init__(
        self,
        backend: PlatformBackend,
        profile: CalibrationProfile,
        on_update: UpdateCallback,
        on_log: LogCallback,
        *,
        dry_run: bool = False,
        diagnostics_enabled: bool = False,
        record_crops: bool = False,
        profile_store: ProfileStore | None = None,
    ) -> None:
        self.backend = backend
        self.game_name = getattr(backend, "game_name", "Roblox")
        self.profile = profile
        self.on_update = on_update
        self.on_log = on_log
        self.dry_run = dry_run
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.engine = TimingEngine(profile)
        self.detector = MeterDetector(profile)
        self.diagnostics = DiagnosticSession(diagnostics_enabled, record_crops)
        self.window: WindowInfo | None = None
        self.profile_store = profile_store
        self.adaptive_ready = bool(
            LIVE_ADAPTIVE_FEEDBACK_SUPPORTED
            and sys.platform == "win32" and not dry_run and profile.adaptive.enabled
            and profile_store is not None
        )
        self.feedback = FeedbackCollector(self.engine.timing)
        self._feedback_prompt_count: int | None = None

    @property
    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    @property
    def paused(self) -> bool:
        return self.pause_event.is_set()

    def start(self) -> None:
        if self.running:
            return
        self.stop_event.clear()
        self.pause_event.clear()
        self.backend.begin_input_session()
        self.thread = threading.Thread(target=self._run, name="sharkman-runner", daemon=True)
        self.thread.start()

    def toggle_pause(self) -> None:
        if not self.running:
            self.start()
            return
        if self.pause_event.is_set():
            self.backend.begin_input_session()
            self.pause_event.clear()
            self.on_log(f"Resumed — focus {self.game_name} before the next meter.")
        else:
            self.pause_event.set()
            self.feedback.cancel()
            self._feedback_prompt_count = None
            released = True
            try:
                self.backend.release_all()
            except Exception as exc:
                released = False
                self.stop_event.set()
                self.on_log(f"Stopped: Space release failed on pause: {exc}")
            self.engine.reset()
            if released:
                self.on_log("Paused — Space released and pending action cleared.")

    def stop(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()
        self.feedback.cancel()
        self._feedback_prompt_count = None
        try:
            self.backend.release_all()
        except Exception as exc:
            self.on_log(f"Stopped: Space release failed: {exc}")
        finally:
            if self.backend is not None:
                self.backend.stop_capture()

    def join(self) -> None:
        if self.thread:
            self.thread.join(self.profile.timing.shutdown_join_ms / 1000.0)

    def _record_feedback(self, observation: MeterObservation, focused: bool) -> None:
        if not self.adaptive_ready or self.profile_store is None:
            return
        if self.feedback.pending is None:
            return
        if self._feedback_prompt_count != self.engine.prompt_count:
            self.feedback.cancel()
            self._feedback_prompt_count = None
            self.on_log("Adaptive: result unknown; a new prompt appeared before feedback settled.")
            return
        result = self.feedback.observe(observation, focused=focused)
        if result is None:
            return
        self._feedback_prompt_count = None
        if self.stop_event.is_set() or self.pause_event.is_set() or not focused:
            return
        candidate, latency = propose_center_learning(self.profile, result, self.engine.timing)
        try:
            self.profile_store.save(candidate, activate=False)
        except (OSError, ValueError) as exc:
            self.adaptive_ready = False
            self.on_log(f"Adaptive disabled: profile save failed ({exc}). Timing was not changed.")
            return
        self.profile.adaptive = candidate.adaptive
        self.profile.timing_overrides = candidate.timing_overrides
        self.profile.updated_at = candidate.updated_at
        if latency is not None:
            self.engine.timing.set("input_latency_ms", latency)
            self.on_log(f"Adaptive: {candidate.adaptive.last_reason} Saved to {candidate.name!r}.")
        else:
            self.on_log(f"Adaptive: {candidate.adaptive.last_outcome}; {candidate.adaptive.last_reason}")

    def _run(self) -> None:
        timing = self.profile.timing
        try:
            self.backend.configure_timing(timing)
            errors = self.profile.calibration_errors()
            if errors:
                raise BackendError("Calibration is incomplete: " + " ".join(errors))
            if self.profile.platform != sys.platform:
                raise BackendError(
                    f"Profile platform is {self.profile.platform!r}; this host is {sys.platform!r}. "
                    "Create a separate calibration for this platform."
                )
            self.window = self.backend.find_window(self.profile.window_title)
            geometry_errors = self.profile.geometry_errors(self.window.fingerprint)
            if geometry_errors:
                raise BackendError("Calibration does not match this window: " + " ".join(geometry_errors))
            privilege = self.backend.privilege_error(self.window)
            if privilege:
                raise BackendError(privilege)
            if not self.dry_run:
                input_error = self.backend.input_error(self.window)
                if input_error:
                    raise BackendError(input_error)
            mode = "DRY RUN" if self.dry_run else "LIVE"
            self.on_log(f"{mode}: found {self.window.title!r} at {self.window.width}×{self.window.height}.")
            if self.profile.adaptive.enabled and not self.adaptive_ready:
                self.on_log("Adaptive timing is disabled; saved toggle is ignored. No automatic timing changes will be made.")
            elif self.adaptive_ready:
                self.on_log("Experimental visual centering is on. Frozen bar position is not a game hit grade.")
            if self.stop_event.wait(timing.startup_delay_ms / 1000.0):
                return
            self.on_log(
                f"Monitoring started. Input remains blocked unless {self.game_name} is foreground."
            )
            next_focus_poll = 0.0
            focused = False
            last_report = -1e9
            last_state = None
            last_capture_issue = None
            while not self.stop_event.is_set():
                if self.pause_event.is_set():
                    self.backend.release_all()
                    self.stop_event.wait(timing.focus_poll_ms / 1000.0)
                    continue
                now = time.perf_counter()
                if now >= next_focus_poll:
                    focused = self.backend.is_foreground(self.window)
                    next_focus_poll = now + timing.focus_poll_ms / 1000.0
                    if not focused:
                        self.backend.release_all()
                        self.feedback.cancel()
                        self._feedback_prompt_count = None
                if not focused:
                    # Screen-coordinate backends could otherwise record another
                    # application's pixels after the user switches windows.
                    self.engine.update(MeterObservation(now, False, 0.0), focused=False)
                    self.stop_event.wait(timing.focus_poll_ms / 1000.0)
                    continue
                try:
                    frame = self.backend.capture(self.window)
                except CaptureUnavailable as exc:
                    self.engine.capture_unavailable()
                    self.feedback.cancel()
                    if str(exc) != last_capture_issue:
                        self.on_log(f"Capture unavailable; input inhibited: {exc}")
                        last_capture_issue = str(exc)
                        # UI/diagnostic status only: never feed a fake absence to the engine.
                        unavailable = MeterObservation(
                            time.perf_counter(), False, 0.0, reasons=("capture unavailable", str(exc)))
                        self.on_update(unavailable, self.engine.snapshot(), None, self.window)
                    continue
                if last_capture_issue is not None:
                    self.on_log("Fresh capture resumed; rebuilding marker motion history.")
                    last_capture_issue = None
                if self.stop_event.is_set() or self.pause_event.is_set():
                    continue
                if not self.backend.is_foreground(self.window):
                    # Discard the whole frame before detection or recording if
                    # focus/geometry changed while capture was in progress.
                    focused = False
                    self.backend.release_all()
                    self.feedback.cancel()
                    self._feedback_prompt_count = None
                    self.engine.update(
                        MeterObservation(time.perf_counter(), False, 0.0), focused=False)
                    continue
                crop = self.profile.meter_roi.crop(frame)
                observed_at = time.perf_counter()
                observation = self.detector.detect(crop, observed_at)
                action = self.engine.update(observation, focused)
                self._record_feedback(observation, focused)
                if action is None:
                    scheduled_at = self.engine.snapshot().scheduled_fire_at
                    if scheduled_at is not None:
                        remaining = scheduled_at - time.perf_counter()
                        precision_window = timing.precision_wait_threshold_ms / 1000.0
                        if 0.0 < remaining <= precision_window:
                            self.stop_event.wait(remaining)
                            if not self.stop_event.is_set() and not self.pause_event.is_set():
                                action = self.engine.fire_scheduled(time.perf_counter())
                if action and not self.dry_run:
                    if self.stop_event.is_set() or self.pause_event.is_set():
                        continue
                    # Foreground is checked again at the irreversible boundary.
                    if not self.backend.is_foreground(self.window):
                        self.backend.release_all()
                        self.engine.update(observation, focused=False)
                        action = None
                        self.on_log(
                            f"Action cancelled: {self.game_name} lost focus or its window geometry changed."
                        )
                    else:
                        self.backend.press_space(timing.key_hold_ms)
                        pressed_at = time.perf_counter()
                        if self.adaptive_ready:
                            if self.engine.adaptive_prompt_kind == "normal":
                                self.feedback.begin(
                                    action, pressed_at,
                                    source_zone=observation.zone_bounds,
                                )
                                self._feedback_prompt_count = self.engine.prompt_count
                            else:
                                self.on_log(f"Adaptive: skipped {self.engine.adaptive_prompt_kind} phase.")
                        self.on_log(
                            f"SPACE — {action.reason}; marker={action.marker_center:.1f}, "
                            f"predicted={action.predicted_center:.1f}, zone={action.zone_bounds[0]:.1f}–{action.zone_bounds[1]:.1f}."
                        )
                elif action and self.dry_run:
                    self.on_log(
                        f"WOULD PRESS — {action.reason}; marker={action.marker_center:.1f}, "
                        f"predicted={action.predicted_center:.1f}."
                    )
                snapshot = self.engine.snapshot()
                self.diagnostics.record(observation, snapshot, action)
                if (
                    snapshot.state != last_state
                    or action is not None
                    or observed_at - last_report >= timing.diagnostic_refresh_ms / 1000.0
                ):
                    self.on_update(observation, snapshot, action, self.window)
                    last_report = observed_at
                    last_state = snapshot.state
                if timing.scan_interval_ms > 0:
                    self.stop_event.wait(timing.scan_interval_ms / 1000.0)
        except Exception as exc:
            self.on_log(f"Stopped: {exc}")
        finally:
            self.feedback.cancel()
            released = True
            try:
                self.backend.release_all()
            except Exception as exc:
                released = False
                self.on_log(f"Stopped: emergency Space release failed: {exc}")
            finally:
                try:
                    self.backend.stop_capture()
                finally:
                    self.diagnostics.close()
            self.on_log(
                "Monitoring stopped; Space is released." if released else
                "Monitoring stopped; Space release could not be confirmed."
            )
