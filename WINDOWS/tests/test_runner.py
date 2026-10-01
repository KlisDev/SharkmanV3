from __future__ import annotations

from collections.abc import Callable

import numpy as np

from sharkman.backends.base import PlatformBackend
from sharkman.config import CalibrationProfile
from sharkman.models import WindowInfo
from sharkman.runner import LiveRunner

from .helpers import calibrated_profile


class FakeBackend(PlatformBackend):
    def __init__(self, *, capture_error: bool = False) -> None:
        self.capture_error = capture_error
        self.releases = 0
        self.callbacks: tuple[Callable[[], None], ...] = ()

    def find_window(self, _title: str) -> WindowInfo:
        return WindowInfo(1, "Roblox", 0, 0, 30, 240, 96, "windowed")

    def capture(self, _window: WindowInfo) -> np.ndarray:
        if self.capture_error:
            raise RuntimeError("capture failed")
        return np.zeros((240, 30, 3), dtype=np.uint8)

    def is_foreground(self, _window: WindowInfo) -> bool:
        return True

    def press_space(self, _hold_ms: float) -> None:
        return

    def release_all(self) -> None:
        self.releases += 1

    def bind_hotkeys(
        self,
        toggle: Callable[[], None],
        stop: Callable[[], None],
        debug: Callable[[], None],
    ) -> Callable[[], None]:
        self.callbacks = (toggle, stop, debug)
        return lambda: None


def make_runner(backend: FakeBackend) -> tuple[LiveRunner, list[str]]:
    profile = calibrated_profile()
    profile.mark_validated()
    profile.timing_overrides = {"startup_delay_ms": 0}
    logs: list[str] = []
    runner = LiveRunner(backend, profile, lambda *_: None, logs.append, dry_run=True)
    return runner, logs


def test_exception_shutdown_always_releases_space() -> None:
    backend = FakeBackend(capture_error=True)
    runner, logs = make_runner(backend)

    runner.start()
    runner.join()

    assert backend.releases >= 1
    assert any("capture failed" in line for line in logs)
    assert logs[-1] == "Monitoring stopped; Space is released."


def test_shared_runner_uses_the_linux_backend_game_name() -> None:
    class SoberBackend(FakeBackend):
        game_name = "Sober/Roblox"

    backend = SoberBackend(capture_error=True)
    runner, logs = make_runner(backend)

    runner.start()
    runner.join()

    assert any("Sober/Roblox is foreground" in line for line in logs)
    assert backend.releases >= 1


def test_stop_always_releases_space_even_before_thread_start() -> None:
    backend = FakeBackend()
    runner, _logs = make_runner(backend)

    runner.stop()

    assert backend.releases == 1


def test_runner_refuses_unvalidated_calibration_before_capture() -> None:
    backend = FakeBackend()
    logs: list[str] = []
    profile = CalibrationProfile(name="incomplete")
    profile.timing_overrides = {"startup_delay_ms": 0}
    runner = LiveRunner(backend, profile, lambda *_: None, logs.append, dry_run=True)

    runner.start()
    runner.join()

    assert any("Calibration is incomplete" in line for line in logs)
    assert backend.releases >= 1


def test_live_runner_refuses_missing_input_before_capture() -> None:
    class MissingInputBackend(FakeBackend):
        def input_error(self, _window: WindowInfo) -> str:
            return "/dev/uinput is unavailable"

    backend = MissingInputBackend()
    profile = calibrated_profile()
    profile.mark_validated()
    profile.timing_overrides = {"startup_delay_ms": 0}
    logs: list[str] = []
    runner = LiveRunner(backend, profile, lambda *_: None, logs.append, dry_run=False)

    runner.start()
    runner.join()

    assert any("/dev/uinput is unavailable" in line for line in logs)
    assert backend.releases >= 1


def test_unavailable_frames_preserve_lockout_and_cleanup_capture() -> None:
    from sharkman.backends.base import CaptureUnavailable
    from sharkman.models import EngineState

    class GapBackend(FakeBackend):
        calls = 0
        closed = 0

        def capture(self, _window):
            self.calls += 1
            if self.calls <= 2:
                raise CaptureUnavailable("idle")
            raise RuntimeError("end test")

        def stop_capture(self):
            self.closed += 1

    backend = GapBackend()
    runner, logs = make_runner(backend)
    runner.engine.state = EngineState.WAIT_GONE
    runner.engine._fired_at = 1
    runner.engine._gone_since = 2
    runner.start()
    runner.join()
    assert runner.engine.state == EngineState.WAIT_GONE
    assert runner.engine._gone_since is None
    assert runner.engine._fired_at == 1
    assert backend.closed == 1
    assert sum("Capture unavailable" in line for line in logs) == 1


def test_stop_during_capture_never_runs_detection_or_input() -> None:
    class StoppingBackend(FakeBackend):
        def capture(self, _window):
            runner.stop()
            return super().capture(_window)

        def press_space(self, _hold):
            raise AssertionError("Input after Stop")

    backend = StoppingBackend()
    runner, _logs = make_runner(backend)
    runner.dry_run = False
    runner.detector.detect = lambda *_: (_ for _ in ()).throw(AssertionError("Detection after Stop"))
    runner.start()
    runner.join()
    assert runner.engine.action_count == 0
    assert backend.releases >= 2


def test_unfocused_window_is_not_captured_or_recorded() -> None:
    class UnfocusedBackend(FakeBackend):
        def is_foreground(self, _window):
            runner.stop_event.set()
            return False

        def capture(self, _window):
            captured.append(True)
            return super().capture(_window)

    captured = []
    recorded = []
    backend = UnfocusedBackend()
    runner, _logs = make_runner(backend)
    runner.diagnostics.record = lambda *args: recorded.append(args)
    runner._run()
    assert not captured
    assert not recorded
    assert backend.releases >= 1


def test_focus_lost_during_capture_discards_frame_before_detection_and_recording() -> None:
    class SwitchingBackend(FakeBackend):
        focused = True

        def capture(self, _window):
            self.focused = False
            return super().capture(_window)

        def is_foreground(self, _window):
            if not self.focused:
                runner.stop_event.set()
            return self.focused

    detected = []
    recorded = []
    backend = SwitchingBackend()
    runner, _logs = make_runner(backend)
    runner.detector.detect = lambda *args: detected.append(args)
    runner.diagnostics.record = lambda *args: recorded.append(args)
    runner._run()
    assert not detected
    assert not recorded
    assert runner.engine.action_count == 0
    assert backend.releases >= 1
