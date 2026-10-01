"""macOS safety policy, independent of the native bridge for mock testing."""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sharkman.backends.base import BackendError, PlatformBackend, WindowSelectionRequired
from sharkman.config import TimingConfig
from sharkman.models import WindowInfo

from .capture import FreshFrames
from .preflight import permission_report

ROBLOX_BUNDLE = "com.roblox.RobloxPlayer"


@dataclass(frozen=True)
class MacWindow:
    handle: int
    pid: int
    bundle: str
    title: str
    x: float
    y: float
    point_width: float
    point_height: float
    scale: float
    display_id: int
    display_mode: str
    visible: bool = True

    @property
    def info(self) -> WindowInfo:
        # left/top are global Quartz points on Mac, never capture pixels.
        # The overlay converts ROI pixels with this binding's exact scale.
        return WindowInfo(
            self.handle, self.title, round(self.x), round(self.y),
            round(self.point_width * self.scale), round(self.point_height * self.scale),
            round(96 * self.scale), self.display_mode,
        )


def choose_window(windows: list[MacWindow], title: str, selected: int | None,
                  *, test_pid: int | None = None) -> MacWindow:
    eligible = [w for w in windows if w.visible and w.point_width > 0 and w.point_height > 0
                and (w.pid == test_pid if test_pid is not None else w.bundle == ROBLOX_BUNDLE)
                and title.casefold() in w.title.casefold()]
    if selected is not None:
        eligible = [w for w in eligible if w.handle == selected]
    if not eligible:
        raise BackendError("No matching visible Roblox client. Open Roblox and bind its window in Setup.")
    if len(eligible) != 1:
        raise WindowSelectionRequired(tuple(w.info for w in eligible))
    return eligible[0]


class MacBackend(PlatformBackend):
    name = "macOS ScreenCaptureKit / Quartz (experimental)"
    game_name = "Roblox for Mac"
    overlay_supported = True

    def __init__(self, native: object | None = None, *, test_pid: int | None = None) -> None:
        if native is None:
            from .native import NativeMac

            native = NativeMac()
        self.native = native
        self.test_pid = test_pid
        self.timing = TimingConfig()
        self.selected: int | None = None
        self.bound: MacWindow | None = None
        self.frames: FreshFrames | None = None
        self.stream = None
        self._capture_lock = threading.RLock()
        self._input_lock = threading.RLock()
        self._cancel = threading.Event()
        self._cancel.set()
        self._space_down = False
        self.listener = None
        self.overlay_error = ""

    def configure_timing(self, timing: TimingConfig) -> None:
        self.timing = timing

    def select_window(self, handle: int | None) -> None:
        self.selected = handle
        self.bound = None
        self.stop_capture()

    def find_window(self, title: str) -> WindowInfo:
        error = self.privilege_error(None)
        if error:
            raise BackendError(error)
        windows = self.native.windows(self.timing.mac_capture_start_ms)
        candidate = choose_window(windows, title, self.selected, test_pid=self.test_pid)
        if self.bound != candidate:
            self.stop_capture()
        self.bound = candidate
        return candidate.info

    def _binding_matches(self, window: WindowInfo) -> bool:
        return bool(self.bound and self.bound.info == window and self.native.matches(self.bound))

    def capture(self, window: WindowInfo) -> np.ndarray:
        if self.privilege_error(window):
            self.release_all()
            raise BackendError("Screen Recording permission was revoked; recheck Mac readiness.")
        if not self._binding_matches(window):
            self.release_all()
            raise BackendError("Roblox window geometry/display changed. Stop and rebind in Setup.")
        with self._capture_lock:
            started = self.stream is None
            if started:
                self.frames = FreshFrames()
                try:
                    self.stream = self.native.start_stream(self.bound, self.frames, self.timing)
                except Exception:
                    self.frames.close()
                    self.frames = None
                    raise
            frames = self.frames
        image = frames.take(
            self.timing.mac_capture_start_ms if started else self.timing.mac_frame_wait_ms,
            self.timing.mac_frame_age_ms,
        )
        if image.shape != (window.height, window.width, 3):
            self.release_all()
            raise BackendError("Capture pixel size changed. Rebind and recalibrate the Mac window.")
        return image

    def stop_capture(self) -> None:
        with self._capture_lock:
            stream, self.stream = self.stream, None
            frames, self.frames = self.frames, None
            if frames:
                frames.close()
            if stream:
                self.native.stop_stream(stream, self.timing.mac_capture_stop_ms)

    def is_foreground(self, window: WindowInfo) -> bool:
        return bool(self._binding_matches(window) and self.native.foreground(self.bound))

    def activate_window(self, window: WindowInfo) -> bool:
        return bool(self._binding_matches(window) and self.native.activate(self.bound))

    def privilege_error(self, _window: WindowInfo | None) -> str | None:
        if not self.native.permissions().get("screen"):
            return "Screen Recording is required for calibration and dry run. See Mac readiness."
        return None

    def input_error(self, _window: WindowInfo) -> str | None:
        report = permission_report(self.native.permissions(), self.hotkeys_ready)
        return " ".join(report.live_blockers) or None

    @property
    def hotkeys_ready(self) -> bool:
        return bool(self.listener and self.listener.healthy)

    def readiness(self) -> dict[str, list[str]]:
        return permission_report(self.native.permissions(), self.hotkeys_ready).as_dict()

    def permission_actions(self) -> dict[str, Callable[[], None]]:
        return {
            "Request Screen Recording": lambda: self.native.request_permission("screen"),
            "Request Accessibility": lambda: self.native.request_permission("post"),
            "Request Input Monitoring": lambda: self.native.request_permission("listen"),
            "Open Privacy settings": self.native.open_settings,
        }

    def begin_input_session(self) -> None:
        with self._input_lock:
            self._cancel.clear()

    def press_space(self, hold_ms: float) -> None:
        with self._input_lock:
            if self._cancel.is_set():
                raise BackendError("Input cancelled; use Start/Resume after checking focus and permissions.")
            if self.bound is None or not self.is_foreground(self.bound.info):
                self.release_all()
                raise BackendError("Space cancelled: Roblox lost focus or its binding changed.")
            error = self.input_error(self.bound.info) or self.privilege_error(self.bound.info)
            if error:
                self.release_all()
                raise BackendError(error)
            if self.frames is None or not self.frames.fresh_for_input(self.timing.mac_frame_age_ms):
                self.release_all()
                raise BackendError("Space cancelled: the captured frame is no longer fresh.")
            if self._cancel.is_set():
                raise BackendError("Space cancelled during final input checks.")
            # Mark first: if native posting raises after posting, still send key-up.
            self._space_down = True
            try:
                self.native.space(True)
            except Exception:
                self.release_all()
                raise
        try:
            self._cancel.wait(hold_ms / 1000)
        finally:
            with self._input_lock:
                if self._space_down:
                    self.native.space(False)
                    self._space_down = False

    def release_all(self) -> None:
        self._cancel.set()
        with self._input_lock:
            if self._space_down:
                self.native.space(False)
                self._space_down = False

    def bind_hotkeys(self, toggle, stop, debug):
        if self.listener:
            self.listener.close()

        def emergency() -> None:
            try:
                self.release_all()
            finally:
                stop()

        self.listener = self.native.hotkeys(toggle, emergency, debug, self.timing)
        return self.listener.close

    def create_overlay(self, root: object) -> object:
        from .overlay import MacOverlay, NullOverlay

        try:
            return MacOverlay(self)
        except Exception as exc:
            self.overlay_error = f"Mac overlay unavailable: {exc}. Diagnostic logging still works."
            return NullOverlay(self.overlay_error)

    def close(self) -> None:
        try:
            super().close()
        finally:
            if self.listener:
                self.listener.close()
