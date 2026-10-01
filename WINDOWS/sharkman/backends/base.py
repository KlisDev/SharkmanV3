from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from ..models import WindowInfo


class BackendError(RuntimeError):
    pass


class CaptureUnavailable(BackendError):
    """No fresh frame; this is not evidence that the meter disappeared."""


class WindowSelectionRequired(BackendError):
    def __init__(self, windows: tuple[WindowInfo, ...]) -> None:
        super().__init__("Multiple Roblox clients found. Select one in Setup.")
        self.windows = windows


class PlatformBackend(ABC):
    """Small OS boundary shared by the GUI, live runner and tests."""

    name = "unsupported"
    overlay_supported = False
    default_window_title = "Roblox"
    game_name = "Roblox"
    overlay_error = ""

    def configure_timing(self, timing: object) -> None:
        """Apply platform capture settings before starting a session."""
        return

    def stop_capture(self) -> None:
        """Release optional streaming resources and wake blocked capture calls."""
        return

    def begin_input_session(self) -> None:
        """Clear a platform cancellation latch after explicit Start/Resume."""
        return

    def readiness(self) -> dict[str, list[str]]:
        return {}

    def permission_actions(self) -> dict[str, Callable[[], None]]:
        return {}

    def select_window(self, handle: int | None) -> None:
        """None requests a fresh selection on the next explicit Bind action."""
        if handle is not None:
            raise BackendError("This backend does not support explicit window selection.")

    @abstractmethod
    def find_window(self, title: str) -> WindowInfo:
        raise NotImplementedError

    @abstractmethod
    def capture(self, window: WindowInfo) -> np.ndarray:
        """Return the visible game client as a BGR uint8 image."""

    @abstractmethod
    def is_foreground(self, window: WindowInfo) -> bool:
        raise NotImplementedError

    @abstractmethod
    def press_space(self, hold_ms: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def release_all(self) -> None:
        raise NotImplementedError

    def privilege_error(self, window: WindowInfo) -> str | None:
        return None

    def input_error(self, window: WindowInfo) -> str | None:
        """Return a live-input blocker; dry runs do not need an input device."""
        return None

    def create_overlay(self, root: object) -> object | None:
        """Return a platform overlay, or None for the shared Windows renderer."""
        return None

    def activate_window(self, window: WindowInfo) -> bool:
        """Bring the bound client forward before visible-screen capture."""
        return self.is_foreground(window)

    def bind_hotkeys(
        self,
        toggle: Callable[[], None],
        stop: Callable[[], None],
        debug: Callable[[], None],
    ) -> Callable[[], None]:
        """Bind global F2/F4/F8 keys and return an unbind callback."""
        raise BackendError("global hotkeys are unavailable on this platform")

    def close(self) -> None:
        try:
            self.release_all()
        finally:
            self.stop_capture()
