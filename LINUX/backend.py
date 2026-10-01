"""Sober/X11 window, capture, hotkey, and virtual Space-key adapter."""
from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import mss
import numpy as np

from sharkman.backends.base import BackendError, PlatformBackend
from sharkman.models import WindowInfo

SOBER_CLASSES = {"sober", "org.vinegarhq.sober"}
MIN_CLIENT_WIDTH = 400
MIN_CLIENT_HEIGHT = 300


def _window_name(window: object) -> str:
    name = window.get_wm_name() or ""
    return name.decode("utf-8", "replace") if isinstance(name, bytes) else str(name)


def _window_classes(window: object) -> set[str]:
    classes = window.get_wm_class() or ()
    return {
        (item.decode("utf-8", "replace") if isinstance(item, bytes) else str(item)).casefold()
        for item in classes
    }


def _matches_sober(window: object, title: str) -> bool:
    class_names = _window_classes(window)
    name = _window_name(window).casefold()
    wanted = title.strip().casefold()
    return bool(class_names & SOBER_CLASSES or "sober" in name or (wanted and wanted in name))


def _sober_priority(window: object) -> int:
    if _window_classes(window) & SOBER_CLASSES:
        return 2
    return 1 if "sober" in _window_name(window).casefold() else 0


def _walk(root: object) -> Iterator[object]:
    stack = [root]
    seen: set[int] = set()
    while stack:
        window = stack.pop()
        if int(window.id) in seen:
            continue
        seen.add(int(window.id))
        yield window
        try:
            stack.extend(window.query_tree().children)
        except Exception:
            continue


def _xft_dpi(root: object, display: object) -> int:
    """Xft.dpi is the X11 logical scale; geometry still uses captured pixels."""
    try:
        property_value = root.get_full_property(display.intern_atom("RESOURCE_MANAGER"), 0)
        raw = property_value.value if property_value is not None else b""
        settings = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        match = re.search(r"(?im)^\s*Xft\.dpi\s*:\s*(\d+(?:\.\d+)?)", settings)
        if match:
            return max(50, min(500, round(float(match.group(1)))))
    except Exception:
        pass
    return 96


class LinuxX11Backend(PlatformBackend):
    """Host-side X11 backend; no Wayland or desktop-fallback capture."""

    name = "Linux Sober/X11 / uinput"
    game_name = "Sober/Roblox"
    default_window_title = "Sober"
    overlay_supported = True

    def __init__(self, display_factory: Callable[[], object] | None = None) -> None:
        if os.environ.get("XDG_SESSION_TYPE", "").casefold() == "wayland":
            raise BackendError("Wayland is not supported yet. Log into an X11 session.")
        if (os.environ.get("WAYLAND_DISPLAY")
                and os.environ.get("XDG_SESSION_TYPE", "").casefold() != "x11"):
            raise BackendError("XWayland is not supported. Log into an actual X11 session.")
        if not os.environ.get("DISPLAY"):
            raise BackendError("DISPLAY is missing; Sober capture requires X11.")
        try:
            if display_factory is None:
                from Xlib import display as xdisplay

                display_factory = xdisplay.Display
            self._display = display_factory()
        except Exception as exc:
            raise BackendError(f"Could not connect to X11: {exc}") from exc
        self._xlock = threading.RLock()
        self._local = threading.local()
        self._input_lock = threading.Lock()
        self._uinput = None
        self._space_down = False
        self.overlay_error = ""

    def _screen(self) -> mss.mss:
        screen = getattr(self._local, "screen", None)
        if screen is None:
            screen = mss.mss()
            self._local.screen = screen
        return screen

    def _display_mode(self, window: object, left: int, top: int, width: int, height: int) -> str:
        try:
            from Xlib import Xatom

            state = window.get_full_property(self._display.intern_atom("_NET_WM_STATE"), Xatom.ATOM)
            fullscreen = self._display.intern_atom("_NET_WM_STATE_FULLSCREEN")
            if state is not None and fullscreen in state.value:
                return "fullscreen"
        except Exception:
            pass
        try:
            for monitor in self._screen().monitors[1:]:
                if (abs(left - int(monitor["left"])) <= 2
                        and abs(top - int(monitor["top"])) <= 2
                        and abs(width - int(monitor["width"])) <= 2
                        and abs(height - int(monitor["height"])) <= 2):
                    return "fullscreen"
        except Exception:
            pass
        return "windowed"

    def find_window(self, title: str) -> WindowInfo:
        with self._xlock:
            root = self._display.screen().root
            candidates: list[tuple[int, int, object, str, object, object]] = []
            for window in _walk(root):
                if int(window.id) == int(root.id):
                    continue
                try:
                    if not _matches_sober(window, title):
                        continue
                    geometry = window.get_geometry()
                    if geometry.width < MIN_CLIENT_WIDTH or geometry.height < MIN_CLIENT_HEIGHT:
                        continue
                    attributes = window.get_attributes()
                    if attributes.map_state != 2:  # X.IsViewable
                        continue
                    absolute = window.translate_coords(root, 0, 0)
                    candidates.append((_sober_priority(window), geometry.width * geometry.height, window,
                                       _window_name(window), geometry, absolute))
                except Exception:
                    continue
            if not candidates:
                raise BackendError("No visible Sober/Roblox X11 client was found. Open Sober, then bind it in Setup.")
            _priority, _area, window, name, geometry, absolute = max(
                candidates, key=lambda item: (item[0], item[1]))
            left, top = int(absolute.x), int(absolute.y)
            width, height = int(geometry.width), int(geometry.height)
            return WindowInfo(
                int(window.id), name, left, top, width, height,
                _xft_dpi(root, self._display),
                self._display_mode(window, left, top, width, height),
            )

    def capture(self, window: WindowInfo) -> np.ndarray:
        try:
            shot = self._screen().grab({
                "left": window.left, "top": window.top,
                "width": window.width, "height": window.height,
            })
            frame = np.asarray(shot, dtype=np.uint8)[:, :, :3].copy()
        except Exception as exc:
            raise BackendError(f"Could not capture the Sober client: {exc}") from exc
        if frame.shape != (window.height, window.width, 3):
            raise BackendError("Sober capture has the wrong size; rebind the window.")
        if int(frame.max()) <= 3:
            raise BackendError("Sober capture is blank/black; check X11 capture before calibration or running.")
        return frame

    def _related(self, active_id: int, target_id: int) -> bool:
        if active_id == target_id:
            return True
        root_id = int(self._display.screen().root.id)
        if active_id == root_id or target_id == root_id:
            return False
        for child_id, ancestor_id in ((active_id, target_id), (target_id, active_id)):
            try:
                window = self._display.create_resource_object("window", child_id)
                for _ in range(16):
                    parent = window.query_tree().parent
                    if parent is None or int(parent.id) in (int(window.id), root_id):
                        break
                    if int(parent.id) == ancestor_id:
                        return True
                    window = parent
            except Exception:
                continue
        return False

    def is_foreground(self, window: WindowInfo) -> bool:
        with self._xlock:
            try:
                root = self._display.screen().root
                client = self._display.create_resource_object("window", window.handle)
                geometry = client.get_geometry()
                absolute = client.translate_coords(root, 0, 0)
                if (int(absolute.x), int(absolute.y), int(geometry.width), int(geometry.height)) != (
                    window.left, window.top, window.width, window.height,
                ):
                    return False
                if _xft_dpi(root, self._display) != window.dpi:
                    return False
                if self._display_mode(client, window.left, window.top, window.width, window.height) != window.display_mode:
                    return False
                atom = self._display.intern_atom("_NET_ACTIVE_WINDOW")
                active = root.get_full_property(atom, 0)
                if (active is not None and len(active.value)
                        and self._related(int(active.value[0]), window.handle)):
                    return True
                focus = self._display.get_input_focus().focus
                return self._related(int(focus.id), window.handle)
            except Exception:
                return False

    def activate_window(self, window: WindowInfo) -> bool:
        """Raise Sober for calibration only, then confirm actual X11 focus."""
        try:
            from Xlib import X

            with self._xlock:
                target = self._display.create_resource_object("window", window.handle)
                target.configure(stack_mode=X.Above)
                target.set_input_focus(X.RevertToParent, X.CurrentTime)
                self._display.sync()
            deadline = time.monotonic() + 0.35
            while time.monotonic() < deadline:
                if self.is_foreground(window):
                    return True
                time.sleep(0.02)
        except Exception:
            pass
        return False

    def input_error(self, _window: WindowInfo) -> str | None:
        path = Path("/dev/uinput")
        if not path.exists() or not os.access(path, os.W_OK):
            return "Live input needs writable /dev/uinput. Install LINUX/install-udev.sh and log out/in."
        try:
            self._device()  # register the virtual keyboard before a timing prompt
        except BackendError as exc:
            return str(exc)
        return None

    def _device(self):
        with self._input_lock:
            if self._uinput is None:
                try:
                    from evdev import UInput, ecodes

                    self._uinput = UInput({ecodes.EV_KEY: [ecodes.KEY_SPACE]}, name="SharkmanV3 Space")
                    time.sleep(0.2)  # allow X11/Sober to register the new keyboard
                except Exception as exc:
                    raise BackendError(f"/dev/uinput could not create a keyboard: {exc}") from exc
            return self._uinput

    def press_space(self, hold_ms: float) -> None:
        from evdev import ecodes

        device = self._device()
        with self._input_lock:
            device.write(ecodes.EV_KEY, ecodes.KEY_SPACE, 1)
            self._space_down = True
            try:
                device.syn()
                time.sleep(max(0.0, hold_ms) / 1000.0)
            finally:
                if not self._release_locked():
                    raise BackendError("Could not release Space after three /dev/uinput attempts.")

    def _release_locked(self) -> bool:
        if not self._space_down or self._uinput is None:
            return True
        from evdev import ecodes

        for _attempt in range(3):
            try:
                self._uinput.write(ecodes.EV_KEY, ecodes.KEY_SPACE, 0)
                self._uinput.syn()
                self._space_down = False
                return True
            except Exception:
                time.sleep(0.01)
        return False

    def release_all(self) -> None:
        with self._input_lock:
            if not self._release_locked():
                raise BackendError("Could not release Space after three /dev/uinput attempts.")

    def bind_hotkeys(
        self,
        toggle: Callable[[], None],
        stop: Callable[[], None],
        debug: Callable[[], None],
    ) -> Callable[[], None]:
        try:
            from pynput.keyboard import GlobalHotKeys

            hotkeys = GlobalHotKeys({"<f2>": toggle, "<f4>": stop, "<f8>": debug})
            hotkeys.daemon = True
            hotkeys.start()
        except Exception as exc:
            raise BackendError(f"X11 global hotkeys unavailable: {exc}") from exc

        def unbind() -> None:
            hotkeys.stop()
            if hotkeys.is_alive() and threading.current_thread() is not hotkeys:
                hotkeys.join(timeout=0.5)

        return unbind

    def create_overlay(self, root: object) -> object:
        from overlay import NullOverlay, X11Overlay

        try:
            return X11Overlay(root)
        except Exception as exc:
            self.overlay_supported = False
            self.overlay_error = str(exc)
            return NullOverlay()

    def close(self) -> None:
        try:
            self.release_all()
        finally:
            if self._uinput is not None:
                self._uinput.close()
                self._uinput = None
            with self._xlock:
                self._display.close()
