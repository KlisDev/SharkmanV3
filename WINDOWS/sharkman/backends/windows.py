from __future__ import annotations

import ctypes
import os
import threading
import time
from collections.abc import Callable
from ctypes import wintypes

import mss
import numpy as np

from ..models import WindowInfo
from .base import BackendError, PlatformBackend

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

# ctypes defaults pointer-returning Win32 functions to a 32-bit c_int. Declare
# the 64-bit-safe return types used by the backend before the first call.
user32.GetForegroundWindow.restype = wintypes.HWND
user32.MonitorFromWindow.restype = wintypes.HANDLE
user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetClientRect.restype = wintypes.BOOL
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.ClientToScreen.restype = wintypes.BOOL
user32.IsIconic.argtypes = [wintypes.HWND]
user32.IsIconic.restype = wintypes.BOOL
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL
user32.BringWindowToTop.argtypes = [wintypes.HWND]
user32.BringWindowToTop.restype = wintypes.BOOL
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
if hasattr(user32, "GetDpiForWindow"):
    user32.GetDpiForWindow.argtypes = [wintypes.HWND]
    user32.GetDpiForWindow.restype = wintypes.UINT
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL
advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
advapi32.OpenProcessToken.restype = wintypes.BOOL
advapi32.GetTokenInformation.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
]
advapi32.GetTokenInformation.restype = wintypes.BOOL

try:
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # per-monitor v2
except (AttributeError, OSError):
    try:
        user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TOKEN_QUERY = 0x0008
TOKEN_ELEVATION_CLASS = 20
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
SPACE_SCAN = 0x39
SW_RESTORE = 9
ULONG_PTR = wintypes.WPARAM


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUTUNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", INPUTUNION)]


user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT


class TOKEN_ELEVATION(ctypes.Structure):
    _fields_ = [("TokenIsElevated", wintypes.DWORD)]


def _send_space(up: bool) -> None:
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
    item = INPUT(type=INPUT_KEYBOARD, union=INPUTUNION(
        ki=KEYBDINPUT(0, SPACE_SCAN, flags, 0, 0)))
    ctypes.set_last_error(0)
    if user32.SendInput(1, ctypes.byref(item), ctypes.sizeof(INPUT)) != 1:
        raise BackendError(f"SendInput failed ({ctypes.get_last_error()})")


def _process_elevated(process_id: int) -> bool | None:
    process = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, process_id)
    if not process:
        return None
    token = wintypes.HANDLE()
    try:
        if not advapi32.OpenProcessToken(process, TOKEN_QUERY, ctypes.byref(token)):
            return None
        elevation = TOKEN_ELEVATION()
        returned = wintypes.DWORD()
        if not advapi32.GetTokenInformation(
            token,
            TOKEN_ELEVATION_CLASS,
            ctypes.byref(elevation),
            ctypes.sizeof(elevation),
            ctypes.byref(returned),
        ):
            return None
        return bool(elevation.TokenIsElevated)
    finally:
        if token:
            kernel32.CloseHandle(token)
        kernel32.CloseHandle(process)


class WindowsBackend(PlatformBackend):
    name = "Windows SendInput"
    overlay_supported = True

    def __init__(self) -> None:
        self._local = threading.local()
        self._input_lock = threading.Lock()
        self._space_down = False

    def _screen(self) -> mss.mss:
        screen = getattr(self._local, "screen", None)
        if screen is None:
            screen = mss.mss()
            self._local.screen = screen
        return screen

    def find_window(self, title: str) -> WindowInfo:
        matches: list[int] = []
        needle = title.casefold().strip()
        enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        @enum_proc
        def visit(hwnd: int, _lparam: int) -> bool:
            if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            if needle in buffer.value.casefold():
                matches.append(hwnd)
            return True

        user32.EnumWindows(visit, 0)
        if not matches:
            raise BackendError(f"No visible window containing {title!r} was found.")
        return self._window_info(matches[0], include_elevation=True)

    def _window_info(self, hwnd: int, *, include_elevation: bool = False) -> WindowInfo:
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            raise BackendError("The bound Roblox window is hidden or minimized.")
        rect = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            raise BackendError("Could not read the Roblox client rectangle.")
        origin = wintypes.POINT(0, 0)
        if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
            raise BackendError("Could not map the Roblox client to the screen.")
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width <= 0 or height <= 0:
            raise BackendError("The Roblox client is minimized or has no visible area.")
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        dpi = int(user32.GetDpiForWindow(hwnd)) if hasattr(user32, "GetDpiForWindow") else 96
        monitor = user32.MonitorFromWindow(hwnd, 2)
        mode = "windowed"
        if monitor:
            class MONITORINFO(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]
            info = MONITORINFO(cbSize=ctypes.sizeof(MONITORINFO))
            if user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
                mw = info.rcMonitor.right - info.rcMonitor.left
                mh = info.rcMonitor.bottom - info.rcMonitor.top
                if width >= mw and height >= mh:
                    mode = "fullscreen"
        elevation = None
        if include_elevation:
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            elevation = _process_elevated(pid.value)
        return WindowInfo(
            int(hwnd), buffer.value, origin.x, origin.y, width, height, dpi, mode, elevation
        )

    def capture(self, window: WindowInfo) -> np.ndarray:
        shot = self._screen().grab(
            {"left": window.left, "top": window.top, "width": window.width, "height": window.height}
        )
        return np.asarray(shot, dtype=np.uint8)[:, :, :3].copy()

    def is_foreground(self, window: WindowInfo) -> bool:
        if int(user32.GetForegroundWindow()) != window.handle:
            return False
        try:
            current = self._window_info(window.handle)
        except BackendError:
            return False
        return self._same_binding(window, current)

    @staticmethod
    def _same_binding(bound: WindowInfo, current: WindowInfo) -> bool:
        return (
            bound.handle == current.handle
            and bound.title == current.title
            and (bound.left, bound.top) == (current.left, current.top)
            and bound.fingerprint == current.fingerprint
        )

    def activate_window(self, window: WindowInfo) -> bool:
        hwnd = wintypes.HWND(window.handle)
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, SW_RESTORE)
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        return self.is_foreground(window)

    def privilege_error(self, window: WindowInfo) -> str | None:
        current = _process_elevated(os.getpid())
        if window.elevated is True and current is False:
            return "Roblox is elevated but SharkmanV3 is not. Restart SharkmanV3 as administrator."
        return None

    def press_space(self, hold_ms: float) -> None:
        with self._input_lock:
            _send_space(False)
            self._space_down = True
            try:
                time.sleep(hold_ms / 1000.0)
            finally:
                _send_space(True)
                self._space_down = False

    def release_all(self) -> None:
        with self._input_lock:
            if self._space_down:
                try:
                    _send_space(True)
                finally:
                    self._space_down = False

    def bind_hotkeys(
        self,
        toggle: Callable[[], None],
        stop: Callable[[], None],
        debug: Callable[[], None],
    ) -> Callable[[], None]:
        try:
            import keyboard

            handles = [
                keyboard.add_hotkey("f2", toggle),
                keyboard.add_hotkey("f4", stop),
                keyboard.add_hotkey("f8", debug),
            ]
        except Exception as exc:
            raise BackendError(f"global hotkeys unavailable: {exc}") from exc

        def unbind() -> None:
            for handle in handles:
                try:
                    keyboard.remove_hotkey(handle)
                except Exception:
                    pass

        return unbind
