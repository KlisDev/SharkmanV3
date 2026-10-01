"""Host-side Sober/X11 readiness checks with separate dry/live gates."""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping


@dataclass(frozen=True)
class PreflightReport:
    gui_blockers: tuple[str, ...] = ()
    live_blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def can_open_gui(self) -> bool:
        return not self.gui_blockers

    @property
    def can_send_input(self) -> bool:
        return self.can_open_gui and not self.live_blockers


def inspect_host(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    module_available: Callable[[str], bool] | None = None,
    exists: Callable[[Path], bool] | None = None,
    writable: Callable[[Path], bool] | None = None,
    is_root: bool | None = None,
    check_dependencies: bool = True,
) -> PreflightReport:
    env = os.environ if environ is None else environ
    host = sys.platform if platform is None else platform
    module_available = module_available or (lambda name: importlib.util.find_spec(name) is not None)
    exists = exists or Path.exists
    writable = writable or (lambda path: os.access(path, os.W_OK))
    if is_root is None:
        is_root = bool(getattr(os, "geteuid", lambda: -1)() == 0)

    gui: list[str] = []
    live: list[str] = []
    warnings: list[str] = []
    if not host.startswith("linux"):
        gui.append("This launcher requires Linux; use WINDOWS/easy_run.py on Windows.")
    if env.get("XDG_SESSION_TYPE", "").casefold() == "wayland":
        gui.append("Wayland is not supported yet; log into an X11/Xorg session.")
    elif env.get("WAYLAND_DISPLAY") and env.get("XDG_SESSION_TYPE", "").casefold() != "x11":
        gui.append("XWayland is not supported; log into an actual X11/Xorg desktop session.")
    if not env.get("DISPLAY", "").strip():
        gui.append("DISPLAY is missing; start the launcher inside an X11 desktop session.")
    if is_root:
        gui.append("Do not run SharkmanV3 as root; use the one-time udev installer instead.")
    if sys.version_info < (3, 11):
        gui.append("Python 3.11 or newer is required; Python 3.12 is recommended.")
    if check_dependencies:
        for name in ("tkinter", "customtkinter", "mss", "numpy", "cv2", "PIL", "Xlib", "pynput"):
            if not module_available(name):
                gui.append(f"Missing {name}; install system Tk and LINUX/requirements-linux.txt.")
        if not module_available("evdev"):
            live.append("Missing evdev; install LINUX/requirements-linux.txt for live Space input.")
    uinput = Path("/dev/uinput")
    if not exists(uinput):
        live.append("/dev/uinput is missing; install the supplied udev rule and log out/in.")
    elif not writable(uinput):
        live.append("/dev/uinput is not writable; install the udev rule and log out/in.")
    if env.get("XDG_SESSION_TYPE", "").casefold() not in ("", "x11", "wayland"):
        warnings.append("Session type is unknown; X11 capture and hotkeys must be checked in Setup.")
    return PreflightReport(tuple(gui), tuple(live), tuple(warnings))


def preflight_errors() -> list[str]:
    """Compatibility wrapper for older callers and repository smoke tests."""
    report = inspect_host()
    return [*report.gui_blockers, *report.live_blockers]
