"""Read-only host checks; privacy prompts are always explicit GUI actions."""
from __future__ import annotations

import importlib.util
import os
import platform
import sys
from dataclasses import dataclass, field


@dataclass
class MacReadiness:
    gui_blockers: list[str] = field(default_factory=list)
    capture_blockers: list[str] = field(default_factory=list)
    live_blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "Setup blocked": self.gui_blockers,
            "Capture blocked": self.capture_blockers,
            "Live input blocked": self.live_blockers,
            "Notice": self.warnings,
        }


def inspect_host(check_dependencies: bool = True) -> MacReadiness:
    report = MacReadiness()
    if sys.platform != "darwin":
        report.gui_blockers.append("Use this launcher on macOS 14 or newer.")
        return report
    version = platform.mac_ver()[0]
    if not version or int(version.split(".")[0]) < 14:
        report.gui_blockers.append("macOS 14 or newer is required.")
    if platform.machine() not in ("arm64", "x86_64"):
        report.gui_blockers.append("Only Apple Silicon and Intel Macs are supported.")
    if sys.version_info < (3, 11):
        report.gui_blockers.append("Python 3.11+ is required; Python 3.12 with Tk is recommended.")
    if getattr(os, "geteuid", lambda: -1)() == 0:
        report.gui_blockers.append("Do not run SharkmanV3 as root or with sudo.")
    modules = ["tkinter"]
    if check_dependencies:
        modules += ["AppKit", "Quartz", "CoreMedia", "ScreenCaptureKit", "ApplicationServices", "dispatch"]
    for module in modules:
        if importlib.util.find_spec(module) is None:
            report.gui_blockers.append(f"Missing {module}; use the source launcher to install dependencies (Tk comes with Python).")
    report.warnings.append("Experimental and gameplay-unverified. No Mac gameplay acceptance test has been completed.")
    return report


def permission_report(permissions: dict[str, bool], hotkeys_ready: bool) -> MacReadiness:
    report = MacReadiness()
    if not permissions.get("screen"):
        report.capture_blockers.append("Grant Screen Recording to this Python/Terminal application, then relaunch if requested.")
    if not permissions.get("post"):
        report.live_blockers.append("Grant Accessibility permission to allow Space input.")
    if not permissions.get("listen"):
        report.live_blockers.append("Grant Input Monitoring permission for global F2/F4/F8.")
    if not hotkeys_ready:
        report.live_blockers.append("The global F4 listener is not ready. Use Check again after granting permissions.")
    report.warnings += [
        "Experimental · gameplay-unverified.",
        "Use Fn/Globe + F2/F4/F8 if these keys control media. Permission changes may require a relaunch.",
    ]
    return report
