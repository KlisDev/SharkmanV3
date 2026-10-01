from __future__ import annotations

import sys

from .base import BackendError, PlatformBackend


def create_backend() -> PlatformBackend:
    if sys.platform == "win32":
        from .windows import WindowsBackend

        return WindowsBackend()
    if sys.platform.startswith("linux"):
        raise BackendError("Use LINUX/easy_run_linux.py to load the Sober/X11 backend.")
    if sys.platform == "darwin":
        raise BackendError("Use MAC/easy_run_mac.py to load the experimental macOS backend.")
    raise BackendError(f"{sys.platform} does not have a SharkmanV3 backend yet")


__all__ = ["BackendError", "PlatformBackend", "create_backend"]
