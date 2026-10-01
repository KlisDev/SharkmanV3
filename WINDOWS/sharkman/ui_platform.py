"""Shared wheel units: X11 notches, Windows delta/120, fine macOS gestures."""
from __future__ import annotations

import sys

CANVAS_FONT = "Helvetica" if sys.platform == "darwin" else "Segoe UI"


def wheel_steps(event: object, platform: str | None = None) -> float:
    number = getattr(event, "num", None)
    if number in (4, 5):
        return 1.0 if number == 4 else -1.0
    delta = float(getattr(event, "delta", 0))
    if (platform or sys.platform) == "darwin":
        return max(-4.0, min(4.0, delta / 10.0))
    return max(-4.0, min(4.0, delta / 120.0))
