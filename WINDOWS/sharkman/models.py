from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NormalizedRect:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def valid(self) -> bool:
        return (
            0.0 <= self.left < self.right <= 1.0
            and 0.0 <= self.top < self.bottom <= 1.0
        )

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        left = int(round(self.left * width))
        top = int(round(self.top * height))
        right = int(round(self.right * width))
        bottom = int(round(self.bottom * height))
        return (
            max(0, min(left, width - 1)),
            max(0, min(top, height - 1)),
            max(1, min(right, width)),
            max(1, min(bottom, height)),
        )

    def crop(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        left, top, right, bottom = self.pixels(width, height)
        return image[top:bottom, left:right]

    def to_dict(self) -> dict[str, float]:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NormalizedRect":
        return cls(*(float(data[k]) for k in ("left", "top", "right", "bottom")))


@dataclass(frozen=True)
class ColorSample:
    rgb: tuple[int, int, int]
    tolerance: float = 45.0

    def to_dict(self) -> dict[str, Any]:
        # Keep validation signatures stable across a JSON round-trip.
        return {"rgb": list(self.rgb), "tolerance": float(self.tolerance)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColorSample":
        rgb = tuple(int(v) for v in data["rgb"])
        if len(rgb) != 3:
            raise ValueError("a color sample must contain three RGB channels")
        return cls(rgb=rgb, tolerance=float(data.get("tolerance", 45.0)))


@dataclass(frozen=True)
class WindowFingerprint:
    width: int = 0
    height: int = 0
    dpi: int = 96
    display_mode: str = "windowed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "display_mode": self.display_mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "WindowFingerprint":
        data = data or {}
        return cls(
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            dpi=int(data.get("dpi", 96)),
            display_mode=str(data.get("display_mode", "windowed")),
        )


@dataclass(frozen=True)
class WindowInfo:
    handle: int
    title: str
    left: int
    top: int
    width: int
    height: int
    dpi: int = 96
    display_mode: str = "windowed"
    elevated: bool | None = None

    @property
    def fingerprint(self) -> WindowFingerprint:
        return WindowFingerprint(self.width, self.height, self.dpi, self.display_mode)


@dataclass
class MeterObservation:
    timestamp: float
    present: bool
    confidence: float
    marker_center: float | None = None
    marker_span: tuple[int, int] | None = None
    zone_bounds: tuple[int, int] | None = None
    track_bounds: tuple[int, int] | None = None
    velocity: float = 0.0
    reasons: tuple[str, ...] = ()
    crop: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def complete(self) -> bool:
        return self.present and self.marker_center is not None and self.zone_bounds is not None


class EngineState(str, Enum):
    IDLE = "IDLE"
    ACQUIRING = "ACQUIRING"
    TRACKING = "TRACKING"
    ARMED = "ARMED"
    FIRED = "FIRED"
    WAIT_GONE = "WAIT_GONE"


@dataclass(frozen=True)
class FireAction:
    timestamp: float
    marker_center: float
    predicted_center: float
    zone_bounds: tuple[float, float]
    velocity: float
    reason: str


@dataclass(frozen=True)
class EngineSnapshot:
    state: EngineState
    velocity: float
    predicted_center: float | None
    scheduled_fire_at: float | None
    action_count: int
    prompt_count: int
    focused: bool
