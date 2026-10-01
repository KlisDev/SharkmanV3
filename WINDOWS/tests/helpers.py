from __future__ import annotations

import cv2
import numpy as np

from sharkman.config import CalibrationProfile
from sharkman.models import ColorSample, NormalizedRect, WindowFingerprint


def calibrated_profile(name: str = "test") -> CalibrationProfile:
    profile = CalibrationProfile(name=name)
    profile.fingerprint = WindowFingerprint(30, 240, 96, "windowed")
    profile.meter_roi = NormalizedRect(0.0, 0.0, 1.0, 1.0)
    profile.samples = {
        "green": [ColorSample((35, 245, 30), 55)],
        "marker": [ColorSample((5, 5, 5), 30)],
        "track": [ColorSample((220, 30, 15), 70), ColorSample((245, 220, 30), 70)],
    }
    profile.detection.min_track_height_fraction = 0.7
    profile.detection.min_colored_fraction = 0.85
    return profile


def synthetic_meter(marker_y: int = 40, height: int = 240, width: int = 30) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    # A saturated red -> yellow -> green -> yellow -> red rail.
    stops = [
        (0.0, (15, 20, 220)),
        (0.32, (30, 220, 245)),
        (0.43, (30, 245, 35)),
        (0.57, (30, 245, 35)),
        (0.68, (30, 220, 245)),
        (1.0, (15, 20, 220)),
    ]
    for y in range(height):
        position = y / max(1, height - 1)
        for (p0, c0), (p1, c1) in zip(stops, stops[1:], strict=False):
            if p0 <= position <= p1:
                ratio = (position - p0) / max(1e-6, p1 - p0)
                rgb = tuple(int(a + (b - a) * ratio) for a, b in zip(c0, c1, strict=True))
                image[y, :] = rgb[::-1]
                break
    cv2.rectangle(image, (0, max(0, marker_y - 3)), (width - 1, min(height - 1, marker_y + 3)), (5, 5, 5), -1)
    return image
