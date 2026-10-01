from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import CalibrationProfile
from .models import ColorSample, MeterObservation


def sample_median_rgb(
    image_bgr: np.ndarray,
    x: int,
    y: int,
    radius: int = 2,
) -> tuple[int, int, int]:
    """Sample a small median patch, returning RGB for profile persistence."""
    if image_bgr.size == 0 or image_bgr.ndim != 3:
        raise ValueError("a BGR image is required")
    height, width = image_bgr.shape[:2]
    x = max(0, min(width - 1, int(x)))
    y = max(0, min(height - 1, int(y)))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    bgr = np.median(image_bgr[y0:y1, x0:x1].reshape(-1, 3), axis=0)
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def color_sample_error(group: str, rgb: tuple[int, int, int]) -> str | None:
    """Reject clicks that cannot represent the selected meter element."""
    red, green, blue = rgb
    if group == "green":
        if green < 80 or green < red + 25 or green < blue + 25:
            return "That pixel is not green enough for the target band."
    elif group == "marker":
        if max(rgb) > 100:
            return "That pixel is too bright for the black marker."
    elif group == "track":
        if red < 120 or red - blue < 45 or max(rgb) - min(rgb) < 60:
            return "That pixel is not a saturated red, orange, or yellow track color."
    return None


def color_mask_bgr(image: np.ndarray, samples: list[ColorSample]) -> np.ndarray:
    if not samples:
        return np.zeros(image.shape[:2], dtype=bool)
    pixels = image.astype(np.int32)
    mask = np.zeros(image.shape[:2], dtype=bool)
    for sample in samples:
        reference = np.array(sample.rgb[::-1], dtype=np.int32)
        delta = pixels - reference
        distance = np.sqrt(np.sum(delta * delta, axis=2, dtype=np.int32))
        mask |= distance <= sample.tolerance
    return mask


def _longest_run(mask: np.ndarray, max_gap: int = 2) -> tuple[int, int] | None:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    best_start = current_start = previous = int(indices[0])
    best_end = previous
    for raw in indices[1:]:
        value = int(raw)
        if value - previous <= max_gap + 1:
            previous = value
            continue
        if previous - current_start > best_end - best_start:
            best_start, best_end = current_start, previous
        current_start = previous = value
    if previous - current_start > best_end - best_start:
        best_start, best_end = current_start, previous
    return best_start, best_end


@dataclass
class DetectionDebug:
    saturated_fraction: float = 0.0
    track_height_fraction: float = 0.0
    green_row_score: float = 0.0
    marker_row_score: float = 0.0


class MeterDetector:
    def __init__(self, profile: CalibrationProfile) -> None:
        self.profile = profile
        self.last_debug = DetectionDebug()

    def detect(self, crop: np.ndarray, timestamp: float | None = None) -> MeterObservation:
        timestamp = time.perf_counter() if timestamp is None else timestamp
        reasons: list[str] = []
        if crop.size == 0 or crop.ndim != 3 or crop.shape[0] < 20 or crop.shape[1] < 8:
            return MeterObservation(timestamp, False, 0.0, reasons=("meter crop is too small",), crop=crop)

        height, width = crop.shape[:2]
        x0 = max(0, int(round(width * 0.16)))
        x1 = min(width, max(x0 + 2, int(round(width * 0.84))))
        center = crop[:, x0:x1]
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        saturated = (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 48)
        colored_row_fraction = saturated.mean(axis=1)
        colored_rows = colored_row_fraction >= 0.46
        track_run = _longest_run(colored_rows, max_gap=max(2, int(height * 0.025)))
        track_height_fraction = 0.0
        if track_run:
            track_height_fraction = (track_run[1] - track_run[0] + 1) / height
        saturated_fraction = float(saturated.mean())

        green_mask = color_mask_bgr(center, self.profile.samples.get("green", []))
        green_scores = green_mask.mean(axis=1)
        green_rows = green_scores >= 0.34
        zone_run = _longest_run(green_rows, max_gap=2)
        if zone_run:
            zone_height = zone_run[1] - zone_run[0] + 1
            if zone_height < self.profile.detection.min_zone_height_px:
                zone_run = None
                reasons.append("green run is too short")
            elif zone_height > height * self.profile.detection.max_zone_height_fraction:
                zone_run = None
                reasons.append("green run is too tall")
        if zone_run is None:
            reasons.append("green target not found")

        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY).astype(np.float32)
        row_luminance = np.median(gray, axis=1)
        smooth_window = max(9, min(61, (height // 10) | 1))
        smooth = np.convolve(row_luminance, np.ones(smooth_window) / smooth_window, mode="same")
        relative_dark = np.divide(
            row_luminance,
            np.maximum(smooth, 1.0),
            out=np.ones_like(row_luminance),
            where=smooth > 18,
        )
        marker_sample_mask = color_mask_bgr(center, self.profile.samples.get("marker", []))
        marker_sample_score = marker_sample_mask.mean(axis=1)
        marker_rows = (
            (marker_sample_score >= self.profile.detection.min_marker_width_fraction)
            & (relative_dark <= self.profile.detection.marker_darkness_ratio)
            & (smooth > 18)
        )
        marker_run = _longest_run(marker_rows, max_gap=2)
        if marker_run and marker_run[1] - marker_run[0] + 1 > max(25, int(height * 0.09)):
            marker_run = None
            reasons.append("dark run is too tall to be the marker")
        if marker_run is None:
            reasons.append("black marker not found")

        track_samples = color_mask_bgr(center, self.profile.samples.get("track", []))
        sampled_track_fraction = float(track_samples.mean())
        # Track samples contribute confidence, while physical presence requires
        # the saturated vertical gradient itself. A red statue or plant can
        # match one sample but cannot safely arm the input path.
        track_signal = saturated_fraction
        confidence = (
            min(1.0, track_height_fraction / max(self.profile.detection.min_track_height_fraction, 0.01)) * 0.34
            + min(1.0, track_signal / max(self.profile.detection.min_colored_fraction, 0.01)) * 0.17
            + min(1.0, sampled_track_fraction * 5.0) * 0.05
            + (0.22 if zone_run else 0.0)
            + (0.22 if marker_run else 0.0)
        )
        present = (
            track_height_fraction >= self.profile.detection.min_track_height_fraction
            and track_signal >= self.profile.detection.min_colored_fraction
            and zone_run is not None
        )
        if not present:
            reasons.append("vertical colored-track geometry not confirmed")

        marker_center = None if marker_run is None else (marker_run[0] + marker_run[1]) / 2.0
        self.last_debug = DetectionDebug(
            saturated_fraction=saturated_fraction,
            track_height_fraction=track_height_fraction,
            green_row_score=float(green_scores.max(initial=0.0)),
            marker_row_score=float(marker_sample_score.max(initial=0.0)),
        )
        return MeterObservation(
            timestamp=timestamp,
            present=present,
            confidence=float(min(1.0, confidence)),
            marker_center=marker_center,
            marker_span=marker_run,
            zone_bounds=zone_run,
            track_bounds=track_run,
            reasons=tuple(dict.fromkeys(reasons)),
            crop=crop,
        )

    def annotate(self, crop: np.ndarray, observation: MeterObservation) -> np.ndarray:
        output = crop.copy()
        height, width = output.shape[:2]
        if observation.zone_bounds:
            y0, y1 = observation.zone_bounds
            cv2.rectangle(output, (0, y0), (width - 1, y1), (255, 255, 255), 1)
        if observation.marker_span:
            y0, y1 = observation.marker_span
            cv2.rectangle(output, (0, y0), (width - 1, y1), (0, 0, 255), 1)
        color = (0, 220, 0) if observation.present else (0, 0, 220)
        cv2.rectangle(output, (0, 0), (width - 1, height - 1), color, 1)
        return output
