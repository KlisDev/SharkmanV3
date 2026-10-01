"""Testable capture transport. No Apple frameworks are imported here."""
from __future__ import annotations

import threading
import time

import numpy as np

from sharkman.backends.base import BackendError, CaptureUnavailable


def bgra_to_bgr(data: object, width: int, height: int, stride: int) -> np.ndarray:
    if width <= 0 or height <= 0 or stride < width * 4:
        raise ValueError("Invalid BGRA pixel-buffer geometry")
    rows = np.frombuffer(data, dtype=np.uint8, count=height * stride).reshape(height, stride)
    return rows[:, :width * 4].reshape(height, width, 4)[:, :, :3].copy()


class FreshFrames:
    """One-slot mailbox, consumed once; native timestamps reject reordered frames."""

    def __init__(self, clock=time.monotonic) -> None:
        self.clock = clock
        self.condition = threading.Condition()
        self.latest: tuple[np.ndarray, float] | None = None
        self.last_sequence = -float("inf")
        self.error = ""
        self.closed = False
        self.delivered_at: float | None = None
        self.input_valid = False

    def publish(self, image: np.ndarray, sequence: float, captured_at: float) -> None:
        with self.condition:
            if self.closed or sequence <= self.last_sequence:
                return
            self.last_sequence = sequence
            # Reject blank/solid buffers, including locked-screen captures.
            if image.size == 0 or int(np.ptp(image)) < 3:
                self.latest = None
                self.input_valid = False
            else:
                self.latest = image, captured_at
            self.condition.notify_all()

    def invalidate(self) -> None:
        with self.condition:
            self.latest = None
            self.input_valid = False
            self.condition.notify_all()

    def fail(self, reason: str) -> None:
        with self.condition:
            self.error = reason
            self.latest = None
            self.input_valid = False
            self.condition.notify_all()

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.latest = None
            self.input_valid = False
            self.condition.notify_all()

    def take(self, wait_ms: float, age_ms: float) -> np.ndarray:
        deadline = self.clock() + wait_ms / 1000
        with self.condition:
            while True:
                if self.error:
                    raise BackendError(self.error)
                if self.closed:
                    raise CaptureUnavailable("Capture stopped")
                if self.latest is not None:
                    image, captured_at = self.latest
                    self.latest = None
                    age = self.clock() - captured_at
                    if 0 <= age <= age_ms / 1000:
                        self.delivered_at = captured_at
                        self.input_valid = True
                        return image
                    self.input_valid = False
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise CaptureUnavailable("No fresh complete ScreenCaptureKit frame")
                self.condition.wait(remaining)

    def fresh_for_input(self, age_ms: float) -> bool:
        with self.condition:
            return bool(not self.closed and not self.error and self.input_valid
                        and self.delivered_at is not None
                        and 0 <= self.clock() - self.delivered_at <= age_ms / 1000)
