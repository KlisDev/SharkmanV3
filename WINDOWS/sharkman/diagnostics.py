from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import cv2

from .config import default_data_root
from .models import EngineSnapshot, FireAction, MeterObservation


class DiagnosticSession:
    """Local-only JSONL logging and optional meter-crop recording."""

    def __init__(self, enabled: bool = False, record_crops: bool = False, root: Path | None = None) -> None:
        self.enabled = enabled
        self.record_crops = record_crops
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self.root = (root or default_data_root()) / "diagnostics" / f"session_{stamp}"
        self._stream = None
        self._lock = threading.Lock()
        self._frame_index = 0

    def _ensure_open(self) -> None:
        if self._stream is not None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self._stream = (self.root / "events.jsonl").open("a", encoding="utf-8")

    def record(
        self,
        observation: MeterObservation,
        snapshot: EngineSnapshot,
        action: FireAction | None,
    ) -> None:
        if not self.enabled and not self.record_crops:
            return
        with self._lock:
            self._ensure_open()
            payload = {
                "logged_at": datetime.now(UTC).isoformat(),
                "timestamp": observation.timestamp,
                "present": observation.present,
                "confidence": observation.confidence,
                "marker_center": observation.marker_center,
                "zone_bounds": observation.zone_bounds,
                "track_bounds": observation.track_bounds,
                "reasons": observation.reasons,
                "state": snapshot.state.value,
                "velocity": snapshot.velocity,
                "predicted_center": snapshot.predicted_center,
                "focused": snapshot.focused,
                "action": None if action is None else {
                    "reason": action.reason,
                    "predicted_center": action.predicted_center,
                    "zone_bounds": action.zone_bounds,
                },
            }
            assert self._stream is not None
            self._stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self._stream.flush()
            if self.record_crops and observation.crop is not None:
                frames = self.root / "meter_crops"
                frames.mkdir(exist_ok=True)
                cv2.imwrite(str(frames / f"{self._frame_index:07d}.png"), observation.crop)
            self._frame_index += 1

    def close(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
