from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sharkman.models import MeterObservation, NormalizedRect
from tools import build_feedback_model

from .helpers import calibrated_profile


def _manifest(tmp_path: Path, *, overlapping: bool) -> Path:
    profile = calibrated_profile()
    profile.meter_roi = NormalizedRect(0, 0, 0, 0)
    (tmp_path / "profile.json").write_text(json.dumps(profile.to_dict()), encoding="utf-8")
    samples = []
    for label in ("critical", "edge"):
        for index in range(9):
            samples.append({
                "image": f"{'same' if overlapping else label}-{index}.png",
                "label": label,
                "split": "train" if index < 6 else "holdout",
                "meter_crop": True,
            })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"profile": "profile.json", "samples": samples}), encoding="utf-8")
    return manifest


def _fake_detector(monkeypatch) -> None:
    class FakeDetector:
        def __init__(self, _profile) -> None:
            pass

        def detect(self, crop, _timestamp) -> MeterObservation:
            assert crop.shape == (568, 16, 3)
            marker = 56.0 if crop[0, 0, 0] else 50.0
            return MeterObservation(0.0, True, 1.0, marker, zone_bounds=(40, 60), track_bounds=(0, 100))

    monkeypatch.setattr(build_feedback_model, "MeterDetector", FakeDetector)
    monkeypatch.setattr(
        build_feedback_model.cv2, "imread",
        lambda path, _mode: np.full((568, 16, 3), int("edge" in path), dtype=np.uint8),
    )


def test_builder_accepts_gauge_only_crops_without_window_roi(monkeypatch, tmp_path: Path) -> None:
    _fake_detector(monkeypatch)
    output = tmp_path / "model.json"
    model = build_feedback_model.build(_manifest(tmp_path, overlapping=False), output)
    assert model.validated
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["features"]) == 18
    assert "image" not in payload["features"][0]


def test_builder_refuses_overlapping_labels(monkeypatch, tmp_path: Path) -> None:
    _fake_detector(monkeypatch)
    output = tmp_path / "model.json"
    with pytest.raises(ValueError, match="overlap"):
        build_feedback_model.build(_manifest(tmp_path, overlapping=True), output)
    assert not output.exists()
