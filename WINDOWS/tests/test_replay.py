from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

from sharkman.config import CalibrationProfile
from sharkman.models import MeterObservation
from sharkman.replay import replay_observations

FIXTURES = Path(__file__).parent / "fixtures"


def _observations(encoded: dict) -> list[MeterObservation]:
    compressed = base64.b85decode("".join(encoded["observations_b85"]))
    rows = json.loads(zlib.decompress(compressed))
    fps = encoded["fps"]
    return [
        MeterObservation(
            timestamp=index / fps,
            present=bool(row[0]),
            confidence=row[1],
            marker_center=row[2],
            marker_span=None if row[3] is None else tuple(row[3]),
            zone_bounds=None if row[4] is None else tuple(row[4]),
            track_bounds=None if row[5] is None else tuple(row[5]),
        )
        for index, row in enumerate(rows)
    ]


def test_historical_meter_observations_have_an_exact_regression_baseline() -> None:
    """This always runs in CI without restoring or distributing raw video."""
    profile = CalibrationProfile.from_dict(
        json.loads((FIXTURES / "profile.json").read_text(encoding="utf-8"))
    )
    traces = json.loads((FIXTURES / "observation_traces.json").read_text(encoding="utf-8"))
    assert traces["schema"] == 1
    assert set(traces["videos"]) == {"0917_1_meter.mp4", "0917_1_long_meter.mp4"}
    for trace in traces["videos"].values():
        observations = _observations(trace)
        result = replay_observations(observations, profile)
        actual_frames = [action["frame"] for action in result["actions"]]
        assert result["frames"] == trace["frames"]
        assert result["prompts"] == trace["current_prompt_count"]
        assert actual_frames == trace["current_action_frames"]
        assert len(actual_frames) == len(set(actual_frames))
        assert all(observations[frame].complete for frame in actual_frames)
        for action in result["actions"]:
            low, high = action["zone_bounds"]
            assert low <= action["predicted_center"] <= high


def test_historical_annotations_are_not_mistaken_for_current_acceptance() -> None:
    """Keep the old human targets visible until a gameplay-validated fix exists."""
    traces = json.loads((FIXTURES / "observation_traces.json").read_text(encoding="utf-8"))
    annotations = json.loads((FIXTURES / "annotations.json").read_text(encoding="utf-8"))
    assert len(annotations["0917_1_meter.mp4"]["expected_action_frames"]) == 3
    assert len(annotations["0917_1_long_meter.mp4"]["expected_action_frames"]) == 37
    assert len(traces["videos"]["0917_1_meter.mp4"]["current_action_frames"]) == 3
    assert len(traces["videos"]["0917_1_long_meter.mp4"]["current_action_frames"]) == 35
