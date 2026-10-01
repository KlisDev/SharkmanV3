from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import cv2

from .config import CalibrationProfile
from .engine import TimingEngine
from .models import MeterObservation
from .vision import MeterDetector


def replay_observations(
    observations: Iterable[MeterObservation], profile: CalibrationProfile,
) -> dict:
    """Replay sanitized detector output with the same scheduling path as live capture."""
    engine = TimingEngine(profile)
    actions: list[dict] = []
    frame_count = 0
    present_frames = 0
    for frame_index, observation in enumerate(observations):
        frame_count += 1
        present_frames += int(observation.present)
        action = engine.update(observation, focused=True)
        if action is None:
            scheduled_at = engine.snapshot().scheduled_fire_at
            if (
                scheduled_at is not None
                and 0.0 < scheduled_at - observation.timestamp
                <= profile.timing.precision_wait_threshold_ms / 1000.0
            ):
                action = engine.fire_scheduled(scheduled_at)
        if action:
            actions.append({
                "frame": frame_index,
                "timestamp": round(action.timestamp, 6),
                "marker_center": round(action.marker_center, 3),
                "predicted_center": round(action.predicted_center, 3),
                "zone_bounds": [round(value, 3) for value in action.zone_bounds],
                "velocity": round(action.velocity, 3),
                "reason": action.reason,
            })
    return {
        "frames": frame_count,
        "present_frames": present_frames,
        "prompts": engine.prompt_count,
        "actions": actions,
    }


def replay_video(video_path: Path, profile: CalibrationProfile, *, meter_only: bool = False) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 60.0)
    detector = MeterDetector(profile)

    def observations() -> Iterable[MeterObservation]:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            crop = frame if meter_only else profile.meter_roi.crop(frame)
            yield detector.detect(crop, frame_index / fps)
            frame_index += 1

    try:
        result = replay_observations(observations(), profile)
    finally:
        capture.release()
    return {"video": str(video_path), "fps": fps, **result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a recording without sending input")
    parser.add_argument("video", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--meter-only", action="store_true", help="video is already cropped to the meter")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    profile = CalibrationProfile.from_dict(json.loads(args.profile.read_text(encoding="utf-8")))
    result = replay_video(args.video, profile, meter_only=args.meter_only)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
