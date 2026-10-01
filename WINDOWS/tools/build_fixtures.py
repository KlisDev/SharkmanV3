"""Create narrow, sanitized regression videos from local full recordings.

Raw source videos are never copied into the repository. Only the 30×568 meter
ROI is encoded. Run this deliberately when replacing the private source clips.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

ROI = (1525, 273, 30, 568)


def crop_video(source: Path, destination: Path) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 60.0)
    left, top, width, height = ROI
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create {destination}")
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            crop = frame[top : top + height, left : left + width]
            if crop.shape[:2] != (height, width):
                raise RuntimeError(f"frame {count} is smaller than the configured ROI")
            writer.write(crop)
            count += 1
    finally:
        capture.release()
        writer.release()
    print(f"{source.name}: {count} sanitized frames -> {destination}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures"))
    args = parser.parse_args()
    crop_video(args.first, args.output / "0917_1_meter.mp4")
    crop_video(args.second, args.output / "0917_1_long_meter.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

