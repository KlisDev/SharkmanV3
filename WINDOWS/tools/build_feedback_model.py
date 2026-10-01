"""Turn privately held, labeled frozen-gauge screenshots into numeric feedback data.

Usage: python tools/build_feedback_model.py manifest.json output.json

The manifest has a profile path and samples with image, label
(critical/edge/miss), split (train/holdout), and either meter_crop=true for an
already-cropped gauge or an optional normalized roi for a full screenshot.
Images are read only; the output contains no image paths or pixels. Do not put
raw screenshots in Git.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from statistics import median

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharkman.adaptive import FeedbackModel, normalized_landing  # noqa: E402
from sharkman.config import CalibrationProfile  # noqa: E402
from sharkman.models import NormalizedRect  # noqa: E402
from sharkman.vision import MeterDetector  # noqa: E402


def build(manifest_path: Path, output_path: Path) -> FeedbackModel:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile_path = (manifest_path.parent / manifest["profile"]).resolve()
    profile = CalibrationProfile.from_dict(json.loads(profile_path.read_text(encoding="utf-8")))
    detector = MeterDetector(profile)
    features: list[dict[str, str | float]] = []
    for item in manifest["samples"]:
        label = str(item["label"]).lower()
        split = str(item["split"]).lower()
        if label not in ("critical", "edge", "miss") or split not in ("train", "holdout"):
            raise ValueError("Each sample needs critical/edge/miss label and train/holdout split")
        image_path = (manifest_path.parent / item["image"]).resolve()
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Cannot read {image_path}")
        if item.get("meter_crop", False):
            crop = frame
        else:
            roi = (NormalizedRect.from_dict(item["roi"]) if "roi" in item else profile.meter_roi)
            if not roi.valid:
                raise ValueError(f"Invalid meter region for {image_path}")
            crop = roi.crop(frame)
        observation = detector.detect(crop, 0.0)
        if not observation.complete or observation.confidence < 0.8:
            raise ValueError(f"Meter not confidently detected in {image_path}: {observation.reasons}")
        assert observation.marker_center is not None and observation.zone_bounds is not None
        position = normalized_landing(observation.marker_center, observation.zone_bounds)
        features.append({"label": label, "split": split, "position": round(position, 5)})
    critical = [float(item["position"]) for item in features if item["split"] == "train" and item["label"] == "critical"]
    edge = [float(item["position"]) for item in features if item["split"] == "train" and item["label"] == "edge"]
    holdout = [item for item in features if item["split"] == "holdout"]
    if (len(critical) < 6 or len(edge) < 6
            or sum(item["label"] == "critical" for item in holdout) < 3
            or sum(item["label"] == "edge" for item in holdout) < 3):
        raise ValueError("Need six training examples per class and three holdout examples per class")
    center = float(median(critical))
    critical_outer = max(abs(value - center) for value in critical)
    edge_inner = min(abs(value - center) for value in edge)
    if edge_inner - critical_outer < 0.06:
        raise ValueError("Critical and edge positions overlap; do not enable adaptive learning")
    boundary = (critical_outer + edge_inner) / 2.0
    digest = hashlib.sha256(json.dumps(features, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    provisional = FeedbackModel(digest, center, boundary, len(holdout), 1.0)
    if any(provisional.classify(float(item["position"])) != item["label"]
           for item in features if item["split"] == "train"):
        raise ValueError("Training labels were not separable; no model written")
    correct = sum(provisional.classify(float(item["position"])) == item["label"] for item in holdout)
    model = FeedbackModel(digest, center, boundary, len(holdout), correct / len(holdout))
    if not model.validated:
        raise ValueError(f"Held-out validation failed: {correct}/{len(holdout)}; no model written")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        **model.to_dict(), "features": features,
    }, indent=2) + "\n", encoding="utf-8")
    return model


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python tools/build_feedback_model.py manifest.json output.json")
    print(build(Path(sys.argv[1]), Path(sys.argv[2])))
