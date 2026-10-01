from __future__ import annotations

import numpy as np

from sharkman.models import ColorSample
from sharkman.vision import (
    MeterDetector,
    color_mask_bgr,
    color_sample_error,
    sample_median_rgb,
)

from .helpers import calibrated_profile, synthetic_meter


def test_detector_finds_vertical_meter_marker_and_green() -> None:
    profile = calibrated_profile()
    observation = MeterDetector(profile).detect(synthetic_meter(marker_y=35), 1.0)
    assert observation.present
    assert observation.confidence >= profile.detection.required_confidence
    assert observation.marker_center is not None
    assert abs(observation.marker_center - 35) <= 2
    assert observation.zone_bounds is not None
    assert observation.zone_bounds[0] < 120 < observation.zone_bounds[1]


def test_detector_rejects_matching_pixel_without_track_geometry() -> None:
    profile = calibrated_profile()
    image = np.zeros((240, 30, 3), dtype=np.uint8)
    image[100:120, :] = (30, 245, 35)[::-1]
    observation = MeterDetector(profile).detect(image, 1.0)
    assert not observation.present


def test_color_distance_does_not_overflow_uint8() -> None:
    profile = calibrated_profile()
    image = np.full((1, 1, 3), 255, dtype=np.uint8)
    mask = color_mask_bgr(image, profile.samples["marker"])
    assert not bool(mask[0, 0])


def test_five_by_five_median_sampling_rejects_edge_noise() -> None:
    image = np.full((9, 9, 3), (30, 245, 35)[::-1], dtype=np.uint8)
    image[4, 4] = (255, 0, 255)

    assert sample_median_rgb(image, 4, 4) == (30, 245, 35)


def test_extra_color_variants_form_a_union_mask() -> None:
    image = np.zeros((1, 3, 3), dtype=np.uint8)
    image[0, 0] = (30, 245, 35)[::-1]
    image[0, 1] = (70, 220, 55)[::-1]
    samples = [ColorSample((30, 245, 35), 4), ColorSample((70, 220, 55), 4)]

    mask = color_mask_bgr(image, samples)

    assert mask.tolist() == [[True, True, False]]


def test_sample_semantics_reject_impossible_calibration_colors() -> None:
    assert color_sample_error("track", (0, 0, 0)) is not None
    assert color_sample_error("track", (252, 225, 27)) is None
    assert color_sample_error("track", (244, 98, 1)) is None
    assert color_sample_error("marker", (0, 0, 0)) is None
    assert color_sample_error("marker", (240, 240, 240)) is not None
    assert color_sample_error("green", (12, 251, 20)) is None
    assert color_sample_error("green", (252, 225, 27)) is not None
