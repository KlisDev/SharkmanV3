from __future__ import annotations

from sharkman.config import CalibrationProfile
from sharkman.models import MeterObservation, NormalizedRect, WindowInfo

from .test_repository_layout import ROOT, _load

overlay = _load("sharkman_linux_overlay_contract", ROOT / "LINUX" / "overlay.py")


def test_overlay_never_draws_in_the_calibrated_meter_crop() -> None:
    profile = CalibrationProfile(name="overlay")
    profile.meter_roi = NormalizedRect(0.70, 0.15, 0.75, 0.85)
    window = WindowInfo(3, "Sober", 100, 80, 640, 480)
    observation = MeterObservation(
        timestamp=1.0, present=True, confidence=0.95,
        marker_center=115, zone_bounds=(80, 125),
    )

    shapes = overlay.overlay_primitives(window, profile, observation, 1920, 1080)
    left, top, right, bottom = profile.meter_roi.pixels(window.width, window.height)
    protected = (window.left + left, window.top + top, right - left, bottom - top)

    assert len(shapes) >= 6
    assert all(not overlay.intersects(item[:4], protected) for item in shapes)
    assert all(x >= 0 and y >= 0 and x + width <= 1920 and y + height <= 1080
               for x, y, width, height, _color in shapes)
    pill = overlay.status_pill(window, profile, 1920, 1080)
    assert pill is not None
    assert not overlay.intersects(pill, protected)


def test_overlay_omits_out_of_screen_shapes_instead_of_clipping_inside_crop() -> None:
    profile = CalibrationProfile(name="edge")
    profile.meter_roi = NormalizedRect(0.0, 0.0, 0.5, 0.8)
    window = WindowInfo(3, "Sober", 0, 0, 640, 480)
    observation = MeterObservation(1.0, True, 0.9, marker_center=100)

    shapes = overlay.overlay_primitives(window, profile, observation, 640, 480)

    assert all(x >= 0 and y >= 0 for x, y, _w, _h, _shade in shapes)
