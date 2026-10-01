from __future__ import annotations

import os
import sys
import tkinter as tk

import numpy as np
import pytest

from sharkman.config import CalibrationProfile
from sharkman.models import MeterObservation, NormalizedRect

from .test_repository_layout import ROOT, _load


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or os.environ.get("SHARKMAN_XVFB_SMOKE") != "1",
    reason="requires a Linux Xvfb job",
)
def test_x11_client_capture_focus_and_overlay_do_not_change_meter_pixels() -> None:
    linux = _load("sharkman_xvfb_backend", ROOT / "LINUX" / "backend.py")
    app = tk.Tk()
    app.title("Sober test client")
    app.geometry("640x480+50+50")
    app.configure(bg="#f0aa40")
    app.update()
    backend = linux.LinuxX11Backend()
    overlay = None
    try:
        window = backend.find_window("Sober")
        assert (window.width, window.height) == (640, 480)
        assert backend.activate_window(window)
        profile = CalibrationProfile(name="xvfb")
        profile.meter_roi = NormalizedRect(0.65, 0.2, 0.75, 0.8)
        before = profile.meter_roi.crop(backend.capture(window))
        overlay = backend.create_overlay(app)
        if not backend.overlay_supported:
            pytest.skip("Xvfb lacks the X Shape extension")
        overlay.show(window)
        observation = MeterObservation(1.0, True, 0.9, marker_center=50, zone_bounds=(40, 70))
        overlay.update(window, profile, observation, None)
        after = profile.meter_roi.crop(backend.capture(window))
        assert overlay.visible
        assert np.array_equal(before, after)
    finally:
        if overlay is not None:
            overlay.close()
        backend.close()
        app.destroy()
