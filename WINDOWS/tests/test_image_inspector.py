from __future__ import annotations

import hashlib
import os
import time
from types import SimpleNamespace

import customtkinter as ctk
import numpy as np
import pytest
from PIL import Image

from sharkman import gui
from sharkman.image_inspector import ImageViewport, viewport_image
from sharkman.models import ColorSample, NormalizedRect

from .helpers import calibrated_profile


def test_pointer_centered_zoom_fit_limits_pan_and_resize() -> None:
    view = ImageViewport(3840, 2160, 960, 540)
    view.fit()
    before = view.source_at(320, 210)
    for _ in range(8):
        view.zoom(1.2, 320, 210)
        assert view.source_at(320, 210) == pytest.approx(before)
    view.zoom(1e8, 320, 210)
    assert view.scale == 8.0
    view.x, view.y = -1e9, 1e9
    view.clamp()
    assert view.x == 960 - 3840 * 8
    assert view.y == 0
    view.zoom(1e-10, 320, 210)
    assert view.scale == 0.25
    view.resize(480, 270)
    assert view.scale == 0.125
    view.resize(800, 800)
    assert view.y > 0


def test_large_source_render_stays_viewport_sized_and_read_only() -> None:
    source = Image.new("RGB", (3840, 2160), "#f0a020")
    before = hashlib.sha256(source.tobytes()).digest()
    view = ImageViewport(*source.size, 640, 360)
    view.fit()
    for _ in range(15):
        view.zoom(1.2, 320, 180)
        rendered = viewport_image(source, view)
        assert rendered.size == (640, 360)
        assert rendered.getpixel((320, 180)) == (240, 160, 32)
    assert hashlib.sha256(source.tobytes()).digest() == before


def test_reference_path_is_shared_with_thumbnail_resolver() -> None:
    for task, filename in gui.CALIBRATION_REFERENCE_FILES.items():
        assert gui._calibration_reference_path(task) == gui.CALIBRATION_ASSETS / filename
    assert gui._calibration_reference_path("missing") is None


def test_screenshot_inspection_uses_original_pixels_and_normalized_roi(monkeypatch) -> None:
    profile = calibrated_profile()
    profile.meter_roi = NormalizedRect(0.15, 0.20, 0.85, 0.90)
    profile.mark_validated()
    source = np.zeros((101, 203, 3), dtype=np.uint8)
    source[50, 100] = (7, 29, 211)
    calibrator = gui.Calibrator.__new__(gui.Calibrator)
    calibrator.profile = profile
    calibrator.image_bgr = source
    calibrator.selected = "region"
    calibrator._inspector = None
    calibrator._close_inspector = lambda: None
    before_profile = profile.to_dict()
    before_pixels = source.tobytes()
    captured = {}

    def inspect(_parent, image, _title, overlay, *, redraw_delay_ms):
        captured["size"] = image.size
        captured["pixel"] = image.getpixel((100, 50))
        captured["overlay"] = overlay
        captured["redraw_delay_ms"] = redraw_delay_ms
        return SimpleNamespace()

    monkeypatch.setattr(gui, "ImageInspector", inspect)
    gui.Calibrator._inspect_screenshot(calibrator)

    assert captured["size"] == (203, 101)
    assert captured["pixel"] == (211, 29, 7)
    assert captured["overlay"][0] == "box"
    assert captured["overlay"][1] == profile.meter_roi.pixels(203, 101)
    assert captured["redraw_delay_ms"] == profile.timing.inspector_redraw_ms
    assert source.tobytes() == before_pixels
    after_profile = profile.to_dict()
    before_profile.pop("updated_at")
    after_profile.pop("updated_at")
    assert after_profile == before_profile


def test_zoom_screenshot_shows_selected_sample_union_only_inside_roi(monkeypatch) -> None:
    profile = calibrated_profile()
    profile.meter_roi = NormalizedRect(0.25, 0.25, 0.75, 0.75)
    profile.samples["green"] = [
        ColorSample((35, 245, 30), 0),
        ColorSample((40, 235, 35), 0),
    ]
    source = np.zeros((8, 8, 3), dtype=np.uint8)
    source[3, 3] = (30, 245, 35)
    source[4, 4] = (35, 235, 40)
    source[1, 1] = (30, 245, 35)  # Same color, outside the meter region.
    calibrator = gui.Calibrator.__new__(gui.Calibrator)
    calibrator.profile = profile
    calibrator.image_bgr = source
    calibrator.selected = "green"
    calibrator._inspector = None
    calibrator._close_inspector = lambda: None
    captured = {}
    monkeypatch.setattr(
        gui, "ImageInspector",
        lambda _parent, image, _title, _overlay, *, redraw_delay_ms: (
            captured.update(image=image.copy()) or SimpleNamespace()
        ),
    )
    before = source.tobytes()

    gui.Calibrator._inspect_screenshot(calibrator)

    image = captured["image"]
    for x, y, original in ((3, 3, (35, 245, 30)), (4, 4, (40, 235, 35))):
        expected = tuple(int(channel * 0.42 + magenta * 0.58)
                         for channel, magenta in zip(original, (232, 80, 222), strict=True))
        assert image.getpixel((x, y)) == expected
    assert image.getpixel((1, 1)) == (35, 245, 30)
    assert image.getpixel((5, 5)) == (0, 0, 0)
    assert source.tobytes() == before


def test_reference_inspection_uses_original_file_and_missing_is_nonfatal(monkeypatch, tmp_path) -> None:
    path = tmp_path / "guide.png"
    Image.new("RGB", (811, 367), "#208050").save(path)
    calibrator = gui.Calibrator.__new__(gui.Calibrator)
    calibrator.profile = calibrated_profile()
    calibrator._reference_key = "green"
    calibrator._inspector = None
    calibrator._close_inspector = lambda: None
    messages = []
    calibrator._set_status = lambda message, _color: messages.append(message)
    sizes = []
    monkeypatch.setattr(gui, "_calibration_reference_path", lambda _task: path)
    monkeypatch.setattr(
        gui, "ImageInspector",
        lambda _parent, image, _title, *, redraw_delay_ms: sizes.append(image.size),
    )

    gui.Calibrator._inspect_reference(calibrator)
    assert sizes == [(811, 367)]
    path.unlink()
    gui.Calibrator._inspect_reference(calibrator)
    assert sizes == [(811, 367)]
    assert "could not be opened" in messages[-1]


@pytest.mark.skipif(os.environ.get("SHARKMAN_TEST_GUI") != "1", reason="opt-in real Tk desktop test")
def test_real_zoom_events_compact_access_and_immediate_parent_close(monkeypatch, tmp_path) -> None:
    def capture_if_requested(window, name):
        directory = os.environ.get("SHARKMAN_TEST_SCREENSHOTS")
        if not directory:
            return
        from PIL import ImageGrab

        window.deiconify()
        window.attributes("-topmost", True)
        window.lift()
        root.update()
        time.sleep(0.25)
        assert window.winfo_viewable()
        x, y = window.winfo_rootx(), window.winfo_rooty()
        ImageGrab.grab(bbox=(
            x, y, x + window.winfo_width(), y + window.winfo_height(),
        )).save(os.path.join(directory, name))
        window.attributes("-topmost", False)

    errors = []
    root = ctk.CTk()
    root.withdraw()
    root.report_callback_exception = lambda *args: errors.append(args)
    root.profile = calibrated_profile()
    root.backend = None
    root.backend_error = "No backend in UI test"
    real_shoot = gui.Calibrator.shoot
    monkeypatch.setattr(gui.Calibrator, "shoot", lambda _calibrator: None)
    calibrator = gui.Calibrator(root)
    try:
        calibrator.image_bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)
        calibrator.image_bgr[100, 100] = (30, 245, 35)
        calibrator.selected = "green"
        calibrator.zoom_shot_btn.configure(state="normal")
        calibrator.update_idletasks()
        root.update()
        before_profile = calibrator.profile.to_dict()
        before_profile.pop("updated_at")
        before = (before_profile, calibrator.scale, calibrator.image_bgr.tobytes())
        calibrator.zoom_shot_btn.invoke()
        viewer = calibrator._inspector
        assert viewer is not None
        assert viewer.source.getpixel((100, 100)) == (149, 149, 141)
        root.update()
        capture_if_requested(viewer, "screenshot-inspector.png")
        for delta in (120, -120, 120):
            viewer.canvas.event_generate("<MouseWheel>", delta=delta, x=250, y=160)
        viewer.canvas.event_generate("<Button-1>", x=250, y=160)
        viewer.canvas.event_generate("<B1-Motion>", x=210, y=120)
        viewer.canvas.event_generate("<ButtonRelease-1>", x=210, y=120)
        viewer.canvas.event_generate("<Button-4>", x=250, y=160)
        viewer.canvas.event_generate("<Button-5>", x=250, y=160)
        viewer._actual_size()
        viewer._fit_view()
        time.sleep(0.03)
        root.update()
        assert viewer._pending is None
        after_profile = calibrator.profile.to_dict()
        after_profile.pop("updated_at")
        assert before == (after_profile, calibrator.scale, calibrator.image_bgr.tobytes())
        calibrator.geometry("1000x680")
        root.update()
        assert calibrator.reference_shell.winfo_ismapped()
        assert calibrator.canvas.winfo_height() >= 330
        assert calibrator.reference_image.image is not None
        assert calibrator.reference_image.image.cget("size")[0] >= 220
        assert calibrator.zoom_shot_btn.winfo_ismapped()
        capture_if_requested(calibrator, "calibration-compact.png")
        for button in (
            child for child in calibrator.winfo_children()
            if isinstance(child, ctk.CTkButton)
        ):
            assert button.winfo_ismapped()
        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)
        for button in descendants(calibrator):
            if isinstance(button, ctk.CTkButton) and button.cget("text") in (
                "Save calibration", "Re-shoot", "Reset selected",
            ):
                assert button.winfo_viewable()
                assert button.winfo_rootx() >= calibrator.winfo_rootx()
                assert button.winfo_rootx() + button.winfo_width() <= (
                    calibrator.winfo_rootx() + calibrator.winfo_width()
                )

        real_shoot(calibrator)
        assert viewer._closed
        assert calibrator._inspector is None
        guide = tmp_path / "reference.png"
        Image.new("RGB", (811, 367), "#40a070").save(guide)
        monkeypatch.setattr(gui, "_calibration_reference_path", lambda _task: guide)
        calibrator._inspect_reference()
        reference_viewer = calibrator._inspector
        assert reference_viewer is not None
        assert reference_viewer.source.size == (811, 367)
        capture_if_requested(reference_viewer, "reference-inspector.png")
        calibrator.geometry("1440x900")
        root.update()
        assert calibrator.reference_shell.winfo_ismapped()

        calibrator._inspect_screenshot()
        viewer = calibrator._inspector
        assert viewer is not None
        assert reference_viewer._closed
        viewer._zoom(1.2)
        assert viewer._pending is not None
        calibrator.destroy()
        assert viewer._closed
        assert viewer._pending is None
        root.update()
    finally:
        if calibrator.winfo_exists():
            calibrator.destroy()
        root.destroy()
    assert errors == []
