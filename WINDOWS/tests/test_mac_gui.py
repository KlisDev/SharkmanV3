"""Shared Mac readiness/workflow smoke test using only an injected fake backend."""
from __future__ import annotations

import os

import pytest

from sharkman import __version__
from sharkman.config import TIMING_FIELDS, ProfileStore
from sharkman.gui import WARNING, Calibrator, CooldownEditor, SharkmanApp

from .helpers import calibrated_profile, synthetic_meter
from .test_runner import FakeBackend


class ReadinessBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.allowed = False
        self.stopped_captures = 0

    def readiness(self):
        return {"Capture blocked": [] if self.allowed else ["Screen Recording required"],
                "Notice": ["Experimental · gameplay-unverified"]}

    def permission_actions(self):
        return {"Grant simulated permission": self.grant}

    def grant(self):
        self.allowed = True

    def privilege_error(self, _window):
        return None if self.allowed else "Screen Recording required"

    def capture(self, _window):
        return synthetic_meter(40)

    def stop_capture(self):
        self.stopped_captures += 1

    def create_overlay(self, _root):
        # No native panels and no keyboard events in this UI test.
        from .test_mac_backend import overlay

        return overlay.NullOverlay()


@pytest.mark.skipif(os.environ.get("SHARKMAN_TEST_GUI") != "1", reason="opt-in real Tk desktop test")
def test_readiness_calibration_cooldowns_preparation_and_back(tmp_path, monkeypatch):
    store = ProfileStore(tmp_path)
    profile = calibrated_profile()
    profile.timing_overrides = {"calibration_capture_delay_ms": 0}
    store.save(profile)
    backend = ReadinessBackend()
    app = SharkmanApp(store=store, backend=backend)
    errors = []
    app.report_callback_exception = lambda *args: errors.append(args)
    try:
        app.update()
        assert "Screen Recording" in app.platform_status.cget("text")
        assert app.session_mode.get() == "Live input"
        assert app.runner is None
        assert app.title() == f"SharkmanV3 v{__version__}"
        assert app.version_label.cget("text") == f"v{__version__}"
        app._permission_action(backend.grant)
        app._recheck_platform()
        assert "Capture blocked" not in app.platform_status.cget("text")
        app._open_calibration()
        app.update()
        calibrator: Calibrator = app._calibrator
        assert calibrator.image_bgr is not None
        assert backend.stopped_captures == 1
        calibrator.test_detection()
        assert calibrator.profile.validation_current
        calibrator.save()
        assert calibrator.save_button.cget("text") == "✓ Saved"
        assert store.active().validation_current
        calibrator.select("green")
        calibrator._inspect_screenshot()
        app.update()
        assert calibrator._inspector is not None
        calibrator.destroy()
        app._open_cooldowns()
        editor: CooldownEditor = app._cooldown_editor
        assert f"v{__version__}" in editor.title()
        assert set(editor.entries) == {field.name for field in TIMING_FIELDS}
        assert editor.entries["input_latency_ms"].cget("border_color") == WARNING
        assert editor.entries["input_latency_ms"].master.cget("border_width") == 2
        assert editor.entries["scan_interval_ms"].master.cget("border_width") == 0
        editor.destroy()
        app.session_mode.set("Dry run")
        app._apply_setup()
        app.update()
        assert app._page == "preparation"
        assert app.version_label.cget("text") == f"v{__version__}"
        assert app.finish_button.cget("state") == "disabled"
        app.preflight_ready.set(True)
        app._finish_preparation()
        app.update()
        assert app._page == "runner"
        assert app.runner is None  # Never automatically start.
        app._back_to_setup()
        app.update()
        assert app._page == "setup"
        assert not errors
    finally:
        app.close()
