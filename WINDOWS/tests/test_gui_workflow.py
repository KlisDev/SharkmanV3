from __future__ import annotations

from types import SimpleNamespace

from sharkman.config import ColorSample, ProfileStore
from sharkman.gui import (
    CALIBRATION_ASSETS,
    CALIBRATION_REFERENCE_FILES,
    DEFAULT_SESSION_MODE,
    Calibrator,
    DebugOverlay,
    SharkmanApp,
    _sample_text_color,
)
from sharkman.models import MeterObservation, WindowInfo

from .helpers import calibrated_profile
from .test_runner import FakeBackend


def test_live_input_is_the_nonpersistent_launch_default() -> None:
    assert DEFAULT_SESSION_MODE == "Live input"
    assert "session_mode" not in calibrated_profile().to_dict()


class Flag:
    def __init__(self, value: bool) -> None:
        self.value = value

    def get(self) -> bool:
        return self.value


class Label:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def configure(self, **values: str) -> None:
        self.values.update(values)


class Progress:
    def set(self, _value: float) -> None:
        pass


class Logbox:
    def winfo_exists(self) -> bool:
        return True

    def insert(self, _where: str, _text: str) -> None:
        pass

    def delete(self, _start: str, _end: str) -> None:
        pass

    def see(self, _where: str) -> None:
        pass


def test_finish_requires_the_explicit_preparation_confirmation() -> None:
    app = SharkmanApp.__new__(SharkmanApp)
    app.preflight_ready = Flag(False)
    app.preflight_message = Label()
    app._preflight_errors = lambda: []
    transitions: list[str] = []
    app._build_runner = lambda: transitions.append("runner")

    SharkmanApp._finish_preparation(app)

    assert transitions == []
    assert "Confirm" in app.preflight_message.values["text"]


def test_setup_preparation_runner_and_back_transitions() -> None:
    app = SharkmanApp.__new__(SharkmanApp)
    transitions: list[str] = []
    app._save_setup = lambda: transitions.append("save")
    app._build_preparation = lambda: transitions.append("preparation")

    SharkmanApp._apply_setup(app)

    app.preflight_ready = Flag(True)
    app.preflight_message = Label()
    app._preflight_errors = lambda: []
    app._build_runner = lambda: transitions.append("runner")
    SharkmanApp._finish_preparation(app)

    app.runner = None
    app.overlay = type("Overlay", (), {"hide": lambda self: transitions.append("hide")})()
    app._build_form = lambda: transitions.append("setup")
    SharkmanApp._back_to_setup(app)

    assert transitions == ["save", "preparation", "runner", "hide", "setup"]


def test_global_hotkeys_route_to_start_stop_and_diagnostics() -> None:
    app = SharkmanApp.__new__(SharkmanApp)
    backend = FakeBackend()
    app.backend = backend
    called: list[str] = []
    app.toggle_run = lambda: called.append("f2")
    app.stop_run = lambda: called.append("f4")
    app.toggle_debug = lambda: called.append("f8")
    app.after = lambda _delay, callback: callback()

    SharkmanApp._bind_hotkeys(app)
    for callback in backend.callbacks:
        callback()

    assert called == ["f2", "f4", "f8"]


def test_dry_run_does_not_require_linux_input_device_but_live_mode_does() -> None:
    class MissingInputBackend(FakeBackend):
        def input_error(self, _window: WindowInfo) -> str:
            return "/dev/uinput is not writable"

    app = SharkmanApp.__new__(SharkmanApp)
    app.profile = calibrated_profile()
    app.profile.mark_validated()
    app.backend = MissingInputBackend()
    app.backend_error = ""
    app.session_mode = SimpleNamespace(get=lambda: "Dry run")

    assert SharkmanApp._preflight_errors(app) == []
    app.session_mode = SimpleNamespace(get=lambda: "Live input")
    assert "/dev/uinput is not writable" in SharkmanApp._preflight_errors(app)
    app.hotkey_error = "listener failed"
    assert any("Global F4 emergency hotkey" in error for error in SharkmanApp._preflight_errors(app))


def test_f4_releases_input_without_an_active_runner() -> None:
    app = SharkmanApp.__new__(SharkmanApp)
    backend = FakeBackend()
    app.backend = backend
    app.runner = None
    app._page = "setup"
    app.overlay = type("Overlay", (), {"hide": lambda self: None})()

    SharkmanApp.stop_run(app)

    assert backend.releases == 1


def test_f2_pause_hides_the_diagnostic_overlay() -> None:
    class PausableRunner:
        running = True
        paused = False

        def toggle_pause(self) -> None:
            self.paused = not self.paused

    app = SharkmanApp.__new__(SharkmanApp)
    app._page = "runner"
    app.backend = FakeBackend()
    app.runner = PausableRunner()
    app.start_button = Label()
    app.run_badge = Label()
    hidden: list[bool] = []
    app.overlay = type("Overlay", (), {"hide": lambda self: hidden.append(True)})()

    SharkmanApp.toggle_run(app)

    assert app.runner.paused
    assert hidden == [True]


def test_diagnostic_overlay_never_creates_a_full_client_surface() -> None:
    window = WindowInfo(1, "Roblox", 40, 60, 1920, 1009)
    profile = calibrated_profile()
    observation = MeterObservation(
        1.0, True, 0.9, marker_center=100.0, zone_bounds=(80, 120))

    layout = DebugOverlay._geometry(window, profile, observation)

    assert {"outline_top", "outline_bottom", "outline_left", "outline_right",
            "status", "zone_top", "zone_bottom", "marker"} == set(layout)
    assert all(width < window.width or height < window.height
               for _x, _y, width, height in layout.values())


def test_runner_error_remains_visible_after_cleanup_log() -> None:
    app = SharkmanApp.__new__(SharkmanApp)
    app._page = "runner"
    app.logbox = Logbox()
    app._log_lines = 0
    app._run_failed = False
    app.after = lambda _delay, callback: callback()
    app.run_badge = Label()
    app.run_metrics = Label()
    app.start_button = Label()
    app.stop_button = Label()
    app.overlay = type("Overlay", (), {"hide": lambda self: None})()

    SharkmanApp._log(app, "Stopped: SendInput failed (87)")
    SharkmanApp._log(app, "Monitoring stopped; Space is released.")

    assert app.run_badge.values["text"] == "●  ERROR — SEE LOG"
    assert app.run_metrics.values["text"] == "Stopped: SendInput failed (87)"


def test_every_calibration_task_has_a_replaceable_reference_image_slot() -> None:
    assert CALIBRATION_REFERENCE_FILES == {
        "region": "complete_meter.png",
        "green": "green_target.png",
        "marker": "black_marker.png",
        "track": "track_gradient.png",
    }
    assert (CALIBRATION_ASSETS / "README.md").is_file()


def test_sample_swatch_text_contrasts_with_dark_and_bright_colors() -> None:
    assert _sample_text_color((0, 0, 0)) == "#ffffff"
    assert _sample_text_color((252, 225, 27)) == "#07111f"


def test_calibration_shows_saved_only_after_profile_is_written(tmp_path) -> None:
    profile = calibrated_profile("Timing profile")
    profile.mark_validated()
    store = ProfileStore(tmp_path)
    calibrator = Calibrator.__new__(Calibrator)
    calibrator.profile = profile
    calibrator.saved_signature = ""
    calibrator.selected = "region"
    calibrator.progress_text = Label()
    calibrator.progress = Progress()
    calibrator.nav_buttons = {key: Label() for key in Calibrator.TASKS}
    calibrator.save_button = Label()
    calibrator.result_label = Label()
    calibrator.interaction = Label()
    calibrator.master_app = SimpleNamespace(
        store=store, profile=None, _build_form=lambda: None)

    Calibrator._refresh_state(calibrator)
    assert calibrator.save_button.values["text"] == "Save calibration"
    assert calibrator.save_button.values["state"] == "normal"

    Calibrator.save(calibrator)
    assert store.active().validation_current
    assert calibrator.save_button.values["text"] == "✓ Saved"
    assert calibrator.save_button.values["state"] == "disabled"
    assert "Saved to profile" in calibrator.interaction.values["text"]

    profile.invalidate_validation()
    Calibrator._refresh_state(calibrator)
    assert calibrator.save_button.values["text"] == "Test detection to save"
    assert calibrator.save_button.values["state"] == "disabled"


def test_remove_invalid_samples_marks_an_explicit_unsaved_change() -> None:
    profile = calibrated_profile()
    profile.samples["track"] = [
        ColorSample((0, 0, 0), 45),
        ColorSample((252, 225, 27), 45),
        ColorSample((244, 98, 1), 45),
    ]
    profile.mark_validated()
    calibrator = Calibrator.__new__(Calibrator)
    calibrator.profile = profile
    calibrator.selected = "track"
    calibrator.selected_sample = 0
    calibrator.observation = object()
    calibrator.interaction = Label()
    calibrator._build_controls = lambda: None
    calibrator._refresh_state = lambda: None
    calibrator.redraw = lambda: None

    Calibrator.remove_invalid_samples(calibrator)

    assert [sample.rgb for sample in profile.samples["track"]] == [
        (252, 225, 27), (244, 98, 1)]
    assert not profile.validation_current
    assert calibrator.observation is None
    assert "Unsaved change" in calibrator.interaction.values["text"]
