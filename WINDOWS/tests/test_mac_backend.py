from __future__ import annotations

import importlib
import sys
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from sharkman.backends.base import BackendError, CaptureUnavailable, WindowSelectionRequired
from sharkman.config import TIMING_BY_NAME, TimingConfig
from sharkman.engine import TimingEngine
from sharkman.models import EngineState, NormalizedRect
from sharkman.ui_platform import wheel_steps

from .helpers import calibrated_profile
from .test_engine import observed
from .test_repository_layout import ROOT

sys.path.insert(0, str(ROOT))
mac = importlib.import_module("MAC.backend")
capture = importlib.import_module("MAC.capture")
preflight = importlib.import_module("MAC.preflight")
overlay = importlib.import_module("MAC.overlay")


def window(**changes):
    base = mac.MacWindow(12, 123, mac.ROBLOX_BUNDLE, "Roblox", 100, 50, 30, 120, 2, 1, "windowed")
    return replace(base, **changes)


class FakeNative:
    def __init__(self):
        self.allowed = {"screen": True, "post": True, "listen": True}
        self.items = [window()]
        self.focused = True
        self.match = True
        self.started = self.stopped = 0
        self.events = []
        self.pressed = threading.Event()
        self.fail_down = False
        self.callbacks = None
        self.listener = SimpleNamespace(healthy=True, close=self.close_listener)

    def close_listener(self):
        self.listener.healthy = False

    def permissions(self):
        return self.allowed.copy()

    def windows(self, _timeout):
        return self.items

    def matches(self, bound):
        return self.match and bound in self.items

    def foreground(self, _bound):
        return self.focused

    def activate(self, _bound):
        return self.focused

    def start_stream(self, bound, frames, _timing):
        self.started += 1
        image = np.zeros((bound.info.height, bound.info.width, 3), dtype=np.uint8)
        image[3:6] = 240
        frames.publish(image, self.started, time.monotonic())
        return object()

    def stop_stream(self, _stream, _timeout):
        self.stopped += 1

    def space(self, down):
        self.events.append(down)
        if down:
            self.pressed.set()
            if self.fail_down:
                raise BackendError("post failed")

    def hotkeys(self, *args):
        self.callbacks = args[:3]
        return self.listener


def bound_backend():
    native = FakeNative()
    backend = mac.MacBackend(native)
    info = backend.find_window("Roblox")
    backend.bind_hotkeys(lambda: None, lambda: None, lambda: None)
    backend.begin_input_session()
    return backend, native, info


def test_selects_only_roblox_player_identity_not_title_or_studio():
    good = window()
    browser = window(handle=13, bundle="com.apple.Safari")
    studio = window(handle=14, bundle="com.roblox.RobloxStudio")
    assert mac.choose_window([browser, studio, good], "Roblox", None) == good
    with pytest.raises(BackendError):
        mac.choose_window([browser, studio], "Roblox", None)
    with pytest.raises(BackendError):
        mac.choose_window([replace(good, visible=False)], "Roblox", None)


def test_multiple_clients_require_selection_and_never_silently_fallback():
    first, second = window(), window(handle=17, pid=127)
    with pytest.raises(WindowSelectionRequired) as error:
        mac.choose_window([first, second], "Roblox", None)
    assert len(error.value.windows) == 2
    assert mac.choose_window([first, second], "Roblox", 17) == second
    with pytest.raises(BackendError):
        mac.choose_window([first], "Roblox", 17)


def test_self_test_only_accepts_its_own_process():
    own = window(handle=23, pid=789, bundle="org.python.python", title="synthetic")
    assert mac.choose_window([own, window()], "synthetic", None, test_pid=789) == own
    with pytest.raises(BackendError):
        mac.choose_window([window()], "", None, test_pid=789)


@pytest.mark.parametrize("scale, pixels, dpi", [(1, 30, 96), (2, 60, 192)])
def test_capture_pixels_are_distinct_from_global_points(scale, pixels, dpi):
    info = window(scale=scale, x=-800).info
    assert info.left == -800
    assert (info.width, info.dpi) == (pixels, dpi)


@pytest.mark.parametrize("change", [{"x": 101}, {"point_height": 121}, {"scale": 1},
                                    {"display_id": 2}, {"display_mode": "fullscreen"}])
def test_geometry_change_blocks_capture_and_input(change):
    backend, native, info = bound_backend()
    native.items = [replace(native.items[0], **change)]
    assert not backend.is_foreground(info)
    with pytest.raises(BackendError):
        backend.capture(info)
    with pytest.raises(BackendError):
        backend.press_space(1)
    assert native.events == []


def test_dry_capture_works_without_input_permissions_or_hotkeys():
    backend, native, info = bound_backend()
    native.allowed.update(post=False, listen=False)
    native.listener.healthy = False
    image = backend.capture(info)
    assert image.shape == (240, 60, 3)
    assert "Accessibility" in backend.input_error(info)
    with pytest.raises(BackendError):
        backend.press_space(1)
    backend.close()
    assert native.stopped == 1
    assert not native.events


def test_screen_permission_denial_and_revocation_fail_closed():
    backend, native, info = bound_backend()
    native.allowed["screen"] = False
    with pytest.raises(BackendError, match="Screen Recording"):
        backend.find_window("Roblox")
    with pytest.raises(BackendError, match="revoked"):
        backend.capture(info)
    assert native.started == 0


@pytest.mark.parametrize("reason", ["focus", "post", "listen", "hotkeys", "cancel"])
def test_final_input_guard(reason):
    backend, native, _info = bound_backend()
    if reason == "focus":
        native.focused = False
    elif reason in ("post", "listen"):
        native.allowed[reason] = False
    elif reason == "hotkeys":
        native.listener.healthy = False
    else:
        backend.release_all()
    with pytest.raises(BackendError):
        backend.press_space(1)
    assert native.events == []


def test_space_released_on_normal_tap_and_post_exception():
    backend, native, info = bound_backend()
    backend.capture(info)
    backend.press_space(1)
    assert native.events == [True, False]
    native.fail_down = True
    with pytest.raises(BackendError, match="post failed"):
        backend.press_space(1)
    assert native.events == [True, False, True, False]


def test_emergency_interrupts_hold_without_duplicate_release():
    backend, native, info = bound_backend()
    backend.capture(info)
    thread = threading.Thread(target=backend.press_space, args=(1000,))
    thread.start()
    assert native.pressed.wait(1)
    native.callbacks[1]()
    thread.join(1)
    assert not thread.is_alive()
    assert native.events == [True, False]
    backend.close()
    assert native.events == [True, False]


def test_blank_or_expired_frame_blocks_a_scheduled_press():
    backend, native, info = bound_backend()
    backend.capture(info)
    backend.frames.invalidate()
    with pytest.raises(BackendError, match="no longer fresh"):
        backend.press_space(1)
    assert native.events == []
    backend.close()


def test_capture_is_consumed_once_and_shutdown_wakes_waiters():
    backend, native, info = bound_backend()
    backend.capture(info)
    backend.configure_timing(TimingConfig({"mac_frame_wait_ms": 10}))
    with pytest.raises(CaptureUnavailable):
        backend.capture(info)
    assert native.started == 1
    frames = backend.frames
    backend.stop_capture()
    with pytest.raises(CaptureUnavailable, match="stopped"):
        frames.take(0, 50)
    backend.stop_capture()
    assert native.stopped == 1


def test_bgra_conversion_respects_stride_and_owns_memory():
    raw = bytearray([1, 2, 3, 255, 4, 5, 6, 255, 99, 99, 99, 99] * 2)
    image = capture.bgra_to_bgr(raw, 2, 2, 12)
    assert image.tolist() == [[[1, 2, 3], [4, 5, 6]]] * 2
    raw[0] = 88
    assert image[0, 0, 0] == 1
    with pytest.raises(ValueError):
        capture.bgra_to_bgr(raw, 2, 2, 7)


def test_frame_mailbox_rejects_stale_reordered_duplicate_and_blank_frames():
    frames = capture.FreshFrames(clock=lambda: 10.0)
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    frames.publish(image, 1, 9.0)
    with pytest.raises(CaptureUnavailable):
        frames.take(0, 50)
    frames.publish(image, 2, 9.99)
    assert frames.take(0, 50) is image
    for sequence in (2, 1):
        frames.publish(image, sequence, 10.0)
        with pytest.raises(CaptureUnavailable):
            frames.take(0, 50)
    frames.publish(np.zeros_like(image), 3, 10.0)
    with pytest.raises(CaptureUnavailable):
        frames.take(0, 50)
    frames.publish(image, 4, 10.0)
    frames.invalidate()  # Idle/blank native attachment cancels buffered data.
    with pytest.raises(CaptureUnavailable):
        frames.take(0, 50)
    frames.fail("stream stopped")
    with pytest.raises(BackendError, match="stream stopped"):
        frames.take(0, 50)


@pytest.mark.parametrize("state", [EngineState.FIRED, EngineState.WAIT_GONE])
def test_unavailable_capture_never_rearms_a_fired_meter(state):
    engine = TimingEngine(calibrated_profile())
    engine.state = state
    engine._fired_at = 1.0
    engine._gone_since = 1.1
    engine._scheduled_fire_at = 1.2
    engine.capture_unavailable()
    assert engine.state == state
    assert engine._fired_at == 1.0
    assert engine._gone_since is None
    assert engine.snapshot().scheduled_fire_at is None


def test_unavailable_capture_forces_fresh_motion_without_incrementing_prompt():
    engine = TimingEngine(calibrated_profile())
    engine.update(observed(0, 10), True)
    engine.update(observed(0.02, 20), True)
    before = engine.prompt_count
    engine.capture_unavailable()
    assert engine.prompt_count == before
    assert not engine._history and engine._last_zone is None
    assert engine.update(observed(10.0, 130), True) is None
    assert engine.state != EngineState.WAIT_GONE


@pytest.mark.parametrize("scale,x,screen", [(1, 100, (0, 0, 1920, 1080)),
                                          (2, -800, (-1920, 0, 1920, 1080))])
def test_overlay_is_outside_roi_in_point_coordinates(scale, x, screen):
    binding = window(scale=scale, x=x)
    profile = calibrated_profile()
    profile.meter_roi = NormalizedRect(0.2, 0.1, 0.8, 0.9)
    shapes = overlay.overlay_rects(binding, profile, observed(0, 60), screen)
    roi = (x + 6, 62, 18, 96)
    assert shapes
    assert all(not overlay.intersects(shape[:4], roi) for shape in shapes)
    assert overlay.cocoa_rect((100, 50, 30, 120), 1080) == ((100, 910), (30, 120))


def test_platform_wheel_normalization():
    assert wheel_steps(SimpleNamespace(delta=120), "win32") == 1
    assert wheel_steps(SimpleNamespace(delta=1), "darwin") == 0.1
    assert wheel_steps(SimpleNamespace(delta=-30), "darwin") == -3
    assert wheel_steps(SimpleNamespace(num=4), "linux") == 1
    assert wheel_steps(SimpleNamespace(num=5), "linux") == -1
    assert wheel_steps(SimpleNamespace(delta=0), "darwin") == 0


def test_mac_readiness_groups_and_timing_defaults():
    report = preflight.permission_report({"screen": False, "post": False, "listen": False}, False)
    assert not report.gui_blockers
    assert len(report.capture_blockers) == 1
    assert len(report.live_blockers) == 3
    for name, default in [("mac_capture_hz", 240), ("mac_frame_wait_ms", 100),
                          ("mac_frame_age_ms", 50), ("mac_capture_start_ms", 2000),
                          ("mac_capture_stop_ms", 1000)]:
        assert TIMING_BY_NAME[name].default == default


def test_mac_profile_location_and_launcher_environment(monkeypatch):
    from sharkman import config

    monkeypatch.delenv("SHARKMAN_DATA_DIR", raising=False)
    monkeypatch.setattr(config, "sys", SimpleNamespace(platform="darwin"))
    assert config.default_data_root().parts[-3:] == ("Library", "Application Support", "SharkmanV3")
    launcher = importlib.import_module("MAC.easy_run_mac")
    assert ROOT not in launcher.environment_path().parents
    assert launcher.environment_path().parent.name == "SharkmanV3"


@pytest.mark.parametrize("version,machine,uid,blocked", [
    ("14.0", "arm64", 501, False), ("14.0", "x86_64", 501, False),
    ("13.6", "arm64", 501, True), ("14.0", "arm64", 0, True),
    ("14.0", "other", 501, True),
])
def test_mac_host_requirements(monkeypatch, version, machine, uid, blocked):
    monkeypatch.setattr(preflight, "sys", SimpleNamespace(platform="darwin", version_info=(3, 12)))
    monkeypatch.setattr(preflight, "platform", SimpleNamespace(mac_ver=lambda: (version, (), ""),
                                                            machine=lambda: machine))
    monkeypatch.setattr(preflight, "os", SimpleNamespace(geteuid=lambda: uid))
    monkeypatch.setattr(preflight, "importlib", SimpleNamespace(util=SimpleNamespace(find_spec=lambda _: True)))
    assert bool(preflight.inspect_host().gui_blockers) == blocked


def test_explicit_rebind_allows_selecting_another_client():
    backend, native, _info = bound_backend()
    native.items.append(window(handle=18, pid=125))
    backend.select_window(12)
    assert backend.find_window("Roblox").handle == 12
    backend.select_window(None)
    with pytest.raises(WindowSelectionRequired):
        backend.find_window("Roblox")
    backend.select_window(18)
    assert backend.find_window("Roblox").handle == 18


@pytest.mark.skipif(sys.platform != "darwin", reason="Apple frameworks need a Mac runner")
def test_native_imports_selectors_and_event_construction_without_posting():
    native = importlib.import_module("MAC.native")
    config = native.SC.SCStreamConfiguration.alloc().init()
    config.setIgnoreShadowsSingleWindow_(True)
    config.setCapturesAudio_(False)
    config.setShowsCursor_(False)
    config.setColorSpaceName_(native.Q.kCGColorSpaceSRGB)
    assert config.ignoreShadowsSingleWindow()
    assert not config.capturesAudio() and not config.showsCursor()
    event = native.Q.CGEventCreateKeyboardEvent(None, 49, False)
    assert event is not None  # Construction only; never post from CI.
    assert native.SharkmanStreamOutput.alloc().init() is not None
    assert overlay.panel_class() is not None
    permissions = native.NativeMac().permissions()
    assert set(permissions) == {"screen", "post", "listen"}


@pytest.mark.skipif(sys.platform != "darwin", reason="CoreVideo buffer bridge needs a Mac runner")
def test_native_pixel_buffer_bridge_without_capture_permission():
    native = importlib.import_module("MAC.native")
    q = native.Q
    error, pixel = q.CVPixelBufferCreate(None, 2, 2, q.kCVPixelFormatType_32BGRA, None, None)
    assert error == 0
    assert q.CVPixelBufferLockBaseAddress(pixel, 0) == 0
    try:
        stride = q.CVPixelBufferGetBytesPerRow(pixel)
        buffer = q.CVPixelBufferGetBaseAddress(pixel).as_buffer(stride * 2)
        raw = np.frombuffer(buffer, dtype=np.uint8)
        raw[:] = 0
        raw[:4] = [7, 13, 29, 255]
        image = capture.bgra_to_bgr(buffer, 2, 2, stride)
        assert tuple(image[0, 0]) == (7, 13, 29)
    finally:
        q.CVPixelBufferUnlockBaseAddress(pixel, 0)
