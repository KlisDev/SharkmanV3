from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from sharkman.backends.base import BackendError

from .test_repository_layout import ROOT, _load

linux = _load("sharkman_linux_backend_contract", ROOT / "LINUX" / "backend.py")


class FakeWindow:
    def __init__(self, identity: int, title: str, classes=(), *, parent=None, width=640, height=480,
                 x=0, y=0):
        self.id = identity
        self.title = title
        self.classes = classes
        self.parent = parent
        self.children: list[FakeWindow] = []
        self.width, self.height = width, height
        self.x, self.y = x, y
        self.focused = False
        if parent is not None:
            parent.children.append(self)

    def query_tree(self):
        return SimpleNamespace(children=self.children, parent=self.parent)

    def get_wm_name(self):
        return self.title

    def get_wm_class(self):
        return self.classes

    def get_geometry(self):
        return SimpleNamespace(width=self.width, height=self.height)

    def get_attributes(self):
        return SimpleNamespace(map_state=2)

    def _root_position(self):
        if self.parent is None:
            return self.x, self.y
        px, py = self.parent._root_position()
        return px + self.x, py + self.y

    def translate_coords(self, source, x, y):
        # Match Xlib: source-local coordinates become receiver-local coordinates.
        sx, sy = source._root_position()
        dx, dy = self._root_position()
        return SimpleNamespace(x=sx + x - dx, y=sy + y - dy)

    def get_full_property(self, atom, _type):
        if atom == "RESOURCE_MANAGER":
            return SimpleNamespace(value=b"Xft.dpi: 144\n")
        if atom == "_NET_ACTIVE_WINDOW":
            return SimpleNamespace(value=[self.active])
        return None

    def configure(self, **_kwargs):
        pass

    def set_input_focus(self, *_args):
        self.focused = True


class FakeDisplay:
    def __init__(self):
        self.root = FakeWindow(1, "root")
        self.sober = FakeWindow(2, "Roblox", ("sober", "org.vinegarhq.Sober"), parent=self.root,
                                x=120, y=80)
        self.child = FakeWindow(3, "", parent=self.sober, width=100, height=100)
        self.root.active = self.child.id
        self.windows = {1: self.root, 2: self.sober, 3: self.child}

    def screen(self):
        return SimpleNamespace(root=self.root)

    def intern_atom(self, name):
        return name

    def create_resource_object(self, _kind, identity):
        return self.windows[identity]

    def sync(self):
        pass

    def close(self):
        pass


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    display = FakeDisplay()
    adapter = linux.LinuxX11Backend(display_factory=lambda: display)
    adapter._screen = lambda: SimpleNamespace(monitors=[{}, {
        "left": 0, "top": 0, "width": 1920, "height": 1080,
    }])
    yield adapter
    adapter.close()


def test_sober_class_finds_client_even_when_title_is_roblox(backend) -> None:
    window = backend.find_window("Sober")

    assert window.handle == 2
    assert (window.left, window.top, window.width, window.height) == (120, 80, 640, 480)
    assert window.dpi == 144
    assert window.display_mode == "windowed"
    assert backend.is_foreground(window)  # active X11 child belongs to this client
    backend._display.sober.width = 700
    assert not backend.is_foreground(window)  # a resize disarms the old calibration
    backend._display.sober.width = 640
    backend._display.root.active = 1
    assert not backend.is_foreground(window)


def test_client_coordinates_include_reparenting_and_movement_disarms(backend) -> None:
    display = backend._display
    decoration = FakeWindow(4, "frame", parent=display.root, x=30, y=40)
    display.root.children.remove(display.sober)
    display.sober.parent = decoration
    decoration.children.append(display.sober)
    display.windows[4] = decoration

    window = backend.find_window("Sober")

    assert (window.left, window.top) == (150, 120)
    assert backend.is_foreground(window)
    decoration.x += 10
    assert not backend.is_foreground(window)
    rebound = backend.find_window("Sober")
    assert (rebound.left, rebound.top) == (160, 120)
    assert backend.is_foreground(rebound)


def test_sober_class_beats_a_larger_unrelated_roblox_title(backend) -> None:
    decoy = FakeWindow(4, "Roblox guide", ("browser",), parent=backend._display.root,
                       width=1000, height=700)
    backend._display.windows[4] = decoy

    assert backend.find_window("Roblox").handle == backend._display.sober.id


def test_sober_class_matches_byte_strings_from_x11(backend) -> None:
    backend._display.sober.classes = (b"sober", b"org.vinegarhq.Sober")
    backend._display.sober.title = b"Roblox"

    assert backend.find_window("Sober").handle == backend._display.sober.id


def test_capture_is_client_only_and_rejects_blank_frames(backend) -> None:
    window = backend.find_window("Sober")
    calls = []
    colored = np.full((480, 640, 4), 90, dtype=np.uint8)
    backend._screen = lambda: SimpleNamespace(grab=lambda rect: calls.append(rect) or colored)

    frame = backend.capture(window)

    assert frame.shape == (480, 640, 3)
    assert calls == [{"left": 120, "top": 80, "width": 640, "height": 480}]
    backend._screen = lambda: SimpleNamespace(grab=lambda _rect: np.zeros_like(colored))
    with pytest.raises(BackendError, match="blank/black"):
        backend.capture(window)


def test_calibration_activation_requires_confirmed_focus(backend, monkeypatch) -> None:
    xlib = types.ModuleType("Xlib")
    xlib.X = SimpleNamespace(Above=0, RevertToParent=0, CurrentTime=0)
    monkeypatch.setitem(sys.modules, "Xlib", xlib)
    window = backend.find_window("Sober")
    backend._display.root.active = window.handle

    assert backend.activate_window(window)
    assert backend._display.sober.focused


def test_space_release_retries_after_a_failed_key_up(backend, monkeypatch) -> None:
    events = []
    fail_release = [True]

    class Device:
        def write(self, _kind, _code, value):
            if value == 0 and fail_release[0]:
                fail_release[0] = False
                raise OSError("transient release failure")
            events.append(value)

        def syn(self):
            pass

        def close(self):
            pass

    evdev = types.ModuleType("evdev")
    evdev.ecodes = SimpleNamespace(EV_KEY=1, KEY_SPACE=57)
    monkeypatch.setitem(sys.modules, "evdev", evdev)
    monkeypatch.setattr(linux.time, "sleep", lambda _duration: None)
    backend._uinput = Device()

    backend.press_space(2)

    assert events == [1, 0]
    assert not backend._space_down


def test_missing_shape_extension_disables_only_visible_overlay(backend, monkeypatch) -> None:
    fallback = object()
    fake_overlay = types.ModuleType("overlay")
    fake_overlay.X11Overlay = lambda _root: (_ for _ in ()).throw(RuntimeError("Shape absent"))
    fake_overlay.NullOverlay = lambda: fallback
    monkeypatch.setitem(sys.modules, "overlay", fake_overlay)

    result = backend.create_overlay(None)

    assert result is fallback
    assert not backend.overlay_supported
    assert "Shape absent" in backend.overlay_error
