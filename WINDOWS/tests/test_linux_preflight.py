from __future__ import annotations

from pathlib import Path

from .test_repository_layout import ROOT, _load

preflight = _load("sharkman_linux_preflight_contract", ROOT / "LINUX" / "preflight.py")


def inspect(*, uinput: bool, wayland: bool = False, root: bool = False):
    return preflight.inspect_host(
        environ={"DISPLAY": ":1", "XDG_SESSION_TYPE": "wayland" if wayland else "x11"},
        platform="linux",
        module_available=lambda _name: True,
        exists=lambda _path: uinput,
        writable=lambda _path: uinput,
        is_root=root,
    )


def test_missing_uinput_blocks_live_input_but_not_the_gui_or_dry_run() -> None:
    report = inspect(uinput=False)

    assert report.can_open_gui
    assert not report.can_send_input
    assert any("/dev/uinput" in item for item in report.live_blockers)


def test_x11_with_uinput_is_ready_for_live_input() -> None:
    report = inspect(uinput=True)

    assert report.can_open_gui
    assert report.can_send_input


def test_wayland_and_root_are_gui_blockers() -> None:
    report = inspect(uinput=True, wayland=True, root=True)

    assert not report.can_open_gui
    assert any("Wayland" in item for item in report.gui_blockers)
    assert any("root" in item for item in report.gui_blockers)


def test_xwayland_without_a_real_x11_session_is_blocked() -> None:
    report = preflight.inspect_host(
        environ={"DISPLAY": ":1", "WAYLAND_DISPLAY": "wayland-0"},
        platform="linux",
        module_available=lambda _name: True,
        exists=lambda _path: True,
        writable=lambda _path: True,
        is_root=False,
    )

    assert not report.can_open_gui
    assert any("XWayland" in item for item in report.gui_blockers)


def test_missing_tk_blocks_gui_but_missing_evdev_only_blocks_live() -> None:
    report = preflight.inspect_host(
        environ={"DISPLAY": ":1", "XDG_SESSION_TYPE": "x11"},
        platform="linux",
        module_available=lambda name: name not in ("tkinter", "evdev"),
        exists=lambda path: path == Path("/dev/uinput"),
        writable=lambda _path: True,
        is_root=False,
    )

    assert any("tkinter" in item for item in report.gui_blockers)
    assert any("evdev" in item for item in report.live_blockers)
