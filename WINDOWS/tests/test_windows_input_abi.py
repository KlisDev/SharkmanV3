from __future__ import annotations

import ctypes
import sys

import pytest

from sharkman.models import WindowInfo

if sys.platform == "win32":
    from sharkman.backends.windows import INPUT, KEYBDINPUT, MOUSEINPUT, WindowsBackend


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ABI only")
def test_sendinput_structures_match_the_native_windows_abi() -> None:
    pointer_size = ctypes.sizeof(ctypes.c_void_p)

    assert ctypes.sizeof(KEYBDINPUT) == (24 if pointer_size == 8 else 16)
    assert ctypes.sizeof(MOUSEINPUT) == (32 if pointer_size == 8 else 24)
    assert ctypes.sizeof(INPUT) == (40 if pointer_size == 8 else 28)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows binding only")
def test_bound_window_changes_disarm_even_if_it_stays_foreground(monkeypatch) -> None:
    backend = WindowsBackend()
    bound = WindowInfo(42, "Roblox", 100, 200, 1920, 1009, 96, "windowed")
    monkeypatch.setattr("sharkman.backends.windows.user32.GetForegroundWindow", lambda: 42)
    for changed in (
        WindowInfo(42, "Roblox", 100, 200, 1280, 720, 96, "windowed"),
        WindowInfo(42, "Roblox", 100, 200, 1920, 1009, 144, "windowed"),
        WindowInfo(42, "Roblox", 100, 200, 1920, 1009, 96, "fullscreen"),
        WindowInfo(42, "Roblox", 101, 200, 1920, 1009, 96, "windowed"),
        WindowInfo(42, "Unrelated", 100, 200, 1920, 1009, 96, "windowed"),
    ):
        monkeypatch.setattr(backend, "_window_info", lambda _hwnd, value=changed: value)
        assert not backend.is_foreground(bound)
    monkeypatch.setattr(backend, "_window_info", lambda _hwnd: bound)
    assert backend.is_foreground(bound)
