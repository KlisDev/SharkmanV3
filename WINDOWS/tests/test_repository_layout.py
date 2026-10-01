from __future__ import annotations

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from sharkman import __version__
from sharkman.backends.base import PlatformBackend

ROOT = Path(__file__).parents[2]


def test_release_version_is_single_source() -> None:
    metadata = tomllib.loads((ROOT / "WINDOWS" / "pyproject.toml").read_text())
    assert "version" in metadata["project"]["dynamic"]
    assert "version" not in metadata["project"]
    assert metadata["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "sharkman.__version__"
    assert __version__ == "1.0"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_required_windows_linux_split_exists() -> None:
    assert (ROOT / "WINDOWS" / "easy_run.py").is_file()
    assert (ROOT / "WINDOWS" / "sharkman" / "engine.py").is_file()
    assert (ROOT / "LINUX" / "easy_run_linux.py").is_file()
    assert (ROOT / "LINUX" / "backend.py").is_file()
    assert not (ROOT / "src" / "sharkman").exists()


def test_windows_launcher_and_linux_backend_import_without_side_effects() -> None:
    launcher = _load("sharkman_windows_launcher", ROOT / "WINDOWS" / "easy_run.py")
    linux_backend = _load("sharkman_linux_backend", ROOT / "LINUX" / "backend.py")

    assert launcher.ROOT == ROOT / "WINDOWS"
    assert ROOT not in launcher.VENV.parents
    assert issubclass(linux_backend.LinuxX11Backend, PlatformBackend)


def test_linux_preflight_rejects_wayland_and_missing_display(monkeypatch) -> None:
    preflight = _load("sharkman_linux_preflight", ROOT / "LINUX" / "preflight.py")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.delenv("DISPLAY", raising=False)

    errors = preflight.preflight_errors()

    assert any("Wayland" in error for error in errors)
    assert any("DISPLAY" in error for error in errors)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows backend smoke test")
def test_windows_backend_constructs_and_releases() -> None:
    from sharkman.backends.windows import WindowsBackend

    backend = WindowsBackend()
    assert backend.name == "Windows SendInput"
    backend.release_all()


def test_private_artifacts_are_ignored_by_git() -> None:
    private_paths = (
        "0917-1.mp4",
        "profiles/private.json",
        "logs/session.log",
        "screenshots/calibration.png",
        "Blox Fruits Fishing Macro.zip",
        "video-edit/private-project.json",
        "video-edit/crypto_key_store.dat",
        "WINDOWS/recording.mkv",
        "LINUX/gameplay.mp4",
        "MAC/session.jsonl",
        ".env",
        ".env.local",
        ".aws/credentials",
        "private.pem",
        "profile.json.bak",
    )
    for path in private_paths:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", path],
            cwd=ROOT,
            check=False,
        )
        assert result.returncode == 0, path

    reference = subprocess.run(
        [
            "git",
            "check-ignore",
            "--no-index",
            "-q",
            "WINDOWS/sharkman/assets/calibration/complete_meter.png",
        ],
        cwd=ROOT,
        check=False,
    )
    assert reference.returncode == 1
