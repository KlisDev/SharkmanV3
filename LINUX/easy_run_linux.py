#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import venv
from pathlib import Path

from preflight import inspect_host

ROOT = Path(__file__).resolve().parent
WINDOWS = ROOT.parent / "WINDOWS"
VENV = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "sharkman-v3" / "venv-linux"
STAMP = VENV / ".sharkman-requirements-v2"


def _venv_python() -> Path:
    return VENV / "bin" / "python"


def _bootstrap() -> int:
    python = _venv_python()
    if not python.exists():
        print("Creating SharkmanV3's Linux environment …")
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV)
    requirements = ROOT / "requirements-linux.txt"
    shared_requirements = WINDOWS / "requirements.txt"
    fingerprint = f"{requirements.stat().st_mtime_ns}:{shared_requirements.stat().st_mtime_ns}"
    if not STAMP.exists() or STAMP.read_text(encoding="utf-8").strip() != fingerprint:
        print("Installing SharkmanV3 dependencies …")
        subprocess.check_call(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)]
        )
        STAMP.write_text(fingerprint, encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(WINDOWS), str(ROOT)))
    env["SHARKMAN_LINUX_BOOTSTRAPPED"] = "1"
    return subprocess.call([str(python), str(Path(__file__).resolve())], cwd=ROOT, env=env)


def main() -> int:
    early = inspect_host(check_dependencies=False)
    if early.gui_blockers:
        for error in early.gui_blockers:
            print(f"[setup blocked] {error}")
        return 1
    if os.environ.get("SHARKMAN_LINUX_BOOTSTRAPPED") != "1":
        return _bootstrap()
    report = inspect_host()
    for warning in report.warnings:
        print(f"[warning] {warning}")
    for blocker in report.live_blockers:
        print(f"[live input unavailable; dry run works] {blocker}")
    if report.gui_blockers:
        for error in report.gui_blockers:
            print(f"[setup blocked] {error}")
        return 1
    from backend import LinuxX11Backend

    from sharkman.gui import launch

    try:
        backend = LinuxX11Backend()
    except Exception as exc:
        print(f"[setup blocked] X11 backend failed: {exc}")
        return 1
    return launch(backend=backend)


if __name__ == "__main__":
    raise SystemExit(main())
