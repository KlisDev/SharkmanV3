#!/usr/bin/env python3
"""Source launcher. No installation, permission requests, or UI on import."""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def environment_path() -> Path:
    version = f"py{sys.version_info.major}.{sys.version_info.minor}-{platform.machine()}"
    return Path.home() / "Library" / "Caches" / "SharkmanV3" / f"venv-mac-{version}"


def main() -> int:
    sys.path.insert(0, str(PROJECT))
    from MAC.preflight import inspect_host

    report = inspect_host(check_dependencies=False)
    if report.gui_blockers:
        print("\n".join(report.gui_blockers))
        return 1
    environment = environment_path()
    python = environment / "bin" / "python"
    # Check the interpreter itself, not an inherited environment variable.
    if Path(sys.prefix).resolve() != environment.resolve():
        if not python.exists():
            print(f"Creating isolated environment: {environment}")
            venv.EnvBuilder(with_pip=True).create(environment)
        requirements = ROOT / "requirements-mac.txt"
        digest = hashlib.sha256(requirements.read_bytes() + (PROJECT / "WINDOWS" / "requirements.txt").read_bytes()).hexdigest()
        stamp = environment / ".requirements-sha256"
        if not stamp.exists() or stamp.read_text().strip() != digest:
            subprocess.check_call([str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)])
            stamp.write_text(digest, encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(PROJECT / "WINDOWS"), str(PROJECT)))
        return subprocess.call([str(python), str(Path(__file__).resolve()), *sys.argv[1:]], env=env, cwd=ROOT)
    report = inspect_host()
    if report.gui_blockers:
        print("\n".join(report.gui_blockers))
        return 1
    print("\n".join(report.warnings))
    sys.path.insert(0, str(PROJECT / "WINDOWS"))
    if "--self-test" in sys.argv[1:]:
        from MAC.self_test import main as self_test

        return self_test()
    from MAC.backend import MacBackend
    from sharkman.gui import launch

    return launch(backend=MacBackend())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Mac launcher failed: {exc}")
        raise SystemExit(1) from exc
