"""Double-clickable source launcher with an isolated per-user environment."""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "SharkmanV3" / "venv-windows"
STAMP = VENV / ".sharkman-requirements-v2"


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def _bootstrap() -> Path:
    python = _venv_python()
    if not python.exists():
        print("Creating SharkmanV3's local Python environment …")
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV)
    requirements = ROOT / "requirements.txt"
    fingerprint = str(requirements.stat().st_mtime_ns)
    if not STAMP.exists() or STAMP.read_text(encoding="utf-8").strip() != fingerprint:
        print("Installing SharkmanV3 dependencies …")
        subprocess.check_call([str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)])
        STAMP.write_text(fingerprint, encoding="utf-8")
    return python


def main() -> int:
    if sys.platform != "win32":
        print("Use LINUX/easy_run_linux.py on Linux/Sober.")
        return 1
    if sys.version_info < (3, 11):
        print("SharkmanV3 needs Python 3.11 or newer. Python 3.12 is recommended.")
        input("Press Enter to close …")
        return 2
    python = _bootstrap()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.call([str(python), "-m", "sharkman", *sys.argv[1:]], cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
