from __future__ import annotations

import argparse

from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sharkman", description="SharkmanV3 timing assistant")
    parser.add_argument("--version", action="version", version=f"SharkmanV3 {__version__}")
    parser.parse_args(argv)
    from .gui import launch

    return launch()


if __name__ == "__main__":
    raise SystemExit(main())

