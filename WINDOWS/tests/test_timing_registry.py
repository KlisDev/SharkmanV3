from __future__ import annotations

import ast
from pathlib import Path

from sharkman.config import TIMING_FIELDS


def test_all_timing_names_are_unique() -> None:
    names = [item.name for item in TIMING_FIELDS]
    assert len(names) == len(set(names))


def test_runtime_has_no_literal_waits_or_timeouts() -> None:
    root = Path(__file__).parents[1] / "sharkman"
    violations = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            function = node.func
            if isinstance(function, ast.Attribute) and function.attr in {"sleep", "wait", "join", "after"}:
                if (
                    isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, (int, float))
                    and node.args[0].value != 0
                ):
                    violations.append((path.name, node.lineno))
    assert violations == []


def test_gui_is_generated_from_timing_registry() -> None:
    gui = (Path(__file__).parents[1] / "sharkman" / "gui.py").read_text(encoding="utf-8")
    assert "for group, specs in timing_groups()" in gui
