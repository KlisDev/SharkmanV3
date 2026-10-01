"""Keep the offline first-run guide navigable and free of remote dependencies."""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from sharkman import __version__

ROOT = Path(__file__).parents[2]


class GuideParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.links = []
        self.resources = []
        self.stack = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "a":
            self.links.append(attrs["href"])
        if "src" in attrs:
            self.resources.append(attrs["src"])
        if tag == "link":
            self.resources.append(attrs.get("href", ""))
        if tag not in {"meta", "br", "hr", "img", "input", "link", "source", "wbr"}:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        assert self.stack and self.stack.pop() == tag, f"Unbalanced HTML tag: {tag}"


def test_guide_is_offline_navigable_and_matches_release():
    content = (ROOT / "GUIDE.html").read_text(encoding="utf-8")
    parser = GuideParser()
    parser.feed(content)
    parser.close()
    assert not parser.stack
    assert len(parser.ids) == len(set(parser.ids))
    assert not parser.resources, "The guide should not fetch external assets to display"
    for link in parser.links:
        url = urlsplit(link)
        if url.scheme:
            assert url.scheme == "https"
        elif link.startswith("#"):
            assert url.fragment in parser.ids
        else:
            assert (ROOT / unquote(url.path)).is_file(), link
    assert {"windows", "linux", "mac", "calibration", "run", "privacy"} <= set(parser.ids)
    assert f"Version {__version__}" in content
    for launcher in ("easy_run.py", "easy_run_linux.py", "easy_run_mac.py"):
        assert launcher in content
    assert "Download ZIP" in content
    assert "gameplay-unverified" in content
