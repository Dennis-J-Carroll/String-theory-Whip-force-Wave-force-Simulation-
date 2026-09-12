"""Emoji-free UI policy, enforced.

All user-facing surfaces (dashboard, report, docs, theme output) use
inline SVGs from ``djc_icons`` instead of emoji. Typographic symbols stay
legal: check/cross marks (U+2713/U+2717) in terminal output, arrows and
math arrows in prose, Greek letters, the section sign — none of those
render as emoji.

The blocklist is the emoji-presentation pictograph planes plus the code
points that force emoji presentation in browsers and terminals.
"""

import pathlib
import re

import pytest

import djc_icons

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF"   # emoji planes (includes dingbats-adjacent)
    "\u2600-\u26FF"            # misc symbols block (U+2600..U+26FF)
    "\u2705\u274C\u2728\u274E"  # emoji check/cross/sparkles
    "\u2B00-\u2BFF"            # stars, arrows-as-emoji
    "\uFE0F"                   # variation selector-16 (forces emoji render)
    "]"
)

# Scanned everywhere user-visible; generated artifacts and caches skipped.
SCAN_SUFFIXES = {".py", ".md", ".html", ".txt", ".css", ".js", ".cfg", ".toml"}
SKIP_PARTS = {".freebuff", "output", "__pycache__", ".git"}


def _iter_project_files():
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
            continue
        if SKIP_PARTS & set(path.parts):
            continue
        yield path


def test_no_emoji_in_any_user_facing_file():
    offenders = {}
    for path in _iter_project_files():
        hits = set(EMOJI.findall(path.read_text(errors="ignore")))
        if hits:
            offenders[str(path.relative_to(REPO_ROOT))] = sorted(hits)
    assert not offenders, (
        "emoji found (use djc_icons SVGs in UI, plain text in docs): "
        + "; ".join(f"{k}: {v}" for k, v in offenders.items())
    )


def test_icon_registry_covers_every_named_icon():
    """Every documented icon name renders, and nothing else resolves."""
    for name in ("pluck", "draw", "check", "cross", "warn", "diamond",
                 "play", "pause"):
        svg = djc_icons.svg_icon(name)
        assert svg.startswith("<svg ")
        assert svg.endswith("</svg>")
        assert 'viewBox="0 0 24 24"' in svg
        assert "stroke=" in svg or "fill=" in svg
    with pytest.raises(KeyError):
        djc_icons.svg_icon("particle")


def test_icons_carry_requested_color_and_size():
    svg = djc_icons.svg_icon("check", size=17, color=djc_icons.AMBER)
    assert 'width="17"' in svg and 'height="17"' in svg
    assert 'stroke="#f4b840"' in svg
    assert "color:#f4b840" in svg  # currentColor fills follow the stroke


def test_icon_module_itself_is_emoji_free():
    assert not EMOJI.search(djc_icons.__doc__ or "")
