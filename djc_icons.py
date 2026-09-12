"""Inline SVG icons — no emoji anywhere in the UI.

One spec table, two renderers:

- ``svg_icon(name)`` — a raw SVG string, for the report's template/JS.
- ``dash_icon(name)`` — a Dash ``html.Svg`` component tree, for the
  dashboard (Dash components reject raw-HTML injection props).

Every icon draws in the DJC palette; ``color`` overrides it. All shapes
live on a 24x24 grid.
"""

from __future__ import annotations

from html import escape

# DJC palette anchors (match djc_theme / report CSS variables).
CYAN = "#4cc9f0"      # fg1 — primary, healthy states
GREEN = "#4ade80"     # bound / success
AMBER = "#f4b840"     # near escape / warning
RED = "#ff5d73"       # unbound / error
SLATE = "#8b9bb4"     # muted guidance

# Icon geometry: "stroke" paths inherit the svg's stroke; "thick" paths
# get a heavier stroke; "filled" paths take the icon color outright;
# "dot" is a small filled circle (cx, cy, r).
_SPECS = {
    "pluck":   {"stroke": ["M4 3l7 17 2.4-6.6L20 11z"]},
    "draw":    {"stroke": ["M17 3a2.83 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5z"]},
    "check":   {"stroke": ["M4 12.5l5.5 5.5L20 6.5"]},
    "cross":   {"stroke": ["M6 6l12 12M18 6L6 18"]},
    "warn":    {"stroke": ["M12 3.5L22 20H2z"],
                "thick": ["M12 9.5v5"],
                "dot": (12, 17.5, 0.9)},
    "diamond": {"filled": ["M12 3l7.5 9-7.5 9-7.5-9z"]},
    "play":    {"filled": ["M7 4.5v15l13-7.5z"]},
    "pause":   {"filled": ["M7.5 4.5h3.5v15H7.5z",
                           "M13 4.5h3.5v15H13z"]},
}


def _resolve(name: str) -> dict:
    if name not in _SPECS:
        raise KeyError(f"unknown icon: {name!r} (known: {sorted(_SPECS)})")
    return _SPECS[name]


def svg_icon(name: str, *, size: int = 14, color: str = CYAN) -> str:
    """The named icon as a raw inline-SVG string."""
    spec = _resolve(name)
    parts = []
    for d in spec.get("stroke", []):
        parts.append(f'<path d="{d}"/>')
    for d in spec.get("thick", []):
        parts.append(f'<path d="{d}" stroke-width="2.4"/>')
    for d in spec.get("filled", []):
        parts.append(f'<path d="{d}" fill="{escape(color, quote=True)}" '
                     f'stroke="none" opacity=".9"/>')
    if "dot" in spec:
        cx, cy, r = spec["dot"]
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" '
                     f'fill="{escape(color, quote=True)}" stroke="none"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(size)}" '
        f'height="{int(size)}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{escape(color, quote=True)}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'aria-hidden="true" '
        f'style="vertical-align:-2px;color:{escape(color, quote=True)}">'
        f'{"".join(parts)}</svg>'
    )


def dash_icon(name: str, *, size: int = 14, color: str = CYAN):
    """The named icon as a Dash ``html.Img`` component.

    Dash 4.x ships no SVG components, so the icon is embedded as a
    base64 ``data:image/svg+xml`` URI — supported by every Dash
    version, styled inline like the raw-SVG variant.
    """
    import base64

    import dash.html as html

    svg = svg_icon(name, size=size, color=color)
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return html.Img(
        src=f"data:image/svg+xml;base64,{b64}",
        alt="",
        width=size, height=size,
        style={"verticalAlign": "-2px", "display": "inline-block",
               "marginRight": "6px"},
    )
