"""
DJC Design System — Plotly bridge.

Ports the navy/teal/glass aesthetic from ``djc_theme`` (matplotlib land) to
Plotly, so the interactive surfaces (``interactive_visualization.py`` and
``dashboard_app.py``) share one look with the static charts instead of
Plotly's stock white/blue defaults.

Usage:
    import djc_plotly as P          # registering the template is automatic
    fig.update_layout(template=P.TEMPLATE)

The module re-exports the palette tokens so callers never import raw hex
codes: ``P.TEAL``, ``P.ACCENT``, ``P.FG_*``, and the signature colorscales
``P.WAVE_SCALE`` (sequential), ``P.DIVERGING_SCALE`` and ``P.TIME_SCALE``.
"""

import plotly.graph_objects as go
import plotly.io as pio

import djc_theme as T

__all__ = [
    "TEMPLATE", "TEAL", "ACCENT", "NAVY", "FG_1", "FG_2", "FG_3", "FG_4",
    "WAVE_SCALE", "DIVERGING_SCALE", "TIME_SCALE",
    "hexa", "glow_traces", "style_figure", "write_html",
]

# Re-exported tokens ----------------------------------------------------------
TEAL = T.TEAL
ACCENT = T.ACCENT
NAVY = T.NAVY
FG_1, FG_2, FG_3, FG_4 = T.FG_1, T.FG_2, T.FG_3, T.FG_4

GRID = "#2b3f58"       # same hairline grid as the matplotlib theme
EDGE = "#223246"
GLASS = "rgba(20,41,63,0.55)"

TEMPLATE = "djc_dark"

# Signature colorscales — same ramps as djc_theme's matplotlib colormaps ------
WAVE_SCALE = [
    [0.00, TEAL.N900],
    [0.35, TEAL.N600],
    [0.70, TEAL.N300],
    [1.00, ACCENT.CYAN],
]
DIVERGING_SCALE = [
    [0.00, ACCENT.VIOLET],
    [0.50, NAVY.N700],
    [1.00, ACCENT.CYAN],
]
TIME_SCALE = [
    [0.00, NAVY.N500],
    [0.55, TEAL.N400],
    [1.00, ACCENT.CYAN],
]


def hexa(hex_color: str, alpha: float) -> str:
    """#rrggbb + alpha -> rgba() string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# The DJC Plotly template ------------------------------------------------------
_DJC_TEMPLATE = {
    "layout": {
        "paper_bgcolor": T.BG_BASE,
        "plot_bgcolor": T.BG_SURFACE,
        "font": {"family": "'Space Grotesk','Segoe UI',sans-serif", "color": FG_2, "size": 13},
        "title": {"font": {"family": "'Orbitron','Space Grotesk',sans-serif",
                           "size": 17, "color": FG_1}},
        "colorway": [TEAL.N500, ACCENT.CYAN, ACCENT.VIOLET, ACCENT.GREEN,
                     ACCENT.AMBER, NAVY.N300, ACCENT.RED],
        "hovermode": "x unified",
        "hoverlabel": {"bgcolor": NAVY.N800, "bordercolor": TEAL.N600,
                       "font": {"color": FG_1}},
        "xaxis": {"gridcolor": GRID, "gridwidth": 0.8, "zerolinecolor": GRID,
                  "linecolor": EDGE},
        "yaxis": {"gridcolor": GRID, "gridwidth": 0.8, "zerolinecolor": GRID,
                  "linecolor": EDGE},
        "legend": {"bgcolor": GLASS, "bordercolor": GRID, "borderwidth": 1,
                   "font": {"color": FG_2}},
        "colorscale": {"sequential": WAVE_SCALE, "diverging": DIVERGING_SCALE},
    }
}

if TEMPLATE not in pio.templates:
    pio.templates[TEMPLATE] = _DJC_TEMPLATE


def glow_traces(x, y, *, color=None, width=2.4, fill_to_zero=False,
                name=None, showlegend=True):
    """The design system's glow line, as a list of Plotly traces.

    Mirrors ``djc_theme.glow_line``: the series is drawn three times (wide
    faint halo, medium halo, crisp core) so bright curves appear to emit
    light against the navy canvas. Fill (if any) belongs to the core trace
    only, so stacked halos never darken the area fill.
    """
    color = color or ACCENT.CYAN
    core = dict(x=x, y=y, mode="lines", name=name, showlegend=showlegend,
                line=dict(color=color, width=width),
                hovertemplate="%{x:.2f}, %{y:.3f}<extra></extra>")
    if fill_to_zero:
        core.update(fill="tozeroy",
                    fillcolor=hexa(color, 0.16),
                    hovertemplate=None)
    return [
        go.Scatter(x=x, y=y, mode="lines", showlegend=False,
                   line=dict(color=color, width=width * 3.4),
                   opacity=0.08, hoverinfo="skip"),
        go.Scatter(x=x, y=y, mode="lines", showlegend=False,
                   line=dict(color=color, width=width * 1.7),
                   opacity=0.20, hoverinfo="skip"),
        go.Scatter(**core),
    ]


def style_figure(fig, *, title=None, height=520):
    """Apply the DJC template + title to a figure. Returns the figure."""
    fig.update_layout(template=TEMPLATE, height=height)
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, xanchor="left"))
    return fig


def write_html(fig, path, *, include_plotlyjs="cdn"):
    """Save a figure as HTML. The Plotly runtime is included inline when
    ``include_plotlyjs=True`` so the file works offline; 'cdn' keeps the
    file small for online viewing."""
    fig.write_html(path, include_plotlyjs=include_plotlyjs)
    return path
