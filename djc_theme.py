"""
DJC Design System — theme for the Wave Force Simulator.

Palette, typography and glassmorphism are ported from the DJC Design System
that powers dennisjcarroll.com:

    Teal / deep navy / futuristic neural aesthetic

    - Deep navy backgrounds (#070e1a → #0f1d31) instead of white canvas
    - Core teal (#14b89a) as the primary accent
    - Electric cyan (#3ef0e2) for hi-energy highlights
    - Electric violet (#7c4dff) as the secondary "neural" accent
    - Glass cards: translucent navy panels with hairline teal borders
    - Display font: Orbitron · Body: Space Grotesk · Mono: JetBrains Mono

This module is the single source of truth for the look. Call ``apply()``
before plotting (visualization.py / analysis.py do this at import), and use
the console helpers (``banner``, ``section``, ``kv``, ``meter`` ...) in CLI
code instead of raw prints.

Fonts are fetched once from the Google Fonts repository and cached under
``~/.cache/djc_theme/fonts``; set ``DJC_THEME_NO_FONTS=1`` to skip the
download. Everything degrades gracefully offline to DejaVu faces.
"""

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "TEAL", "NAVY", "ACCENT", "apply", "ensure_fonts",
    "glow_line", "glass_legend", "finish_figure",
    "banner", "section", "kv", "meter", "success", "warn", "error", "rule",
]

# ============================================================================
# PALETTE — ported verbatim from the DJC Design System tokens
# ============================================================================


class TEAL:
    """Core teal ramp (--teal-* tokens)."""
    N50 = "#e6fbf7"
    N100 = "#b8f2e5"
    N200 = "#7ce5cf"
    N300 = "#44d6b8"
    N400 = "#2dc7a6"
    N500 = "#14b89a"   # primary accent
    N600 = "#0fa387"
    N700 = "#0b7e68"
    N800 = "#075844"
    N900 = "#043327"


class NAVY:
    """Deep navy ramp (--navy-* tokens)."""
    N50 = "#eaf0f5"
    N100 = "#cdd9e5"
    N200 = "#9fb5ca"
    N300 = "#6d89a8"
    N400 = "#46678a"
    N500 = "#2a4d6d"
    N600 = "#1d3852"
    N700 = "#14293f"
    N800 = "#0f1d31"
    N900 = "#0a1423"
    N950 = "#070e1a"   # deepest background


class ACCENT:
    """Electric / neural accents (--electric-* and signal colors)."""
    CYAN = "#3ef0e2"     # hi-energy highlight
    VIOLET = "#7c4dff"   # secondary from codebase
    GREEN = "#4ade80"    # success / live
    AMBER = "#f4b840"
    RED = "#ff5d73"


# Semantic foregrounds (--fg-*)
FG_1 = "#ecf4fb"   # hi-contrast white
FG_2 = "#b8c6d8"   # body
FG_3 = "#7d92ab"   # muted
FG_4 = "#4e6480"   # hint

# Semantic surfaces (--bg-*)
BG_BASE = NAVY.N950
BG_SURFACE = NAVY.N900
BG_RAISED = NAVY.N800
GLASS_BG = (20 / 255, 41 / 255, 63 / 255, 0.55)          # --bg-glass
BORDER_SUBTLE = (124 / 255, 180 / 255, 200 / 255, 0.08)  # --border-subtle
BORDER_LINE = (124 / 255, 180 / 255, 200 / 255, 0.18)    # --border-line

FONT_DISPLAY = "Orbitron"
FONT_SANS = "Space Grotesk"
FONT_MONO = "JetBrains Mono"

# ============================================================================
# COLOORMAPS — signature DJC ramps
# ============================================================================

#: Sequential: deep teal → electric cyan (energy fields, wave magnitude)
_DJC_WAVE_STOPS = [TEAL.N900, TEAL.N800, TEAL.N700, TEAL.N500, TEAL.N300, TEAL.N200, ACCENT.CYAN]

#: Diverging: violet (negative) → navy (zero) → cyan (positive) — displacement
_DJC_DIVERGING_STOPS = [ACCENT.VIOLET, "#4a3f8f", NAVY.N700, TEAL.N600, ACCENT.CYAN]

#: Time progression for overlaid snapshots: dim slate past → glowing cyan now
_DJC_TIME_STOPS = [NAVY.N500, TEAL.N500, TEAL.N300, ACCENT.CYAN]


def _register_colormaps() -> None:
    ramps = {
        "djc_wave": _DJC_WAVE_STOPS,
        "djc_diverging": _DJC_DIVERGING_STOPS,
        "djc_time": _DJC_TIME_STOPS,
    }
    for name, stops in ramps.items():
        if name not in mpl.colormaps:
            mpl.colormaps.register(LinearSegmentedColormap.from_list(name, stops), name=name)


# ============================================================================
# FONT PROVISIONING (best-effort, cache-first, offline-safe)
# ============================================================================

_FONT_FILES = {
    FONT_DISPLAY: "ofl/orbitron/Orbitron%5Bwght%5D.ttf",
    FONT_SANS: "ofl/spacegrotesk/SpaceGrotesk%5Bwght%5D.ttf",
    FONT_MONO: "ofl/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf",
}

_CACHE_DIR = Path.home() / ".cache" / "djc_theme" / "fonts"


def _download_fonts() -> None:
    """Fetch the brand typefaces once; any failure is silently ignored."""
    if os.environ.get("DJC_THEME_NO_FONTS") == "1":
        return
    import urllib.request

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for family, remote in _FONT_FILES.items():
        target = _CACHE_DIR / f"{family.replace(' ', '')}.ttf"
        if target.exists() and target.stat().st_size > 10_000:
            continue
        url = f"https://raw.githubusercontent.com/google/fonts/main/{remote}"
        try:
            with urllib.request.urlopen(url, timeout=4) as resp:
                target.write_bytes(resp.read())
        except Exception:
            continue  # offline — fall back to system faces


def ensure_fonts() -> None:
    """Register DJC typefaces with matplotlib when available."""
    try:
        _download_fonts()
        from matplotlib import font_manager

        for family in _FONT_FILES:
            path = _CACHE_DIR / f"{family.replace(' ', '')}.ttf"
            if path.exists():
                font_manager.fontManager.addfont(str(path))
    except Exception:
        pass  # theme must never break the simulation


# ============================================================================
# MATPLOTLIB THEME
# ============================================================================


def apply() -> None:
    """Apply the DJC rcParams and register the signature colormaps."""
    ensure_fonts()
    _register_colormaps()

    cycle_colors = [TEAL.N500, ACCENT.CYAN, ACCENT.VIOLET, ACCENT.GREEN, ACCENT.AMBER, NAVY.N300, ACCENT.RED]

    mpl.rcParams.update(
        {
            # Canvas — deep navy, glass axes
            "figure.facecolor": BG_BASE,
            "axes.facecolor": BG_SURFACE,
            "savefig.facecolor": BG_BASE,
            # Ink
            "text.color": FG_1,
            "axes.labelcolor": FG_2,
            "axes.titlecolor": FG_1,
            "xtick.color": FG_3,
            "ytick.color": FG_3,
            "axes.edgecolor": "#223246",
            # Grid — hairline, subtle
            "axes.grid": True,
            "grid.color": "#2b3f58",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.55,
            "axes.axisbelow": True,
            # Clean frame (design system: no top/right spines)
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Typography
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_SANS, "DejaVu Sans"],
            "font.monospace": [FONT_MONO, "DejaVu Sans Mono"],
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "axes.labelsize": 11,
            # Lines & markers
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            # Legend — glass card defaults
            "legend.facecolor": GLASS_BG,
            "legend.edgecolor": "#2b3f58",
            "legend.framealpha": 0.75,
            "legend.labelcolor": FG_2,
            # Series colors
            "axes.prop_cycle": mpl.cycler(color=cycle_colors),
            # Output quality
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
        }
    )


def glow_line(ax, x, y, *, color=None, lw=2.0, label=None, zorder=3, glow=True, **kwargs):
    """Plot a line with the design system's signature teal glow.

    Draws the line three times (wide faint, medium, crisp core) so bright
    series appear to emit light against the navy canvas.
    """
    if glow:
        ax.plot(x, y, color=color, lw=lw * 3.6, alpha=0.08, zorder=zorder - 2,
                solid_capstyle="round", **kwargs)
        ax.plot(x, y, color=color, lw=lw * 1.9, alpha=0.22, zorder=zorder - 1,
                solid_capstyle="round", **kwargs)
    return ax.plot(x, y, color=color, lw=lw, label=label, zorder=zorder, **kwargs)


def glass_legend(ax, **kwargs):
    """Attach a legend styled as a glassmorphism card."""
    leg = ax.legend(
        facecolor=GLASS_BG,
        edgecolor="#2b3f58",
        framealpha=0.8,
        labelcolor=FG_2,
        fancybox=True,
        borderpad=0.9,
        labelspacing=0.7,
        **kwargs,
    )
    leg.get_frame().set_linewidth(1.0)
    return leg


def finish_figure(fig, save_path=None, show=False, print_label=None, tight=True):
    """Save/close a figure consistently. Returns the saved path (or None).

    Non-blocking by design: figures are closed after saving so long runs
    never leak memory or pop dozens of windows.
    """
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    saved = None
    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        saved = save_path
        label = print_label or os.path.basename(save_path)
        print(success(f"saved {label} → {os.path.dirname(save_path) or '.'}"))
    if show:
        plt.show()
    plt.close(fig)
    return saved


# ============================================================================
# CONSOLE UI — same aesthetic for the terminal
# ============================================================================

# Enable ANSI on Windows terminals (harmless no-op elsewhere).
os.system("")

_RESET, _BOLD, _DIM = "\033[0m", "\033[1m", "\033[2m"

_ANSI = {
    "teal": "\033[38;2;20;184;154m",
    "teal_bright": "\033[38;2;68;214;184m",
    "cyan": "\033[38;2;62;240;226m",
    "violet": "\033[38;2;124;77;255m",
    "green": "\033[38;2;74;222;128m",
    "amber": "\033[38;2;244;184;64m",
    "red": "\033[38;2;255;93;115m",
    "white": "\033[38;2;236;244;251m",
    "body": "\033[38;2;184;198;216m",
    "muted": "\033[38;2;125;146;171m",
    "rule": "\033[38;2;15;88;72m",  # dark teal hairline
}


def style(text: str, color: str = "body", bold: bool = False, dim: bool = False) -> str:
    prefix = _ANSI.get(color, "")
    if bold:
        prefix += _BOLD
    if dim:
        prefix += _DIM
    return f"{prefix}{text}{_RESET}"


def rule(width: int = 72) -> str:
    return style("─" * width, "rule")


def banner(title: str, subtitle: str, eyebrow: str = "SIMULATION SUITE") -> None:
    """Design-system hero block: eyebrow, glowing title, divider rules."""
    w = max(len(title) * 2 + 8, 56)
    print()
    print(rule(w))
    print(f"  {style('◆ ' + eyebrow, 'teal_bright', bold=True)}")
    spaced = " ".join(title)
    print(f"  {style(spaced, 'cyan', bold=True)}")
    print(f"  {style(subtitle, 'body', dim=True)}")
    print(rule(w))


def section(title: str) -> None:
    print()
    print(f"  {style('◆ ' + title.upper(), 'teal_bright', bold=True)}")
    print(style("  " + "─" * 60, "rule"))


def kv(key: str, value: str, color: str = "white") -> None:
    """Aligned key/value row: muted key, bright value."""
    print(f"    {style(f'{key:<26}', 'muted')}{style(value, color)}")


def meter(frac: float, width: int = 24, color: str | None = None) -> str:
    """Block meter (█ filled / ░ empty), auto-colored by fraction."""
    frac = max(0.0, min(frac, 1.999))
    filled = int(round(min(frac, 1.0) * width))
    if color is None:
        color = "green" if frac <= 1.0 else "red"
    bar = style("█" * filled, color) + style("░" * (width - filled), "muted", dim=True)
    return bar


def success(text: str) -> str:
    return style("✓ ", "green", bold=True) + style(text, "body")


def warn(text: str) -> str:
    return style("▲ ", "amber", bold=True) + style(text, "body")


def error(text: str) -> str:
    return style("✗ ", "red", bold=True) + style(text, "body")


# ============================================================================
# EQUATION PANELS — mathtext captions binding the math to each chart
# ============================================================================


def cfl_badge(cfl: float) -> str:
    """Format a CFL number as a green/red stamp showing distance to the limit."""
    state = "OK" if cfl <= 1.0 else "UNSTABLE"
    return f"CFL = {cfl:.2f} {state}"


def draw_cfl_badge(fig, cfl: float) -> None:
    """Stamp a green/red CFL badge in the top-right corner of a figure."""
    ok = cfl <= 1.0
    fig.text(0.995, 0.988, cfl_badge(cfl), ha="right", va="top", fontsize=8,
             color=t_green if ok else ACCENT.RED, family=FONT_MONO)


def equation_footnote(fig, *lines: str, badge: str | None = None) -> None:
    """Render a mathtext caption strip along the bottom of a figure.

    Lines may mix plain text with $...$ mathtext. A badge (e.g. the CFL
    stamp) rides at the right edge of the last line. Never raises: the plot
    itself must survive any bad math string.
    """
    try:
        n = len(lines)
        for i, line in enumerate(lines):
            y = 0.008 + (n - 1 - i) * 0.032
            if badge and i == n - 1:
                fig.text(0.01, y, line, fontsize=9, color=FG_3, ha="left", va="bottom")
                fig.text(0.995, y, badge, fontsize=9,
                         color=t_green if "OK" in badge else ACCENT.RED,
                         ha="right", va="bottom", family=FONT_MONO)
            else:
                fig.text(0.01, y, line, fontsize=9, color=FG_3, ha="left", va="bottom")
    except Exception:
        pass  # a broken caption must never kill a plot


# Short aliases used inside module helpers
t_green = ACCENT.GREEN


# Apply eagerly so `import djc_theme` is enough for themed plots.
apply()
