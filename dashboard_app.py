"""
Interactive Dash Dashboard for Wave Simulation — DJC Design System edition.

A web-based interactive dashboard using Plotly Dash, sharing the deep-navy /
teal / glass aesthetic of every other surface in this repo (via
``djc_plotly``, the Plotly bridge to ``djc_theme``).

Physics notes (the reason this file was rewritten rather than re-skinned):

- Defaults use the *calibrated wave-scale* Lennard-Jones well (u* = 2 m,
  omega0 = 5 rad/s — the same constants main.py and the shim use). The
  previous cosmological defaults (K1 = 10^35) detonated instantly.
- The string is initialized at the well equilibrium u*, so the force is zero
  at rest and the pulse you launch is the whole story.
- K1/K2 sliders move *log10 around the calibration*, so the scale is
  explorable in both directions from a stable center.
- dt defaults to 80% of the CFL limit for the current c/dx (auto-recomputed
  on every relevant change) and the CFL badge flips red past 1.0 — the sim
  then refuses to run until dt is sane, exactly like the playground.
- Energy is kinetic (1/2 * v^2) vs potential (the true LJ integral), not
  displacement squared.

Usage:
    python dashboard_app.py

Then open your browser to http://localhost:8050
"""

import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from solver import wave_solver, potential_function, force_function
from solver import saved_frame_velocities, well_properties, crest_energy
from solver import WAVE_SCALE_K1, WAVE_SCALE_K2, WELL_U_STAR, WELL_OMEGA0
from pluck import pluck_displacement
from regimes import escape_landscape, stability_landscape
from interactive_visualization import (
    create_animated_wave,
    create_3d_wave_surface,
    create_interactive_potential_force,
    create_energy_monitor,
    create_phase_space,
    create_escape_landscape,
    create_stability_landscape,
)

import djc_plotly as P
import djc_icons as _svg


def _icon(name, size=13, color=None):
    """Inline SVG icon as a Dash component tree (emoji-free UI)."""
    return _svg.dash_icon(name, size=size, color=color or _svg.CYAN)


def pluck_mode_hint(mode):
    """One-line pointer hint under the pluck/draw radio, with its icon."""
    if mode == "draw":
        return html.Small([
            _icon("draw", color=_svg.SLATE),
            "draw mode — drag on the wave chart to sketch a shape "
            "(the drag is your pen), then press RUN",
        ], className="text-djc-muted")
    return html.Small([
        _icon("pluck", color=_svg.SLATE),
        "pluck mode — click the string anywhere to displace it there "
        "and watch it evolve (drag still zooms)",
    ], className="text-djc-muted")

# ============================================================================
# App shell — the DJC dark surfaces, expressed in Bootstrap classes
# ============================================================================

app = dash.Dash(
    __name__,
    title="DJC Wave Lab",
    external_stylesheets=[dbc.themes.CYBORG],
)
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>DJC Wave Lab</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Orbitron:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        {%css%}
        <style>
            /* DJC surfaces over the CYBORG base.
               Type stack matches dennisjcarroll.com: Orbitron display,
               Space Grotesk body, Fira Code mono (loaded above). */
            body { background: #070e1a !important;
                   font-family: 'Space Grotesk', sans-serif; }
            .bg-djc       { background: #0f1d31 !important; border: 1px solid #2b3f58 !important; }
            .card-header  { background: #14293f !important; color: #ecf4fb !important;
                            border-bottom: 1px solid #2b3f58 !important;
                            font-family: 'Orbitron', sans-serif; letter-spacing: .06em; }
            .card-body    { background: transparent; }
            .text-djc-muted { color: #7d92ab !important; }
            .eyebrow   { color: #44d6b8; font-family: 'Orbitron', sans-serif;
                         letter-spacing: .35em; font-size: .8rem; }
            .hero      { color: #3ef0e2; font-family: 'Orbitron', sans-serif; }
            .btn-djc   { background: #14b89a; border: none; color: #043327; font-weight: 700;
                         font-family: 'Orbitron', sans-serif; letter-spacing: .05em; }
            .btn-djc:hover { background: #3ef0e2; color: #043327; }
            .nav-link  { color: #b8c6d8 !important; }
            .nav-link.active { background: #0f1d31 !important; color: #3ef0e2 !important;
                               border-color: #2b3f58 !important; }
            .cfl-badge { font-family: 'Fira Code', monospace; font-size: .9rem; }
            .cfl-ok    { color: #4ade80; }
            .cfl-bad   { color: #ff5d73; font-weight: 700; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

GRID = P.GRID

# Shared graph config — the wave tab relies on plain clicks firing
# clickData for the pluck (drag still zooms: the two gestures coexist).
GRAPH_CONFIG = {"displaylogo": False}


def _pluck_hit_traces(x_range=(0, 50), y_range=(-1.0, 4.5)):
    """Invisible marker lattice so ANY canvas click fires plotly_click.

    Plotly only emits click events near trace data — a bare axes canvas
    (or a chart whose trace is a thin line) swallows clicks that land in
    open space. A dense opacity-0 lattice gives every click a nearby
    point; it renders nothing and never enters the legend.
    """
    gx = np.arange(x_range[0], x_range[1] + 0.5, 0.5)
    gy = np.arange(y_range[0], y_range[1] + 0.001, 0.25)
    X, Y = np.meshgrid(gx, gy)
    return go.Scatter(
        x=X.ravel(), y=Y.ravel(), mode="markers",
        marker=dict(opacity=0, size=8, color=P.GRID),
        # "none" keeps the point hover/click-participating (no label);
        # "skip" would remove it from Plotly's hit-testing entirely.
        hoverinfo="none", showlegend=False, name="hit-grid",
    )


def _empty_wave_graph():
    """The Wave tab's pre-RUN state: an empty stage inviting a pluck.

    Rendered statically in the layout (so the pluck callback's Input
    exists the moment the page loads — no wiring errors) and reused by
    the tab renderer whenever there is no sim data yet.
    """
    fig = go.Figure()
    fig.add_trace(_pluck_hit_traces())
    fig.add_hline(y=WELL_U_STAR, line=dict(color=P.GRID, dash="dot"))
    fig.update_layout(
        template=P.TEMPLATE,
        title="Click anywhere on the string to pluck it — "
              "or set parameters and RUN",
        xaxis=dict(range=[0, 50], title="x (m)"),
        yaxis=dict(range=[-1, 4.5], title="u (m)"),
        height=600,
    )
    return dcc.Graph(figure=fig,
                     id="wave-graph",
                     config=GRAPH_CONFIG,
                     style={"height": "600px"})

# ============================================================================
# CFL plumbing — the badge drives the dt slider, like the playground dial
# ============================================================================


def cfl_limit(c: float, dx: float) -> float:
    """Largest stable dt for the explicit scheme: dx / c."""
    return dx / max(c, 1e-12)


def cfl_ratio(c: float, dt: float, dx: float) -> float:
    return c * dt / max(dx, 1e-12)


# ============================================================================
# Layout
# ============================================================================

app.layout = dbc.Container([
    dcc.Store(id="simulation-data"),
    dcc.Store(id="sim-params", data={}),   # remembers the last run's params
    dcc.Store(id="store-drawn", data=None),  # freehand shape waiting for RUN

    dbc.Row([
        dbc.Col([
            html.Div("WAVE FORCE SIMULATOR", className="eyebrow"),
            html.H1("DJC Wave Lab", className="hero mb-0"),
            html.P("Interactive exploration of waves in a Lennard-Jones "
                   "well — u* = 2 m · ω₀ = 5 rad/s",
                   className="text-djc-muted mb-0"),
        ]),
        dbc.Col(html.Div(id="cfl-badge", className="cfl-badge cfl-ok text-end"),
                width="auto", align="center"),
    ], className="mb-4 mt-2"),

    dbc.Row([
        # ------------------------------------------------------------------
        # Control column
        # ------------------------------------------------------------------
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("PHYSICS"),
                dbc.CardBody([
                    html.Label("Attractive constant k₂ (log₁₀ offset)"),
                    dcc.Slider(
                        id="k2-offset", min=-2, max=2, step=0.25, value=0,
                        marks={-2: "-2", -1: "-1", 0: "0", 1: "+1", 2: "+2"},
                        tooltip={"placement": "bottom"},
                    ),
                    html.Small("k₁ follows k₂ so the well stays at u* = 2 m",
                               className="text-djc-muted d-block mb-3"),

                    html.Label("Wave speed c (m/s)"),
                    dcc.Slider(
                        id="c-slider", min=0.2, max=5.0, step=0.1, value=1.0,
                        marks={0.5: "0.5", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"},
                        tooltip={"placement": "bottom"},
                    ),

                    html.Hr(),
                    html.Div(id="well-info", className="text-djc-muted"),
                    html.Div(id="escape-readout", className="mt-2"),
                ]),
            ], className="bg-djc mb-3"),

            dbc.Card([
                dbc.CardHeader("NUMERICS"),
                dbc.CardBody([
                    html.Label("Time step dt (s)"),
                    dcc.Slider(
                        id="dt-frac", min=0.1, max=1.2, step=0.05, value=0.8,
                        marks={0.2: "0.2×", 0.5: "0.5×", 0.8: "0.8×", 1.0: "1.0×", 1.2: "1.2×"},
                        tooltip={"placement": "bottom"},
                    ),
                    html.Small("fraction of the CFL limit — past 1.0 the grid "
                               "detonates (this is a feature: try it)",
                               className="text-djc-muted d-block mb-3"),

                    html.Label("Grid spacing dx (m)"),
                    dcc.Slider(
                        id="dx-slider", min=0.05, max=0.5, step=0.05, value=0.1,
                        marks={0.05: ".05", 0.1: ".1", 0.2: ".2", 0.3: ".3", 0.4: ".4", 0.5: ".5"},
                        tooltip={"placement": "bottom"},
                    ),

                    html.Label("Total time T (s)"),
                    dcc.Slider(
                        id="time-slider", min=1, max=20, step=1, value=5,
                        marks={1: "1", 5: "5", 10: "10", 15: "15", 20: "20"},
                        tooltip={"placement": "bottom"},
                    ),
                ]),
            ], className="bg-djc mb-3"),

            dbc.Card([
                dbc.CardHeader("INITIAL PULSE"),
                dbc.CardBody([
                    html.Label("Amplitude (m)"),
                    dcc.Slider(
                        id="amplitude-slider", min=0.1, max=4.0, step=0.1, value=1.0,
                        marks={1: "1", 2: "2", 3: "3", 4: "4"},
                        tooltip={"placement": "bottom"},
                    ),
                    html.Div(id="amplitude-warning",
                             className="text-djc-muted d-block mb-3"),

                    dbc.RadioItems(
                        id="pluck-mode",
                        options=[
                            {"label": html.Span([_icon("pluck"), " pluck"]),
                             "value": "pluck"},
                            {"label": html.Span([_icon("draw"), " draw"]),
                             "value": "draw"},
                        ],
                        value="pluck",
                        inline=True,
                        className="mb-1",
                    ),
                    html.Div(id="pluck-mode-banner",
                             children=pluck_mode_hint("pluck")),

                    html.Hr(),
                    html.Label("Center position (m)"),
                    dcc.Slider(
                        id="center-slider", min=10, max=40, step=1, value=25,
                        marks={10: "10", 20: "20", 30: "30", 40: "40"},
                        tooltip={"placement": "bottom"},
                    ),

                    html.Label("Gaussian width σ (m)"),
                    dcc.Slider(
                        id="width-slider", min=0.5, max=5.0, step=0.1, value=2.0,
                        marks={1: "1", 2: "2", 3: "3", 4: "4", 5: "5"},
                        tooltip={"placement": "bottom"},
                    ),
                ]),
            ], className="bg-djc mb-3"),

            dbc.Button("RUN SIMULATION", id="run-button", className="btn-djc w-100 mb-3", size="lg"),
            html.Div(id="status-message", className="text-center"),
        ], width=12, lg=3),

        # ------------------------------------------------------------------
        # Charts column
        # ------------------------------------------------------------------
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="Wave", tab_id="tab-animation"),
                dbc.Tab(label="3D Surface", tab_id="tab-3d"),
                dbc.Tab(label="Energy", tab_id="tab-energy"),
                dbc.Tab(label="Phase Space", tab_id="tab-phase"),
                dbc.Tab(label="Potential & Force", tab_id="tab-potential"),
                dbc.Tab(label="Landscapes", tab_id="tab-landscapes"),
            ], id="tabs", active_tab="tab-animation", className="mb-2"),

            html.Div(id="tab-content",
                     children=[_empty_wave_graph()],
                     className="mt-3"),
        ], width=12, lg=9),
    ]),
], fluid=True, className="p-4")


# ============================================================================
# Callbacks
# ============================================================================


@app.callback(
    Output("cfl-badge", "children"),
    Output("cfl-badge", "className"),
    Input("c-slider", "value"),
    Input("dt-frac", "value"),
    Input("dx-slider", "value"),
)
def update_cfl_badge(c, dt_frac, dx):
    """Live CFL readout — green OK, red UNSTABLE past 1.0."""
    dt = dt_frac * cfl_limit(c, dx)
    ratio = cfl_ratio(c, dt, dx)
    cls = "cfl-badge cfl-ok text-end" if ratio <= 1.0 else "cfl-badge cfl-bad text-end"
    state = "OK" if ratio <= 1.0 else "UNSTABLE"
    return f"CFL = {ratio:.2f} {state}", cls


@app.callback(
    Output("well-info", "children"),
    Input("k2-offset", "value"),
)
def update_well_info(offset):
    """Show the actual force constants the sliders imply."""
    k2 = WAVE_SCALE_K2 * 10.0 ** offset
    k1 = WAVE_SCALE_K1 * 10.0 ** offset
    return [
        html.Span(f"k₁ = {k1:.3g}", className="d-block"),
        html.Span(f"k₂ = {k2:.3g}", className="d-block"),
        html.Span(f"u* = 2 m (equilibrium) · ω₀ ≈ {WELL_OMEGA0 * 10.0 ** (offset / 2):.2g} rad/s",
                  className="d-block"),
    ]


@app.callback(
    Output("escape-readout", "children"),
    Output("amplitude-warning", "children"),
    Output("amplitude-warning", "className"),
    Input("k2-offset", "value"),
    Input("amplitude-slider", "value"),
    Input("width-slider", "value"),
    Input("c-slider", "value"),
)
def update_escape_readout(offset, amplitude, width, c):
    """Well depth vs crest energy — can the pulse escape the well?

    Both numbers are exact properties of the initial condition (no
    simulation): the well depth V(u_out) − V(u*) is the energy per unit
    mass a node needs to reach the V = 0 crossing on the wall side, and
    crest_energy() sums the lifted crest's well energy with the elastic
    energy of its steepest slope.
    """
    scale = 10.0 ** offset
    k1, k2 = WAVE_SCALE_K1 * scale, WAVE_SCALE_K2 * scale
    props = well_properties(k1, k2)
    crest = crest_energy(amplitude, width, c=c, k1=k1, k2=k2)

    ratio = crest["total"] / props["depth"]
    pct = 100.0 * ratio

    # State machine: green bound / amber near escape / red unbound.
    # Icons are inline SVGs (see djc_icons) — no emoji in the UI.
    if ratio >= 1.0:
        state_cls, icon = "text-danger fw-bold", _icon("warn", color=_svg.RED)
        verdict = "crest exceeds the well — wall slams and punch-through likely"
    elif ratio >= 0.85:
        state_cls, icon = "text-warning", _icon("diamond", color=_svg.AMBER)
        verdict = "near escape — wave focusing can still slam the wall"
    else:
        state_cls, icon = "text-success", _icon("check", color=_svg.GREEN)
        verdict = "bound — the well recaptures the crest"

    readout = html.Div([
        html.Span(f"well depth: {props['depth']:.3g} J/kg",
                  className="d-block"),
        html.Span(f"crest energy: {crest['total']:.3g} J/kg "
                  f"({crest['well']:.3g} well + {crest['elastic']:.3g} elastic)",
                  className="d-block"),
        html.Span([icon, f" escape ratio: {pct:.0f}% — {verdict}"],
                  className=f"d-block {state_cls}"),
    ])

    warning = (f"crest carries {pct:.0f}% of the escape energy — "
               f"expect wall interaction" if ratio >= 0.85
               else "measured from the well floor u* = 2 m")
    warn_cls = ("text-warning d-block mb-3 small" if 0.85 <= ratio < 1.0
                else "text-danger fw-bold d-block mb-3 small" if ratio >= 1.0
                else "text-djc-muted d-block mb-3 small")
    return readout, warning, warn_cls


@app.callback(
    [Output("simulation-data", "data"),
     Output("status-message", "children"),
     Output("sim-params", "data")],
    Input("run-button", "n_clicks"),
    [State("k2-offset", "value"),
     State("c-slider", "value"),
     State("dt-frac", "value"),
     State("dx-slider", "value"),
     State("time-slider", "value"),
     State("amplitude-slider", "value"),
     State("center-slider", "value"),
     State("width-slider", "value"),
     State("store-drawn", "data")],
    prevent_initial_call=True,
)
def run_simulation(n_clicks, k2_offset, c, dt_frac, dx, total_time,
                   amplitude, center, width, drawn):
    """Run the simulation with the current parameter values."""
    if n_clicks is None:
        return None, "", {}
    label = "Simulation complete"
    if drawn and drawn.get("bumps"):
        label += " — with drawn shape"
    return _solve_and_pack(k2_offset, c, dt_frac, dx, total_time,
                           amplitude, center, width, label, drawn=drawn)


def _solve_and_pack(k2_offset, c, dt_frac, dx, total_time,
                    amplitude, center, width, label, drawn=None):
    """The one solver path every entry point shares.

    RUN button, click-to-pluck, and freehand-drawn shapes all funnel
    through here, so the CFL guard, the calibrated well, the conserved
    energy ledger, and the escape story apply identically no matter how
    the initial condition was born. ``amplitude`` may be negative (a
    pluck *below* the floor is a real, physical inverted pulse).
    """
    dt = dt_frac * cfl_limit(c, dx)
    ratio = cfl_ratio(c, dt, dx)
    if ratio > 1.0:
        status = dbc.Alert(
            [f"CFL condition violated: c·dt/dx = {ratio:.2f} > 1 — the grid "
             f"would detonate. Pull dt below {1.0:.1f}× (playground rules)."],
            color="danger",
        )
        return None, status, {}

    try:
        x = np.arange(0, 50 + dx, dx)

        # Rest state is the well equilibrium; the pulse rides on top of it.
        # K1/K2 scale together so F(u*) = 0 for every offset — the well moves,
        # the equilibrium doesn't.
        scale = 10.0 ** k2_offset
        K1 = WAVE_SCALE_K1 * scale
        K2 = WAVE_SCALE_K2 * scale

        u0 = amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))
        if drawn:
            # Freehand bumps (draw mode) ride on top of the base pulse.
            for bump in drawn.get("bumps", []):
                u0 = u0 + bump["amp"] * np.exp(
                    -0.5 * ((x - bump["center"]) / bump["width"]) ** 2)
        u0_prev = u0.copy()  # rest start

        u_history, time_points, v_history = wave_solver(
            x, u0, u0_prev, c, dt, total_time, K1, K2, return_velocity=True)

        # --- Real energies -------------------------------------------------
        # The conserved energy of u_tt = c^2 u_xx + F(u) is the line integral
        # of [1/2 u_t^2 + 1/2 c^2 u_x^2 + V(u)] dx, with V(u) taken on the
        # *absolute* displacement (the calibration puts F = 0 at u* = 2 m, so
        # V(u) - V(u*) is the well energy; no |u - u*| reflection — that is a
        # different, non-conserved potential). KE uses velocities honest to
        # the leapfrog recurrence, not np.gradient over saved frames.
        ke = 0.5 * np.mean(v_history ** 2, axis=1)

        pe_elastic_density = 0.5 * (c ** 2) * np.diff(
            u_history, axis=1) ** 2 / dx ** 2
        pe_elastic = pe_elastic_density.mean(axis=1)
        pe_well_density = potential_function(u_history, K1, K2) \
            - potential_function(WELL_U_STAR, K1, K2)
        pe_well = pe_well_density.mean(axis=1)
        pe = pe_elastic + pe_well

        # Phase-space probe at the grid center
        center_idx = len(x) // 2
        position = u_history[:, center_idx]
        velocity_center = v_history[:, center_idx]

        data = {
            "x": x.tolist(),
            "u_history": u_history.tolist(),
            "time_points": time_points.tolist(),
            "kinetic_energy": ke.tolist(),
            "potential_energy": pe.tolist(),
            "position": position.tolist(),
            "velocity": velocity_center.tolist(),
            "center_x": float(x[center_idx]),
            "K1": float(K1),
            "K2": float(K2),
        }

        # Honest status: warn when the frame count was too big to be useful
        status = dbc.Alert(
            [_icon("check", color=_svg.GREEN),
             f" {label} — {len(time_points)} frames "
             f"(dt = {dt:.4g} s, CFL = {ratio:.2f})"],
            color="success",
        )
        params = {"K1": float(K1), "K2": float(K2), "k2_offset": k2_offset}
        return data, status, params

    except Exception as exc:  # keep the dashboard alive, surface the error
        status = dbc.Alert(
            [_icon("cross", color=_svg.RED), f" Error: {exc}"],
            color="danger",
        )
        return None, status, {}


@app.callback(
    [Output("simulation-data", "data", allow_duplicate=True),
     Output("status-message", "children", allow_duplicate=True),
     Output("sim-params", "data", allow_duplicate=True)],
    Input("wave-graph", "clickData"),
    [State("k2-offset", "value"),
     State("c-slider", "value"),
     State("dt-frac", "value"),
     State("dx-slider", "value"),
     State("time-slider", "value"),
     State("width-slider", "value")],
    prevent_initial_call=True,
)
def pluck_from_click(click_data, k2_offset, c, dt_frac, dx, total_time,
                     width):
    """Click anywhere on the string → Gaussian pluck there, solved live.

    The clicked y is the pluck amplitude (signed — you can pluck below the
    floor); the clicked x is the pluck center. Everything downstream is
    the exact RUN path, so CFL, the energy ledger, and the escape story
    all apply to a plucked run for free.
    """
    if not click_data or "points" not in click_data:
        return None, "", {}
    point = click_data["points"][0]
    click_x, click_y = float(point["x"]), float(point.get("y") or 0.0)

    # Pluck amplitude from the pointer, sanity-clamped; σ follows the
    # width slider so the finger's pluck matches the slider's story.
    amp = float(np.clip(click_y - WELL_U_STAR, -4.0, 4.0))
    if abs(amp) < 0.05:
        amp = 1.0  # clicked the flat floor — give them a real pluck

    label = f"Plucked at x = {click_x:.1f} m (A = {amp:+.2f} m)"
    return _solve_and_pack(k2_offset, c, dt_frac, dx, total_time,
                           amp, click_x, width, label)


@app.callback(
    Output("pluck-mode-banner", "children"),
    Input("pluck-mode", "value"),
)
def pluck_mode_banner(mode):
    """A one-line hint of what the pointer currently does."""
    return pluck_mode_hint(mode)


@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"),
     Input("simulation-data", "data")],
    [State("sim-params", "data")],
)
def render_tab_content(active_tab, sim_data, params):
    """Render the appropriate visualization based on selected tab."""
    if active_tab == "tab-landscapes":
        return _render_landscapes(params)

    if active_tab == "tab-potential":
        # Potential/force view needs no simulation — but it does want the
        # K1/K2 currently selected, falling back to the calibration.
        offset = (params or {}).get("k2_offset", 0.0)
        scale = 10.0 ** offset
        K1, K2 = WAVE_SCALE_K1 * scale, WAVE_SCALE_K2 * scale
        # Window chosen to show the well bowl and its minimum at u* = 2 —
        # including u -> 0 would put the 1/u^12 wall at ~1e21 on screen and
        # flatten everything else into a line.
        u_range = np.linspace(1.5, 4.5, 500)
        potential = potential_function(u_range, K1, K2)
        force = force_function(u_range, K1, K2)
        fig = create_interactive_potential_force(
            u_range, potential, force,
            title=f"Lennard-Jones well — k₁ = {K1:.3g}, k₂ = {K2:.3g} "
                  f"(equilibrium u* = 2 m)",
        )
        return dcc.Graph(figure=fig,
                         config={"displaylogo": False},
                         style={"height": "600px"})

    if active_tab == "tab-animation" and sim_data is None:
        # No sim yet — keep inviting the pointer.
        return _empty_wave_graph()

    if sim_data is None:
        return dbc.Alert(
            ["Set your parameters and hit RUN SIMULATION — or switch to "
             "the Wave tab and click the string directly."],
            color="info",
        )

    x = np.array(sim_data["x"])
    u_history = [np.array(u) for u in sim_data["u_history"]]
    time_points = np.array(sim_data["time_points"])
    ke = np.array(sim_data["kinetic_energy"])
    pe = np.array(sim_data["potential_energy"])

    if active_tab == "tab-animation":
        # Sample frames for a snappy animation (max ~120 frames)
        step = max(1, len(u_history) // 120)
        fig = create_animated_wave(x, u_history[::step], time_points[::step],
                                   baseline=WELL_U_STAR)
        # Keep the whole canvas pluckable after a run, too.
        fig.add_trace(_pluck_hit_traces())
        return dcc.Graph(figure=fig,
                         id="wave-graph",
                         config=GRAPH_CONFIG,
                         style={"height": "600px"})

    if active_tab == "tab-3d":
        step = max(1, len(u_history) // 60)
        fig = create_3d_wave_surface(x, u_history[::step], time_points[::step])
        return dcc.Graph(figure=fig,
                         config={"displaylogo": False},
                         style={"height": "620px"})

    if active_tab == "tab-energy":
        total = ke + pe
        fig = create_energy_monitor(time_points, ke, pe, total)
        return dcc.Graph(figure=fig,
                         config={"displaylogo": False},
                         style={"height": "600px"})

    if active_tab == "tab-phase":
        fig = create_phase_space(
            np.array(sim_data["position"]),
            np.array(sim_data["velocity"]),
            time_points,
            title=f"Phase space at x = {sim_data['center_x']:.2f} m "
                  f"(green = start, red = end)",
        )
        return dcc.Graph(figure=fig,
                         config={"displaylogo": False},
                         style={"height": "600px"})

    return html.Div("Select a tab to view a visualization")


@app.callback(
    Output("store-drawn", "data"),
    Input("wave-graph", "relayoutData"),
    State("store-drawn", "data"),
    State("pluck-mode", "value"),
    prevent_initial_call=True,
)
def capture_drawn_shape(relayout, existing, mode):
    """Rasterize a drag-drawn pen path while in draw mode.

    Plotly has no freehand pen, so draw mode repurposes box-select: the
    box's span becomes a raised bump between its corners. Clicks that are
    pure zooms/pan (xaxis.range changes etc.) pass through untouched.
    """
    if mode != "draw" or not relayout:
        return existing  # draw signal only when a selection arrives
    if "selections" in relayout and relayout["selections"]:
        sel = relayout["selections"][-1]
        try:
            x0, x1 = float(sel["x0"]), float(sel["x1"])
            y0, y1 = float(sel["y0"]), float(sel["y1"])
        except (KeyError, TypeError, ValueError):
            return existing
        if x1 < x0:
            x0, x1 = x1, x0
        amp = float(np.clip(y1 - WELL_U_STAR, -4.0, 4.0))
        if abs(amp) < 0.05:
            amp = 1.0
        center, width = 0.5 * (x0 + x1), max(0.2, 0.5 * (x1 - x0))
        drawn = {"center": center, "width": width, "amp": amp}
        return {"bumps": [drawn]}
    return existing


def _render_landscapes(params):
    """The Landscapes tab: two regime maps, drawn from cached sweeps."""
    offset = (params or {}).get("k2_offset", 0.0)
    scale = 10.0 ** offset
    K1, K2 = WAVE_SCALE_K1 * scale, WAVE_SCALE_K2 * scale

    # Landscape 1 is pure analytics — instant on every visit.
    esc = escape_landscape(k1=K1, k2=K2)
    escape_fig = create_escape_landscape(
        esc["ratio"], esc["axes"]["amplitude"], esc["axes"]["width"],
        esc["depth"],
        title=f"Escape landscape — well depth {esc['depth']:.3g} J/kg; "
              f"dotted amber = ratio 1 (punch-through boundary)")

    # Landscape 2 runs real short sims across the grid; keep the coarse
    # default so the tab stays responsive (a few seconds, parallel).
    try:
        stab = stability_landscape(n=9)
        stab_fig = create_stability_landscape(
            stab["drift"], stab["axes"]["dt_frac"], stab["axes"]["dx"])
        stability_block = dcc.Graph(figure=stab_fig,
                                    config={"displaylogo": False},
                                    style={"height": "560px"})
    except Exception as exc:
        stability_block = dbc.Alert(
            f"stability sweep failed: {exc}", color="warning")

    return html.Div([
        dcc.Graph(figure=escape_fig, config={"displaylogo": False},
                  style={"height": "560px"}),
        html.P("Above the dotted line the initial crest already carries "
               "more energy than the well depth — Mission 2's punch-through "
               "regime. Below it, the well recaptures the crest.",
               className="text-djc-muted"),
        stability_block,
        html.P("dt past the CFL limit (1.0×) lights up: the explicit scheme "
               "detonates exactly where the Courant number exceeds one, "
               "independent of dx.", className="text-djc-muted"),
    ])


if __name__ == "__main__":
    print("=" * 60)
    print("  DJC WAVE LAB — interactive dashboard")
    print("=" * 60)
    print("\nStarting server...")
    print("Open your browser to: http://localhost:8050")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    app.run(debug=False, host="127.0.0.1", port=8050)
