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
from interactive_visualization import (
    create_animated_wave,
    create_3d_wave_surface,
    create_interactive_potential_force,
    create_energy_monitor,
    create_phase_space,
)

import djc_plotly as P

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
            ], id="tabs", active_tab="tab-animation", className="mb-2"),

            html.Div(id="tab-content", className="mt-3"),
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
    if ratio >= 1.0:
        state_cls, icon = "text-danger fw-bold", "⚠"
        verdict = "crest exceeds the well — wall slams and punch-through likely"
    elif ratio >= 0.85:
        state_cls, icon = "text-warning", "◆"
        verdict = "near escape — wave focusing can still slam the wall"
    else:
        state_cls, icon = "text-success", "✓"
        verdict = "bound — the well recaptures the crest"

    readout = html.Div([
        html.Span(f"well depth: {props['depth']:.3g} J/kg",
                  className="d-block"),
        html.Span(f"crest energy: {crest['total']:.3g} J/kg "
                  f"({crest['well']:.3g} well + {crest['elastic']:.3g} elastic)",
                  className="d-block"),
        html.Span(f"{icon} escape ratio: {pct:.0f}% — {verdict}",
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
     State("width-slider", "value")],
    prevent_initial_call=True,
)
def run_simulation(n_clicks, k2_offset, c, dt_frac, dx, total_time,
                   amplitude, center, width):
    """Run the simulation with the current parameter values."""
    if n_clicks is None:
        return None, "", {}

    dt = dt_frac * cfl_limit(c, dx)
    ratio = cfl_ratio(c, dt, dx)
    if ratio > 1.0:
        status = dbc.Alert(
            f"✗ CFL condition violated: c·dt/dx = {ratio:.2f} > 1 — the grid "
            f"would detonate. Pull dt below {1.0:.1f}× (playground rules).",
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
            f"✓ Simulation complete — {len(time_points)} frames "
            f"(dt = {dt:.4g} s, CFL = {ratio:.2f})",
            color="success",
        )
        params = {"K1": float(K1), "K2": float(K2), "k2_offset": k2_offset}
        return data, status, params

    except Exception as exc:  # keep the dashboard alive, surface the error
        status = dbc.Alert(f"✗ Error: {exc}", color="danger")
        return None, status, {}


@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"),
     Input("simulation-data", "data")],
    [State("sim-params", "data")],
)
def render_tab_content(active_tab, sim_data, params):
    """Render the appropriate visualization based on selected tab."""
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

    if sim_data is None:
        return dbc.Alert(
            "👈 Set your parameters and hit RUN SIMULATION",
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
        return dcc.Graph(figure=fig,
                         config={"displaylogo": False},
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


if __name__ == "__main__":
    print("=" * 60)
    print("  DJC WAVE LAB — interactive dashboard")
    print("=" * 60)
    print("\nStarting server...")
    print("Open your browser to: http://localhost:8050")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    app.run(debug=False, host="127.0.0.1", port=8050)
