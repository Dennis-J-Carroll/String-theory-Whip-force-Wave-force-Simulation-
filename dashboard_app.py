"""
Interactive Dash Dashboard for Wave Simulation

This module provides a web-based interactive dashboard using Plotly Dash.
Run this script to launch a local web server with real-time parameter controls.

Usage:
    python dashboard_app.py

Then open your browser to http://localhost:8050
"""

import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from solver import wave_solver, potential_function, force_function
from interactive_visualization import (
    create_animated_wave,
    create_3d_wave_surface,
    create_interactive_potential_force,
    create_energy_monitor,
    create_phase_space
)

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🌊 Wave Simulation Dashboard", className="text-center mb-4"),
            html.P("Interactive exploration of wave dynamics with string theory-inspired forces",
                  className="text-center text-muted")
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("⚙️ Physics Parameters")),
                dbc.CardBody([
                    html.Label("Repulsive Constant (log₁₀ K1)"),
                    dcc.Slider(
                        id='k1-slider',
                        min=30, max=40, step=0.5, value=35,
                        marks={i: str(i) for i in range(30, 41, 2)},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    html.Br(),

                    html.Label("Attractive Constant (log₁₀ K2)"),
                    dcc.Slider(
                        id='k2-slider',
                        min=15, max=22, step=0.5, value=18,
                        marks={i: str(i) for i in range(15, 23, 2)},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    html.Br(),

                    html.Label("Wave Speed (c)"),
                    dcc.Slider(
                        id='c-slider',
                        min=0.1, max=5.0, step=0.1, value=1.0,
                        marks={i: f'{i}' for i in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                ])
            ], className="mb-3"),

            dbc.Card([
                dbc.CardHeader(html.H4("🔢 Numerical Parameters")),
                dbc.CardBody([
                    html.Label("Time Step (dt)"),
                    dcc.Slider(
                        id='dt-slider',
                        min=0.001, max=0.05, step=0.001, value=0.01,
                        marks={0.01: '0.01', 0.02: '0.02', 0.03: '0.03', 0.04: '0.04', 0.05: '0.05'},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    html.Br(),

                    html.Label("Space Step (dx)"),
                    dcc.Slider(
                        id='dx-slider',
                        min=0.05, max=0.5, step=0.05, value=0.1,
                        marks={i/10: f'{i/10:.1f}' for i in range(1, 6)},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    html.Br(),

                    html.Label("Total Time"),
                    dcc.Slider(
                        id='time-slider',
                        min=1, max=20, step=1, value=5,
                        marks={i: str(i) for i in range(1, 21, 4)},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                ])
            ], className="mb-3"),

            dbc.Card([
                dbc.CardHeader(html.H4("📍 Initial Conditions")),
                dbc.CardBody([
                    html.Label("Amplitude"),
                    dcc.Slider(
                        id='amplitude-slider',
                        min=0.1, max=5.0, step=0.1, value=1.0,
                        marks={i: str(i) for i in range(1, 6)},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    html.Br(),

                    html.Label("Center Position"),
                    dcc.Slider(
                        id='center-slider',
                        min=10, max=40, step=1, value=25,
                        marks={i: str(i) for i in range(10, 41, 10)},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    html.Br(),

                    html.Label("Gaussian Width"),
                    dcc.Slider(
                        id='width-slider',
                        min=0.5, max=5.0, step=0.1, value=2.0,
                        marks={i: str(i) for i in range(1, 6)},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                ])
            ], className="mb-3"),

            dbc.Button("🚀 Run Simulation", id="run-button", color="success",
                      size="lg", className="w-100 mb-3"),

            html.Div(id="status-message", className="text-center")

        ], width=3),

        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="📊 Animated Wave", tab_id="tab-animation"),
                dbc.Tab(label="🎨 3D Surface", tab_id="tab-3d"),
                dbc.Tab(label="⚡ Energy Monitor", tab_id="tab-energy"),
                dbc.Tab(label="🔄 Phase Space", tab_id="tab-phase"),
                dbc.Tab(label="📈 Potential & Force", tab_id="tab-potential"),
            ], id="tabs", active_tab="tab-animation"),

            html.Div(id="tab-content", className="mt-3")

        ], width=9)
    ]),

    # Store simulation results
    dcc.Store(id='simulation-data'),

], fluid=True, className="p-4")


@app.callback(
    [Output('simulation-data', 'data'),
     Output('status-message', 'children')],
    Input('run-button', 'n_clicks'),
    [State('k1-slider', 'value'),
     State('k2-slider', 'value'),
     State('c-slider', 'value'),
     State('dt-slider', 'value'),
     State('dx-slider', 'value'),
     State('time-slider', 'value'),
     State('amplitude-slider', 'value'),
     State('center-slider', 'value'),
     State('width-slider', 'value')],
    prevent_initial_call=True
)
def run_simulation(n_clicks, k1_exp, k2_exp, c, dt, dx, total_time,
                   amplitude, center, width):
    """
    Run the wave simulation with current parameter values.
    """
    if n_clicks is None:
        return None, ""

    try:
        # Convert K1, K2 from log scale
        K1 = 10**k1_exp
        K2 = 10**k2_exp

        # Create spatial grid
        x = np.arange(0, 50, dx)

        # Initial conditions
        u0 = amplitude * np.exp(-((x - center) / width)**2)
        u0_prev = u0.copy()

        # Run simulation
        u_history, time_points = wave_solver(x, u0, u0_prev, c, dt, total_time, K1, K2)

        # Calculate energies
        kinetic_energy = []
        potential_energy = []

        for u in u_history:
            # Kinetic energy approximation
            ke = 0.5 * np.sum(u**2) * dx
            kinetic_energy.append(ke)

            # Potential energy
            pe = np.sum(potential_function(np.abs(u) + 0.1, K1, K2)) * dx
            potential_energy.append(pe)

        # Phase space data (center point)
        center_idx = len(x) // 2
        position = [u[center_idx] for u in u_history]
        velocity = np.gradient(position, dt).tolist()

        # Store data
        data = {
            'x': x.tolist(),
            'u_history': [u.tolist() for u in u_history],
            'time_points': time_points.tolist(),
            'kinetic_energy': kinetic_energy,
            'potential_energy': potential_energy,
            'K1': K1,
            'K2': K2,
            'position': position,
            'velocity': velocity,
            'center_x': x[center_idx]
        }

        status = dbc.Alert(
            f"✓ Simulation complete! {len(u_history)} time steps computed.",
            color="success"
        )

        return data, status

    except Exception as e:
        status = dbc.Alert(
            f"✗ Error: {str(e)}",
            color="danger"
        )
        return None, status


@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'active_tab'),
     Input('simulation-data', 'data')],
    [State('k1-slider', 'value'),
     State('k2-slider', 'value')]
)
def render_tab_content(active_tab, sim_data, k1_exp, k2_exp):
    """
    Render the appropriate visualization based on selected tab.
    """
    if active_tab == "tab-potential":
        # Show potential and force functions (doesn't require simulation)
        K1 = 10**k1_exp
        K2 = 10**k2_exp
        u_range = np.linspace(0.1, 5.0, 500)
        potential = potential_function(u_range, K1, K2)
        force = force_function(u_range, K1, K2)

        fig = create_interactive_potential_force(
            u_range, potential, force,
            title=f"Potential & Force Functions (K1=10^{k1_exp:.1f}, K2=10^{k2_exp:.1f})"
        )
        return dcc.Graph(figure=fig, style={'height': '600px'})

    if sim_data is None:
        return dbc.Alert(
            "👈 Click 'Run Simulation' to generate visualizations",
            color="info"
        )

    # Convert stored data back to numpy arrays
    x = np.array(sim_data['x'])
    u_history = [np.array(u) for u in sim_data['u_history']]
    time_points = np.array(sim_data['time_points'])

    if active_tab == "tab-animation":
        # Sample frames for animation (max 100 frames)
        step = max(1, len(u_history) // 100)
        u_sampled = u_history[::step]
        t_sampled = time_points[::step]

        fig = create_animated_wave(x, u_sampled, t_sampled)
        return dcc.Graph(figure=fig, style={'height': '600px'})

    elif active_tab == "tab-3d":
        # Sample for 3D (max 50 time slices)
        step = max(1, len(u_history) // 50)
        u_sampled = u_history[::step]
        t_sampled = time_points[::step]

        fig = create_3d_wave_surface(x, u_sampled, t_sampled)
        return dcc.Graph(figure=fig, style={'height': '600px'})

    elif active_tab == "tab-energy":
        kinetic_energy = np.array(sim_data['kinetic_energy'])
        potential_energy = np.array(sim_data['potential_energy'])
        total_energy = kinetic_energy + potential_energy

        fig = create_energy_monitor(time_points, kinetic_energy,
                                    potential_energy, total_energy)
        return dcc.Graph(figure=fig, style={'height': '600px'})

    elif active_tab == "tab-phase":
        position = np.array(sim_data['position'])
        velocity = np.array(sim_data['velocity'])
        center_x = sim_data['center_x']

        fig = create_phase_space(position, velocity, time_points,
                                title=f"Phase Space at x={center_x:.2f}")
        return dcc.Graph(figure=fig, style={'height': '600px'})

    return html.Div("Select a tab to view visualization")


if __name__ == '__main__':
    print("=" * 60)
    print("🌊 Wave Simulation Dashboard")
    print("=" * 60)
    print("\nStarting server...")
    print("Open your browser to: http://localhost:8050")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)

    app.run_server(debug=True, host='0.0.0.0', port=8050)
