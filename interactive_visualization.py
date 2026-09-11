"""
Interactive visualization module using Plotly for wave simulations.

This module provides animated and interactive visualizations including:
- Animated wave propagation
- 3D surface plots of wave evolution
- Interactive potential and force function plots
- Energy monitoring over time
- Phase space visualization

All figures share the DJC design system (deep navy canvas, teal/cyan glow
series, glass legends) via ``djc_plotly`` — the same look as the static
matplotlib charts, so every visualization surface in the repo reads as one
product.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, List, Optional

import djc_plotly as P


# ============================================================================
# Shared styling helpers
# ============================================================================

def _animation_controls(frames, n_steps: int = 60):
    """Themed Play/Pause buttons + time slider shared by animated figures."""
    buttons = [
        {
            "label": "▶ Play",
            "method": "animate",
            "args": [None, {
                "frame": {"duration": n_steps, "redraw": True},
                "fromcurrent": True,
                "transition": {"duration": 0},
            }],
        },
        {
            "label": "⏸ Pause",
            "method": "animate",
            "args": [[None], {
                "frame": {"duration": 0, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": 0},
            }],
        },
    ]
    return [
        {
            "type": "buttons",
            "showactive": False,
            "x": 0.0, "y": 1.12, "xanchor": "left", "yanchor": "top",
            "bgcolor": P.NAVY.N800,
            "bordercolor": P.GRID,
            "font": {"color": P.FG_2},
            "buttons": buttons,
        }
    ], [{
        "active": 0,
        "currentvalue": {"prefix": "t = ", "font": {"color": P.FG_2}},
        "x": 0.0, "len": 0.92, "xanchor": "left", "y": -0.18, "yanchor": "top",
        "bgcolor": P.NAVY.N800,
        "activebgcolor": P.NAVY.N700,
        "bordercolor": P.GRID,
        "tickcolor": P.GRID,
        "font": {"color": P.FG_3, "size": 11},
        "steps": [
            {
                "args": [[f.name], {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                }],
                "label": f"{float(f.name):.3f}",
                "method": "animate",
            }
            for f in frames
        ],
    }]


# ============================================================================
# 1. Animated wave
# ============================================================================

def create_animated_wave(x: np.ndarray, u_history: List[np.ndarray],
                         time_points: np.ndarray,
                         title: str = "Wave Evolution Animation",
                         baseline: Optional[float] = None) -> go.Figure:
    """
    Create an animated visualization of wave propagation over time.

    Args:
        x: Spatial grid points
        u_history: List of displacement arrays at each time step
        time_points: Array of time values
        title: Plot title
        baseline: Optional rest level (e.g. the well floor u*) drawn as a
            dotted reference line. Displacement is absolute, so no area fill
            is drawn — filling to zero would imply a physical quantity that
            does not exist.

    Returns:
        Plotly Figure object with animation
    """
    # Glow traces per frame (halo, medium, crisp core) so the pulse emits
    # light. The baseline reference rides in EVERY frame's data: animation
    # frames replace the figure's trace list, so a trace added only to the
    # base figure would vanish on Play.
    def frame_traces(u):
        traces = P.glow_traces(x, u, color=P.ACCENT.CYAN, width=2.6,
                               name="u(x, t)")
        if baseline is not None:
            traces.append(go.Scatter(
                x=[float(x[0]), float(x[-1])], y=[baseline, baseline],
                mode="lines", name=f"well floor u* = {baseline:g} m",
                line=dict(color=P.FG_4, width=1.4, dash="dot"),
                hoverinfo="skip",
            ))
        return traces

    frames = []
    for u, t in zip(u_history, time_points):
        frames.append(go.Frame(
            data=frame_traces(u),
            name=f"{t:.3f}",
            layout=go.Layout(title_text=f"{title}  ·  t = {t:.3f} s"),
        ))

    updatemenus, sliders = _animation_controls(frames)

    fig = go.Figure(
        data=frame_traces(u_history[0]),
        layout=go.Layout(
            title=dict(text=f"{title}  ·  t = {time_points[0]:.3f} s",
                       x=0.01, xanchor="left"),
            xaxis=dict(title="Position x (m)"),
            yaxis=dict(title="Displacement u (m)"),
            updatemenus=updatemenus,
            sliders=sliders,
            template=P.TEMPLATE,
            height=520,
        ),
        frames=frames,
    )

    return fig


# ============================================================================
# 2. 3D wave surface
# ============================================================================

def create_3d_wave_surface(x: np.ndarray, u_history: List[np.ndarray],
                           time_points: np.ndarray,
                           title: str = "3D Wave Evolution") -> go.Figure:
    """
    Create a 3D surface plot showing wave evolution in space-time.

    Args:
        x: Spatial grid points
        u_history: List of displacement arrays at each time step
        time_points: Array of time values
        title: Plot title

    Returns:
        Plotly Figure object with 3D surface
    """
    X, T = np.meshgrid(x, time_points)
    Z = np.array(u_history)

    fig = go.Figure(data=[go.Surface(
        x=X, y=T, z=Z,
        colorscale=P.WAVE_SCALE,
        colorbar=dict(
            title=dict(text="u", side="right"),
            tickfont={"color": P.FG_3},
            outlinecolor=P.GRID,
            outlinewidth=1,
        ),
        contours={
            "z": {"show": True, "usecolormap": True,
                  "highlightcolor": P.ACCENT.CYAN,
                  "project": {"z": True}},
        },
        hovertemplate="x: %{x:.2f}<br>t: %{y:.3f}<br>u: %{z:.3f}<extra></extra>",
    )])

    axis = dict(
        backgroundcolor=P.NAVY.N900,
        gridcolor=P.GRID,
        zerolinecolor=P.GRID,
        showbackground=True,
    )
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        scene=dict(
            xaxis_title="Position x (m)",
            yaxis_title="Time t (s)",
            zaxis_title="Displacement u (m)",
            xaxis=axis, yaxis=axis, zaxis=axis,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
        ),
        template=P.TEMPLATE,
        height=620,
    )
    return fig


# ============================================================================
# 3. Potential & force functions
# ============================================================================

def create_interactive_potential_force(u_range: np.ndarray,
                                       potential: np.ndarray,
                                       force: np.ndarray,
                                       title: str = "Potential & Force Functions") -> go.Figure:
    """
    Create interactive plot showing both potential and force functions.

    Args:
        u_range: Range of displacement values
        potential: Potential energy values
        force: Force values
        title: Plot title

    Returns:
        Plotly Figure with dual y-axes
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Potential — cyan (matches the static DJC potential chart)
    fig.add_trace(
        go.Scatter(x=u_range, y=potential, name="Potential V(u)",
                   line=dict(color=P.ACCENT.CYAN, width=3),
                   hovertemplate="u: %{x:.3f}<br>V(u): %{y:.3e}<extra></extra>"),
        secondary_y=False,
    )
    # Force — amber, dashed
    fig.add_trace(
        go.Scatter(x=u_range, y=force, name="Force F(u)",
                   line=dict(color=P.ACCENT.AMBER, width=2.5, dash="dash"),
                   hovertemplate="u: %{x:.3f}<br>F(u): %{y:.3e}<extra></extra>"),
        secondary_y=True,
    )

    # F = 0 equilibrium marker
    fig.add_hline(y=0, line_dash="dot", line_color=P.FG_4, secondary_y=True,
                  annotation_text="F = 0 (equilibrium)",
                  annotation_font_color=P.FG_3)

    fig.update_xaxes(title_text="Displacement u (m)")
    fig.update_yaxes(title_text="Potential V(u)", secondary_y=False,
                     exponentformat="e")
    fig.update_yaxes(title_text="Force F(u)", secondary_y=True,
                     exponentformat="e")

    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        hovermode="x unified",
        template=P.TEMPLATE,
        height=520,
        legend=dict(x=0.68, y=0.98),
    )
    return fig


# ============================================================================
# 4. Energy monitor
# ============================================================================

def create_energy_monitor(time_points: np.ndarray,
                          kinetic_energy: np.ndarray,
                          potential_energy: np.ndarray,
                          total_energy: np.ndarray,
                          title: str = "Energy Conservation Monitor") -> go.Figure:
    """
    Create interactive energy monitoring plot showing kinetic, potential,
    and total energy.

    Args:
        time_points: Time values
        kinetic_energy: Kinetic energy at each time step
        potential_energy: Potential energy at each time step
        total_energy: Total energy at each time step
        title: Plot title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Same series colors as analysis.py's static energy chart:
    # KE violet · PE teal · Total cyan glow.
    fig.add_trace(go.Scatter(
        x=time_points, y=kinetic_energy,
        name="Kinetic Energy",
        line=dict(color=P.ACCENT.VIOLET, width=2),
        fill="tozeroy", fillcolor=P.hexa(P.ACCENT.VIOLET, 0.16),
        hovertemplate="t: %{x:.3f}<br>KE: %{y:.3e}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=time_points, y=potential_energy,
        name="Potential Energy",
        line=dict(color=P.TEAL.N300, width=2),
        fill="tozeroy", fillcolor=P.hexa(P.TEAL.N300, 0.16),
        hovertemplate="t: %{x:.3f}<br>PE: %{y:.3e}<extra></extra>",
    ))
    # Total drawn last, with the signature glow
    for tr in P.glow_traces(time_points, total_energy, color=P.ACCENT.CYAN,
                            width=2.6, name="Total Energy"):
        fig.add_trace(tr)

    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        xaxis_title="Time t (s)",
        yaxis_title="Energy (J)",
        template=P.TEMPLATE,
        height=520,
        legend=dict(x=0.02, y=0.98),
        yaxis=dict(exponentformat="e"),
    )
    return fig


# ============================================================================
# 5. Phase space
# ============================================================================

def create_phase_space(position: np.ndarray,
                       velocity: np.ndarray,
                       time_points: Optional[np.ndarray] = None,
                       title: str = "Phase Space Trajectory") -> go.Figure:
    """
    Create phase space plot (position vs velocity).

    Args:
        position: Position values
        velocity: Velocity values
        time_points: Optional time values for color coding
        title: Plot title

    Returns:
        Plotly Figure object
    """
    if time_points is not None:
        fig = go.Figure(data=go.Scatter(
            x=position, y=velocity,
            mode="lines+markers",
            name="trajectory",
            marker=dict(
                size=4.5,
                color=time_points,
                colorscale=P.TIME_SCALE,
                showscale=True,
                colorbar=dict(
                    title=dict(text="t (s)", side="right"),
                    tickfont={"color": P.FG_3},
                    outlinecolor=P.GRID, outlinewidth=1,
                ),
            ),
            line=dict(width=1, color=P.hexa(P.NAVY.N300, 0.35)),
            hovertemplate="u: %{x:.3f}<br>v: %{y:.3f}<extra></extra>",
        ))
    else:
        fig = go.Figure(data=P.glow_traces(
            position, velocity, color=P.ACCENT.CYAN, width=2.2,
        ))
        fig.data[0].update(mode="lines+markers",
                           marker=dict(size=4, color=P.ACCENT.CYAN))

    # Start (green) and end (red) markers — matches the static phase plot
    fig.add_trace(go.Scatter(
        x=[position[0]], y=[velocity[0]],
        mode="markers", name="Start",
        marker=dict(size=11, color=P.ACCENT.GREEN, symbol="circle",
                    line=dict(color=P.TEAL.N50, width=1.2)),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[position[-1]], y=[velocity[-1]],
        mode="markers", name="End",
        marker=dict(size=11, color=P.ACCENT.RED, symbol="x",
                    line=dict(color=P.TEAL.N50, width=1.2)),
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        xaxis_title="Displacement u (m)",
        yaxis_title="Velocity v (m/s)",
        template=P.TEMPLATE,
        height=520,
        hovermode="closest",
    )
    return fig


# ============================================================================
# 6. Comprehensive dashboard layout
# ============================================================================

def create_dashboard_layout(x: np.ndarray,
                            u_history: List[np.ndarray],
                            time_points: np.ndarray,
                            potential: np.ndarray,
                            force: np.ndarray,
                            u_range: np.ndarray,
                            kinetic_energy: Optional[np.ndarray] = None,
                            potential_energy: Optional[np.ndarray] = None) -> go.Figure:
    """
    Create a comprehensive dashboard with multiple subplots.

    Args:
        x: Spatial grid points
        u_history: Wave displacement history
        time_points: Time values
        potential: Potential energy function values
        force: Force function values
        u_range: Range for potential/force plots
        kinetic_energy: Optional kinetic energy values
        potential_energy: Optional potential energy values

    Returns:
        Plotly Figure with subplots
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Wave at Current Time", "Potential & Force Functions",
                        "Wave Evolution (3D)", "Energy Monitor"),
        specs=[[{"type": "xy"}, {"type": "xy", "secondary_y": True}],
               [{"type": "scene"}, {"type": "xy"}]],
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
    )

    # Wave (glow) at mid time — absolute displacement, so no zero fill
    mid_idx = len(u_history) // 2
    for tr in P.glow_traces(x, u_history[mid_idx], color=P.ACCENT.CYAN,
                            width=2.2, name="u(x, t)"):
        fig.add_trace(tr, row=1, col=1)

    # Potential & force
    fig.add_trace(
        go.Scatter(x=u_range, y=potential, name="V(u)",
                   line=dict(color=P.ACCENT.CYAN, width=2.4)),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=u_range, y=force, name="F(u)",
                   line=dict(color=P.ACCENT.AMBER, width=2, dash="dash")),
        row=1, col=2, secondary_y=True,
    )

    # 3D surface
    X, T = np.meshgrid(x, time_points)
    Z = np.array(u_history)
    fig.add_trace(
        go.Surface(x=X, y=T, z=Z, colorscale=P.WAVE_SCALE, showscale=False),
        row=2, col=1,
    )

    # Energy monitor
    if kinetic_energy is not None and potential_energy is not None:
        total_energy = kinetic_energy + potential_energy
        fig.add_trace(
            go.Scatter(x=time_points, y=kinetic_energy, name="KE",
                       line=dict(color=P.ACCENT.VIOLET, width=2)),
            row=2, col=2,
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=potential_energy, name="PE",
                       line=dict(color=P.TEAL.N300, width=2)),
            row=2, col=2,
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=total_energy, name="Total",
                       line=dict(color=P.ACCENT.CYAN, width=2.4, dash="dash")),
            row=2, col=2,
        )

    fig.update_layout(
        height=900,
        showlegend=True,
        title=dict(text="Wave Simulation Interactive Dashboard", x=0.01,
                   xanchor="left"),
        template=P.TEMPLATE,
    )

    fig.update_xaxes(title_text="Position x (m)", row=1, col=1)
    fig.update_yaxes(title_text="Displacement u (m)", row=1, col=1)
    fig.update_xaxes(title_text="u (m)", row=1, col=2)
    fig.update_yaxes(title_text="V(u)", row=1, col=2)
    fig.update_xaxes(title_text="t (s)", row=2, col=2)
    fig.update_yaxes(title_text="Energy (J)", row=2, col=2)

    return fig
