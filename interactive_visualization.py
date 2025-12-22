"""
Interactive visualization module using Plotly for wave simulations.

This module provides animated and interactive visualizations including:
- Animated wave propagation
- 3D surface plots of wave evolution
- Interactive potential and force function plots
- Energy monitoring over time
- Phase space visualization
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Tuple, List, Optional


def create_animated_wave(x: np.ndarray, u_history: List[np.ndarray],
                        time_points: np.ndarray,
                        title: str = "Wave Evolution Animation") -> go.Figure:
    """
    Create an animated visualization of wave propagation over time.

    Args:
        x: Spatial grid points
        u_history: List of displacement arrays at each time step
        time_points: Array of time values
        title: Plot title

    Returns:
        Plotly Figure object with animation
    """
    # Create frames for animation
    frames = []
    for i, (u, t) in enumerate(zip(u_history, time_points)):
        frame = go.Frame(
            data=[go.Scatter(x=x, y=u, mode='lines',
                           line=dict(color='#1f77b4', width=3),
                           fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.3)')],
            name=f'{t:.3f}',
            layout=go.Layout(title_text=f"{title}<br>Time: {t:.3f}")
        )
        frames.append(frame)

    # Create initial figure
    fig = go.Figure(
        data=[go.Scatter(x=x, y=u_history[0], mode='lines',
                        line=dict(color='#1f77b4', width=3),
                        fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.3)')],
        layout=go.Layout(
            title=f"{title}<br>Time: {time_points[0]:.3f}",
            xaxis=dict(title="Position (x)", gridcolor='lightgray'),
            yaxis=dict(title="Displacement (u)", gridcolor='lightgray'),
            hovermode='x unified',
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 50, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 0}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ],
                'x': 0.1,
                'y': 1.15,
            }],
            sliders=[{
                'active': 0,
                'steps': [
                    {
                        'args': [[f.name], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }],
                        'label': f'{float(f.name):.3f}',
                        'method': 'animate'
                    }
                    for f in frames
                ],
                'x': 0.1,
                'len': 0.85,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top',
            }],
            template='plotly_white',
            height=500
        ),
        frames=frames
    )

    return fig


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
    # Create meshgrid for 3D surface
    X, T = np.meshgrid(x, time_points)
    Z = np.array(u_history)

    fig = go.Figure(data=[go.Surface(
        x=X, y=T, z=Z,
        colorscale='Viridis',
        contours={
            "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project": {"z": True}}
        }
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Position (x)',
            yaxis_title='Time (t)',
            zaxis_title='Displacement (u)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        template='plotly_white',
        height=600
    )

    return fig


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
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add potential trace
    fig.add_trace(
        go.Scatter(x=u_range, y=potential, name="Potential V(u)",
                  line=dict(color='#ff7f0e', width=3),
                  hovertemplate='u: %{x:.2f}<br>V(u): %{y:.2e}<extra></extra>'),
        secondary_y=False,
    )

    # Add force trace
    fig.add_trace(
        go.Scatter(x=u_range, y=force, name="Force F(u)",
                  line=dict(color='#2ca02c', width=3, dash='dash'),
                  hovertemplate='u: %{x:.2f}<br>F(u): %{y:.2e}<extra></extra>'),
        secondary_y=True,
    )

    # Add zero line for force
    fig.add_hline(y=0, line_dash="dot", line_color="gray",
                 secondary_y=True, annotation_text="F=0")

    # Update axes
    fig.update_xaxes(title_text="Displacement (u)", gridcolor='lightgray')
    fig.update_yaxes(title_text="Potential V(u)", secondary_y=False,
                    gridcolor='lightgray', exponentformat='e')
    fig.update_yaxes(title_text="Force F(u)", secondary_y=True,
                    gridcolor='lightgray', exponentformat='e')

    fig.update_layout(
        title=title,
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(x=0.7, y=0.98)
    )

    return fig


def create_energy_monitor(time_points: np.ndarray,
                         kinetic_energy: np.ndarray,
                         potential_energy: np.ndarray,
                         total_energy: np.ndarray,
                         title: str = "Energy Conservation Monitor") -> go.Figure:
    """
    Create interactive energy monitoring plot showing kinetic, potential, and total energy.

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

    # Add kinetic energy
    fig.add_trace(go.Scatter(
        x=time_points, y=kinetic_energy,
        name='Kinetic Energy',
        line=dict(color='#d62728', width=2),
        fill='tozeroy',
        fillcolor='rgba(214, 39, 40, 0.2)',
        hovertemplate='Time: %{x:.3f}<br>KE: %{y:.2e}<extra></extra>'
    ))

    # Add potential energy
    fig.add_trace(go.Scatter(
        x=time_points, y=potential_energy,
        name='Potential Energy',
        line=dict(color='#9467bd', width=2),
        fill='tozeroy',
        fillcolor='rgba(148, 103, 189, 0.2)',
        hovertemplate='Time: %{x:.3f}<br>PE: %{y:.2e}<extra></extra>'
    ))

    # Add total energy
    fig.add_trace(go.Scatter(
        x=time_points, y=total_energy,
        name='Total Energy',
        line=dict(color='#17becf', width=3, dash='dash'),
        hovertemplate='Time: %{x:.3f}<br>Total: %{y:.2e}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (t)",
        yaxis_title="Energy",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(x=0.02, y=0.98),
        yaxis=dict(exponentformat='e')
    )

    return fig


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
        # Color by time
        fig = go.Figure(data=go.Scatter(
            x=position, y=velocity,
            mode='lines+markers',
            marker=dict(
                size=4,
                color=time_points,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="Time (t)")
            ),
            line=dict(width=1, color='rgba(0,0,0,0.3)'),
            hovertemplate='Position: %{x:.3f}<br>Velocity: %{y:.3f}<extra></extra>'
        ))
    else:
        # Simple trajectory
        fig = go.Figure(data=go.Scatter(
            x=position, y=velocity,
            mode='lines+markers',
            marker=dict(size=4, color='#1f77b4'),
            line=dict(width=2, color='#1f77b4'),
            hovertemplate='Position: %{x:.3f}<br>Velocity: %{y:.3f}<extra></extra>'
        ))

    # Add origin marker
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=10, color='red', symbol='x'),
        name='Origin',
        showlegend=False
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Position",
        yaxis_title="Velocity",
        template='plotly_white',
        height=500,
        hovermode='closest'
    )

    return fig


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
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Wave at Current Time', 'Potential & Force Functions',
                       'Wave Evolution (3D)', 'Energy Monitor'),
        specs=[[{"type": "scatter"}, {"type": "scatter", "secondary_y": True}],
               [{"type": "surface"}, {"type": "scatter"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Add current wave state (top left)
    mid_idx = len(u_history) // 2
    fig.add_trace(
        go.Scatter(x=x, y=u_history[mid_idx],
                  line=dict(color='#1f77b4', width=2),
                  fill='tozeroy', name='Wave'),
        row=1, col=1
    )

    # Add potential and force (top right)
    fig.add_trace(
        go.Scatter(x=u_range, y=potential, name="V(u)",
                  line=dict(color='#ff7f0e', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=u_range, y=force, name="F(u)",
                  line=dict(color='#2ca02c', width=2, dash='dash')),
        row=1, col=2, secondary_y=True
    )

    # Add 3D wave surface (bottom left)
    X, T = np.meshgrid(x, time_points)
    Z = np.array(u_history)
    fig.add_trace(
        go.Surface(x=X, y=T, z=Z, colorscale='Viridis', showscale=False),
        row=2, col=1
    )

    # Add energy monitor (bottom right) if provided
    if kinetic_energy is not None and potential_energy is not None:
        total_energy = kinetic_energy + potential_energy
        fig.add_trace(
            go.Scatter(x=time_points, y=kinetic_energy, name='KE',
                      line=dict(color='#d62728', width=2)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=potential_energy, name='PE',
                      line=dict(color='#9467bd', width=2)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=total_energy, name='Total',
                      line=dict(color='#17becf', width=2, dash='dash')),
            row=2, col=2
        )

    # Update layout
    fig.update_layout(
        height=900,
        showlegend=True,
        title_text="Wave Simulation Interactive Dashboard",
        template='plotly_white'
    )

    # Update axes labels
    fig.update_xaxes(title_text="Position", row=1, col=1)
    fig.update_yaxes(title_text="Displacement", row=1, col=1)
    fig.update_xaxes(title_text="u", row=1, col=2)
    fig.update_yaxes(title_text="V(u)", row=1, col=2)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    fig.update_yaxes(title_text="Energy", row=2, col=2)

    return fig
