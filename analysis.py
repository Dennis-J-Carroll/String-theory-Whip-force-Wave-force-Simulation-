"""
Analysis tools for wave simulations.

Provides functions for energy tracking, phase space analysis,
and physical validation of simulation results.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import solver

import djc_theme as t

t.apply()  # idempotent; guarantees theme when imported standalone


def analyze_energy_conservation(
    time_history: List[float],
    energy_history: List[Tuple[float, float, float]],
    tolerance: float = 0.05,
    plot: bool = True,
    save_path: Optional[str] = None,
    footnote: tuple = (),
    badge: str = None,
) -> dict:
    """
    Analyze energy conservation in the simulation.

    Args:
        time_history: List of time values
        energy_history: List of (KE, PE, Total) tuples
        tolerance: Maximum allowed relative energy drift
        plot: Whether to generate plot
        save_path: Path to save the plot

    Returns:
        Dictionary with energy statistics
    """
    if len(energy_history) == 0:
        raise ValueError("No energy history provided")

    # Extract energy components
    ke = np.array([e[0] for e in energy_history])
    pe = np.array([e[1] for e in energy_history])
    total = np.array([e[2] for e in energy_history])
    time = np.array(time_history[: len(energy_history)])

    # Calculate statistics
    initial_energy = total[0]
    final_energy = total[-1]
    energy_drift = final_energy - initial_energy
    relative_drift = abs(energy_drift / initial_energy) if initial_energy != 0 else float("inf")

    max_energy = np.max(total)
    min_energy = np.min(total)
    energy_range = max_energy - min_energy
    relative_range = energy_range / initial_energy if initial_energy != 0 else float("inf")

    is_conserved = relative_drift < tolerance

    stats = {
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "energy_drift": energy_drift,
        "relative_drift": relative_drift,
        "max_energy": max_energy,
        "min_energy": min_energy,
        "energy_range": energy_range,
        "relative_range": relative_range,
        "is_conserved": is_conserved,
        "tolerance": tolerance,
    }

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})
        fig.subplots_adjust(hspace=0.38)

        # Panel 1: energy exchange over time
        t.glow_line(ax1, time, ke, color=t.ACCENT.VIOLET, lw=1.8, label="Kinetic")
        t.glow_line(ax1, time, pe, color=t.TEAL.N300, lw=1.8, label="Potential")
        t.glow_line(ax1, time, total, color=t.ACCENT.CYAN, lw=2.4, label="Total")
        ax1.set_xlabel("Time  [s]")
        ax1.set_ylabel("Energy")
        ax1.set_title("Energy Evolution", fontfamily=t.FONT_DISPLAY)
        t.glass_legend(ax1, loc="best", fontsize=9)

        # Panel 2: relative drift inside the tolerance band
        relative_total = (total - initial_energy) / initial_energy * 100
        ax2.fill_between(time, -tolerance * 100, tolerance * 100,
                         color=t.ACCENT.GREEN, alpha=0.10, label=f"Tolerance ±{tolerance*100:.1f}%")
        t.glow_line(ax2, time, relative_total, color=t.ACCENT.CYAN, lw=1.6)
        ax2.set_xlabel("Time  [s]")
        ax2.set_ylabel("Relative Drift (%)")
        drift_ok = relative_drift < tolerance
        ax2.set_title(
            f"Energy Conservation — drift {relative_drift*100:.3f}%",
            fontfamily=t.FONT_DISPLAY,
            color=t.ACCENT.GREEN if drift_ok else t.ACCENT.RED,
        )
        t.glass_legend(ax2, loc="upper right", fontsize=9)

        if footnote or badge:
            t.equation_footnote(fig, *footnote, badge=badge)

        if save_path:
            fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
            print(t.success(f"energy plot → {save_path}"))
        plt.close(fig)

    return stats


def plot_phase_space(
    displacement_history: List[np.ndarray],
    velocity_history: List[np.ndarray],
    node_index: int = None,
    num_nodes: int = 3,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot phase space trajectories (displacement vs velocity).

    Args:
        displacement_history: List of displacement arrays
        velocity_history: List of velocity arrays
        node_index: Specific node to plot (if None, plots multiple nodes)
        num_nodes: Number of nodes to plot if node_index is None
        save_path: Path to save the plot
    """
    if len(displacement_history) == 0 or len(velocity_history) == 0:
        raise ValueError("Empty history provided")

    fig, ax = plt.subplots(figsize=(8, 8))

    if node_index is not None:
        # Trajectory of a single node
        u = [disp[node_index] for disp in displacement_history]
        v = [vel[node_index] for vel in velocity_history]
        t.glow_line(ax, u, v, color=t.ACCENT.CYAN, lw=1.6)
        ax.scatter(u[0], v[0], c=t.ACCENT.GREEN, s=110, marker="o",
                   edgecolors=t.TEAL.N50, linewidths=1.2, label="Start", zorder=5)
        ax.scatter(u[-1], v[-1], c=t.ACCENT.RED, s=110, marker="x",
                   linewidths=2.2, label="End", zorder=5)
        ax.set_title(f"Phase Space Trajectory — Node {node_index}", fontfamily=t.FONT_DISPLAY)
    else:
        # Trajectories of several nodes, colored along the teal time ramp
        num_points = len(displacement_history[0])
        indices = np.linspace(0, num_points - 1, num_nodes, dtype=int)
        cmap = plt.get_cmap("djc_time")

        for j, idx in enumerate(indices):
            u = [disp[idx] for disp in displacement_history]
            v = [vel[idx] for vel in velocity_history]
            t.glow_line(ax, u, v, color=cmap(j / max(num_nodes - 1, 1)), lw=1.6,
                        label=f"Node {idx}")

        ax.set_title(f"Phase Space Trajectories — {num_nodes} Nodes", fontfamily=t.FONT_DISPLAY)
        t.glass_legend(ax, loc="best", fontsize=9)

    ax.set_xlabel("Displacement  u(x, t)")
    ax.set_ylabel("Velocity  ∂u/∂t")
    ax.axhline(y=0, color="#3a5270", linewidth=1)
    ax.axvline(x=0, color="#3a5270", linewidth=1)

    try:
        fig.tight_layout()
    except Exception:
        pass

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(t.success(f"phase space → {save_path}"))

    plt.close(fig)


def plot_spacetime_heatmap(
    displacement_history: List[np.ndarray],
    x: np.ndarray,
    time_history: List[float],
    colormap: str = "djc_diverging",
    save_path: Optional[str] = None,
    footnote: tuple = (),
    badge: str = None,
) -> None:
    """
    Create a space-time heatmap showing wave evolution.

    Args:
        displacement_history: List of displacement arrays
        x: Spatial grid points
        time_history: List of time values
        colormap: Matplotlib colormap name
        save_path: Path to save the plot
    """
    if len(displacement_history) == 0:
        raise ValueError("Empty displacement history")

    # Create 2D array: time x space
    data = np.array(displacement_history)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Space-time heatmap — violet (negative) / navy (zero) / cyan (positive)
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], time_history[0], time_history[-1]],
        cmap=colormap,
        interpolation="bilinear",
    )

    ax.set_xlabel("Position  x  [m]")
    ax.set_ylabel("Time  t  [s]")
    ax.set_title("Space-Time Evolution of Wave Displacement", fontfamily=t.FONT_DISPLAY)
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("Displacement  u(x, t)", color=t.FG_2)
    cbar.ax.yaxis.set_tick_params(color=t.FG_3)
    plt.setp(cbar.ax.get_yticklabels(), color=t.FG_3)
    cbar.outline.set_edgecolor("#2b3f58")

    if footnote or badge:
        t.equation_footnote(fig, *footnote, badge=badge)

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(t.success(f"heatmap → {save_path}"))

    plt.close(fig)


def plot_tip_velocity(
    time_history: List[float],
    velocity_history: List[np.ndarray],
    sound_speed: float = 343.0,
    save_path: Optional[str] = None,
) -> dict:
    """
    Plot the velocity at the tip of the string (for whip simulations).

    Args:
        time_history: List of time values
        velocity_history: List of velocity arrays
        sound_speed: Speed of sound for reference (m/s)
        save_path: Path to save the plot

    Returns:
        Dictionary with tip velocity statistics
    """
    if len(velocity_history) == 0:
        raise ValueError("Empty velocity history")

    # Extract tip velocities (last spatial point)
    tip_velocities = np.array([v[-1] for v in velocity_history])
    time = np.array(time_history[: len(velocity_history)])

    # Statistics
    max_tip_velocity = np.max(np.abs(tip_velocities))
    max_tip_velocity_time = time[np.argmax(np.abs(tip_velocities))]
    is_supersonic = max_tip_velocity > sound_speed

    stats = {
        "max_tip_velocity": max_tip_velocity,
        "max_tip_velocity_time": max_tip_velocity_time,
        "sound_speed": sound_speed,
        "is_supersonic": is_supersonic,
        "mach_number": max_tip_velocity / sound_speed,
    }

    # Plot — tip velocity with the sonic barrier drawn as a red danger line
    fig, ax = plt.subplots(figsize=(10, 6))

    t.glow_line(ax, time, tip_velocities, color=t.ACCENT.CYAN, lw=2.2, label="Tip Velocity")
    ax.axhline(y=sound_speed, color=t.ACCENT.RED, ls="--", lw=1.6, alpha=0.9,
               label=f"Sonic barrier ({sound_speed:.0f} m/s)")
    ax.axhline(y=-sound_speed, color=t.ACCENT.RED, ls="--", lw=1.6, alpha=0.9)

    ax.set_xlabel("Time  [s]")
    ax.set_ylabel("Tip Velocity  [m/s]")
    ax.set_title(
        f"Whip Tip Velocity — Mach {stats['mach_number']:.2f}"
        f"{'  —  SUPERSONIC' if is_supersonic else ''}",
        fontfamily=t.FONT_DISPLAY,
    )
    t.glass_legend(ax, loc="upper right", fontsize=9)

    # Mark the moment of the crack
    v_peak = tip_velocities[np.argmax(np.abs(tip_velocities))]
    ax.plot(max_tip_velocity_time, v_peak, "o", color=t.ACCENT.AMBER, markersize=9,
            mec=t.TEAL.N50, mew=1.4, label=f"Crack @ t={max_tip_velocity_time:.3f} s", zorder=5)

    try:
        fig.tight_layout()
    except Exception:
        pass

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(t.success(f"tip velocity → {save_path}"))

    plt.close(fig)

    return stats


def check_numerical_stability(displacement_history: List[np.ndarray], threshold: float = 1e6) -> dict:
    """
    Check for numerical instabilities (NaN, Inf, explosion).

    Args:
        displacement_history: List of displacement arrays
        threshold: Maximum allowed displacement

    Returns:
        Dictionary with stability information
    """
    has_nan = False
    has_inf = False
    has_explosion = False
    first_nan_step = None
    first_inf_step = None
    first_explosion_step = None

    for i, disp in enumerate(displacement_history):
        if np.any(np.isnan(disp)):
            has_nan = True
            if first_nan_step is None:
                first_nan_step = i

        if np.any(np.isinf(disp)):
            has_inf = True
            if first_inf_step is None:
                first_inf_step = i

        if np.max(np.abs(disp)) > threshold:
            has_explosion = True
            if first_explosion_step is None:
                first_explosion_step = i

    is_stable = not (has_nan or has_inf or has_explosion)

    return {
        "is_stable": is_stable,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "has_explosion": has_explosion,
        "first_nan_step": first_nan_step,
        "first_inf_step": first_inf_step,
        "first_explosion_step": first_explosion_step,
        "threshold": threshold,
    }


def print_simulation_summary(
    solver_name: str,
    energy_stats: dict,
    stability_stats: dict,
    total_time: float,
    num_steps: int,
) -> None:
    """
    Print a formatted summary of simulation results.

    Args:
        solver_name: Name of the solver used
        energy_stats: Dictionary from analyze_energy_conservation
        stability_stats: Dictionary from check_numerical_stability
        total_time: Total simulation time
        num_steps: Number of time steps
    """
    print()
    print(t.rule(70))
    print(f"  {t.style('SIMULATION SUMMARY', 'cyan', bold=True)}{t.style('  ·  ' + solver_name, 'muted')}")
    print(t.rule(70))

    print(f"\nSimulation Parameters:")
    print(f"  Total Time:        {total_time:.4f}")
    print(f"  Number of Steps:   {num_steps}")
    print(f"  Time Step:         {total_time / num_steps:.6f}")

    print(f"\nEnergy Conservation:")
    print(f"  Initial Energy:    {energy_stats['initial_energy']:.6e}")
    print(f"  Final Energy:      {energy_stats['final_energy']:.6e}")
    print(f"  Relative Drift:    {energy_stats['relative_drift']*100:.4f}%")
    print(f"  Tolerance:         {energy_stats['tolerance']*100:.1f}%")
    status = t.success("CONSERVED") if energy_stats["is_conserved"] else t.error("NOT CONSERVED")
    print(f"  Status:            {status}")

    print(f"\nNumerical Stability:")
    stable = t.success("YES") if stability_stats["is_stable"] else t.error("NO")
    print(f"  Stable:            {stable}")
    if stability_stats["has_nan"]:
        print(f"  NaN detected:      Step {stability_stats['first_nan_step']}")
    if stability_stats["has_inf"]:
        print(f"  Inf detected:      Step {stability_stats['first_inf_step']}")
    if stability_stats["has_explosion"]:
        print(f"  Explosion:         Step {stability_stats['first_explosion_step']}")

    print()
    print(t.rule(70))
    print()
