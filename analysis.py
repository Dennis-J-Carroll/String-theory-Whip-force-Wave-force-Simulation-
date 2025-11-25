"""
Analysis tools for wave simulations.

Provides functions for energy tracking, phase space analysis,
and physical validation of simulation results.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import solver


def analyze_energy_conservation(
    time_history: List[float],
    energy_history: List[Tuple[float, float, float]],
    tolerance: float = 0.05,
    plot: bool = True,
    save_path: Optional[str] = None,
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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot 1: Energy components over time
        ax1.plot(time, ke, label="Kinetic Energy", linewidth=2)
        ax1.plot(time, pe, label="Potential Energy", linewidth=2)
        ax1.plot(time, total, label="Total Energy", linewidth=2, linestyle="--", color="black")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Energy")
        ax1.set_title("Energy Evolution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Relative energy drift
        relative_total = (total - initial_energy) / initial_energy * 100
        ax2.plot(time, relative_total, linewidth=2, color="red")
        ax2.axhline(y=tolerance * 100, color="orange", linestyle="--", label=f"Tolerance ({tolerance*100:.1f}%)")
        ax2.axhline(y=-tolerance * 100, color="orange", linestyle="--")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Relative Energy Drift (%)")
        ax2.set_title(f"Energy Conservation (Drift: {relative_drift*100:.3f}%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Energy plot saved to {save_path}")

        plt.show()

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
        # Plot single node
        u = [disp[node_index] for disp in displacement_history]
        v = [vel[node_index] for vel in velocity_history]
        ax.plot(u, v, linewidth=1.5, alpha=0.8)
        ax.scatter(u[0], v[0], c="green", s=100, marker="o", label="Start", zorder=5)
        ax.scatter(u[-1], v[-1], c="red", s=100, marker="x", label="End", zorder=5)
        ax.set_title(f"Phase Space Trajectory - Node {node_index}")
    else:
        # Plot multiple nodes
        num_points = len(displacement_history[0])
        indices = np.linspace(0, num_points - 1, num_nodes, dtype=int)

        for idx in indices:
            u = [disp[idx] for disp in displacement_history]
            v = [vel[idx] for vel in velocity_history]
            ax.plot(u, v, linewidth=1.5, alpha=0.7, label=f"Node {idx}")

        ax.set_title(f"Phase Space Trajectories - {num_nodes} Nodes")
        ax.legend()

    ax.set_xlabel("Displacement u(x, t)")
    ax.set_ylabel("Velocity ∂u/∂t")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Phase space plot saved to {save_path}")

    plt.show()


def plot_spacetime_heatmap(
    displacement_history: List[np.ndarray],
    x: np.ndarray,
    time_history: List[float],
    colormap: str = "RdBu_r",
    save_path: Optional[str] = None,
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

    # Create heatmap
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], time_history[0], time_history[-1]],
        cmap=colormap,
        interpolation="bilinear",
    )

    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Time (t)")
    ax.set_title("Space-Time Evolution of Wave Displacement")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Displacement u(x, t)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to {save_path}")

    plt.show()


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

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time, tip_velocities, linewidth=2, label="Tip Velocity")
    ax.axhline(
        y=sound_speed, color="red", linestyle="--", linewidth=2, label=f"Sound Speed ({sound_speed} m/s)"
    )
    ax.axhline(y=-sound_speed, color="red", linestyle="--", linewidth=2)

    ax.set_xlabel("Time")
    ax.set_ylabel("Tip Velocity (m/s)")
    ax.set_title(
        f"Whip Tip Velocity\n"
        f"Max: {max_tip_velocity:.2f} m/s (Mach {stats['mach_number']:.2f}) at t={max_tip_velocity_time:.3f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Highlight supersonic region
    if is_supersonic:
        supersonic_mask = np.abs(tip_velocities) > sound_speed
        ax.fill_between(
            time,
            -sound_speed,
            sound_speed,
            where=supersonic_mask,
            alpha=0.2,
            color="red",
            label="Supersonic",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Tip velocity plot saved to {save_path}")

    plt.show()

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
    print("\n" + "=" * 70)
    print(f"SIMULATION SUMMARY - {solver_name}")
    print("=" * 70)

    print(f"\nSimulation Parameters:")
    print(f"  Total Time:        {total_time:.4f}")
    print(f"  Number of Steps:   {num_steps}")
    print(f"  Time Step:         {total_time / num_steps:.6f}")

    print(f"\nEnergy Conservation:")
    print(f"  Initial Energy:    {energy_stats['initial_energy']:.6e}")
    print(f"  Final Energy:      {energy_stats['final_energy']:.6e}")
    print(f"  Relative Drift:    {energy_stats['relative_drift']*100:.4f}%")
    print(f"  Tolerance:         {energy_stats['tolerance']*100:.1f}%")
    print(f"  Status:            {'✓ CONSERVED' if energy_stats['is_conserved'] else '✗ NOT CONSERVED'}")

    print(f"\nNumerical Stability:")
    print(f"  Stable:            {'✓ YES' if stability_stats['is_stable'] else '✗ NO'}")
    if stability_stats["has_nan"]:
        print(f"  NaN detected:      Step {stability_stats['first_nan_step']}")
    if stability_stats["has_inf"]:
        print(f"  Inf detected:      Step {stability_stats['first_inf_step']}")
    if stability_stats["has_explosion"]:
        print(f"  Explosion:         Step {stability_stats['first_explosion_step']}")

    print("\n" + "=" * 70 + "\n")
