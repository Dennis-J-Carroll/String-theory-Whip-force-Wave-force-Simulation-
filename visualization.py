"""
Visualization functions for the wave simulation.

Provides comprehensive visualization capabilities including:
- Wave evolution plots
- Animated GIF generation
- Potential and force function plots
- Integration with analysis module
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import constants as const
import solver
from typing import Optional, List


def plot_wave_evolution(
    u: np.ndarray,
    x: np.ndarray,
    dt: float = None,
    time_history: List[float] = None,
    num_snapshots: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the wave displacement at different time steps.

    Parameters:
        u (numpy.ndarray): Wave displacement data (time x space)
        x (numpy.ndarray): Spatial grid
        dt (float): Time step (optional if time_history provided)
        time_history (List[float]): List of time values (optional)
        num_snapshots (int): Number of time snapshots to show
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 7))

    num_times = u.shape[0]
    # Select exactly num_snapshots evenly spaced time points
    indices = np.linspace(0, num_times - 1, num_snapshots, dtype=int)

    # Use colormap for time progression
    colors = plt.cm.viridis(np.linspace(0, 1, num_snapshots))

    # Limit to exactly num_snapshots
    snapshot_count = 0
    for idx, i in enumerate(range(0, num_times, step)):
        if snapshot_count >= num_snapshots:
            break

        if time_history is not None:
            time_label = f"t={time_history[i]:.2f}"
        elif dt is not None:
            time_label = f"t={i*dt:.2f}"
        else:
            time_label = f"step {i}"

        plt.plot(x, u[i, :], label=time_label, color=colors[snapshot_count], linewidth=2, alpha=0.8)
        snapshot_count += 1

    plt.xlabel("Position (x)", fontsize=12)
    plt.ylabel("Displacement u(x, t)", fontsize=12)
    plt.title("Wave Propagation Evolution", fontsize=14, fontweight="bold")
    plt.legend(loc="best", ncol=2, fontsize=9)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Wave evolution plot saved to {save_path}")

    plt.show()


def animate_wave(
    displacement_history: List[np.ndarray],
    x: np.ndarray,
    time_history: List[float],
    fps: int = 30,
    skip_frames: int = 1,
    save_path: Optional[str] = None,
    show_energy: bool = False,
    energy_history: Optional[List[tuple]] = None,
) -> None:
    """
    Create an animated visualization of wave propagation.

    Parameters:
        displacement_history: List of displacement arrays
        x: Spatial grid
        time_history: List of time values
        fps: Frames per second for animation
        skip_frames: Show every Nth frame (for speed)
        save_path: Path to save animation (must end in .gif)
        show_energy: Whether to show energy subplot
        energy_history: List of (KE, PE, Total) tuples
    """
    if len(displacement_history) == 0:
        raise ValueError("Empty displacement history")

    # Setup figure
    if show_energy and energy_history is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # Determine y-axis limits
    all_displacements = np.array(displacement_history)
    y_min = np.min(all_displacements) * 1.1
    y_max = np.max(all_displacements) * 1.1

    # Initialize wave line
    (line,) = ax1.plot([], [], "b-", linewidth=2)
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel("Position (x)")
    ax1.set_ylabel("Displacement u(x, t)")
    ax1.grid(True, alpha=0.3)
    time_text = ax1.text(0.02, 0.95, "", transform=ax1.transAxes, fontsize=12, verticalalignment="top")

    # Setup energy subplot if requested
    if show_energy and energy_history is not None:
        ke = [e[0] for e in energy_history]
        pe = [e[1] for e in energy_history]
        total = [e[2] for e in energy_history]
        energy_times = time_history[: len(energy_history)]

        ax2.plot(energy_times, ke, "r-", label="Kinetic", alpha=0.6)
        ax2.plot(energy_times, pe, "b-", label="Potential", alpha=0.6)
        ax2.plot(energy_times, total, "k-", label="Total", linewidth=2)
        (time_marker,) = ax2.plot([], [], "ro", markersize=8)

        ax2.set_xlabel("Time")
        ax2.set_ylabel("Energy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    def init():
        line.set_data([], [])
        time_text.set_text("")
        if show_energy and energy_history is not None:
            time_marker.set_data([], [])
            return line, time_text, time_marker
        return (line, time_text)

    def animate(frame):
        idx = frame * skip_frames
        if idx >= len(displacement_history):
            idx = len(displacement_history) - 1

        line.set_data(x, displacement_history[idx])
        time_text.set_text(f"Time: {time_history[idx]:.3f}")
        ax1.set_title(f"Wave Propagation (frame {idx}/{len(displacement_history)-1})")

        if show_energy and energy_history is not None:
            if idx < len(energy_history):
                time_marker.set_data([energy_times[idx]], [total[idx]])
            return line, time_text, time_marker

        return line, time_text

    num_frames = len(displacement_history) // skip_frames

    anim = FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=1000 / fps, blit=True)

    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_potential_energy(u_min=0.1, u_max=5, num_points=500, k1=None, k2=None, save_path=None):
    """
    Plot the potential energy function.

    Parameters:
        u_min (float): Minimum displacement value
        u_max (float): Maximum displacement value
        num_points (int): Number of points for plotting
        k1 (float): Repulsive force constant
        k2 (float): Attractive force constant
        save_path (str, optional): Path to save the plot
    """
    u_values = np.linspace(u_min, u_max, num_points)
    potential_energy = solver.potential_function(u_values, k1, k2)

    plt.figure(figsize=(8, 6))
    plt.plot(u_values, potential_energy, linewidth=2.5)
    plt.xlabel("Displacement u", fontsize=12)
    plt.ylabel("V(u)", fontsize=12)
    plt.title("Potential Energy Function (Lennard-Jones-like)", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="k", linewidth=0.5)

    # Find and mark equilibrium point
    min_idx = np.argmin(potential_energy)
    plt.plot(u_values[min_idx], potential_energy[min_idx], "ro", markersize=10, label="Equilibrium")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Potential energy plot saved to {save_path}")

    plt.show()


def plot_force_function(u_min=0.1, u_max=5, num_points=500, k1=None, k2=None, save_path=None):
    """
    Plot the force function.

    Parameters:
        u_min (float): Minimum displacement value
        u_max (float): Maximum displacement value
        num_points (int): Number of points for plotting
        k1 (float): Repulsive force constant
        k2 (float): Attractive force constant
        save_path (str, optional): Path to save the plot
    """
    u_values = np.linspace(u_min, u_max, num_points)
    force = solver.force_function(u_values, k1, k2)

    plt.figure(figsize=(8, 6))
    plt.plot(u_values, force, linewidth=2.5, color="darkred")
    plt.xlabel("Displacement u", fontsize=12)
    plt.ylabel("F(u) = -dV/du", fontsize=12)
    plt.title("Force Function", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="k", linewidth=0.5)
    plt.axvline(x=0, color="k", linewidth=0.5)

    # Mark zero crossing (equilibrium)
    zero_crossings = np.where(np.diff(np.sign(force)))[0]
    if len(zero_crossings) > 0:
        for zc in zero_crossings:
            plt.plot(u_values[zc], 0, "go", markersize=10)
        plt.plot([], [], "go", markersize=10, label="Zero Force (Equilibrium)")
        plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Force function plot saved to {save_path}")

    plt.show()


def plot_density_profile(
    x: np.ndarray, density: np.ndarray, wave_speed: np.ndarray = None, save_path: Optional[str] = None
) -> None:
    """
    Plot the density profile and wave speed along the string.

    Parameters:
        x: Spatial grid
        density: Linear mass density μ(x)
        wave_speed: Wave speed c(x) (optional)
        save_path: Path to save the plot
    """
    if wave_speed is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

    # Plot density
    ax1.plot(x, density, linewidth=2.5, color="blue")
    ax1.set_xlabel("Position (x)")
    ax1.set_ylabel("Linear Density μ(x) [kg/m]")
    ax1.set_title("Density Profile Along String")
    ax1.grid(True, alpha=0.3)

    # Plot wave speed if provided
    if wave_speed is not None:
        ax2.plot(x, wave_speed, linewidth=2.5, color="red")
        ax2.set_xlabel("Position (x)")
        ax2.set_ylabel("Wave Speed c(x) [m/s]")
        ax2.set_title("Wave Speed Profile (c = √(T/μ))")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Density profile plot saved to {save_path}")

    plt.show()


def create_comparison_plot(
    results_dict: dict, x: np.ndarray, time_index: int = -1, save_path: Optional[str] = None
) -> None:
    """
    Create a comparison plot for multiple solver results.

    Parameters:
        results_dict: Dictionary mapping solver names to displacement history arrays
        x: Spatial grid
        time_index: Which time step to compare
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 7))

    for solver_name, displacement_history in results_dict.items():
        plt.plot(x, displacement_history[time_index], label=solver_name, linewidth=2, alpha=0.8)

    plt.xlabel("Position (x)", fontsize=12)
    plt.ylabel("Displacement u(x, t)", fontsize=12)
    plt.title(f"Solver Comparison at Time Index {time_index}", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison plot saved to {save_path}")

    plt.show()
