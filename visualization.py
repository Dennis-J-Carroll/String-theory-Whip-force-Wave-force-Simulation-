"""
Visualization for the wave simulation — DJC Design System edition.

Every plot renders on the deep-navy canvas with teal/cyan glow accents from
``djc_theme``. Figures are saved and closed (never blocking), so full runs
complete without dismissing windows.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Optional, List

import djc_theme as t

t.apply()  # idempotent; guarantees theme when imported standalone

__all__ = [
    "plot_wave_evolution", "animate_wave", "plot_potential_energy",
    "plot_force_function", "plot_density_profile", "create_comparison_plot",
]


def _glow(ax, x, y, color, lw=2.0, label=None, zorder=3, **kwargs):
    return t.glow_line(ax, x, y, color=color, lw=lw, label=label, zorder=zorder, **kwargs)


def plot_wave_evolution(
    u: np.ndarray,
    x: np.ndarray,
    dt: float = None,
    time_history: List[float] = None,
    num_snapshots: int = 10,
    save_path: Optional[str] = None,
    show: bool = False,
    footnote: tuple = (),
    badge: str = None,
) -> None:
    """
    Plot the wave displacement at different time steps.

    Early snapshots render in dim slate (the past); recent ones advance through
    the teal ramp into glowing electric cyan (the present).
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    num_times = u.shape[0]
    step = max(1, num_times // num_snapshots)
    snap_indices = list(range(0, num_times, step))[:num_snapshots]

    n = len(snap_indices)
    colors = [plt.get_cmap("djc_time")(i / max(n - 1, 1)) for i in range(n)]

    for idx, i in enumerate(snap_indices):
        if time_history is not None:
            time_label = f"t={time_history[i]:.2f}"
        elif dt is not None:
            time_label = f"t={i*dt:.2f}"
        else:
            time_label = f"step {i}"
        _glow(ax, x, u[i, :], colors[idx], lw=1.9, label=time_label,
              zorder=3 + idx * 0.1)

    ax.set_xlabel("Position  x  [m]")
    ax.set_ylabel("Displacement  u(x, t)")
    ax.set_title("Wave Propagation Evolution", fontfamily=t.FONT_DISPLAY, fontsize=13)
    t.glass_legend(ax, loc="upper right", ncol=2, fontsize=9)
    if footnote or badge:
        t.equation_footnote(fig, *footnote, badge=badge)

    t.finish_figure(fig, save_path, show=show, print_label="wave evolution")


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
    Create an animated visualization of wave propagation (GIF export).

    The traveling pulse glows cyan; the optional energy panel shows kinetic,
    potential and total energy with a live marker.
    """
    if len(displacement_history) == 0:
        raise ValueError("Empty displacement history")

    if show_energy and energy_history is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    fig.subplots_adjust(hspace=0.35)

    all_displacements = np.array(displacement_history)
    y_min = np.min(all_displacements) * 1.1
    y_max = np.max(all_displacements) * 1.1

    (line,) = ax1.plot([], [], color=t.ACCENT.CYAN, linewidth=2.2)
    (trail,) = ax1.plot([], [], color=t.ACCENT.CYAN, alpha=0.25, linewidth=5)
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(min(y_min, -1e-9), max(y_max, 1e-9))
    ax1.set_xlabel("Position  x  [m]")
    ax1.set_ylabel("Displacement  u(x, t)")
    time_text = ax1.text(
        0.02, 0.95, "", transform=ax1.transAxes, fontsize=12,
        verticalalignment="top", color=t.FG_1, family=t.FONT_MONO,
        bbox=dict(boxstyle="round,pad=0.45", facecolor=t.GLASS_BG, edgecolor="#2b3f58"),
    )

    energy_times = None
    total = None
    (time_marker,) = ax1.plot([], [])  # placeholder if energy panel absent
    if show_energy and energy_history is not None:
        ke = [e[0] for e in energy_history]
        pe = [e[1] for e in energy_history]
        total = [e[2] for e in energy_history]
        energy_times = time_history[: len(energy_history)]

        ax2.plot(energy_times, ke, color=t.ACCENT.VIOLET, alpha=0.75, label="Kinetic")
        ax2.plot(energy_times, pe, color=t.TEAL.N300, alpha=0.75, label="Potential")
        t.glow_line(ax2, energy_times, total, color=t.ACCENT.CYAN, lw=1.8, label="Total")
        (time_marker,) = ax2.plot([], [], "o", color=t.ACCENT.CYAN, markersize=7)

        ax2.set_xlabel("Time")
        ax2.set_ylabel("Energy")
        t.glass_legend(ax2, loc="upper right", fontsize=9)

    def init():
        line.set_data([], [])
        trail.set_data([], [])
        time_text.set_text("")
        return (line, trail, time_text, time_marker)

    def animate(frame):
        idx = min(frame * skip_frames, len(displacement_history) - 1)
        line.set_data(x, displacement_history[idx])
        trail.set_data(x, displacement_history[idx])
        time_text.set_text(f" t = {time_history[idx]:.3f} s ")
        ax1.set_title(f"Wave Propagation — frame {idx}/{len(displacement_history)-1}",
                      fontfamily=t.FONT_DISPLAY, fontsize=12)
        if show_energy and energy_history is not None:
            if idx < len(energy_history):
                time_marker.set_data([energy_times[idx]], [total[idx]])
            return line, trail, time_text, time_marker
        return (line, trail, time_text, time_marker)

    num_frames = len(displacement_history) // skip_frames
    anim = FuncAnimation(fig, animate, init_func=init, frames=num_frames,
                         interval=1000 / fps, blit=True)

    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, savefig_kwargs={"facecolor": t.BG_BASE})
        print(t.success(f"animation → {save_path}"))
    else:
        plt.show()

    plt.close(fig)


def plot_potential_energy(u_min=0.1, u_max=5, num_points=500, k1=None, k2=None,
                          save_path=None, show: bool = False) -> None:
    """Plot the Lennard-Jones-like potential with glow shading."""
    import solver

    u_values = np.linspace(u_min, u_max, num_points)
    potential_energy = solver.potential_function(u_values, k1, k2)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(u_values, potential_energy, color=t.ACCENT.CYAN, lw=2.5, zorder=4)
    ax.fill_between(u_values, potential_energy, potential_energy.min(),
                    color=t.TEAL.N500, alpha=0.10, zorder=2)

    min_idx = np.argmin(potential_energy)
    ax.plot(u_values[min_idx], potential_energy[min_idx], "o", color=t.ACCENT.CYAN,
            markersize=9, mec=t.TEAL.N50, mew=1.5, label="Equilibrium", zorder=5)
    ax.set_xlabel("Displacement  u")
    ax.set_ylabel("V(u)")
    ax.set_title("Potential Energy Function", fontfamily=t.FONT_DISPLAY)
    t.glass_legend(ax, loc="upper right")

    t.finish_figure(fig, save_path, show=show, print_label="potential function")


def plot_force_function(u_min=0.1, u_max=5, num_points=500, k1=None, k2=None,
                        save_path=None, show: bool = False) -> None:
    """Plot the force derived from the potential; equilibrium points in green."""
    import solver

    u_values = np.linspace(u_min, u_max, num_points)
    force = solver.force_function(u_values, k1, k2)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(u_values, force, color=t.ACCENT.VIOLET, lw=2.5, zorder=4)
    ax.axhline(0, color="#3a5270", lw=1)
    ax.axvline(0, color="#3a5270", lw=1)

    zero_crossings = np.where(np.diff(np.sign(force)))[0]
    if len(zero_crossings) > 0:
        ax.plot(u_values[zero_crossings], 0, "o", color=t.ACCENT.GREEN, markersize=9,
                mec=t.TEAL.N50, mew=1.5, ls="none", label="Zero Force (Equilibrium)")
        t.glass_legend(ax, loc="upper right")

    ax.set_xlabel("Displacement  u")
    ax.set_ylabel("F(u) = −dV/du")
    ax.set_title("Force Function", fontfamily=t.FONT_DISPLAY)

    t.finish_figure(fig, save_path, show=show, print_label="force function")


def plot_density_profile(
    x: np.ndarray, density: np.ndarray, wave_speed: np.ndarray = None,
    save_path: Optional[str] = None, show: bool = False,
) -> None:
    """Density and wave-speed profiles for the whip (μ(x), c(x))."""
    if wave_speed is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

    _glow(ax1, x, density, t.TEAL.N500, lw=2.4)
    ax1.fill_between(x, density, color=t.TEAL.N500, alpha=0.12)
    ax1.set_xlabel("Position  x  [m]")
    ax1.set_ylabel("Linear density  μ(x)  [kg/m]")
    ax1.set_title("Density Profile Along the Whip", fontfamily=t.FONT_DISPLAY)

    if wave_speed is not None:
        _glow(ax2, x, wave_speed, t.ACCENT.CYAN, lw=2.4)
        ax2.fill_between(x, wave_speed, color=t.ACCENT.CYAN, alpha=0.10)
        ax2.set_xlabel("Position  x  [m]")
        ax2.set_ylabel("Wave speed  c(x)  [m/s]")
        ax2.set_title("Wave Speed Profile  (c = √(T/μ))", fontfamily=t.FONT_DISPLAY)

    try:
        fig.tight_layout()
    except Exception:
        pass
    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(t.success(f"density profile → {save_path}"))
    if show:
        plt.show()
    plt.close(fig)


def create_comparison_plot(
    results_dict: dict, x: np.ndarray, time_index: int = -1,
    save_path: Optional[str] = None, show: bool = False,
    footnote: tuple = (), badge: str = None,
) -> None:
    """Compare solver outputs, one glow line per integrator."""
    fig, ax = plt.subplots(figsize=(12, 7))
    series_colors = {
        "Central Difference": t.TEAL.N500,
        "RK4": t.ACCENT.VIOLET,
        "Verlet": t.ACCENT.CYAN,
    }
    for solver_name, displacement_history in results_dict.items():
        _glow(ax, x, displacement_history[time_index],
              series_colors.get(solver_name, t.NAVY.N300), lw=2.0, label=solver_name)

    ax.set_xlabel("Position  x  [m]")
    ax.set_ylabel("Displacement  u(x, t)")
    ax.set_title(f"Solver Comparison — t-index {time_index}", fontfamily=t.FONT_DISPLAY)
    t.glass_legend(ax, loc="best")
    if footnote or badge:
        t.equation_footnote(fig, *footnote, badge=badge)

    t.finish_figure(fig, save_path, show=show, print_label="solver comparison")
