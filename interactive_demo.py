"""
Interactive Demo Script for Wave Simulation.

Generates the full set of interactive (Plotly) artifacts in the DJC design
system: animated wave, 3D space-time surface, potential & force curves,
energy monitor, phase space, and the composite dashboard — all as
self-contained HTML files under ``output/``.

Physics: the calibrated wave-scale Lennard-Jones well (u* = 2 m,
omega0 = 5 rad/s) used by main.py and the dashboard, integrated with the
same Verlet solver via ``wave_solver``.

Usage:
    python interactive_demo.py
"""

import os

import numpy as np

from solver import wave_solver, potential_function, force_function
from solver import saved_frame_velocities
from solver import WAVE_SCALE_K1, WAVE_SCALE_K2, WELL_U_STAR, WELL_OMEGA0
from interactive_visualization import (
    create_animated_wave,
    create_3d_wave_surface,
    create_interactive_potential_force,
    create_energy_monitor,
    create_phase_space,
    create_dashboard_layout,
)
from analysis import analyze_energy_conservation, check_numerical_stability, \
    print_simulation_summary

import djc_theme as t
import djc_plotly as P

# ---------------------------------------------------------------------------
# Simulation parameters — the calibrated wave scale (see main.py)
# ---------------------------------------------------------------------------

C = 1.0            # wave speed (m/s) — tension = c^2 on the unit-density string
DX = 0.1           # grid spacing (m)
X = np.arange(0, 50 + DX, DX)
DT = 0.02          # time step (s) — CFL = 0.2, comfortably stable
TOTAL_TIME = 8.0   # seconds

K1 = WAVE_SCALE_K1
K2 = WAVE_SCALE_K2

# Initial pulse: a Gaussian bump on the well floor (wave_solver adds the u*
# background itself, so u0 here is the displacement *relative* to equilibrium)
AMPLITUDE = 1.0
CENTER = 25.0
WIDTH = 2.0

OUT_DIR = "output"


def main():
    t.banner("INTERACTIVE DEMO", "six DJC-themed Plotly artifacts → output/",
             eyebrow="WAVE FORCE SIMULATOR")

    t.section("simulation")
    t.kv("grid", f"{len(X)} points · dx = {DX:g} m")
    t.kv("time", f"T = {TOTAL_TIME:g} s · dt = {DT:g} s (CFL = {C * DT / DX:.2f})")
    t.kv("well", f"u* = {WELL_U_STAR:g} m · ω₀ = {WELL_OMEGA0:g} rad/s")
    t.kv("k1, k2", f"{K1:.3g}, {K2:.3g}")

    t.section("running simulation")
    u0 = AMPLITUDE * np.exp(-((X - CENTER) ** 2) / (2 * WIDTH ** 2))
    u0_prev = u0.copy()  # rest start
    u_history, time_points, v_history = wave_solver(
        X, u0, u0_prev, C, DT, TOTAL_TIME, K1, K2, return_velocity=True)
    print(t.success(f"{len(time_points)} frames computed"))

    # --- Real energies ------------------------------------------------------
    # Conserved ledger: KE(½v²) + elastic(½c²u_x²) + well(V(u) − V(u*)) on the
    # ABSOLUTE displacement — no |u − u*| reflection (that's a different,
    # non-conserved potential). Velocities come from saved_frame_velocities,
    # honest to the leapfrog recurrence.
    ke = 0.5 * np.mean(v_history ** 2, axis=1)
    pe_elastic = (0.5 * C ** 2 * np.diff(u_history, axis=1) ** 2
                  / DX ** 2).mean(axis=1)
    pe_well = np.mean(potential_function(u_history, K1, K2)
                      - potential_function(WELL_U_STAR, K1, K2), axis=1)
    pe = pe_elastic + pe_well
    total = ke + pe

    # --- Honest summary through the themed console --------------------------
    t.section("analysis")
    energy_stats = analyze_energy_conservation(
        time_points.tolist(),
        list(zip(ke, pe, total)),
        tolerance=0.05,
        plot=False,
    )
    stability = check_numerical_stability(list(u_history))
    print_simulation_summary("Verlet (wave_solver shim)", energy_stats,
                             stability, TOTAL_TIME, len(time_points) - 1)

    # --- Themed interactive artifacts ---------------------------------------
    t.section("interactive artifacts")
    os.makedirs(OUT_DIR, exist_ok=True)

    def save(fig, name, *, offline=False):
        # The first (flagship) artifact embeds plotly.js so it works with no
        # network at all; the rest reference the CDN to stay small.
        P.write_html(fig, os.path.join(OUT_DIR, name),
                     include_plotlyjs=True if offline else "cdn")
        print(t.success(name))

    u_sampled = u_history[::max(1, len(u_history) // 120)]
    t_sampled = time_points[::max(1, len(time_points) // 120)]

    save(create_animated_wave(X, u_sampled, t_sampled, baseline=WELL_U_STAR),
         "animation.html")
    save(create_3d_wave_surface(
        X, u_history[::max(1, len(u_history) // 60)],
        time_points[::max(1, len(time_points) // 60)]), "3d_surface.html")
    u_window = np.linspace(1.5, 4.5, 500)  # the bowl; u->0 is a 1e21 wall
    save(create_interactive_potential_force(
        u_window,
        potential_function(u_window, K1, K2),
        force_function(u_window, K1, K2),
        title=f"Lennard-Jones well — k₁ = {K1:.3g}, k₂ = {K2:.3g}",
    ), "potential_force.html")
    save(create_energy_monitor(time_points, ke, pe, total), "energy_monitor.html")

    center_idx = len(X) // 2
    save(create_phase_space(
        u_history[:, center_idx], velocity[:, center_idx], time_points,
        title=f"Phase space at x = {X[center_idx]:.2f} m",
    ), "phase_space.html")
    # Flagship composite — fully offline-capable
    save(create_dashboard_layout(
        X, u_history, time_points,
        potential_function(u_window, K1, K2),
        force_function(u_window, K1, K2),
        u_window,
        ke, pe,
    ), "dashboard.html", offline=True)

    print()
    print(t.rule(70))
    print(t.style(" ✓ ALL VISUALIZATIONS GENERATED ", "green", bold=True))
    print(t.rule(70))
    print(f"\n{t.style('For real-time parameter controls, run:', 'body')}")
    print(f"  {t.style('python dashboard_app.py', 'cyan', bold=True)}"
          f"{t.style('   → http://localhost:8050', 'muted')}")


if __name__ == "__main__":
    main()
