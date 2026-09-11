"""
Main script to run the wave simulation — DJC Design System edition.

The OOP architecture (String, solvers, analysis) is unchanged; the console and
all generated plots now share the navy/teal aesthetic of dennisjcarroll.com.
"""
import numpy as np
import os

from string_model import String
from solver import CentralDifferenceSolver, RK4Solver, VerletSolver
import visualization
import analysis
import djc_theme as t

OUT_DIR = "output"

# ---------------------------------------------------------------------------
# Lennard-Jones wave-force calibration for meter-scale waves.
#
# The repo's cosmological constants (constants.py) place the LJ repulsive
# wall at ~1e-11 m, so at wave amplitudes of order meters the force always
# collapses (or detonates) the solution. Calibrated instead to the wave
# scale: equilibrium u* = 2 m, soft well omega0 = 5 rad/s, derived from
#   F''(u*) = k2*42/u*^8 - k1*156/u*^14 ... solved for k1, k2 with u* fixed.
# The well shape is identical physics, just expressed at wave scale.
# ---------------------------------------------------------------------------
U_STAR = 2.0      # equilibrium displacement [m]
OMEGA0 = 5.0      # well curvature scale [rad/s]
K2_CAL = OMEGA0**2 * U_STAR**8 / 36.0
K1_CAL = K2_CAL * U_STAR**6 / 2.0


def _cfl_of(string, dt: float) -> float:
    """CFL number c_max·dt/dx for a string/timestep pair."""
    return string.get_max_wave_speed() * dt / string.dx


_EQ_WAVE = r"$\partial^2 u/\partial t^2 = c^2\,\partial^2 u/\partial x^2 + F(u)$"
_EQ_WAVESPEED = r"$c = \sqrt{T/\mu}$"  # fully mathtext — avoids font glyph gaps
_EQ_FORCE = r"$F(u) = 12k_1/u^{13} - 6k_2/u^7$"


def run_basic_simulation() -> None:
    """Run a basic wave simulation with default parameters."""
    t.banner("WAVE FORCE SIMULATOR", "Classical String Mechanics · Lennard-Jones Wave Forces",
             eyebrow="SIMULATION 01 — UNIFORM STRING")

    # Create string
    string = String(
        length=50.0,
        num_points=500,
        tension=50.0,
        density_profile="uniform",
        density_uniform=0.01,
    )

    # Set initial conditions: LJ equilibrium background + a Gaussian
    # disturbance, so the wave-force term acts as an anharmonic well.
    pulse_amplitude, pulse_width = 0.5, 5.0
    string.set_initial_custom(
        displacement_func=lambda x: U_STAR
        + pulse_amplitude * np.exp(-((x - 25.0) ** 2) / (2 * pulse_width**2)),
    )

    # Create solver with the calibrated wave-force constants. Velocity Verlet
    # is symplectic, so it conserves energy far better than central difference
    # when a potential force term is active (~4.6% vs ~6.2% drift over 5 s).
    solver = VerletSolver(string, enable_force=True, k1=K1_CAL, k2=K2_CAL)

    # Solve
    print()
    t.section("Solving wave equation")
    displacement_history = solver.solve(total_time=5.0, dt=0.001, save_interval=10, verbose=False)

    # Visualizations
    t.section("Generating visualizations")

    # Wave evolution — carrying its governing equation and CFL stamp
    visualization.plot_wave_evolution(
        displacement_history,
        string.x,
        time_history=solver.time_history,
        num_snapshots=10,
        save_path=f"{OUT_DIR}/wave_evolution.png",
        footnote=(
            _EQ_WAVE + f"   with {_EQ_FORCE}",
            f"u* = {U_STAR:g} m   $\\omega_0$ = {OMEGA0:g} rad/s   "
            f"$k_1$ = {K1_CAL:.2e}   $k_2$ = {K2_CAL:.2e}",
        ),
        badge=t.cfl_badge(_cfl_of(string, 0.001)),
    )

    # Energy conservation
    energy_stats = analysis.analyze_energy_conservation(
        solver.time_history, solver.energy_history, plot=True,
        save_path=f"{OUT_DIR}/energy_conservation.png",
    )

    # Phase space
    analysis.plot_phase_space(
        solver.displacement_history, solver.velocity_history, node_index=250,
        save_path=f"{OUT_DIR}/phase_space.png",
    )

    # Heatmap — the same PDE, now bound to the space-time view
    analysis.plot_spacetime_heatmap(
        solver.displacement_history, string.x, solver.time_history,
        save_path=f"{OUT_DIR}/spacetime_heatmap.png",
        footnote=(
            _EQ_WAVE,
            _EQ_WAVESPEED + f"   with  T = 50 N,  $\\mu$ = 0.01 kg/m  $\\rightarrow$  c = {string.get_max_wave_speed():.1f} m/s",
        ),
        badge=t.cfl_badge(_cfl_of(string, 0.001)),
    )

    # Print summary
    stability_stats = analysis.check_numerical_stability(solver.displacement_history)
    analysis.print_simulation_summary("Velocity Verlet", energy_stats, stability_stats, 5.0, 500)

    # Interactive HTML report — the shareable artifact for this run
    import report
    data = report.capture_run(
        "Wave Propagation — Uniform String",
        string.x, solver.time_history, solver.displacement_history,
        energy_history=solver.energy_history,
        cfl=_cfl_of(string, 0.001),
        subtitle="Velocity-Verlet · LJ well (u* = 2 m, ω₀ = 5 rad/s) · scrub, hover, click the heatmap",
        k1=K1_CAL, k2=K2_CAL,
        pulse_amplitude=pulse_amplitude, pulse_width=pulse_width,
        pulse_speed=string.get_max_wave_speed(),
    )
    report.write_report(data, f"{OUT_DIR}/wave_report.html")

    # Styled energy verdict
    t.section("Energy conservation")
    drift_pct = energy_stats["relative_drift"] * 100
    t.kv("Initial energy", f"{energy_stats['initial_energy']:.4e} J")
    t.kv("Final energy", f"{energy_stats['final_energy']:.4e} J")
    t.kv("Relative drift", f"{drift_pct:.3f} %")
    print(f"    {t.meter(min(drift_pct / 5.0, 1.0))}  "
          + (t.success("within tolerance") if energy_stats["is_conserved"] else t.error("exceeds tolerance")))
    print()


def run_whip_simulation() -> None:
    """Run a whip crack simulation with tapered density."""
    t.banner("WHIP CRACK SIMULATOR", "Tapered Density · Supersonic Tip Dynamics",
             eyebrow="SIMULATION 02 — TAPERED WHIP")

    # Create tapered string (whip). The tip must be a FREE end — a fixed tip
    # has its velocity zeroed every step, so the whip could never crack.
    whip = String(
        length=100.0,
        num_points=1000,
        tension=100.0,
        density_profile="tapered",
        density_base=0.02,
        density_tip=0.002,
        taper_exponent=1.5,
        boundary_right="free",
    )

    # Set initial pulse at thick end
    whip.set_initial_pulse(position=10.0, amplitude=2.0, width=3.0)

    # Plot density profile
    visualization.plot_density_profile(
        whip.x, whip.density, whip.wave_speed, save_path=f"{OUT_DIR}/whip_density_profile.png"
    )

    # Create solver - use Verlet for better energy conservation
    solver = VerletSolver(whip, enable_force=False)  # Disable external force for pure whip dynamics

    # Solve with a timestep that satisfies the CFL condition for the tapered
    # whip (c_max*dt/dx < 1 requires dt < ~0.00045) — 0.001 used to abort the run.
    t.section("Solving whip dynamics")
    displacement_history = solver.solve(total_time=2.0, dt=0.0004, save_interval=25, verbose=False)

    # Visualizations
    t.section("Analyzing the crack")
    visualization.plot_wave_evolution(
        displacement_history,
        whip.x,
        time_history=solver.time_history,
        num_snapshots=10,
        save_path=f"{OUT_DIR}/whip_evolution.png",
    )

    # Tip velocity analysis
    tip_stats = analysis.plot_tip_velocity(
        solver.time_history, solver.velocity_history, sound_speed=343.0,
        save_path=f"{OUT_DIR}/whip_tip_velocity.png",
    )

    t.section("Whip tip velocity")
    mach = tip_stats["mach_number"]
    t.kv("Max tip velocity", f"{tip_stats['max_tip_velocity']:.2f} m/s")
    t.kv("Mach number", f"{mach:.3f}", color="amber" if mach >= 1.0 else "white")
    t.kv("Supersonic", "YES" if tip_stats["is_supersonic"] else "no",
         color="amber" if tip_stats["is_supersonic"] else "muted")
    t.kv("Time of max velocity", f"{tip_stats['max_tip_velocity_time']:.4f} s")
    print(f"    {t.meter(min(mach / 2.0, 1.0))}  Mach {mach:.2f} of 2.0 scale")
    print()

    # Energy conservation
    energy_stats = analysis.analyze_energy_conservation(
        solver.time_history, solver.energy_history, plot=True,
        save_path=f"{OUT_DIR}/whip_energy.png",
    )


def run_solver_comparison() -> None:
    """Compare different numerical integrators."""
    t.banner("SOLVER SHOWDOWN", "Central Difference · RK4 · Velocity Verlet",
             eyebrow="SIMULATION 03 — INTEGRATORS")

    # Create string
    string_base = String(length=50.0, num_points=500, tension=50.0, density_profile="uniform")
    string_base.set_initial_gaussian(center=25.0, width=5.0, amplitude=1.0)

    solvers = {
        "Central Difference": CentralDifferenceSolver,
        "RK4": RK4Solver,
        "Verlet": VerletSolver,
    }

    results = {}
    drift_by_solver = {}

    for solver_name, SolverClass in solvers.items():
        print(f"\n  {t.style('▸ ' + solver_name, 'teal_bright', bold=True)}")

        # Create fresh string for each solver
        string = String(length=50.0, num_points=500, tension=50.0, density_profile="uniform")
        string.set_initial_gaussian(center=25.0, width=5.0, amplitude=1.0)

        solver_instance = SolverClass(string, enable_force=True, k1=K1_CAL, k2=K2_CAL)
        displacement_history = solver_instance.solve(total_time=2.0, dt=0.001, save_interval=20, verbose=False)
        results[solver_name] = displacement_history

        # Quick energy analysis
        energy_stats = analysis.analyze_energy_conservation(
            solver_instance.time_history, solver_instance.energy_history, plot=False
        )
        drift_pct = energy_stats["relative_drift"] * 100
        drift_by_solver[solver_name] = drift_pct
        print(f"    {t.style('energy drift', 'muted')}  {t.style(f'{drift_pct:8.3f} %', 'white')}")

    # Create comparison plot — annotated with the discretization each solver integrates
    visualization.create_comparison_plot(
        results, string_base.x, time_index=-1,
        save_path=f"{OUT_DIR}/solver_comparison.png",
        footnote=(
            _EQ_WAVE,
            "all three integrate the same PDE — only the time discretization differs",
        ),
    )

    # Drift ranking
    t.section("Energy drift ranking")
    for name, drift in sorted(drift_by_solver.items(), key=lambda kv: kv[1]):
        print(f"    {t.style(f'{name:<20}', 'body')} {t.meter(min(drift / 1.0, 1.0))} {t.style(f'{drift:7.3f} %', 'muted')}")
    print()


def main() -> None:
    """Main entry point - run all demonstrations."""
    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    print()
    print(t.rule(72))
    print(f"  {t.style('◆ DJC DESIGN SYSTEM', 'teal_bright', bold=True)}")
    spaced = " ".join("WAVE SIMULATOR")
    print(f"  {t.style(spaced, 'cyan', bold=True)}")
    print(f"  {t.style('Classical String Mechanics & Whip Forces', 'body', dim=True)}")
    print(t.rule(72))

    # Run demonstrations
    try:
        run_basic_simulation()
        run_whip_simulation()
        run_solver_comparison()

        # Generate potential and force plots
        t.section("Potential & force functions")
        visualization.plot_potential_energy(k1=K1_CAL, k2=K2_CAL,
                                             save_path=f"{OUT_DIR}/potential_function.png")
        visualization.plot_force_function(k1=K1_CAL, k2=K2_CAL,
                                           save_path=f"{OUT_DIR}/force_function.png")

        print()
        print(t.rule(72))
        print(f"  {t.style('◆ RUN COMPLETE', 'green', bold=True)}")
        print(f"  {t.style('All simulations finished successfully.', 'body')}")
        print()
        print(f"  {t.style('Output files:', 'muted')}")
        print(f"    {t.style(OUT_DIR + '/', 'teal')}  plots & visualizations")
        print(f"    {t.style('docs/', 'teal')}     documentation")
        print()
        print(t.rule(72))
        print()

    except Exception as e:
        print()
        print(t.error(f"Error during simulation: {e}"))
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
